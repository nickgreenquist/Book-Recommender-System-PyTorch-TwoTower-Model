"""
Stage 1 — Base Preprocessing
Outputs data/base_*.parquet.

Usage:
    python main.py preprocess books         # Step 1: filter books → data/base_books.parquet
    python main.py preprocess interactions  # Step 2: stream interactions → base_interactions_raw + vocab/shelf/ts parquets
    python main.py preprocess split         # Step 3: split raw interactions → base_ratings_read, base_ratings_labels
    python main.py preprocess               # Run all steps in order
"""
import gzip
import json
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────────────

MIN_RATINGS_PER_BOOK  = 10_000
MIN_RATINGS_PER_USER  = 20
MAX_RATINGS_PER_USER  = 1_000
MIN_NUM_SHELVES       = 500     # shelf must appear this many times across all corpus books to be kept
PERCENT_READ_HISTORY  = 0.9    # fraction of each user's ratings used as read history


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def _parse_ts(date_str: str):
    """Parse Goodreads date string to unix timestamp. Returns None if unparseable."""
    if not date_str:
        return None
    try:
        # Format: 'Mon Aug 01 13:41:57 -0700 2011'
        parts = date_str.split()
        if len(parts) == 6:
            parts.pop(4)  # remove timezone offset
        clean = ' '.join(parts)
        dt = datetime.strptime(clean, '%a %b %d %H:%M:%S %Y')
        return int(dt.timestamp())
    except Exception:
        return None


def _parse_timestamp_with_source(record: dict):
    """Priority: read_at → date_updated → date_added → None. Returns (ts, field_used)."""
    for field in ['read_at', 'date_updated', 'date_added']:
        ts = _parse_ts(record.get(field, ''))
        if ts is not None:
            return ts, field
    return None, None


# ── Step 1: Books ─────────────────────────────────────────────────────────────

def run_books(data_dir: str = 'data') -> None:
    """
    Stream goodreads_books.json and goodreads_book_genres_initial.json.
    Filter to books with ratings_count >= MIN_RATINGS_PER_BOOK.
    Uses goodreads_book_works.json for original publication year (not edition year).
    Uses goodreads_book_authors.json for primary author name.
    Saves data/base_books.parquet.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Load genres
    print("Loading goodreads_book_genres_initial.json ...")
    genres: dict = {}
    with open(os.path.join(data_dir, 'goodreads_book_genres_initial.json')) as f:
        for line in f:
            rec = json.loads(line)
            genres[rec['book_id']] = rec.get('genres', {})
    print(f"  Loaded genres for {len(genres):,} books")

    # Load original publication years from works (work_id → original_publication_year)
    work_pub_year: dict = {}
    works_path = os.path.join(data_dir, 'goodreads_book_works.json')
    if os.path.exists(works_path):
        print("\nLoading goodreads_book_works.json ...")
        with open(works_path) as f:
            for line in f:
                rec = json.loads(line)
                wid = rec.get('work_id', '')
                yr  = rec.get('original_publication_year', '')
                if wid and yr:
                    work_pub_year[wid] = str(yr)
        print(f"  Loaded original pub years for {len(work_pub_year):,} works")
    else:
        print("\nWarning: goodreads_book_works.json not found — falling back to edition year")

    # Load author names
    author_names: dict = {}
    authors_path = os.path.join(data_dir, 'goodreads_book_authors.json')
    if os.path.exists(authors_path):
        print("\nLoading goodreads_book_authors.json ...")
        with open(authors_path) as f:
            for line in f:
                rec = json.loads(line)
                aid = rec.get('author_id', '')
                name = rec.get('name', '')
                if aid and name:
                    author_names[aid] = name
        print(f"  Loaded names for {len(author_names):,} authors")
    else:
        print("\nWarning: goodreads_book_authors.json not found — primary_author will be empty")

    # Stream books, filter by ratings_count
    print("\nStreaming goodreads_books.json ...")
    total_books = 0
    rows = []

    with open(os.path.join(data_dir, 'goodreads_books.json')) as f:
        for line in tqdm(f, desc="  filtering books"):
            b = json.loads(line)
            bid = b.get('book_id', '')
            if not bid:
                continue
            total_books += 1
            try:
                rc = int(b.get('ratings_count', 0) or 0)
            except (ValueError, TypeError):
                rc = 0
            if rc < MIN_RATINGS_PER_BOOK:
                continue
            genre_votes = genres.get(bid, {})
            genre_list = sorted(genre_votes, key=lambda g: -genre_votes[g]) if genre_votes else ['unknown']
            author_ids = [a['author_id'] for a in b.get('authors', []) if a.get('author_id')]
            primary_author = author_names.get(author_ids[0], '') if author_ids else ''
            # Prefer original publication year from works; fall back to edition year
            wid  = b.get('work_id', '')
            year = work_pub_year.get(wid) or str(b.get('publication_year', '') or '-1')
            # Prefer ISBN-13, fall back to ISBN-10
            isbn = str(b.get('isbn13', '') or b.get('isbn', '') or '').strip()
            rows.append({
                'book_id':          bid,
                'title':            b.get('title', ''),
                'year':             year,
                'genres':           genre_list,
                'author_ids':       author_ids,
                'primary_author':   primary_author,
                'isbn':             isbn,
                'ratings_count':    rc,
                'popular_shelves':  b.get('popular_shelves', []),
            })

    kept = len(rows)
    print(f"\n  Total books in JSON:       {total_books:,}")
    print(f"  Kept (ratings_count≥{MIN_RATINGS_PER_BOOK}): {kept:,}  ({100*kept/total_books:.1f}%)")
    print(f"  Dropped:                   {total_books-kept:,}  ({100*(total_books-kept)/total_books:.1f}%)")

    books_df = pd.DataFrame(rows)
    out_path = os.path.join(data_dir, 'base_books.parquet')
    books_df.to_parquet(out_path, index=False)
    print(f"\n✓ Wrote {out_path}  ({kept:,} books)")


# ── Step 2: Interactions ──────────────────────────────────────────────────────

def run_interactions(data_dir: str = 'data') -> None:
    """
    Two-pass stream over goodreads_interactions_dedup.json.gz.
    Requires data/base_books.parquet from run_books().
    Saves base_interactions_raw, base_vocab, base_book_shelves, base_timestamps.
    """
    base_books_path = os.path.join(data_dir, 'base_books.parquet')
    if not os.path.exists(base_books_path):
        raise FileNotFoundError(
            f"{base_books_path} not found — run 'python main.py preprocess books' first"
        )

    print("Loading base_books.parquet ...")
    books_df = pd.read_parquet(base_books_path)
    # Rebuild books dict for shelf lookups
    books: dict = {row['book_id']: row.to_dict() for _, row in books_df.iterrows()}
    valid_books_from_metadata = set(books.keys())
    print(f"  Corpus from metadata filter: {len(valid_books_from_metadata):,} books")

    path = os.path.join(data_dir, 'goodreads_interactions_dedup.json.gz')

    # ── Pass 1: count users (books already pre-filtered) ──
    print("\nPass 1: counting interactions ...")
    user_counts: dict = defaultdict(int)
    rating_dist: dict = defaultdict(int)
    total_raw = 0

    with gzip.open(path, 'rt') as f:
        for line in tqdm(f, desc="  pass 1"):
            rec = json.loads(line)
            r = rec.get('rating', 0)
            if r > 0 and rec['book_id'] in valid_books_from_metadata:
                total_raw += 1
                user_counts[rec['user_id']] += 1
                rating_dist[r] += 1

    print(f"\n  Rated interactions (corpus books only): {total_raw:,}")
    print(f"  Rating distribution:")
    for star in sorted(rating_dist):
        pct = 100 * rating_dist[star] / total_raw
        print(f"    {star}★: {rating_dist[star]:,}  ({pct:.1f}%)")

    valid_users = {u for u, c in user_counts.items() if c >= MIN_RATINGS_PER_USER}

    print(f"\n  Total users: {len(user_counts):,}  "
          f"kept (≥{MIN_RATINGS_PER_USER} ratings): {len(valid_users):,}  "
          f"({100*len(valid_users)/len(user_counts):.1f}%)")

    # ── Pass 2: filter and collect ──
    print("\nPass 2: filtering and collecting ...")
    rows = []
    ts_source: dict = defaultdict(int)

    with gzip.open(path, 'rt') as f:
        for line in tqdm(f, desc="  pass 2"):
            rec = json.loads(line)
            if rec.get('rating', 0) <= 0:
                continue
            if rec['user_id'] not in valid_users:
                continue
            if rec['book_id'] not in valid_books_from_metadata:
                continue
            ts, field_used = _parse_timestamp_with_source(rec)
            ts_source[field_used] += 1
            if ts is None:
                continue
            rows.append({
                'user_id':   rec['user_id'],
                'book_id':   rec['book_id'],
                'rating':    int(rec['rating']),
                'timestamp': ts,
            })

    total_pass2 = sum(ts_source.values())
    print(f"\n  Collected {len(rows):,} rows from {total_pass2:,} valid candidates")
    print(f"  Timestamp source breakdown:")
    for field in ['read_at', 'date_updated', 'date_added', None]:
        n = ts_source[field]
        label = field if field else 'skipped (no timestamp)'
        print(f"    {label}: {n:,}  ({100*n/total_pass2:.1f}%)")

    df = pd.DataFrame(rows)
    print(f"\n  Final: {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['book_id'].nunique():,} books")
    ratings_per_user = df.groupby('user_id')['rating'].count()
    print(f"  Ratings per user — min: {ratings_per_user.min()}, "
          f"median: {ratings_per_user.median():.0f}, "
          f"mean: {ratings_per_user.mean():.1f}, "
          f"p95: {ratings_per_user.quantile(0.95):.0f}, "
          f"max: {ratings_per_user.max()}")

    top_books = books_df['book_id'].tolist()

    # ── Vocab ──
    print("\n── Building vocabulary ──")
    genres_lookup: dict = {}
    for _, row in books_df.iterrows():
        genre_votes_raw = row.get('genres', [])
        # genres stored as list (sorted by votes); reconstruct as dict with rank-based votes
        genres_lookup[row['book_id']] = {g: len(genre_votes_raw) - i for i, g in enumerate(genre_votes_raw)}
    vocab_df = _build_vocab(top_books, books, genres_lookup)

    # ── Write parquets ──
    # Note: base_books.parquet is NOT overwritten here — it was written by run_books()
    df.to_parquet(os.path.join(data_dir, 'base_interactions_raw.parquet'), index=False)
    vocab_df.to_parquet(os.path.join(data_dir, 'base_vocab.parquet'), index=False)

    print("\n── Building shelf scores parquet ──")
    book_shelves_df = _build_book_shelf_scores(top_books, books, vocab_df)
    book_shelves_df.to_parquet(os.path.join(data_dir, 'base_book_shelves.parquet'), index=False)

    ts_df = pd.DataFrame({'ts_min': [int(df['timestamp'].min())],
                          'ts_max': [int(df['timestamp'].max())]})
    ts_df.to_parquet(os.path.join(data_dir, 'base_timestamps.parquet'), index=False)

    print(f"\n✓ Wrote base_interactions_raw, base_vocab, base_book_shelves, base_timestamps  →  {data_dir}/")


# ── Step 3: Split ─────────────────────────────────────────────────────────────

def run_split(data_dir: str = 'data') -> None:
    """
    Split base_interactions_raw.parquet into read/label sets.
    Requires base_interactions_raw.parquet from run_interactions().
    Rewrites base_ratings_read.parquet and base_ratings_labels.parquet.
    Safe to re-run without re-streaming the gzip.
    """
    raw_path = os.path.join(data_dir, 'base_interactions_raw.parquet')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found — run 'python main.py preprocess interactions' first"
        )
    books_path = os.path.join(data_dir, 'base_books.parquet')
    if not os.path.exists(books_path):
        raise FileNotFoundError(
            f"{books_path} not found — run 'python main.py preprocess books' first"
        )

    print("Loading base_interactions_raw.parquet ...")
    df = pd.read_parquet(raw_path)
    top_books = pd.read_parquet(books_path, columns=['book_id'])['book_id'].tolist()
    print(f"  {len(df):,} interactions, {df['user_id'].nunique():,} users")

    print("\n── Splitting user histories ──")
    read_df, labels_df = _split_user_history(df, top_books)

    read_df.to_parquet(os.path.join(data_dir, 'base_ratings_read.parquet'), index=False)
    labels_df.to_parquet(os.path.join(data_dir, 'base_ratings_labels.parquet'), index=False)
    print(f"\n✓ Wrote base_ratings_read, base_ratings_labels  →  {data_dir}/")


# ── Vocab building ────────────────────────────────────────────────────────────

def _build_vocab(top_books: list, books: dict, genres: dict) -> pd.DataFrame:
    all_genres: set = set()
    for bid in top_books:
        for g in genres.get(bid, {}):
            all_genres.add(g)
    genres_ordered = sorted(all_genres)

    shelf_total: dict = defaultdict(int)
    for bid in top_books:
        for shelf in books.get(bid, {}).get('popular_shelves', []):
            shelf_total[shelf['name']] += int(shelf['count'])
    final_shelves = sorted(s for s, c in shelf_total.items() if c >= MIN_NUM_SHELVES)

    years_seen: set = set()
    for bid in top_books:
        years_seen.add(str(books.get(bid, {}).get('year', '-1')))
    years_ordered = sorted(years_seen)

    all_authors: set = set()
    for bid in top_books:
        for aid in books.get(bid, {}).get('author_ids', []):
            all_authors.add(str(aid))
    # index 0 = unknown/missing; real authors start at 1
    authors_ordered = ['__unknown__'] + sorted(all_authors)

    rows = []
    for i, g in enumerate(genres_ordered):
        rows.append({'type': 'genre', 'index': i, 'value': g, 'extra': ''})
    for i, s in enumerate(final_shelves):
        rows.append({'type': 'shelf', 'index': i, 'value': s, 'extra': ''})
    for i, y in enumerate(years_ordered):
        rows.append({'type': 'year', 'index': i, 'value': y, 'extra': ''})
    for i, a in enumerate(authors_ordered):
        rows.append({'type': 'author', 'index': i, 'value': a, 'extra': ''})

    print(f"  Vocab sizes — genres: {len(genres_ordered)}, shelves: {len(final_shelves)}, "
          f"years: {len(years_ordered)}, authors: {len(authors_ordered)} (incl. __unknown__ at 0)")
    return pd.DataFrame(rows)


# ── User history splitting ────────────────────────────────────────────────────

def _split_user_history(df: pd.DataFrame, top_books: list) -> tuple:
    top_books_set = set(top_books)
    df = df[df['book_id'].isin(top_books_set)].copy()
    df = df.sort_values(['user_id', 'timestamp'])

    df_agg = df.groupby('user_id').agg(
        book_id   = ('book_id',   list),
        rating    = ('rating',    list),
        timestamp = ('timestamp', list),
    ).reset_index()

    read_rows  = []
    label_rows = []
    too_few = too_many = 0

    for _, row in tqdm(df_agg.iterrows(), total=len(df_agg), desc="Splitting user histories"):
        n = len(row['book_id'])
        if n < MIN_RATINGS_PER_USER:
            too_few += 1
            continue
        if n > MAX_RATINGS_PER_USER:
            too_many += 1
            continue

        uid     = row['user_id']
        split   = int(n * PERCENT_READ_HISTORY)
        bids    = row['book_id']
        ratings = row['rating']
        times   = row['timestamp']

        for i in range(split):
            read_rows.append({'user_id': uid, 'book_id': bids[i],
                              'rating': ratings[i], 'timestamp': times[i]})
        for i in range(split, n):
            label_rows.append({'user_id': uid, 'book_id': bids[i],
                               'rating': ratings[i], 'timestamp': times[i]})

    read_df   = pd.DataFrame(read_rows)
    labels_df = pd.DataFrame(label_rows)

    print(f"  Users kept: {read_df['user_id'].nunique():,}  "
          f"(skipped too_few={too_few}, too_many={too_many})")
    print(f"  Read rows: {len(read_df):,}   Label rows: {len(labels_df):,}")
    return read_df, labels_df


# ── Per-book shelf scores helper ─────────────────────────────────────────────

def _build_book_shelf_scores(top_books: list, books: dict, vocab_df: pd.DataFrame) -> pd.DataFrame:
    import math
    final_shelves = set(vocab_df.loc[vocab_df['type'] == 'shelf', 'value'].tolist())
    N = len(top_books)

    # Pass 1: document frequency — how many books have each shelf (count > 0)
    df_count: dict = defaultdict(int)
    for bid in top_books:
        for s in books.get(bid, {}).get('popular_shelves', []):
            if s['name'] in final_shelves and int(s['count']) > 0:
                df_count[s['name']] += 1

    # Pass 2: TF-IDF scores — TF * log(N / df)
    rows = []
    for bid in tqdm(top_books, desc="Building book shelf scores (TF-IDF)"):
        book = books.get(bid, {})
        raw = {
            s['name']: int(s['count'])
            for s in book.get('popular_shelves', [])
            if s['name'] in final_shelves and int(s['count']) > 0
        }
        total = sum(raw.values())
        if total == 0:
            rows.append({'book_id': bid, 'shelf_names': [], 'scores': []})
            continue
        names  = list(raw.keys())
        scores = [(raw[n] / total) * math.log(N / df_count[n]) for n in names]
        rows.append({'book_id': bid, 'shelf_names': names, 'scores': scores})
    return pd.DataFrame(rows)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(data_dir: str = 'data', step: str = None) -> None:
    if step == 'books':
        run_books(data_dir)
    elif step == 'interactions':
        run_interactions(data_dir)
    elif step == 'split':
        run_split(data_dir)
    else:
        run_books(data_dir)
        print()
        run_interactions(data_dir)
        print()
        run_split(data_dir)
