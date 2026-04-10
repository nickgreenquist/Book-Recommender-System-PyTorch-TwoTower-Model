"""
Stage 2 — Feature Engineering
Reads base_*.parquet, builds per-book and per-user feature vectors.
Re-run this (not preprocess) when iterating on feature ideas.

Usage:
    python main.py features
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


FEATURES_VERSION = 'v1'
MAX_HISTORY_LEN  = 50   # cap read history to most recent N books


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_base(data_dir: str) -> dict:
    files = [
        ('books',        'base_books.parquet'),
        ('vocab',        'base_vocab.parquet'),
        ('read',         'base_ratings_read.parquet'),
        ('labels',       'base_ratings_labels.parquet'),
        ('timestamps',   'base_timestamps.parquet'),
        ('book_shelves', 'base_book_shelves.parquet'),
    ]
    result = {}
    for key, filename in files:
        print(f"  Loading {filename} ...")
        result[key] = pd.read_parquet(os.path.join(data_dir, filename))
    return result


def parse_vocab(vocab_df: pd.DataFrame) -> dict:
    """Extract ordered vocabulary lists and index maps from base_vocab.parquet."""
    g = vocab_df[vocab_df['type'] == 'genre'].sort_values('index')
    s = vocab_df[vocab_df['type'] == 'shelf'].sort_values('index')
    y = vocab_df[vocab_df['type'] == 'year'].sort_values('index')
    a = vocab_df[vocab_df['type'] == 'author'].sort_values('index')

    return {
        'genres_ordered':  g['value'].tolist(),
        'shelves_ordered': s['value'].tolist(),
        'years_ordered':   y['value'].tolist(),
        'authors_ordered': a['value'].tolist(),
        'genre_to_i':      dict(zip(g['value'], g['index'].astype(int))),
        'shelf_to_i':      dict(zip(s['value'], s['index'].astype(int))),
        'year_to_i':       dict(zip(y['value'], y['index'].astype(int))),
        'author_to_i':     dict(zip(a['value'], a['index'].astype(int))),
    }


# ── Per-book features ─────────────────────────────────────────────────────────

def build_book_features(base: dict, vocab: dict) -> pd.DataFrame:
    """
    Returns DataFrame with one row per book:
      book_id, book_idx, year, genre_context, shelf_context, author_indices

    genre_context  — float vector length n_genres, rank-based normalized weights
    shelf_context  — float vector length n_shelves, normalized shelf scores (count/total)
    author_idx     — int, primary author vocab index; 0 (__unknown__) if no authors
    """
    books_df        = base['books']
    book_shelves_df = base['book_shelves']

    genre_to_i  = vocab['genre_to_i']
    shelf_to_i  = vocab['shelf_to_i']
    author_to_i = vocab['author_to_i']
    n_genres    = len(genre_to_i)
    n_shelves   = len(shelf_to_i)

    top_books   = books_df['book_id'].tolist()
    book_to_idx = {bid: i for i, bid in enumerate(top_books)}

    # Shelf context lookup: book_id → dense float vector
    book_shelf_ctx: dict = {}
    for _, row in tqdm(book_shelves_df.iterrows(), total=len(book_shelves_df),
                       desc="Book shelf contexts"):
        bid = row['book_id']
        vec = [0.0] * n_shelves
        for name, score in zip(row['shelf_names'], row['scores']):
            if name in shelf_to_i:
                vec[shelf_to_i[name]] = float(score)
        book_shelf_ctx[bid] = vec

    rows = []
    for _, brow in tqdm(books_df.iterrows(), total=len(books_df), desc="Book features"):
        bid  = brow['book_id']
        year = str(brow['year'])

        # Genre context — rank-based normalized weights (index 0 = top genre by votes)
        genres = list(brow['genres']) if brow['genres'] is not None else []
        n      = len(genres)
        genre_ctx = [0.0] * n_genres
        if n > 0:
            weights = [n - i for i in range(n)]
            total_w = sum(weights)
            for g, w in zip(genres, weights):
                if g in genre_to_i:
                    genre_ctx[genre_to_i[g]] = w / total_w

        # Primary author index — first author in list; fallback to __unknown__ (index 0)
        # Note: only primary author used in v1; multi-author avg-pool is a future improvement
        raw_ids    = list(brow['author_ids']) if brow['author_ids'] is not None else []
        author_idx = author_to_i.get(str(raw_ids[0]), 0) if raw_ids else 0

        rows.append({
            'book_id':       bid,
            'book_idx':      book_to_idx[bid],
            'year':          year,
            'genre_context': genre_ctx,
            'shelf_context': book_shelf_ctx.get(bid, [0.0] * n_shelves),
            'author_idx':    author_idx,
        })

    df = pd.DataFrame(rows)
    print(f"  Book features: {len(df)} books  (genres={n_genres}, shelves={n_shelves})")
    return df


# ── Per-user features ─────────────────────────────────────────────────────────

def build_user_features(base: dict, vocab: dict) -> pd.DataFrame:
    """
    Returns DataFrame with one row per user:
      user_id, avg_rating, genre_context,
      read_history, read_history_ratings,
      label_bookIds, label_ratings, label_timestamps

    genre_context — float vector length 2*n_genres:
                    first half  = debiased avg rating per genre
                    second half = fraction of read history in each genre
    read_history  — list[int] of book_idx values, capped to MAX_HISTORY_LEN most recent

    Note: shelf_context is NOT stored per user. The model looks up shelf vectors
    from a book_shelf_matrix using read_history indices and pools them in the
    forward pass — avoids materializing a 525k × 3032 tensor.
    """
    read_df   = base['read']
    labels_df = base['labels']
    books_df  = base['books']

    genre_to_i = vocab['genre_to_i']
    genres_ord = vocab['genres_ordered']
    n_genres   = len(genre_to_i)

    top_books   = books_df['book_id'].tolist()
    book_to_idx = {bid: i for i, bid in enumerate(top_books)}

    bookId_to_genres = {r['book_id']: (list(r['genres']) if r['genres'] is not None else []) for _, r in books_df.iterrows()}

    # Per-user avg rating
    avg_ratings = read_df.groupby('user_id')['rating'].mean().to_dict()

    # Per-user genre stats — vectorized via explode + groupby
    print("  Computing user genre stats ...")
    _wg = read_df[['user_id', 'book_id', 'rating']].copy()
    _wg['genre'] = _wg['book_id'].map(bookId_to_genres)
    _wg = _wg.explode('genre').dropna(subset=['genre'])
    _agg = (_wg.groupby(['user_id', 'genre'])
               .agg(N=('rating', 'count'), S=('rating', 'sum'))
               .reset_index())

    avg_idx   = {g: i            for i, g in enumerate(genres_ord)}
    genre_frac_idx = {g: n_genres + i for i, g in enumerate(genres_ord)}

    print("  Building genre context matrix ...")
    total_N    = _agg.groupby('user_id')['N'].sum()
    all_uids   = list(total_N.index)
    uid_to_row = {uid: i for i, uid in enumerate(all_uids)}

    _ctx = _agg.copy()
    _ctx['total_N']   = _ctx['user_id'].map(total_N)
    _ctx['avg_rat']   = _ctx['user_id'].map(avg_ratings)
    _ctx['avg_g']     = _ctx['S'] / _ctx['N']
    _ctx['val_avg']   = _ctx['avg_g'] - _ctx['avg_rat']
    _ctx['val_genre_frac'] = _ctx['N'] / _ctx['total_N']
    _ctx['col_avg']   = _ctx['genre'].map(avg_idx)
    _ctx['col_genre_frac'] = _ctx['genre'].map(genre_frac_idx)
    _ctx = _ctx.dropna(subset=['col_avg'])
    _ctx['col_avg']   = _ctx['col_avg'].astype(int)
    _ctx['col_genre_frac'] = _ctx['col_genre_frac'].astype(int)
    _ctx['row']       = _ctx['user_id'].map(uid_to_row).astype(int)

    genre_ctx_matrix = np.zeros((len(all_uids), 2 * n_genres), dtype=np.float32)
    genre_ctx_matrix[_ctx['row'].values, _ctx['col_avg'].values]   = _ctx['val_avg'].values
    genre_ctx_matrix[_ctx['row'].values, _ctx['col_genre_frac'].values] = _ctx['val_genre_frac'].values

    # Aggregate read/label history per user
    read_agg = (read_df
                 .groupby('user_id')
                 .agg(book_ids=('book_id', list), ratings=('rating', list))
                 .reset_index())
    label_agg = (labels_df
                 .groupby('user_id')
                 .agg(book_ids=('book_id', list), ratings=('rating', list),
                      timestamps=('timestamp', list))
                 .reset_index())

    read_by_user = {r['user_id']: r for _, r in read_agg.iterrows()}
    label_by_user = {r['user_id']: r for _, r in label_agg.iterrows()}

    rows = []
    for uid in tqdm(all_uids, desc="User features"):
        avg_rat = float(avg_ratings.get(uid, 3.0))

        # Genre context — O(1) lookup from precomputed matrix
        genre_ctx = genre_ctx_matrix[uid_to_row[uid]].tolist()

        # Read history → book_idx + debiased ratings, cap to MAX_HISTORY_LEN most recent
        rrow = read_by_user.get(uid)
        if rrow is not None:
            pairs = [
                (book_to_idx[bid], float(rat) - avg_rat)
                for bid, rat in zip(rrow['book_ids'], rrow['ratings'])
                if bid in book_to_idx
            ][-MAX_HISTORY_LEN:]
        else:
            pairs = []
        hist_ids     = [p[0] for p in pairs]
        hist_ratings = [p[1] for p in pairs]

        # Labels
        lrow = label_by_user.get(uid)
        if lrow is not None:
            lbl_books = list(lrow['book_ids'])
            lbl_rats  = [float(r) for r in lrow['ratings']]
            lbl_times = [int(t)   for t in lrow['timestamps']]
        else:
            lbl_books = lbl_rats = lbl_times = []

        rows.append({
            'user_id':               uid,
            'avg_rating':            avg_rat,
            'genre_context':         genre_ctx,
            'read_history':          hist_ids,
            'read_history_ratings':  hist_ratings,
            'label_bookIds':         lbl_books,
            'label_ratings':         lbl_rats,
            'label_timestamps':      lbl_times,
        })

    df = pd.DataFrame(rows)
    print(f"  User features: {len(df)} users")
    return df


# ── Parquet writer (handles list columns) ─────────────────────────────────────

def _write_list_parquet(df: pd.DataFrame, path: str) -> None:
    arrays = {}
    for col in tqdm(df.columns, desc=f"  Serializing columns"):
        sample = df[col].iloc[0] if len(df) > 0 else None
        first  = sample[0] if isinstance(sample, list) and sample else None
        if isinstance(sample, list) and isinstance(first, float):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
        elif isinstance(sample, list) and isinstance(first, int):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.int64()))
        elif isinstance(sample, list) and isinstance(first, str):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.string()))
        elif isinstance(sample, list):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
        else:
            arrays[col] = pa.array(df[col].tolist())
    print(f"  Writing {path} ...")
    pq.write_table(pa.table(arrays), path)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(data_dir: str = 'data', version: str = FEATURES_VERSION) -> None:
    print(f"Loading base parquets from {data_dir}/ ...")
    base  = load_base(data_dir)
    vocab = parse_vocab(base['vocab'])

    print("\n── Building book features ──")
    book_df = build_book_features(base, vocab)

    print("\n── Building user features ──")
    user_df = build_user_features(base, vocab)

    book_out = os.path.join(data_dir, f'features_books_{version}.parquet')
    user_out = os.path.join(data_dir, f'features_users_{version}.parquet')

    print(f"\nWriting {book_out} ...")
    _write_list_parquet(book_df, book_out)
    print(f"Writing {user_out} ...")
    _write_list_parquet(user_df, user_out)

    print(f"\n✓ features_books_{version}.parquet and features_users_{version}.parquet → {data_dir}/")
