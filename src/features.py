"""
Stage 2 — Feature Engineering
Reads base_*.parquet, builds per-book and per-user feature vectors.
Re-run this (not preprocess) when iterating on feature ideas.

Usage:
    python main.py features
"""
import os
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


FEATURES_VERSION = 'v1'
VAL_FRACTION     = 0.10   # fraction of users held out for val
VAL_SPLIT_SEED   = 42


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_base(data_dir: str) -> dict:
    files = [
        ('books',         'base_books.parquet'),
        ('vocab',         'base_vocab.parquet'),
        ('interactions',  'base_interactions_raw.parquet'),
        ('timestamps',    'base_timestamps.parquet'),
        ('book_shelves',  'base_book_shelves.parquet'),
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
      book_id, book_idx, year, genre_context, shelf_context, author_idx,
      interaction_count

    genre_context      — float vector length n_genres, rank-based normalized weights
    shelf_context      — float vector length n_shelves, TF-IDF shelf scores
    author_idx         — int, primary author vocab index; 0 (__unknown__) if no authors
    interaction_count  — number of corpus interactions for this book (for popularity debiasing)
    """
    books_df        = base['books']
    book_shelves_df = base['book_shelves']

    # Per-book interaction count across all corpus users
    interaction_counts = base['interactions'].groupby('book_id').size().to_dict()

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
            'book_id':           bid,
            'book_idx':          book_to_idx[bid],
            'year':              year,
            'genre_context':     genre_ctx,
            'shelf_context':     book_shelf_ctx.get(bid, [0.0] * n_shelves),
            'author_idx':        author_idx,
            'interaction_count': float(interaction_counts.get(bid, 0)),
        })

    df = pd.DataFrame(rows)
    print(f"  Book features: {len(df)} books  (genres={n_genres}, shelves={n_shelves})")
    return df


# ── Per-user features ─────────────────────────────────────────────────────────

def build_user_features(base: dict, vocab: dict) -> pd.DataFrame:
    """
    Returns DataFrame with one row per user:
      user_id, split, avg_rating

    split      — 'train' or 'val' (user-level 90/10, seed=VAL_SPLIT_SEED)
    avg_rating — mean raw rating across all of the user's interactions

    Genre context, read history, and rollback slices are computed on-the-fly
    in dataset.build_softmax_dataset() from base_interactions_raw.parquet.
    """
    interactions_df = base['interactions']

    all_users = interactions_df['user_id'].unique().tolist()
    rng = random.Random(VAL_SPLIT_SEED)
    rng.shuffle(all_users)
    n_val    = int(len(all_users) * VAL_FRACTION)
    val_set  = set(all_users[:n_val])
    print(f"  Train users: {len(all_users) - n_val:,}   Val users: {n_val:,}")

    avg_ratings = interactions_df.groupby('user_id')['rating'].mean().to_dict()

    rows = []
    for uid in tqdm(all_users, desc="User features"):
        rows.append({
            'user_id':    uid,
            'split':      'val' if uid in val_set else 'train',
            'avg_rating': float(avg_ratings.get(uid, 3.0)),
        })

    df = pd.DataFrame(rows)
    print(f"  User features: {len(df):,} users  "
          f"({df['split'].eq('train').sum():,} train, {df['split'].eq('val').sum():,} val)")
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
