"""
Stage 3 — Dataset building for in-batch negatives softmax training.
Reads features_*.parquet into a FeatureStore, builds rollback training examples.

Usage:
    from src.dataset import load_features, make_softmax_splits
"""
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


TIMESTAMP_NUM_BINS = 1_500


# ── FeatureStore ──────────────────────────────────────────────────────────────

@dataclass
class FeatureStore:
    # Vocabulary (ordered lists for index reproducibility)
    top_books:       list   # list[str], ordered — index i = embedding row i
    genres_ordered:  list
    shelves_ordered: list
    years_ordered:   list
    authors_ordered: list

    # Vocabulary index maps
    genre_to_i:   dict
    shelf_to_i:   dict
    year_to_i:    dict
    author_to_i:  dict
    bookId_to_idx: dict  # book_id (str) → embedding row index

    # Per-book lookups
    bookId_to_title:         dict
    title_to_bookId:         dict
    bookId_to_year:          dict
    bookId_to_genres:        dict
    bookId_to_genre_context: dict
    bookId_to_shelf_context: dict
    bookId_to_author_idx:    dict  # book_id → primary author vocab index

    # Per-user lookups (absent when loaded via load_book_features)
    user_ids:                        list  = field(default_factory=list)
    user_to_avg_rating:              dict  = field(default_factory=dict)
    user_to_genre_context:           dict  = field(default_factory=dict)
    user_to_read_history:            dict  = field(default_factory=dict)
    user_to_read_history_ratings:    dict  = field(default_factory=dict)
    user_to_book_to_rating_LABEL:    dict  = field(default_factory=dict)
    user_to_book_to_timestamp_LABEL: dict  = field(default_factory=dict)

    # Derived constants
    user_context_size:   int          = 0
    timestamp_num_bins:  int          = TIMESTAMP_NUM_BINS
    timestamp_bins:      torch.Tensor = field(default_factory=lambda: torch.tensor([]))


# ── Loader ────────────────────────────────────────────────────────────────────

def load_features(data_dir: str = 'data', version: str = 'v1') -> FeatureStore:
    """Load feature parquets and base vocab into a FeatureStore."""
    vocab_df  = pd.read_parquet(os.path.join(data_dir, 'base_vocab.parquet'))
    books_df  = pd.read_parquet(os.path.join(data_dir, 'base_books.parquet'))
    ts_df     = pd.read_parquet(os.path.join(data_dir, 'base_timestamps.parquet'))

    book_feat_path = os.path.join(data_dir, f'features_books_{version}.parquet')
    user_feat_path = os.path.join(data_dir, f'features_users_{version}.parquet')

    book_feat_df = pq.read_table(book_feat_path).to_pandas()
    user_feat_df = pq.read_table(user_feat_path).to_pandas()

    # ── Vocab ─────────────────────────────────────────────────────────────────
    g = vocab_df[vocab_df['type'] == 'genre'].sort_values('index')
    s = vocab_df[vocab_df['type'] == 'shelf'].sort_values('index')
    y = vocab_df[vocab_df['type'] == 'year'].sort_values('index')
    a = vocab_df[vocab_df['type'] == 'author'].sort_values('index')

    genres_ordered  = g['value'].tolist()
    shelves_ordered = s['value'].tolist()
    years_ordered   = y['value'].tolist()
    authors_ordered = a['value'].tolist()

    genre_to_i  = dict(zip(g['value'], g['index'].astype(int)))
    shelf_to_i  = dict(zip(s['value'], s['index'].astype(int)))
    year_to_i   = dict(zip(y['value'], y['index'].astype(int)))
    author_to_i = dict(zip(a['value'], a['index'].astype(int)))

    # ── Book metadata ─────────────────────────────────────────────────────────
    top_books     = books_df['book_id'].tolist()
    bookId_to_idx = {bid: i for i, bid in enumerate(top_books)}

    bookId_to_title  = {}
    title_to_bookId  = {}
    bookId_to_year   = {}
    bookId_to_genres = {}
    for _, row in books_df.iterrows():
        bid = row['book_id']
        bookId_to_title[bid]       = row['title']
        title_to_bookId[row['title']] = bid
        bookId_to_year[bid]        = str(row['year'])
        bookId_to_genres[bid]      = list(row['genres']) if row['genres'] is not None else []

    # ── Per-book feature vectors ──────────────────────────────────────────────
    bookId_to_genre_context = {}
    bookId_to_shelf_context = {}
    bookId_to_author_idx    = {}
    for _, row in book_feat_df.iterrows():
        bid = row['book_id']
        bookId_to_genre_context[bid] = list(row['genre_context'])
        bookId_to_shelf_context[bid] = list(row['shelf_context'])
        bookId_to_author_idx[bid]    = int(row['author_idx'])

    # ── Per-user feature vectors ──────────────────────────────────────────────
    user_ids = user_feat_df['user_id'].tolist()

    user_to_avg_rating           = dict(zip(user_feat_df['user_id'],
                                            user_feat_df['avg_rating'].astype(float)))
    user_to_genre_context        = dict(zip(user_feat_df['user_id'],
                                            user_feat_df['genre_context'].apply(list)))
    user_to_read_history         = dict(zip(user_feat_df['user_id'],
                                            user_feat_df['read_history'].apply(list)))
    user_to_read_history_ratings = dict(zip(user_feat_df['user_id'],
                                            user_feat_df['read_history_ratings'].apply(list)))

    lbl_books_col = user_feat_df['label_bookIds'].apply(list)
    lbl_rats_col  = user_feat_df['label_ratings'].apply(list)
    lbl_times_col = user_feat_df['label_timestamps'].apply(list)
    uids          = user_feat_df['user_id']

    user_to_book_to_rating_LABEL    = {uid: dict(zip(b, r))
                                       for uid, b, r in zip(uids, lbl_books_col, lbl_rats_col)}
    user_to_book_to_timestamp_LABEL = {uid: dict(zip(b, t))
                                       for uid, b, t in zip(uids, lbl_books_col, lbl_times_col)}

    # ── Derived constants ─────────────────────────────────────────────────────
    user_context_size = 2 * len(genres_ordered)

    ts_min = int(ts_df['ts_min'].iloc[0])
    ts_max = int(ts_df['ts_max'].iloc[0])
    timestamp_bins = torch.tensor(np.linspace(ts_min, ts_max, TIMESTAMP_NUM_BINS))

    return FeatureStore(
        top_books=top_books,
        genres_ordered=genres_ordered,
        shelves_ordered=shelves_ordered,
        years_ordered=years_ordered,
        authors_ordered=authors_ordered,
        genre_to_i=genre_to_i,
        shelf_to_i=shelf_to_i,
        year_to_i=year_to_i,
        author_to_i=author_to_i,
        bookId_to_idx=bookId_to_idx,
        bookId_to_title=bookId_to_title,
        title_to_bookId=title_to_bookId,
        bookId_to_year=bookId_to_year,
        bookId_to_genres=bookId_to_genres,
        bookId_to_genre_context=bookId_to_genre_context,
        bookId_to_shelf_context=bookId_to_shelf_context,
        bookId_to_author_idx=bookId_to_author_idx,
        user_ids=user_ids,
        user_to_avg_rating=user_to_avg_rating,
        user_to_genre_context=user_to_genre_context,
        user_to_read_history=user_to_read_history,
        user_to_read_history_ratings=user_to_read_history_ratings,
        user_to_book_to_rating_LABEL=user_to_book_to_rating_LABEL,
        user_to_book_to_timestamp_LABEL=user_to_book_to_timestamp_LABEL,
        user_context_size=user_context_size,
        timestamp_num_bins=TIMESTAMP_NUM_BINS,
        timestamp_bins=timestamp_bins,
    )


def load_book_features(data_dir: str = 'data', version: str = 'v1') -> FeatureStore:
    """Load only book-side features — skips the slow 525k-user parquet.
    Suitable for probes and export; user_* fields are empty."""
    vocab_df = pd.read_parquet(os.path.join(data_dir, 'base_vocab.parquet'))
    books_df = pd.read_parquet(os.path.join(data_dir, 'base_books.parquet'))

    book_feat_df = pq.read_table(
        os.path.join(data_dir, f'features_books_{version}.parquet')
    ).to_pandas()

    g = vocab_df[vocab_df['type'] == 'genre'].sort_values('index')
    s = vocab_df[vocab_df['type'] == 'shelf'].sort_values('index')
    y = vocab_df[vocab_df['type'] == 'year'].sort_values('index')
    a = vocab_df[vocab_df['type'] == 'author'].sort_values('index')

    genres_ordered  = g['value'].tolist()
    shelves_ordered = s['value'].tolist()
    years_ordered   = y['value'].tolist()
    authors_ordered = a['value'].tolist()

    genre_to_i  = dict(zip(g['value'], g['index'].astype(int)))
    shelf_to_i  = dict(zip(s['value'], s['index'].astype(int)))
    year_to_i   = dict(zip(y['value'], y['index'].astype(int)))
    author_to_i = dict(zip(a['value'], a['index'].astype(int)))

    top_books     = books_df['book_id'].tolist()
    bookId_to_idx = {bid: i for i, bid in enumerate(top_books)}

    bookId_to_title  = {}
    title_to_bookId  = {}
    bookId_to_year   = {}
    bookId_to_genres = {}
    for _, row in books_df.iterrows():
        bid = row['book_id']
        bookId_to_title[bid]      = row['title']
        title_to_bookId[row['title']] = bid
        bookId_to_year[bid]       = str(row['year'])
        bookId_to_genres[bid]     = list(row['genres']) if row['genres'] is not None else []

    bookId_to_genre_context = {}
    bookId_to_shelf_context = {}
    bookId_to_author_idx    = {}
    for _, row in book_feat_df.iterrows():
        bid = row['book_id']
        bookId_to_genre_context[bid] = list(row['genre_context'])
        bookId_to_shelf_context[bid] = list(row['shelf_context'])
        bookId_to_author_idx[bid]    = int(row['author_idx'])

    return FeatureStore(
        top_books=top_books,
        genres_ordered=genres_ordered,
        shelves_ordered=shelves_ordered,
        years_ordered=years_ordered,
        authors_ordered=authors_ordered,
        genre_to_i=genre_to_i,
        shelf_to_i=shelf_to_i,
        year_to_i=year_to_i,
        author_to_i=author_to_i,
        bookId_to_idx=bookId_to_idx,
        bookId_to_title=bookId_to_title,
        title_to_bookId=title_to_bookId,
        bookId_to_year=bookId_to_year,
        bookId_to_genres=bookId_to_genres,
        bookId_to_genre_context=bookId_to_genre_context,
        bookId_to_shelf_context=bookId_to_shelf_context,
        bookId_to_author_idx=bookId_to_author_idx,
        user_context_size=2 * len(genres_ordered),
    )


# ── Padding helpers ───────────────────────────────────────────────────────────

def pad_history_batch(histories: list, pad_idx: int) -> torch.Tensor:
    max_len = max((len(h) for h in histories), default=1)
    padded  = torch.full((len(histories), max_len), pad_idx, dtype=torch.long)
    for i, hist in enumerate(histories):
        if hist:
            padded[i, :len(hist)] = torch.tensor(hist, dtype=torch.long)
    return padded


def pad_history_ratings_batch(history_ratings: list) -> torch.Tensor:
    max_len = max((len(r) for r in history_ratings), default=1)
    padded  = torch.zeros(len(history_ratings), max_len)
    for i, rats in enumerate(history_ratings):
        if rats:
            padded[i, :len(rats)] = torch.tensor(rats, dtype=torch.float)
    return padded


# ── Softmax dataset (rollback) ────────────────────────────────────────────────
# Each user contributes multiple training examples via "rollback": for a user who
# read [A, B, C, D, E] in order, context=[A] predicts B, context=[A,B,C] predicts D, etc.
# Both train and val use rollback (not last-read holdout) to keep context length
# distributions consistent. MAX_SOFTMAX_EXAMPLES_PER_USER caps examples per user.

MAX_SOFTMAX_EXAMPLES_PER_USER = 10   # rollback examples sampled per user


def _rollback_genre_context(ctx_book_idxs: list, ctx_debiased_ratings: list,
                             fs: FeatureStore) -> list:
    """
    Compute user genre context from a rollback history slice.
    Mirrors the features.py build_user_features genre_context format:
      first half  (indices 0..n_genres-1)   : debiased avg rating per genre
      second half (indices n_genres..2n-1)  : genre fraction of total genre assignments

    ctx_book_idxs       — list of book embedding indices (int)
    ctx_debiased_ratings — list of debiased ratings (float) aligned with ctx_book_idxs
    """
    n_genres           = len(fs.genres_ordered)
    genre_debias_sum   = np.zeros(n_genres, dtype=np.float32)
    genre_count        = np.zeros(n_genres, dtype=np.float32)

    for book_idx, d_rat in zip(ctx_book_idxs, ctx_debiased_ratings):
        bid = fs.top_books[book_idx]
        for genre in fs.bookId_to_genres.get(bid, []):
            g_idx = fs.genre_to_i.get(genre)
            if g_idx is not None:
                genre_debias_sum[g_idx] += d_rat
                genre_count[g_idx]      += 1

    ctx               = np.zeros(2 * n_genres, dtype=np.float32)
    total_assignments = genre_count.sum()
    if total_assignments > 0:
        mask            = genre_count > 0
        ctx[:n_genres][mask] = genre_debias_sum[mask] / genre_count[mask]
        ctx[n_genres:]       = genre_count / total_assignments

    return ctx.tolist()


def build_softmax_dataset(users: list, fs: FeatureStore, raw_df,
                           max_per_user: int = MAX_SOFTMAX_EXAMPLES_PER_USER,
                           seed: int = 42) -> tuple:
    """
    Build rollback training examples for in-batch negatives softmax training.

    For each user, reads are sorted by timestamp. For each read event at index i,
    the context is all reads before index i (the "rollback" approach from YouTube DNN).
    Examples are capped at max_per_user per user; if more exist, a random sample is taken.

    raw_df must have columns: user_id, book_id, rating (raw 1-5), timestamp (unix seconds).

    Returns 5-tuple:
        [0] X_genre            — (N, user_context_size) float
        [1] X_history          — list[list[int]]  padded at training time
        [2] X_history_ratings  — list[list[float]]
        [3] timestamp          — (N,) long  (binned)
        [4] target_book_idx    — (N,) long
    target_genre/year/author_idx are NOT stored — looked up from model buffers
    at training time using target_book_idx as the corpus index.
    """
    from src.features import MAX_HISTORY_LEN
    rng       = random.Random(seed)
    users_set = set(users)
    max_hist  = MAX_HISTORY_LEN
    n_genres  = len(fs.genres_ordered)

    # Precompute book → list of genre indices (avoids repeated dict lookups in inner loop)
    book_genre_idxs = {
        bid: [fs.genre_to_i[g] for g in fs.bookId_to_genres.get(bid, []) if g in fs.genre_to_i]
        for bid in fs.top_books
    }

    # Preprocess already filtered to valid users + corpus books.
    # Filter here only to restrict to this split's users (train or val subset).
    df = raw_df[raw_df['user_id'].isin(users_set)].copy()
    print(f"  Sorting {len(df):,} interactions by user + timestamp ...")
    df = df.sort_values(['user_id', 'timestamp'])
    print(f"  {df['user_id'].nunique():,} users")

    X_genre           = []
    X_history         = []
    X_history_ratings = []
    timestamps_raw    = []
    target_book_idx   = []

    from tqdm import tqdm
    n_users = df['user_id'].nunique()
    for uid, group in tqdm(df.groupby('user_id'), total=n_users, desc="Building softmax examples"):
        avg_rat = fs.user_to_avg_rating.get(uid, 3.0)

        books   = group['book_id'].tolist()
        ratings = group['rating'].tolist()
        ts_vals = group['timestamp'].tolist()
        n       = len(books)

        if n < 2:
            continue

        # Sample target positions upfront — avoids generating all rollbacks then discarding.
        # Valid targets: positions 1..n-1 (position 0 has no prior context).
        # Sorting ensures a single left-to-right scan maintains genre accumulators correctly.
        k               = min(max_per_user, n - 1)
        sampled_targets = sorted(rng.sample(range(1, n), k))
        sampled_set     = set(sampled_targets)

        # Single left-to-right pass: snapshot accumulators at sampled positions,
        # then update accumulators with the current book for future positions.
        running_count = np.zeros(n_genres, dtype=np.float32)
        running_sum   = np.zeros(n_genres, dtype=np.float32)
        ctx_ids_buf   = []   # grows as we advance; sliced at each target position
        ctx_rats_buf  = []

        for pos, (bid, rat, ts) in enumerate(zip(books, ratings, ts_vals)):
            d_rat = rat - avg_rat

            if pos in sampled_set:
                # Snapshot: accumulators and ctx buffers reflect books 0..pos-1
                total_assign = running_count.sum()
                genre_ctx    = np.zeros(2 * n_genres, dtype=np.float32)
                if total_assign > 0:
                    mask = running_count > 0
                    genre_ctx[:n_genres][mask] = running_sum[mask] / running_count[mask]
                    genre_ctx[n_genres:]       = running_count / total_assign

                t_idx = fs.bookId_to_idx[bid]
                X_genre.append(genre_ctx.tolist())
                X_history.append(ctx_ids_buf[-max_hist:])
                X_history_ratings.append(ctx_rats_buf[-max_hist:])
                timestamps_raw.append(ts)
                target_book_idx.append(t_idx)

            # Update accumulators and context buffer with current book
            ctx_ids_buf.append(fs.bookId_to_idx[bid])
            ctx_rats_buf.append(d_rat)
            for g_idx in book_genre_idxs.get(bid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    n = len(target_book_idx)
    print(f"  {n:,} softmax examples — building tensors ...")

    X_genre_t         = torch.from_numpy(np.array(X_genre,        dtype=np.float32))
    target_book_idx_t = torch.from_numpy(np.array(target_book_idx, dtype=np.int64))
    timestamp_t       = torch.bucketize(
        torch.from_numpy(np.array(timestamps_raw, dtype=np.float64)).float(),
        fs.timestamp_bins.float(), right=False)

    return (X_genre_t, X_history, X_history_ratings, timestamp_t, target_book_idx_t)


def make_softmax_splits(fs: FeatureStore, data_dir: str = 'data',
                        max_per_user: int = MAX_SOFTMAX_EXAMPLES_PER_USER,
                        pct_train: float = 0.9, seed: int = 42,
                        max_users: int = None) -> tuple:
    """
    Load base_interactions_raw.parquet, split users 90/10, build rollback datasets.
    max_users: if set, subsample (for fast debug runs, e.g. max_users=10_000).
    Returns (train_data, val_data).
    """
    import pandas as pd
    raw_path = os.path.join(data_dir, 'base_interactions_raw.parquet')
    print(f"Loading {raw_path} ...")
    raw_df = pd.read_parquet(raw_path)
    print(f"  {len(raw_df):,} raw interactions, {raw_df['user_id'].nunique():,} users")

    valid_users = fs.user_ids[:]
    rng = random.Random(seed)
    rng.shuffle(valid_users)

    if max_users is not None:
        valid_users = valid_users[:max_users]
        print(f"  [debug] subsampled to {len(valid_users):,} users")

    split       = int(len(valid_users) * pct_train)
    train_users = valid_users[:split]
    val_users   = valid_users[split:]

    print(f"\nBuilding softmax train dataset ({len(train_users):,} users) ...")
    train_data = build_softmax_dataset(train_users, fs, raw_df, max_per_user, seed)
    print(f"  X_genre_train shape: {train_data[0].shape}")

    print(f"\nBuilding softmax val dataset ({len(val_users):,} users) ...")
    val_data = build_softmax_dataset(val_users, fs, raw_df, max_per_user, seed)
    print(f"  X_genre_val shape:   {val_data[0].shape}")

    return train_data, val_data


def save_softmax_splits(train_data: tuple, val_data: tuple,
                        data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_softmax_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_softmax_val_{version}.pt'))
    print(f"Saved dataset_softmax_train_{version}.pt and dataset_softmax_val_{version}.pt → {data_dir}/")


def load_softmax_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_softmax_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_softmax_val_{version}.pt')

    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data
