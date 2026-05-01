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
MAX_HISTORY_LEN    = 50     # cap per-user read history for rollback buffers


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
    user_ids:           list = field(default_factory=list)
    train_users:        list = field(default_factory=list)
    val_users:          list = field(default_factory=list)
    user_to_avg_rating: dict = field(default_factory=dict)

    # Per-book interaction counts (for popularity debiasing in full softmax)
    book_interaction_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

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
    user_feat_df = pd.read_parquet(user_feat_path)

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

    # ── Per-book interaction counts (popularity debiasing) ───────────────────
    ic_map = dict(zip(book_feat_df['book_id'], book_feat_df['interaction_count'].astype(np.float32)))
    book_interaction_counts = np.array([ic_map.get(bid, 0.0) for bid in top_books], dtype=np.float32)

    # ── Per-user features (user_id, split, avg_rating) ───────────────────────
    train_users      = user_feat_df[user_feat_df['split'] == 'train']['user_id'].tolist()
    val_users        = user_feat_df[user_feat_df['split'] == 'val']['user_id'].tolist()
    user_ids         = train_users + val_users
    user_to_avg_rating = dict(zip(user_feat_df['user_id'],
                                  user_feat_df['avg_rating'].astype(float)))

    print(f"  Users: {len(train_users):,} train | {len(val_users):,} val")

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
        train_users=train_users,
        val_users=val_users,
        user_to_avg_rating=user_to_avg_rating,
        book_interaction_counts=book_interaction_counts,
        user_context_size=user_context_size,
        timestamp_num_bins=TIMESTAMP_NUM_BINS,
        timestamp_bins=timestamp_bins,
    )


def load_book_features(data_dir: str = 'data', version: str = 'v1') -> FeatureStore:
    """Load only book-side features — skips the slow 525k-user parquet.
    Suitable for probes, canary, and export; user_* fields are empty."""
    vocab_df = pd.read_parquet(os.path.join(data_dir, 'base_vocab.parquet'))
    books_df = pd.read_parquet(os.path.join(data_dir, 'base_books.parquet'))
    ts_df    = pd.read_parquet(os.path.join(data_dir, 'base_timestamps.parquet'))

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

    ic_map = dict(zip(book_feat_df['book_id'], book_feat_df['interaction_count'].astype(np.float32)))
    book_interaction_counts = np.array([ic_map.get(bid, 0.0) for bid in top_books], dtype=np.float32)

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
        book_interaction_counts=book_interaction_counts,
        user_context_size=2 * len(genres_ordered),
        timestamp_num_bins=TIMESTAMP_NUM_BINS,
        timestamp_bins=torch.tensor(np.linspace(
            int(ts_df['ts_min'].iloc[0]), int(ts_df['ts_max'].iloc[0]), TIMESTAMP_NUM_BINS)),
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
# distributions consistent. MAX_ROLLBACK_EXAMPLES_PER_USER caps examples per user.

MAX_ROLLBACK_EXAMPLES_PER_USER = 10   # rollback examples sampled per user


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
                           max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
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

    # 1. Counting pass — find exact number of examples for pre-allocation
    print("  Counting pass ...")
    n_examples = 0
    valid_groups = [] # (uid, group)
    for uid, group in df.groupby('user_id'):
        n = len(group)
        if n >= 2:
            n_examples += min(max_per_user, n - 1)
            valid_groups.append((uid, group))

    # 2. Pre-allocate NumPy buffers
    print(f"  Pre-allocating buffers for {n_examples:,} examples ...")
    X_genre         = np.zeros((n_examples, 2 * n_genres), dtype=np.float32)
    pad_idx         = len(fs.top_books)
    X_hist_full     = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_liked    = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_disliked = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_weighted = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_rats_weighted = np.zeros((n_examples, max_hist), dtype=np.float32)
    timestamps_raw  = np.zeros(n_examples, dtype=np.float64)
    target_book_idx = np.zeros(n_examples, dtype=np.int32)

    # 3. Filling pass
    curr_idx = 0
    from tqdm import tqdm
    for uid, group in tqdm(valid_groups, desc="Filling buffers"):
        avg_rat = fs.user_to_avg_rating.get(uid, 3.0)

        books   = group['book_id'].tolist()
        ratings = group['rating'].tolist()
        ts_vals = group['timestamp'].tolist()
        n       = len(books)

        if n < 2:
            continue

        k               = min(max_per_user, n - 1)
        sampled_targets = sorted(rng.sample(range(1, n), k))
        sampled_set     = set(sampled_targets)

        running_count = np.zeros(n_genres, dtype=np.float32)
        running_sum   = np.zeros(n_genres, dtype=np.float32)
        
        # Behavior-based history buffers
        buf_full     = []   # All books seen
        buf_liked    = []   # Books with rating 4 or 5
        buf_disliked = []   # Books with rating 1 or 2
        buf_w_ids    = []   # Same as full, but used with buf_w_rats for weighted pool
        buf_w_rats   = []

        for pos, (bid, rat, ts) in enumerate(zip(books, ratings, ts_vals)):
            d_rat = rat - avg_rat
            b_idx = fs.bookId_to_idx[bid]

            if pos in sampled_set:
                total_assign = running_count.sum()
                if total_assign > 0:
                    mask = running_count > 0
                    X_genre[curr_idx, :n_genres][mask] = running_sum[mask] / running_count[mask]
                    X_genre[curr_idx, n_genres:]       = running_count / total_assign

                # Fill pre-padded history slots (right-aligned)
                def fill_hist(target_arr, source_list):
                    l = len(source_list)
                    if l > 0:
                        take = min(l, max_hist)
                        target_arr[curr_idx, max_hist-take:] = source_list[-take:]

                fill_hist(X_hist_full,     buf_full)
                fill_hist(X_hist_liked,    buf_liked)
                fill_hist(X_hist_disliked, buf_disliked)
                fill_hist(X_hist_weighted, buf_w_ids)
                
                l_w = len(buf_w_ids)
                if l_w > 0:
                    take_w = min(l_w, max_hist)
                    X_rats_weighted[curr_idx, max_hist-take_w:] = buf_w_rats[-take_w:]

                timestamps_raw[curr_idx]  = ts
                target_book_idx[curr_idx] = b_idx
                curr_idx += 1

            # Update buffers with CURRENT book for future target positions
            buf_full.append(b_idx)
            if rat >= 4:
                buf_liked.append(b_idx)
            elif rat <= 2:
                buf_disliked.append(b_idx)
            
            buf_w_ids.append(b_idx)
            buf_w_rats.append(d_rat)

            for g_idx in book_genre_idxs.get(bid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    print(f"  {n_examples:,} softmax examples — building tensors ...")

    X_genre_t         = torch.from_numpy(X_genre)
    X_hist_full_t     = torch.from_numpy(X_hist_full).long()
    X_hist_liked_t    = torch.from_numpy(X_hist_liked).long()
    X_hist_disliked_t = torch.from_numpy(X_hist_disliked).long()
    X_hist_weighted_t = torch.from_numpy(X_hist_weighted).long()
    X_rats_weighted_t = torch.from_numpy(X_rats_weighted)
    target_book_idx_t = torch.from_numpy(target_book_idx).long()
    
    timestamp_t       = torch.bucketize(
        torch.from_numpy(timestamps_raw).float(),
        fs.timestamp_bins.float(), right=False)

    return (X_genre_t, X_hist_full_t, X_hist_liked_t, X_hist_disliked_t, 
            X_hist_weighted_t, X_rats_weighted_t, timestamp_t, target_book_idx_t)


def make_softmax_splits(fs: FeatureStore, data_dir: str = 'data',
                        max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
                        seed: int = 42,
                        max_users: int = None) -> tuple:
    """
    Load base_interactions_raw.parquet and build rollback datasets.
    Train/val split is fixed in features.py (VAL_FRACTION=0.10, VAL_SPLIT_SEED=42).
    max_users: if set, subsample train users only (for fast debug runs).
    Returns (train_data, val_data).
    """
    import pandas as pd
    raw_path = os.path.join(data_dir, 'base_interactions_raw.parquet')
    print(f"Loading {raw_path} ...")
    raw_df = pd.read_parquet(raw_path)
    print(f"  {len(raw_df):,} raw interactions, {raw_df['user_id'].nunique():,} users")

    train_users = fs.train_users[:]
    val_users   = fs.val_users[:]

    if max_users is not None:
        rng = random.Random(seed)
        train_users = rng.sample(train_users, min(max_users, len(train_users)))
        print(f"  [debug] subsampled to {len(train_users):,} train users")

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
