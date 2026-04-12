"""
Stage 3 — Dataset Loading
Reads features_*.parquet into a FeatureStore, builds PyTorch tensors.
No files are written here — pure in-memory.

Usage (from train.py or main.py):
    from src.dataset import load_features, make_splits
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
    user_ids                        = []
    user_to_avg_rating              = {}
    user_to_genre_context           = {}
    user_to_read_history            = {}
    user_to_read_history_ratings    = {}
    user_to_book_to_rating_LABEL    = {}
    user_to_book_to_timestamp_LABEL = {}

    from tqdm import tqdm
    for _, row in tqdm(user_feat_df.iterrows(), total=len(user_feat_df), desc="Loading user features"):
        uid = row['user_id']
        user_ids.append(uid)
        user_to_avg_rating[uid]           = float(row['avg_rating'])
        user_to_genre_context[uid]        = list(row['genre_context'])
        user_to_read_history[uid]         = list(row['read_history'])
        user_to_read_history_ratings[uid] = list(row['read_history_ratings'])

        lbl_books = list(row['label_bookIds'])
        lbl_rats  = list(row['label_ratings'])
        lbl_times = list(row['label_timestamps'])
        user_to_book_to_rating_LABEL[uid]    = dict(zip(lbl_books, lbl_rats))
        user_to_book_to_timestamp_LABEL[uid] = dict(zip(lbl_books, lbl_times))

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


# ── Dataset builder ───────────────────────────────────────────────────────────

BPR_POS_THRESHOLD =  0.5   # debiased rating above this → positive
BPR_NEG_THRESHOLD = -0.5   # debiased rating below this → negative


def build_dataset(users: list, fs: FeatureStore) -> tuple:
    """
    Build training/validation tensors for a list of user IDs.
    Returns a tuple of 10 elements:
      [0] X_genre, [1] X_history (list), [2] X_history_ratings (list),
      [3] timestamp, [4] Y, [5] target_book_idx,
      [6] target_genre_context, [7] target_year, [8] target_author_idx,
      [9] bpr_pairs — LongTensor (N_pairs, 2): same-user (pos_row, neg_row) index pairs

    Indices 0-8 are unchanged from v1 — existing MSE training code is unaffected.
    bpr_pairs uses BPR_POS_THRESHOLD / BPR_NEG_THRESHOLD on debiased Y values.

    Note: shelf context is NOT stored per sample — neither user-side nor item-side.
    The model looks up shelf vectors from book_shelf_matrix (registered buffer) using
    read_history and target_book_idx at forward pass time.
    """
    X_genre               = []
    X_history             = []
    X_history_ratings     = []
    timestamp             = []
    Y                     = []
    target_book_idx       = []
    target_genre_context  = []
    target_year           = []
    target_author_idx     = []
    user_rows             = {}   # user → list of row indices (for BPR pairing)

    from tqdm import tqdm
    for user in tqdm(users, desc="Collecting samples"):
        user_row_indices = []
        for book_id, rating in fs.user_to_book_to_rating_LABEL[user].items():
            if book_id not in fs.bookId_to_idx:
                continue
            row_idx = len(Y)
            X_genre.append(fs.user_to_genre_context[user])
            X_history.append(fs.user_to_read_history[user])
            X_history_ratings.append(fs.user_to_read_history_ratings[user])
            timestamp.append(fs.user_to_book_to_timestamp_LABEL[user][book_id])
            Y.append(float(rating) - fs.user_to_avg_rating[user])
            target_book_idx.append(fs.bookId_to_idx[book_id])
            target_genre_context.append(fs.bookId_to_genre_context[book_id])
            target_year.append(fs.year_to_i.get(fs.bookId_to_year[book_id], 0))
            target_author_idx.append(fs.bookId_to_author_idx[book_id])
            user_row_indices.append(row_idx)
        if user_row_indices:
            user_rows[user] = user_row_indices

    n = len(Y)
    print(f"  {n:,} samples — building tensors ...")

    print("  X_genre, Y ...")
    X_genre_t = torch.from_numpy(np.array(X_genre, dtype=np.float32))
    Y_t       = torch.from_numpy(np.array(Y,       dtype=np.float32))

    print("  target_book_idx, year, author, timestamp ...")
    target_book_idx_t  = torch.from_numpy(np.array(target_book_idx,  dtype=np.int64))
    target_year_t      = torch.from_numpy(np.array(target_year,      dtype=np.int64))
    target_author_idx_t = torch.from_numpy(np.array(target_author_idx, dtype=np.int64))
    timestamp_t = torch.bucketize(
        torch.from_numpy(np.array(timestamp, dtype=np.float64)).float(),
        fs.timestamp_bins.float(), right=False)

    print("  genre context ...")
    target_genre_t = torch.from_numpy(np.array(target_genre_context, dtype=np.float32))

    print("  BPR pairs ...")
    Y_arr = np.array(Y, dtype=np.float32)
    bpr_pairs = []
    for rows in user_rows.values():
        pos = [r for r in rows if Y_arr[r] >  BPR_POS_THRESHOLD]
        neg = [r for r in rows if Y_arr[r] < BPR_NEG_THRESHOLD]
        for p in pos:
            for q in neg:
                bpr_pairs.append((p, q))
    bpr_pairs_t = torch.tensor(bpr_pairs, dtype=torch.long) if bpr_pairs else torch.zeros((0, 2), dtype=torch.long)
    print(f"  {len(bpr_pairs):,} BPR pairs from {sum(1 for r in user_rows.values() if any(Y_arr[i] > BPR_POS_THRESHOLD for i in r) and any(Y_arr[i] < BPR_NEG_THRESHOLD for i in r)):,} users with both pos and neg labels")

    return (X_genre_t, X_history, X_history_ratings, timestamp_t, Y_t,
            target_book_idx_t, target_genre_t,
            target_year_t, target_author_idx_t,
            bpr_pairs_t)


# ── Disk cache helpers ────────────────────────────────────────────────────────

def save_splits(train_data: tuple, val_data: tuple,
                data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_val_{version}.pt'))
    print(f"Saved dataset_train_{version}.pt and dataset_val_{version}.pt → {data_dir}/")


def load_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data


# ── Train / val split ─────────────────────────────────────────────────────────

def make_splits(fs: FeatureStore, pct_train: float = 0.9, seed: int = 42) -> tuple:
    """
    Split users into train/val, build tensors for each.
    Returns (train_data, val_data) where each is the 10-tuple from build_dataset().
    """
    final_users = [
        u for u in fs.user_ids
        if 2 <= len(fs.user_to_book_to_rating_LABEL.get(u, {})) < 500
    ]
    print(f"Final users for training: {len(final_users):,}  "
          f"(skipped {len(fs.user_ids) - len(final_users):,})")

    rng = random.Random(seed)
    rng.shuffle(final_users)
    split       = int(len(final_users) * pct_train)
    train_users = final_users[:split]
    val_users   = final_users[split:]

    print(f"Building train dataset ({len(train_users):,} users) ...")
    train_data = build_dataset(train_users, fs)
    print(f"  X_genre_train shape: {train_data[0].shape}")

    print(f"Building val dataset ({len(val_users):,} users) ...")
    val_data = build_dataset(val_users, fs)
    print(f"  X_genre_val shape:   {val_data[0].shape}")

    return train_data, val_data
