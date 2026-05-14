"""
Stage 5 — Export serving artifacts.

Generates serving/ directory with three files:
  model.pth             — model state_dict (buffers excluded)
  book_embeddings.pt    — {bookId: {BOOK_EMBEDDING_COMBINED, sub-embeddings}}
  feature_store.pt      — inference-only dict (no user data)

book_shelf_matrix and book_author_idx are registered buffers (~135 MB together)
and are excluded from model.pth. They are stored in feature_store.pt so the
Streamlit app can reconstruct the model without needing training data parquets.

Usage:
    python main.py export
    python main.py export <checkpoint_path>
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Goodreads CDN serves three sizes via a single-letter suffix before the filename:
# `s` (50px), `m` (98px), `l` (~180px). The `image_url` field defaults to `m`,
# which is too small for retina displays at our ~100px render width — covers
# look fuzzy. Rewrite to `l` for crisper rendering; files stay ~15–20 KB.
_GR_SIZE_PATTERN = re.compile(r'(\d+)m(/\d+\.jpg)$')

from src.dataset import load_features
from src.evaluate import build_book_embeddings
from src.train import build_model, get_softmax_config

SERVING_DIR = 'serving'


def run_export(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if checkpoint_path is None:
        candidates = sorted(
            glob.glob(os.path.join('saved_models', 'best_full_softmax_4pool_*.pth')) +
            glob.glob(os.path.join('saved_models', 'full_softmax_4pool_*_step_*.pth')) +
            glob.glob(os.path.join('saved_models', 'best_softmax_4pool_*.pth')) +
            glob.glob(os.path.join('saved_models', 'softmax_4pool_*.pth')),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            print("No checkpoint found in saved_models/. Train a model first.")
            return
        checkpoint_path = candidates[0]

    print(f"Checkpoint: {checkpoint_path}")
    config_path = checkpoint_path.replace('.pth', '.json')
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loading config from: {config_path}")
    else:
        config = get_softmax_config()
        print(f"Config sidecar not found, using defaults from get_softmax_config()")

    print("Loading features ...")
    fs = load_features(data_dir, version)

    state_dict = torch.load(checkpoint_path, weights_only=True)
    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()

    print("Building book embeddings ...")
    book_embeddings = build_book_embeddings(model, fs)

    os.makedirs(SERVING_DIR, exist_ok=True)

    # ── model.pth ─────────────────────────────────────────────────────────────
    # Exclude registered buffers — book_shelf_matrix (~135MB) and book_author_idx
    # would push the file well over GitHub's 100MB limit. Both are stored in
    # feature_store.pt and restored at Streamlit startup via the constructor.
    model_state = {k: v for k, v in model.state_dict().items()
                   if k not in ('book_shelf_matrix', 'book_author_idx')}
    model_path  = os.path.join(SERVING_DIR, 'model.pth')
    torch.save(model_state, model_path)
    print(f"Saved {model_path}  ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # ── book_embeddings.pt ────────────────────────────────────────────────────
    emb_path = os.path.join(SERVING_DIR, 'book_embeddings.pt')
    torch.save(book_embeddings, emb_path)
    print(f"Saved {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    # ── Popularity ordering (for app dropdowns) ───────────────────────────────
    print("Computing popularity order ...")
    books_df      = pd.read_parquet(os.path.join(data_dir, 'base_books.parquet'))
    bookId_to_author = dict(zip(books_df['book_id'].astype(str), books_df.get('primary_author', '')))
    bookId_to_isbn   = dict(zip(books_df['book_id'].astype(str), books_df.get('isbn', '')))

    # Goodreads CDN cover URLs (images.gr-assets.com). Faster than Open Library
    # and covers ISBN-less popular books (Twilight, Gatsby, etc). ~67% coverage;
    # the `nophoto` sentinel is mapped to '' so callers can fall back to OL.
    print("Loading image URLs from goodreads_books.json ...")
    corpus_ids = set(fs.top_books)
    bookId_to_image_url = {}
    books_json_path = os.path.join(data_dir, 'goodreads_books.json')
    with open(books_json_path) as f:
        for line in tqdm(f, desc="  scanning books"):
            b = json.loads(line)
            bid = b.get('book_id', '')
            if not bid or bid not in corpus_ids:
                continue
            url = (b.get('image_url') or '').strip()
            if 'nophoto' in url:
                url = ''
            elif url:
                url = _GR_SIZE_PATTERN.sub(r'\1l\2', url)
            bookId_to_image_url[bid] = url
    real_covers = sum(1 for u in bookId_to_image_url.values() if u)
    print(f"  Image URLs: {real_covers:,}/{len(corpus_ids):,} corpus books have a cover "
          f"({100*real_covers/max(len(corpus_ids),1):.1f}%)")
    books_sorted  = books_df.sort_values('ratings_count', ascending=False)
    top_set       = set(fs.top_books)
    popularity_ordered_titles = [
        row['title']
        for _, row in books_sorted.iterrows()
        if row['book_id'] in top_set
    ]
    covered = set(popularity_ordered_titles)
    for bid in fs.top_books:
        t = fs.bookId_to_title.get(bid)
        if t and t not in covered:
            popularity_ordered_titles.append(t)

    # ── Rebuild buffers to store in feature_store ────────────────────────────
    n_books   = len(fs.top_books)
    n_shelves = len(fs.shelves_ordered)
    n_authors = len(fs.authors_ordered)
    n_genres  = len(fs.genres_ordered)

    shelf_matrix = np.array(
        [fs.bookId_to_shelf_context[bid] for bid in fs.top_books],
        dtype=np.float32,
    )
    # Store as float16 to keep feature_store.pt under GitHub's 100MB limit.
    # Cast back to float32 at app startup before passing to the model.
    book_shelf_matrix = torch.from_numpy(
        np.vstack([shelf_matrix, np.zeros((1, n_shelves), dtype=np.float32)])
    ).half()

    author_idx_arr = np.array(
        [fs.bookId_to_author_idx.get(bid, 0) for bid in fs.top_books],
        dtype=np.int64,
    )
    book_author_idx = torch.from_numpy(np.append(author_idx_arr, n_authors))

    # Non-persistent buffers — required at serving time for use_item_pool_for_history=True
    genre_matrix = np.array(
        [fs.bookId_to_genre_context[bid] for bid in fs.top_books], dtype=np.float32)
    book_genre_matrix = torch.from_numpy(
        np.vstack([genre_matrix, np.zeros((1, n_genres), dtype=np.float32)]))
    year_array = np.array(
        [fs.year_to_i.get(fs.bookId_to_year[bid], 0) for bid in fs.top_books], dtype=np.int64)
    book_year_idx = torch.from_numpy(
        np.concatenate([year_array, np.zeros((1,), dtype=np.int64)]))

    # ── feature_store.pt ──────────────────────────────────────────────────────
    feature_store = {
        'popularity_ordered_titles': popularity_ordered_titles,
        # Vocabularies
        'top_books':        fs.top_books,
        'genres_ordered':   fs.genres_ordered,
        'shelves_ordered':  fs.shelves_ordered,
        'years_ordered':    fs.years_ordered,
        'authors_ordered':  fs.authors_ordered,
        # Index maps
        'genre_to_i':       fs.genre_to_i,
        'shelf_to_i':       fs.shelf_to_i,
        'year_to_i':        fs.year_to_i,
        'author_to_i':      fs.author_to_i,
        'bookId_to_idx':    fs.bookId_to_idx,
        # Per-book metadata
        'bookId_to_title':  fs.bookId_to_title,
        'title_to_bookId':  fs.title_to_bookId,
        'bookId_to_year':   fs.bookId_to_year,
        'bookId_to_genres': fs.bookId_to_genres,
        'bookId_to_author': bookId_to_author,
        'bookId_to_isbn':   bookId_to_isbn,
        'bookId_to_image_url': bookId_to_image_url,
        # Context dicts stored as float32 arrays to reduce pickle overhead
        'bookId_to_genre_context': {
            bid: np.array(v, dtype=np.float32)
            for bid, v in fs.bookId_to_genre_context.items()
        },
        # Registered buffers (excluded from model.pth)
        'book_shelf_matrix':  book_shelf_matrix,
        'book_author_idx':    book_author_idx,
        # Non-persistent buffers — needed at serving time for ipool models
        'book_genre_matrix':  book_genre_matrix,
        'book_year_idx':      book_year_idx,
        # Derived constants
        'user_context_size':  fs.user_context_size,
        'timestamp_num_bins': fs.timestamp_num_bins,
        'timestamp_bins':     fs.timestamp_bins,
        # Model config — needed to reconstruct BookRecommender in the app
        'model_config': config,
    }
    fs_path = os.path.join(SERVING_DIR, 'feature_store.pt')
    torch.save(feature_store, fs_path)
    print(f"Saved {fs_path}  ({os.path.getsize(fs_path) / 1e6:.1f} MB)")

    total_mb = sum(
        os.path.getsize(os.path.join(SERVING_DIR, f)) / 1e6
        for f in ('model.pth', 'book_embeddings.pt', 'feature_store.pt')
    )
    print(f"\nTotal serving/ size: {total_mb:.1f} MB")
    print("Done. Verify with:")
    print("  python -c \"import torch; fs=torch.load('serving/feature_store.pt', "
          "weights_only=False); print(len(fs['top_books']), 'books')\"")
