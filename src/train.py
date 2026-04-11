"""
Training loop.

Usage:
    python main.py train
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.model import BookRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_config() -> dict:
    """All training hyperparameters in one place."""
    # Shared dim (user history pool + item book tower — cannot be set independently)
    item_id_embedding_size    = 40

    # User-only
    user_genre_embedding_size = 50
    timestamp_embedding_size  = 10

    # Item-only (shelf and author are no longer on the user side)
    item_genre_embedding_size = 10
    shelf_embedding_size      = 25
    author_embedding_size     = 15
    item_year_embedding_size  = 10

    user_dim = (item_id_embedding_size + user_genre_embedding_size
                + timestamp_embedding_size)
    item_dim = (item_genre_embedding_size + shelf_embedding_size
                + item_id_embedding_size + author_embedding_size
                + item_year_embedding_size)
    assert user_dim == item_dim, (
        f"Tower size mismatch — user={user_dim} "
        f"(history={item_id_embedding_size} + genre={user_genre_embedding_size} "
        f"+ ts={timestamp_embedding_size}), "
        f"item={item_dim} "
        f"(genre={item_genre_embedding_size} + shelf={shelf_embedding_size} "
        f"+ book={item_id_embedding_size} + author={author_embedding_size} "
        f"+ year={item_year_embedding_size})"
    )

    return {
        'item_id_embedding_size':    item_id_embedding_size,
        'author_embedding_size':     author_embedding_size,
        'shelf_embedding_size':      shelf_embedding_size,
        'user_genre_embedding_size': user_genre_embedding_size,
        'timestamp_embedding_size':  timestamp_embedding_size,
        'item_genre_embedding_size': item_genre_embedding_size,
        'item_year_embedding_size':  item_year_embedding_size,
        # Training
        'lr':               0.001,
        'minibatch_size':   64,
        'training_steps':   150_000,
        'log_every':        10_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: FeatureStore) -> BookRecommender:
    n_books   = len(fs.top_books)
    n_shelves = len(fs.shelves_ordered)
    n_authors = len(fs.authors_ordered)  # includes __unknown__ at index 0

    # book_shelf_matrix: (n_books+1, n_shelves) — last row = zeros (padding)
    shelf_matrix = np.array(
        [fs.bookId_to_shelf_context[bid] for bid in fs.top_books],
        dtype=np.float32,
    )
    pad_row = np.zeros((1, n_shelves), dtype=np.float32)
    book_shelf_matrix = torch.from_numpy(np.vstack([shelf_matrix, pad_row]))

    # book_author_idx: (n_books+1,) — maps book_idx → author vocab index
    # last entry = n_authors (author padding index)
    author_idx_arr = np.array(
        [fs.bookId_to_author_idx.get(bid, 0) for bid in fs.top_books],
        dtype=np.int64,
    )
    book_author_idx = torch.from_numpy(
        np.append(author_idx_arr, n_authors)  # padding entry at end
    )

    model = BookRecommender(
        n_genres=len(fs.genres_ordered),
        n_shelves=n_shelves,
        n_books=n_books,
        n_years=len(fs.years_ordered),
        n_authors=n_authors,
        n_timestamp_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        book_shelf_matrix=book_shelf_matrix,
        book_author_idx=book_author_idx,
        item_id_embedding_size=config['item_id_embedding_size'],
        author_embedding_size=config['author_embedding_size'],
        shelf_embedding_size=config['shelf_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        timestamp_embedding_size=config['timestamp_embedding_size'],
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
    )
    return model


def print_model_summary(model: BookRecommender) -> None:
    m = model
    history_dim = m.item_embedding_lookup.embedding_dim
    genre_dim   = m.user_genre_tower[0].out_features
    ts_dim      = m.timestamp_embedding_lookup.embedding_dim
    user_total  = history_dim + genre_dim + ts_dim

    item_genre_dim  = m.item_genre_tower[0].out_features
    item_shelf_dim  = m.item_shelf_tower[0].out_features
    item_book_dim   = m.item_embedding_tower[0].out_features
    item_author_dim = m.author_tower[0].out_features
    year_dim        = m.year_embedding_tower[0].out_features
    item_total      = item_genre_dim + item_shelf_dim + item_book_dim + item_author_dim + year_dim

    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  history({history_dim}) + genre({genre_dim}) + ts({ts_dim})  =  {user_total}")
    print(f"  Item side:  genre({item_genre_dim}) + shelf({item_shelf_dim})"
          f" + book({item_book_dim}) + author({item_author_dim}) + year({year_dim})  =  {item_total}")
    print(f"  Parameters: {n_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: BookRecommender, train_data: tuple, val_data: tuple,
          config: dict, fs: FeatureStore) -> str:
    """
    Run the training loop. Returns the path of the best checkpoint.

    train_data / val_data: 9-tuple from dataset.build_dataset()
      (X_genre, X_history, X_history_ratings, timestamp, Y,
       target_book_idx, target_genre, target_year, target_author_idx)
    """
    (X_genre_train, X_history_train, X_history_ratings_train, timestamp_train,
     Y_train, target_book_idx_train, target_genre_train,
     target_year_train, target_author_idx_train) = train_data

    (X_genre_val, X_history_val, X_history_ratings_val, timestamp_val,
     Y_val, target_book_idx_val, target_genre_val,
     target_year_val, target_author_idx_val) = val_data

    print_model_summary(model)

    pad_idx          = len(fs.top_books)
    loss_fn          = torch.nn.MSELoss()
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'])
    minibatch_size   = config['minibatch_size']
    training_steps   = config['training_steps']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_checkpoint_{run_timestamp}.pth')

    loss_train = []
    loss_val   = []

    print(f"\nStarting training loop  ({training_steps:,} steps) ...")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            val_batch_size = 1024
            n_val          = X_genre_val.shape[0]
            sq_sum         = 0.0
            n_preds        = 0
            model.eval()
            with torch.no_grad():
                for v0 in range(0, n_val, val_batch_size):
                    v1  = min(v0 + val_batch_size, n_val)
                    vix = list(range(v0, v1))
                    hb  = pad_history_batch([X_history_val[j] for j in vix], pad_idx)
                    rb  = pad_history_ratings_batch([X_history_ratings_val[j] for j in vix])
                    vp  = model(X_genre_val[v0:v1], hb, rb, timestamp_val[v0:v1],
                                target_genre_val[v0:v1], target_year_val[v0:v1],
                                target_book_idx_val[v0:v1], target_author_idx_val[v0:v1])
                    sq_sum  += ((vp - Y_val[v0:v1]) ** 2).sum().item()
                    n_preds += (v1 - v0)
            output_val = sq_sum / n_preds
            loss_val.append(output_val)
        else:
            ix         = torch.randint(0, X_genre_train.shape[0], (minibatch_size,)).tolist()
            hist_batch = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
            rat_batch  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
            model.train()
            preds = model(X_genre_train[ix], hist_batch, rat_batch,
                          timestamp_train[ix],
                          target_genre_train[ix], target_year_train[ix],
                          target_book_idx_train[ix], target_author_idx_train[ix])
            output = loss_fn(preds, Y_train[ix])
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            loss_train.append(output.item())

        if is_val:
            elapsed   = time.time() - start
            start     = time.time()
            avg_train = np.mean(loss_train[i-log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            val_loss  = output_val
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'checkpoint_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
