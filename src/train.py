"""
In-batch negatives softmax training.

Usage:
    python main.py train
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.model import BookRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_softmax_config() -> dict:
    """All training hyperparameters in one place."""
    return {
        # Sub-embedding sizes — projection MLP handles cross-feature interactions.
        'item_id_embedding_size':    32,   # shared: item tower lookup + user history pool
        'user_genre_embedding_size': 32,
        'timestamp_embedding_size':  8,
        'item_genre_embedding_size': 10,
        'shelf_embedding_size':      40,   # richest signal: 3032-dim TF-IDF → 2-layer MLP
        'author_embedding_size':     10,
        'item_year_embedding_size':  8,
        # item concat: 10+40+32+10+8 = 100; user concat (ipool): 128+32+8 = 168
        # Projection MLP  (concat → Linear(proj_hidden) → ReLU → Linear(output_dim))
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.001,
        'weight_decay':     1e-5,
        'minibatch_size':   512,    # in-batch negatives: larger batch = more negatives
        'temperature':      0.05,   # softmax temperature (scales logits before cross-entropy)
        'training_steps':   150_000,
        'log_every':        5_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: FeatureStore) -> BookRecommender:
    n_books   = len(fs.top_books)
    n_shelves = len(fs.shelves_ordered)
    n_authors = len(fs.authors_ordered)  # includes __unknown__ at index 0

    # book_shelf_matrix: (n_books+1, n_shelves) — last row = zeros  [persistent]
    shelf_matrix = np.array(
        [fs.bookId_to_shelf_context[bid] for bid in fs.top_books], dtype=np.float32)
    book_shelf_matrix = torch.from_numpy(
        np.vstack([shelf_matrix, np.zeros((1, n_shelves), dtype=np.float32)]))

    # book_author_idx: (n_books+1,) — last entry = n_authors (padding)  [persistent]
    author_idx_arr = np.array(
        [fs.bookId_to_author_idx.get(bid, 0) for bid in fs.top_books], dtype=np.int64)
    book_author_idx = torch.from_numpy(np.append(author_idx_arr, n_authors))

    # book_genre_matrix: (n_books+1, n_genres) — last row = zeros  [non-persistent]
    genre_matrix = np.array(
        [fs.bookId_to_genre_context[bid] for bid in fs.top_books], dtype=np.float32)
    book_genre_matrix = torch.from_numpy(
        np.vstack([genre_matrix, np.zeros((1, genre_matrix.shape[1]), dtype=np.float32)]))

    # book_year_idx: (n_books+1,) — last entry = 0  [non-persistent]
    year_array = np.array(
        [fs.year_to_i.get(fs.bookId_to_year[bid], 0) for bid in fs.top_books], dtype=np.int64)
    book_year_idx = torch.from_numpy(np.concatenate([year_array, np.zeros((1,), dtype=np.int64)]))

    return BookRecommender(
        n_genres=len(fs.genres_ordered),
        n_shelves=n_shelves,
        n_books=n_books,
        n_years=len(fs.years_ordered),
        n_authors=n_authors,
        n_timestamp_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        book_shelf_matrix=book_shelf_matrix,
        book_author_idx=book_author_idx,
        book_genre_matrix=book_genre_matrix,
        book_year_idx=book_year_idx,
        item_id_embedding_size=config['item_id_embedding_size'],
        author_embedding_size=config['author_embedding_size'],
        shelf_embedding_size=config['shelf_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        timestamp_embedding_size=config['timestamp_embedding_size'],
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        proj_hidden=config['proj_hidden'],
        output_dim=config['output_dim'],
    )


def print_model_summary(model: BookRecommender) -> None:
    m = model
    genre_dim = m.user_genre_tower[-2].out_features
    ts_dim    = m.timestamp_embedding_lookup.embedding_dim

    item_genre_dim  = m.item_genre_tower[-2].out_features
    item_shelf_dim  = m.item_shelf_tower[-2].out_features
    item_book_dim   = m.item_embedding_tower[-2].out_features
    item_author_dim = m.author_tower[-2].out_features
    year_dim        = m.year_embedding_tower[-2].out_features
    item_concat     = item_genre_dim + item_shelf_dim + item_book_dim + item_author_dim + year_dim

    output_dim = m.output_dim
    pool_dim   = m.user_projection[2].out_features
    user_total = pool_dim + genre_dim + ts_dim

    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  item_pool({pool_dim}) + genre({genre_dim}) + ts({ts_dim})"
          f"  =  {user_total}  → proj →  {output_dim}")
    print(f"  Item side:  genre({item_genre_dim}) + shelf({item_shelf_dim})"
          f" + book({item_book_dim}) + author({item_author_dim}) + year({year_dim})"
          f"  =  {item_concat}  → proj →  {output_dim}")
    print(f"  Parameters: {n_params:,}")


# ── Softmax training (in-batch negatives) ─────────────────────────────────────

def train_softmax(model: BookRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: FeatureStore) -> str:
    """
    Train with in-batch negatives cross-entropy (softmax).

    train_data / val_data: 5-tuple from dataset.build_softmax_dataset()
      (X_genre, X_history, X_history_ratings, timestamp, target_book_idx)
    Item features are looked up from model buffers during item_embedding().

    Each minibatch of size B computes:
      U = user_embedding(...)         (B, dim)
      V = item_embedding(...)         (B, dim)
      scores = U @ V.T / temperature  (B, B)
      loss   = cross_entropy(scores, arange(B))   target is always on the diagonal
    """
    (X_genre_train, X_history_train, X_history_ratings_train, timestamp_train,
     target_book_idx_train) = train_data

    (X_genre_val, X_history_val, X_history_ratings_val, timestamp_val,
     target_book_idx_val) = val_data

    print_model_summary(model)

    pad_idx          = len(fs.top_books)
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=0)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_proj_softmax_ipool_{run_timestamp}.pth')

    loss_train = []

    print(f"\nStarting softmax training ({training_steps:,} steps, "
          f"batch={minibatch_size}, temp={temperature}) ...")
    print(f"  Train: {n_train:,} examples  |  Val: {n_val:,} examples")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (softmax)")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            model.eval()
            with torch.no_grad():
                if i == 0:
                    # Logit diagnostics on one batch at step 0
                    vidx       = torch.randint(0, n_val, (minibatch_size,)).tolist()
                    vhp        = pad_history_batch([X_history_val[j] for j in vidx], pad_idx)
                    vrp        = pad_history_ratings_batch([X_history_ratings_val[j] for j in vidx])
                    U          = model.user_embedding(X_genre_val[vidx], vhp, vrp, timestamp_val[vidx])
                    V          = model.item_embedding(target_book_idx_val[vidx])
                    raw_scores = U @ V.T
                    scores     = raw_scores / temperature
                    print(f"  [logit diagnostics] raw dot products — "
                          f"mean={raw_scores.mean().item():.6f}  "
                          f"std={raw_scores.std().item():.6f}  "
                          f"min={raw_scores.min().item():.6f}  "
                          f"max={raw_scores.max().item():.6f}")
                    print(f"  [logit diagnostics] after /temp={temperature} — "
                          f"mean={scores.mean().item():.6f}  "
                          f"std={scores.std().item():.6f}  "
                          f"min={scores.min().item():.6f}  "
                          f"max={scores.max().item():.6f}")
                    print(f"  [logit diagnostics] random baseline loss = {np.log(minibatch_size):.4f}")

                # Full val: iterate over all val examples in batches, average cross-entropy
                val_loss_sum = 0.0
                val_batches  = 0
                for v0 in range(0, n_val, minibatch_size):
                    v1   = min(v0 + minibatch_size, n_val)
                    vidx = list(range(v0, v1))
                    vhp  = pad_history_batch([X_history_val[j] for j in vidx], pad_idx)
                    vrp  = pad_history_ratings_batch([X_history_ratings_val[j] for j in vidx])
                    U    = model.user_embedding(X_genre_val[v0:v1], vhp, vrp, timestamp_val[v0:v1])
                    V    = model.item_embedding(target_book_idx_val[v0:v1])
                    scores = (U @ V.T) / temperature
                    labels = torch.arange(len(vidx))
                    val_loss_sum += F.cross_entropy(scores, labels).item()
                    val_batches  += 1
                val_loss = val_loss_sum / val_batches
            avg_train  = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.6f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'proj_softmax_ipool_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            ix  = torch.randint(0, n_train, (minibatch_size,)).tolist()
            hp  = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
            rp  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
            U   = model.user_embedding(X_genre_train[ix], hp, rp, timestamp_train[ix])
            V   = model.item_embedding(target_book_idx_train[ix])
            scores = (U @ V.T) / temperature                                     # (B, B)
            labels = torch.arange(len(ix))
            loss   = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())

    print(f"\nSoftmax training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
