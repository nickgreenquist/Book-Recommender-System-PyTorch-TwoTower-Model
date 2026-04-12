"""
Training loops: BPR/MSE (train) and in-batch negatives softmax (train_softmax).

Usage:
    python main.py train           # BPR or MSE (set in get_config)
    python main.py train softmax   # in-batch negatives cross-entropy
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
        'loss':             'bpr',   # 'mse' or 'bpr'
        'lr':               0.001,
        'weight_decay':     1e-4,
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
    genre_dim   = m.user_genre_tower[-2].out_features  # last Linear before final Tanh
    ts_dim      = m.timestamp_embedding_lookup.embedding_dim
    user_total  = history_dim + genre_dim + ts_dim

    item_genre_dim  = m.item_genre_tower[-2].out_features
    item_shelf_dim  = m.item_shelf_tower[-2].out_features
    item_book_dim   = m.item_embedding_tower[-2].out_features
    item_author_dim = m.author_tower[-2].out_features
    year_dim        = m.year_embedding_tower[-2].out_features
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

    train_data / val_data: 10-tuple from dataset.build_dataset()
      (X_genre, X_history, X_history_ratings, timestamp, Y,
       target_book_idx, target_genre, target_year, target_author_idx,
       bpr_pairs)
    """
    # [:9] ignores bpr_pairs (index 9) — loaded separately below for BPR
    (X_genre_train, X_history_train, X_history_ratings_train, timestamp_train,
     Y_train, target_book_idx_train, target_genre_train,
     target_year_train, target_author_idx_train) = train_data[:9]

    (X_genre_val, X_history_val, X_history_ratings_val, timestamp_val,
     Y_val, target_book_idx_val, target_genre_val,
     target_year_val, target_author_idx_val) = val_data[:9]

    use_bpr = config.get('loss', 'mse') == 'bpr'
    if use_bpr:
        bpr_pairs_train = train_data[9]   # (N_pairs, 2) — same-user (pos_row, neg_row)
        bpr_pairs_val   = val_data[9]
        print(f"  BPR mode: {len(bpr_pairs_train):,} train pairs, {len(bpr_pairs_val):,} val pairs")

    print_model_summary(model)

    pad_idx          = len(fs.top_books)
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'])
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

    print(f"\nStarting training loop  ({training_steps:,} steps, loss={config.get('loss','mse')}) ...")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            model.eval()
            with torch.no_grad():
                if use_bpr:
                    # Sample a fixed set of val pairs for BPR loss
                    n_val_pairs  = len(bpr_pairs_val)
                    sample_size  = min(8192, n_val_pairs)
                    vidx         = torch.randint(0, n_val_pairs, (sample_size,))
                    vpi          = bpr_pairs_val[vidx, 0].tolist()
                    vni          = bpr_pairs_val[vidx, 1].tolist()
                    vhp = pad_history_batch([X_history_val[j] for j in vpi], pad_idx)
                    vrp = pad_history_ratings_batch([X_history_ratings_val[j] for j in vpi])
                    vhn = pad_history_batch([X_history_val[j] for j in vni], pad_idx)
                    vrn = pad_history_ratings_batch([X_history_ratings_val[j] for j in vni])
                    sp  = model(X_genre_val[vpi], vhp, vrp, timestamp_val[vpi],
                                target_genre_val[vpi], target_year_val[vpi],
                                target_book_idx_val[vpi], target_author_idx_val[vpi])
                    sn  = model(X_genre_val[vni], vhn, vrn, timestamp_val[vni],
                                target_genre_val[vni], target_year_val[vni],
                                target_book_idx_val[vni], target_author_idx_val[vni])
                    output_val = -F.logsigmoid(sp - sn).mean().item()
                else:
                    val_batch_size = 1024
                    n_val          = X_genre_val.shape[0]
                    sq_sum         = 0.0
                    n_preds        = 0
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
            model.train()
            if use_bpr:
                n_pairs  = len(bpr_pairs_train)
                pidx     = torch.randint(0, n_pairs, (minibatch_size,))
                pi       = bpr_pairs_train[pidx, 0].tolist()
                ni       = bpr_pairs_train[pidx, 1].tolist()
                hp = pad_history_batch([X_history_train[j] for j in pi], pad_idx)
                rp = pad_history_ratings_batch([X_history_ratings_train[j] for j in pi])
                hn = pad_history_batch([X_history_train[j] for j in ni], pad_idx)
                rn = pad_history_ratings_batch([X_history_ratings_train[j] for j in ni])
                sp = model(X_genre_train[pi], hp, rp, timestamp_train[pi],
                           target_genre_train[pi], target_year_train[pi],
                           target_book_idx_train[pi], target_author_idx_train[pi])
                sn = model(X_genre_train[ni], hn, rn, timestamp_train[ni],
                           target_genre_train[ni], target_year_train[ni],
                           target_book_idx_train[ni], target_author_idx_train[ni])
                output = -F.logsigmoid(sp - sn).mean()
            else:
                ix         = torch.randint(0, X_genre_train.shape[0], (minibatch_size,)).tolist()
                hist_batch = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
                rat_batch  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
                preds  = model(X_genre_train[ix], hist_batch, rat_batch,
                               timestamp_train[ix],
                               target_genre_train[ix], target_year_train[ix],
                               target_book_idx_train[ix], target_author_idx_train[ix])
                output = torch.nn.functional.mse_loss(preds, Y_train[ix])
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


# ── Softmax training (in-batch negatives) ─────────────────────────────────────

def get_softmax_config() -> dict:
    """Hyperparameters for in-batch negatives softmax training."""
    item_id_embedding_size    = 40
    user_genre_embedding_size = 50
    timestamp_embedding_size  = 10
    item_genre_embedding_size = 10
    shelf_embedding_size      = 25
    author_embedding_size     = 15
    item_year_embedding_size  = 10

    user_dim = item_id_embedding_size + user_genre_embedding_size + timestamp_embedding_size
    item_dim = (item_genre_embedding_size + shelf_embedding_size + item_id_embedding_size
                + author_embedding_size + item_year_embedding_size)
    assert user_dim == item_dim, f"Tower size mismatch: user={user_dim} item={item_dim}"

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
        'weight_decay':     1e-5,   # lighter than BPR (1e-4 was for collapse prevention, not needed with softmax)
        'minibatch_size':   256,    # in-batch negatives: larger batch = more negatives
        'temperature':      0.05,   # softmax temperature (scales logits before cross-entropy)
        'training_steps':   150_000,
        'log_every':        10_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


def train_softmax(model: BookRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: FeatureStore) -> str:
    """
    Train with in-batch negatives cross-entropy (softmax).

    train_data / val_data: 8-tuple from dataset.build_softmax_dataset()
      (X_genre, X_history, X_history_ratings, timestamp,
       target_book_idx, target_genre, target_year, target_author_idx)

    Each minibatch of size B computes:
      U = user_embedding(...)         (B, dim)
      V = item_embedding(...)         (B, dim)
      scores = U @ V.T / temperature  (B, B)
      loss   = cross_entropy(scores, arange(B))   target is always on the diagonal
    """
    (X_genre_train, X_history_train, X_history_ratings_train, timestamp_train,
     target_book_idx_train, target_genre_train,
     target_year_train, target_author_idx_train) = train_data

    (X_genre_val, X_history_val, X_history_ratings_val, timestamp_val,
     target_book_idx_val, target_genre_val,
     target_year_val, target_author_idx_val) = val_data

    print_model_summary(model)

    pad_idx          = len(fs.top_books)
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'])
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    training_steps   = config['training_steps']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_softmax_{run_timestamp}.pth')

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
                # Sample a fixed val batch
                vidx = torch.randint(0, n_val, (minibatch_size,)).tolist()
                vhp  = pad_history_batch([X_history_val[j] for j in vidx], pad_idx)
                vrp  = pad_history_ratings_batch([X_history_ratings_val[j] for j in vidx])
                U = model.user_embedding(X_genre_val[vidx], vhp, vrp, timestamp_val[vidx])
                V = model.item_embedding(target_genre_val[vidx], target_year_val[vidx],
                                         target_book_idx_val[vidx], target_author_idx_val[vidx])
                scores    = (U @ V.T) / temperature        # (B, B)
                labels    = torch.arange(len(vidx))
                val_loss  = F.cross_entropy(scores, labels).item()

                if i == 0:
                    raw_scores = U @ V.T
                    print(f"  [logit diagnostics] raw dot products — "
                          f"mean={raw_scores.mean().item():.3f}  "
                          f"std={raw_scores.std().item():.3f}  "
                          f"min={raw_scores.min().item():.3f}  "
                          f"max={raw_scores.max().item():.3f}")
                    print(f"  [logit diagnostics] after /temp={temperature} — "
                          f"mean={scores.mean().item():.3f}  "
                          f"std={scores.std().item():.3f}  "
                          f"min={scores.min().item():.3f}  "
                          f"max={scores.max().item():.3f}")
                    print(f"  [logit diagnostics] random baseline loss = {np.log(minibatch_size):.4f}")
            avg_train = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            elapsed   = time.time() - start
            start     = time.time()
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'softmax_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            ix  = torch.randint(0, n_train, (minibatch_size,)).tolist()
            hp  = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
            rp  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
            U   = model.user_embedding(X_genre_train[ix], hp, rp, timestamp_train[ix])
            V   = model.item_embedding(target_genre_train[ix], target_year_train[ix],
                                       target_book_idx_train[ix], target_author_idx_train[ix])
            scores = (U @ V.T) / temperature               # (B, B)
            labels = torch.arange(len(ix))
            loss   = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

    print(f"\nSoftmax training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
