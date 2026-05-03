"""
In-batch negatives softmax training.

Usage:
    python main.py train
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from src.dataset import FeatureStore
from src.model import BookRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_softmax_config() -> dict:
    """All training hyperparameters in one place."""
    config = {
        # Sub-embedding sizes — projection MLP handles cross-feature interactions.
        'item_id_embedding_size':    32,   # shared: item tower lookup + user history pool
        'user_genre_embedding_size': 16,
        'user_shelf_affinity_embedding_size': 64,
        'timestamp_embedding_size':  8,
        'item_genre_embedding_size': 16,
        'shelf_embedding_size':      64,   # richest signal: 3032-dim TF-IDF → 2-layer MLP
        'author_embedding_size':     16,
        'item_year_embedding_size':  8,
        # item concat: 10+40+32+10+8 = 100; user concat (v2): 4*32+16+64+8 = 216
        # Projection MLP  (concat → Linear(proj_hidden) → ReLU → Linear(output_dim))
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.001,
        'weight_decay':     0.0,
        'adam_eps':         1e-6,
        'minibatch_size':   512,
        'training_steps':   150_000,
        'log_every':        5_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
        # Full softmax scores against all ~11k books per step — temperature needs to be
        # softer than in-batch negatives (0.05) because 11k negatives already provide
        # strong contrast. 0.5/batch_size gave ~0.001 (effectively argmax), causing
        # gradient collapse onto the hardest negative and popularity overfitting.
        'temperature':      0.1,
        # Menon et al. 2021 logit adjustment: add alpha*log1p(count_i) to all logits.
        # Popular items get a free boost → easy positives → embeddings shrink naturally.
        # Rare items must fight for high scores → embeddings grow.
        # Raw dot products at inference are then debiased without any post-hoc correction.
        'popularity_alpha': 0.2,
    }
    return config


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
        user_shelf_affinity_embedding_size=config['user_shelf_affinity_embedding_size'],
        timestamp_embedding_size=config['timestamp_embedding_size'],
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        proj_hidden=config['proj_hidden'],
        output_dim=config['output_dim'],
    )


def print_model_summary(model: BookRecommender) -> None:
    m = model
    genre_dim = m.user_genre_tower[-2].out_features
    shelf_aff_dim = m.user_shelf_affinity_tower[-2].out_features
    ts_dim    = m.timestamp_embedding_lookup.embedding_dim

    item_genre_dim  = m.item_genre_tower[-2].out_features
    item_shelf_dim  = m.item_shelf_tower[-2].out_features
    item_book_dim   = m.item_embedding_tower[-2].out_features
    item_author_dim = m.author_tower[-2].out_features
    year_dim        = m.year_embedding_tower[-2].out_features
    item_concat     = item_genre_dim + item_shelf_dim + item_book_dim + item_author_dim + year_dim

    output_dim = m.output_dim
    pool_dim   = 4 * m.item_embedding_lookup.embedding_dim
    user_total = pool_dim + genre_dim + shelf_aff_dim + ts_dim

    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  shallow_pools({pool_dim}) + genre({genre_dim}) + shelf_aff({shelf_aff_dim}) + ts({ts_dim})"
          f"  =  {user_total}  → proj →  {output_dim}")
    print(f"  Item side:  genre({item_genre_dim}) + shelf({item_shelf_dim})"
          f" + book({item_book_dim}) + author({item_author_dim}) + year({year_dim})"
          f"  =  {item_concat}  → proj →  {output_dim}")
    print(f"  Parameters: {n_params:,}")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ── Softmax training (in-batch negatives) ─────────────────────────────────────

def train_softmax(model: BookRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: FeatureStore) -> str:
    """
    Train with full softmax cross-entropy.

    train_data / val_data: 8-tuple from dataset.build_softmax_dataset()
      (X_genre, X_hist_full, X_hist_liked, X_hist_disliked,
       X_hist_weighted, X_rats_weighted, timestamp, target_book_idx)
    History tensors are pre-padded; item features looked up from model buffers.

    Each minibatch of size B computes:
      U = user_embedding(...)         (B, dim)
      V = item_embedding(...)         (B, dim)
      scores = U @ V.T / temperature  (B, B)
      loss   = cross_entropy(scores, arange(B))   target is always on the diagonal
    """
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)

    (X_genre_train, X_hist_full_train, X_hist_liked_train, X_hist_disliked_train,
     X_hist_weighted_train, X_rats_weighted_train, timestamp_train,
     target_book_idx_train) = train_data

    (X_genre_val, X_hist_full_val, X_hist_liked_val, X_hist_disliked_val,
     X_hist_weighted_val, X_rats_weighted_val, timestamp_val,
     target_book_idx_val) = val_data

    # Move compact tensors to device; history tensors stay on CPU (too large), moved per batch
    X_genre_train = X_genre_train.to(device)
    timestamp_train = timestamp_train.to(device)
    target_book_idx_train = target_book_idx_train.to(device)

    X_genre_val = X_genre_val.to(device)
    timestamp_val = timestamp_val.to(device)
    target_book_idx_val = target_book_idx_val.to(device)

    print_model_summary(model)

    pad_idx          = len(fs.top_books)

    # Popularity logit adjustment (Menon et al. 2021)
    alpha = config.get('popularity_alpha', 0.0)
    counts_cpu = torch.bincount(target_book_idx_train.cpu(), minlength=pad_idx).float()
    popularity_bias = (alpha * torch.log1p(counts_cpu)).to(device)
    print(f"  Popularity bias: alpha={alpha}  "
          f"max_adj={popularity_bias.max():.3f}  min_adj={popularity_bias.min():.3f}")

    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'],
                                        eps=config['adam_eps'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=1e-4)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_tag     = str(alpha).replace('.', '') if alpha != int(alpha) else str(int(alpha))
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_full_softmax_4pool_alpha_{alpha_tag}_{run_timestamp}.pth')

    # Fixed val subset — sampled once so val_loss is comparable across steps
    val_eval_size = min(8_192, n_val)
    rng_val = torch.Generator()
    rng_val.manual_seed(0)
    val_eval_idx = torch.randperm(n_val, generator=rng_val)[:val_eval_size].tolist()

    loss_train = []
    grad_norms = []

    print(f"\nStarting softmax training ({training_steps:,} steps, "
          f"batch={minibatch_size}, temp={temperature:.6f}) ...")
    print(f"  Train: {n_train:,} examples  |  Val: {n_val:,} examples (eval subset: {val_eval_size:,})")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (softmax)")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            model.eval()
            with torch.no_grad():
                V_all = model.full_item_embedding()
                if i == 0:
                    # Logit diagnostics on one batch at step 0
                    vidx = torch.randint(0, n_val, (minibatch_size,))
                    h_full     = X_hist_full_val[vidx].to(device)
                    h_liked    = X_hist_liked_val[vidx].to(device)
                    h_disliked = X_hist_disliked_val[vidx].to(device)
                    h_weighted = X_hist_weighted_val[vidx].to(device)
                    r_weighted = X_rats_weighted_val[vidx].to(device)

                    U = model.user_embedding(X_genre_val[vidx], h_full, h_liked, h_disliked,
                                             h_weighted, r_weighted, timestamp_val[vidx])
                    raw_scores = U @ V_all.T
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
                    print(f"  [logit diagnostics] random baseline loss = {np.log(V_all.shape[0]):.4f}")

                # Fixed val subset — same examples every step for comparable loss
                val_losses = []
                for vs in range(0, val_eval_size, minibatch_size):
                    ve   = min(vs + minibatch_size, val_eval_size)
                    vidx = val_eval_idx[vs:ve]

                    h_full     = X_hist_full_val[vidx].to(device)
                    h_liked    = X_hist_liked_val[vidx].to(device)
                    h_disliked = X_hist_disliked_val[vidx].to(device)
                    h_weighted = X_hist_weighted_val[vidx].to(device)
                    r_weighted = X_rats_weighted_val[vidx].to(device)

                    U = model.user_embedding(X_genre_val[vidx], h_full, h_liked, h_disliked,
                                             h_weighted, r_weighted, timestamp_val[vidx])
                    scores = (U @ V_all.T) / temperature
                    labels = target_book_idx_val[vidx]
                    val_losses.append(F.cross_entropy(scores, labels).item())
                val_loss = float(np.mean(val_losses))
            avg_train     = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            avg_grad_norm = np.mean(grad_norms[i - log_every:i]) if i >= log_every else (grad_norms[-1] if grad_norms else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.6f}  grad_norm={avg_grad_norm:.3f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                with open(best_path.replace('.pth', '.json'), 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'full_softmax_4pool_alpha_{alpha_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                with open(periodic.replace('.pth', '.json'), 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            V_all = model.full_item_embedding()
            ix         = torch.randint(0, n_train, (minibatch_size,))
            h_full     = X_hist_full_train[ix].to(device)
            h_liked    = X_hist_liked_train[ix].to(device)
            h_disliked = X_hist_disliked_train[ix].to(device)
            h_weighted = X_hist_weighted_train[ix].to(device)
            r_weighted = X_rats_weighted_train[ix].to(device)

            U = model.user_embedding(X_genre_train[ix], h_full, h_liked, h_disliked,
                                     h_weighted, r_weighted, timestamp_train[ix])

            scores = (U @ V_all.T) / temperature + popularity_bias
            labels = target_book_idx_train[ix]
            loss   = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())
            grad_norms.append(grad_norm)

    print(f"\nSoftmax training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
