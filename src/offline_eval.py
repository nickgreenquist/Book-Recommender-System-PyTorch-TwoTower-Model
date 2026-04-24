"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Protocol: leave-label-out per user.
  Context = user_to_read_history (90% of reads, pre-mapped book indices)
  Targets = user_to_book_to_rating_LABEL (remaining 10% of reads)

No new splits needed — reuses the existing 90/10 split from preprocess.py.
Eval is restricted to the validation user set (last 10% of the shuffle used
in make_softmax_splits, seed=42, pct_train=0.9) to avoid evaluating on
users seen during training.

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import random

import torch
import torch.nn.functional as F

from src.dataset import FeatureStore
from src.evaluate import build_book_embeddings
from src.model import BookRecommender


def run_offline_eval(model: BookRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int = 5_000,
                     ks: tuple = (1, 5, 10, 20, 50),
                     dataset_seed: int = 42,
                     dataset_pct_train: float = 0.9) -> None:
    model.eval()

    # ── Pre-compute item embedding matrix ────────────────────────────────────
    print("Building book embeddings ...")
    book_embeddings = build_book_embeddings(model, fs)
    all_ids  = list(book_embeddings.keys())
    all_embs = torch.cat(
        [book_embeddings[bid]['BOOK_EMBEDDING_COMBINED'] for bid in all_ids], dim=0
    )  # (n_books, 100)
    bid_to_pos = {bid: i for i, bid in enumerate(all_ids)}

    # ── Derive val users: replicate the same shuffle+split used in make_softmax_splits ──
    # This ensures eval only covers users not seen during training.
    all_users = fs.user_ids[:]
    rng_split = random.Random(dataset_seed)
    rng_split.shuffle(all_users)
    split     = int(len(all_users) * dataset_pct_train)
    val_users_full = all_users[split:]

    eligible = [u for u in val_users_full
                if fs.user_to_read_history.get(u)
                and fs.user_to_book_to_rating_LABEL.get(u)]

    rng = random.Random(dataset_seed)
    eval_users = rng.sample(eligible, min(n_users, len(eligible)))

    # ── Timestamp: use max bin (same as canary) ───────────────────────────────
    ts_max_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False,
    )

    # ── Accumulators ─────────────────────────────────────────────────────────
    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    with torch.no_grad():
        for user in eval_users:
            hist_indices = fs.user_to_read_history[user]          # list[int]
            hist_ratings = fs.user_to_read_history_ratings[user]  # list[float], debiased

            if not hist_indices:
                continue

            target_bids = [b for b in fs.user_to_book_to_rating_LABEL[user]
                           if b in bid_to_pos]
            if not target_bids:
                continue

            # ── Build user embedding ──────────────────────────────────────────
            hist_idx_t = torch.tensor(hist_indices, dtype=torch.long).unsqueeze(0)
            hist_wts_t = torch.tensor([hist_ratings], dtype=torch.float)
            rat_wts    = hist_wts_t.unsqueeze(-1)                  # (1, hist, 1)
            wt_sum     = rat_wts.abs().sum(dim=1).clamp(min=1e-6)  # (1, 1)

            hist_embs   = model.item_embedding_lookup(hist_idx_t)  # (1, hist, D)
            history_emb = (hist_embs * rat_wts).sum(dim=1) / wt_sum  # (1, D)

            genre_ctx = fs.user_to_genre_context[user]
            genre_emb = model.user_genre_tower(torch.tensor([genre_ctx]))
            ts_emb    = model.timestamp_embedding_tower(
                            model.timestamp_embedding_lookup(ts_max_bin))

            concat   = torch.cat([history_emb, genre_emb, ts_emb], dim=1)
            user_emb = model.user_projection(concat) if model.user_projection is not None else concat

            # ── Score all books ───────────────────────────────────────────────
            scores = (all_embs @ user_emb.T).squeeze(-1)  # (n_books,)

            # ── Metrics ───────────────────────────────────────────────────────
            n_targets = len(target_bids)
            target_positions = [bid_to_pos[b] for b in target_bids]
            target_scores    = scores[target_positions]

            # Rank of each target: number of ALL books scoring higher + 1
            # Use broadcasting: (n_books,) vs (n_targets,) → (n_targets,)
            ranks = (scores.unsqueeze(1) > target_scores.unsqueeze(0)).sum(dim=0) + 1
            # ranks: (n_targets,) 1-indexed

            # MRR — best rank among targets
            best_rank = ranks.min().item()
            mrr_sum  += 1.0 / best_rank

            for k in ks:
                hits_k = (ranks <= k).sum().item()
                recall[k]   += hits_k / n_targets
                hit_rate[k] += int(hits_k > 0)
                # NDCG: sum of 1/log2(rank+1) for hits in top-K, normalised by ideal
                dcg   = sum(1.0 / math.log2(r + 1) for r in ranks.tolist() if r <= k)
                ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_targets, k)))
                ndcg[k] += dcg / ideal if ideal > 0 else 0.0

            n_eval += 1

    if n_eval == 0:
        print("No users evaluated — check that features parquets are loaded.")
        return

    # ── Print results ─────────────────────────────────────────────────────────
    # Correct random Hit Rate@K: P(at least 1 of K random items is a target)
    # = 1 - (1 - avg_labels/corpus)^K, where avg_labels accounts for multi-label users.
    n_corpus   = len(all_ids)
    avg_labels = sum(
        len([b for b in fs.user_to_book_to_rating_LABEL[u] if b in bid_to_pos])
        for u in eval_users
    ) / max(n_eval, 1)

    print(f"\n── Offline Evaluation  (n={n_eval:,} val users, leave-label-out) "
          + "─" * 20)
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus: {n_corpus:,} books  |  Avg label books/user: {avg_labels:.1f}")
    print(f"Random Hit Rate baselines: " + "  ".join(
        f"@{k}: {1 - (1 - avg_labels/n_corpus)**k:.3%}" for k in ks
    ) + "\n")

    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    print(header)
    print("─" * len(header))
    for k in ks:
        print(f"{k:>6}  "
              f"{recall[k]/n_eval:>10.4f}  "
              f"{hit_rate[k]/n_eval:>11.4f}  "
              f"{ndcg[k]/n_eval:>8.4f}")
    print("─" * len(header))
    print(f"MRR: {mrr_sum/n_eval:.4f}")
