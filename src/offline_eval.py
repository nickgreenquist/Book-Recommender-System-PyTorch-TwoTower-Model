"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Protocol: leave-label-out per user.
  Context = user_to_read_history (80% of reads, pre-mapped book indices)
  Targets = user_to_book_to_rating_LABEL (remaining 20% of reads)

No new splits needed — reuses the existing 80/20 split from features.py.

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import random

import torch

from src.dataset import FeatureStore
from src.evaluate import build_book_embeddings
from src.model import BookRecommender


def run_offline_eval(model: BookRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int = 5_000,
                     ks: tuple = (1, 5, 10, 20, 50),
                     seed: int = 42) -> None:
    model.eval()

    # ── Pre-compute item embedding matrix ────────────────────────────────────
    print("Building book embeddings ...")
    book_embeddings = build_book_embeddings(model, fs)
    all_ids  = list(book_embeddings.keys())
    all_embs = torch.cat(
        [book_embeddings[bid]['BOOK_EMBEDDING_COMBINED'] for bid in all_ids], dim=0
    )  # (n_books, 100)
    bid_to_pos = {bid: i for i, bid in enumerate(all_ids)}

    # ── Sample eval users ────────────────────────────────────────────────────
    eligible = [u for u in fs.user_ids
                if fs.user_to_read_history.get(u)
                and fs.user_to_book_to_rating_LABEL.get(u)]
    rng = random.Random(seed)
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

            user_emb = torch.cat([history_emb, genre_emb, ts_emb], dim=1)  # (1, 100)

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
    max_k = max(ks)
    random_hit_baseline = max_k / len(all_ids)

    print(f"\n── Offline Evaluation  (n={n_eval:,} users, leave-label-out) "
          + "─" * 20)
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus: {len(all_ids):,} books  |  "
          f"Random Hit Rate@{max_k} baseline: {random_hit_baseline:.3%}\n")

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
