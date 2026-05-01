"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Protocol: rollback examples from val users (same as training).
For each val user, rollback examples are generated (context = all reads before
position i, target = read at position i). Metrics are per-example (single target).

Val users are fixed in features.py (VAL_FRACTION=0.10, VAL_SPLIT_SEED=42).
No new splits needed — reuses fs.val_users directly.

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import os
import random

import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import FeatureStore, build_softmax_dataset, MAX_ROLLBACK_EXAMPLES_PER_USER
from src.evaluate import build_book_embeddings
from src.model import BookRecommender


def run_offline_eval(model: BookRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int = 5_000,
                     ks: tuple = (1, 5, 10, 20, 50),
                     seed: int = 42,
                     data_dir: str = 'data') -> None:
    device = next(model.parameters()).device
    model.eval()

    # ── Pre-compute item embedding matrix ─────────────────────────────────────
    print(f"Building book embeddings on {device} ...")
    book_embeddings = build_book_embeddings(model, fs)
    all_ids  = list(book_embeddings.keys())
    all_embs = torch.cat(
        [book_embeddings[bid]['BOOK_EMBEDDING_COMBINED'] for bid in all_ids], dim=0
    ).to(device)  # (n_books, output_dim)

    # ── Sample val users ──────────────────────────────────────────────────────
    val_users = fs.val_users[:]
    rng = random.Random(seed)
    eval_users = rng.sample(val_users, min(n_users, len(val_users)))
    print(f"Evaluating on {len(eval_users):,} val users ...")

    # ── Build rollback examples ───────────────────────────────────────────────
    print("Loading interactions ...")
    raw_df = pd.read_parquet(os.path.join(data_dir, 'base_interactions_raw.parquet'))

    print("Building rollback examples for val users ...")
    (X_genre, X_hist_full, X_hist_liked, X_hist_disliked,
     X_hist_weighted, X_rats_weighted, timestamp, target_book_idx) = build_softmax_dataset(
        eval_users, fs, raw_df,
        max_per_user=MAX_ROLLBACK_EXAMPLES_PER_USER,
        seed=seed + 1,
    )

    n_examples = target_book_idx.shape[0]
    print(f"  {n_examples:,} rollback examples")

    # ── Score in batches ──────────────────────────────────────────────────────
    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    batch_size = 512

    with torch.no_grad():
        for s in tqdm(range(0, n_examples, batch_size), desc="Scoring"):
            e = min(s + batch_size, n_examples)

            h_full     = X_hist_full[s:e].to(device)
            h_liked    = X_hist_liked[s:e].to(device)
            h_disliked = X_hist_disliked[s:e].to(device)
            h_weighted = X_hist_weighted[s:e].to(device)
            r_weighted = X_rats_weighted[s:e].to(device)
            x_genre    = X_genre[s:e].to(device)
            ts         = timestamp[s:e].to(device)

            user_emb = model.user_embedding(
                x_genre, h_full, h_liked, h_disliked, h_weighted, r_weighted, ts)
            scores = user_emb @ all_embs.T  # (B, n_books)

            target_idxs = target_book_idx[s:e]  # (B,)

            for i in range(e - s):
                t_pos        = target_idxs[i].item()
                target_score = scores[i, t_pos]
                rank         = int((scores[i] > target_score).sum().item()) + 1

                mrr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        recall[k]   += 1.0
                        hit_rate[k] += 1
                        ndcg[k]     += 1.0 / math.log2(rank + 1)
                n_eval += 1

    if n_eval == 0:
        print("No examples evaluated.")
        return

    # ── Print results ─────────────────────────────────────────────────────────
    n_corpus  = len(all_ids)
    rand_mrr  = sum(1.0 / r for r in range(1, n_corpus + 1)) / n_corpus

    print(f"\n── Offline Evaluation  ({n_eval:,} rollback examples, "
          f"{len(eval_users):,} val users) " + "─" * 20)
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus: {n_corpus:,} books\n")

    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    sep    = "─" * len(header)
    print(header)
    print(sep)
    for k in ks:
        rand_recall_k = k / n_corpus
        print(f"{k:>6}  {rand_recall_k:>10.4f}  {rand_recall_k:>11.4f}  "
              f"{sum(1.0/math.log2(r+1) for r in range(1,k+1))/n_corpus:>8.4f}  ← random")
    print("·" * len(header))
    for k in ks:
        print(f"{k:>6}  "
              f"{recall[k]/n_eval:>10.4f}  "
              f"{hit_rate[k]/n_eval:>11.4f}  "
              f"{ndcg[k]/n_eval:>8.4f}  ← model")
    print(sep)
    print(f"MRR  random: {rand_mrr:.4f}   model: {mrr_sum/n_eval:.4f}  "
          f"(+{mrr_sum/n_eval - rand_mrr:.4f})")

    os.makedirs('eval_results', exist_ok=True)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0] if checkpoint_path else 'unknown'
    out_path = os.path.join('eval_results', f'{stem}.txt')
    lines = [
        f"── Offline Evaluation  ({n_eval:,} rollback examples, {len(eval_users):,} val users) ──",
        f"Checkpoint: {checkpoint_path}",
        f"Corpus: {n_corpus:,} books",
        "",
        header,
        sep,
    ]
    for k in ks:
        lines.append(f"{k:>6}  {recall[k]/n_eval:>10.4f}  "
                     f"{hit_rate[k]/n_eval:>11.4f}  {ndcg[k]/n_eval:>8.4f}")
    lines.append(sep)
    lines.append(f"MRR: {mrr_sum/n_eval:.4f}")
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✓ Saved → {out_path}")
