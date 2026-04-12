---
name: Training Learnings
description: What worked and what didn't across training runs — optimizer, loss, collapse, probes
type: project
---

## MSE Causes Embedding Collapse

With MSE loss on debiased ratings, all item embeddings converge to nearly the same direction as training progresses. The model minimizes loss by predicting ~0 for all debiased ratings. Symptoms: probe_similar cosine similarities all → 1.0, probe_genre returning same niche books, probe_similar degrading with more training steps (step 30k better than step 150k).

Weight decay (1e-4) partially mitigates but doesn't fix — probe_similar improved but shelf probes remained broken.

**Fix: BPR loss.**

## SGD Plateaued, Adam Works

SGD (lr=0.005, momentum=0.9) got stuck at val_loss ~0.814 for 60k steps after user tower simplification. Switched to Adam (lr=0.001, weight_decay=1e-4). Adam converges reliably.

## Probe Quality as Training Signal

Use `python main.py probe` at checkpoints to evaluate embedding quality. probe_similar is the most informative — it directly shows whether item embeddings are semantically structured. Genre and shelf probes catch specific failure modes (collapse, common-shelf domination).

Key baseline results per model version:
- MSE baseline: probe_similar decent for HP/Hunger Games/LotR, broken for others. Shelf probes all 1.0.
- MSE + weight_decay 1e-4 + deeper towers: probe_similar improved, shelf probes still all 1.0.
- BPR (30k steps): First time horror/sci-fi shelf probes worked correctly. Genre probes showing adult mystery/fantasy instead of YA crossovers.

## Shelf Probe 1.0 Scores = Data Issue

Shelf probe cosine similarities clustering at 1.0 indicates the shelf tower inputs are dominated by universal shelves (to-read, currently-reading, fiction). TF-IDF partially helps the input quality. Full fix requires EmbeddingBag (future).

## preprocess interactions and split are independent

`preprocess interactions` only builds base parquets. `preprocess split` is separate. Do NOT call run_split() from run_interactions(). If only shelf scores changed (e.g. TF-IDF update), only need to rerun: `preprocess interactions` → `features` → `dataset`. Skip `preprocess books` and `preprocess split`.
