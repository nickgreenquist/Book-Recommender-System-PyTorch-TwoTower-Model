---
name: Architecture Decisions
description: Key design choices proven through experimentation — user tower, shelf scoring, loss function
type: project
---

## User Tower Simplification (proven better)

User tower = history pool + genre context + timestamp only. Shelf and author pooling were removed from the user side.

**Why:** Shared towers (shelf, author) pulled item embeddings in multiple directions during MSE training, degrading probe_similar quality. Probe_similar degraded as training progressed with the full tower. Simplified tower dramatically improved similar book retrieval at 30k steps.

**How to apply:** Never re-add shelf or author avg pooling to the user side. User side is intentionally simpler than item side.

## Deeper Towers (implemented)

- `item_shelf_tower`: 3032 → 128 → 25 (was 3032 → 25)
- `user_genre_tower`: 2*n_genres → 64 → 50 (was 2*n_genres → 50)

**Why:** Shelf inputs are sparse (3% density). Single linear layer from 3032 dims was too aggressive a compression. Intermediate layer learns which shelf combinations matter. User genre also benefits from depth due to combined rating+fraction signals.

**How to apply:** Keep these as 2-layer MLPs. Other towers (author, year, ts, item_genre) are shallow by design — their input dims are small enough that depth doesn't help.

## TF-IDF Shelf Scoring (implemented)

Shelf score = `(shelf_count / total_vocab_shelf_count_for_book) * log(N / df)` where N = corpus size, df = books with that shelf.

**Why:** Raw TF scoring gave near-zero IDF weight to universal shelves like `to-read` (appears on nearly every book), which caused all book shelf embeddings to collapse toward the same direction. TF-IDF for LotR correctly surfaces `tolkien`, `epic-fantasy`, `middle-earth` instead of `to-read`.

**How to apply:** TF is normalized over vocab shelves only (not all shelves) — this is intentional, not a bug. IDF computed in `_build_book_shelf_scores` in preprocess.py. Rerun `preprocess interactions`, `features`, `dataset` to apply.

## BPR Loss (current best — major improvement)

Using Bayesian Personalized Ranking with same-user hard negatives instead of MSE.

Loss = `-logsigmoid(score_pos - score_neg).mean()`

Positives: debiased rating > 0.5. Negatives: debiased rating < -0.5. Pairs stored at index 9 of dataset tuple.

**Why:** MSE caused embedding collapse — all item embeddings converged toward same direction as model learned to predict ~0 for debiased ratings. BPR forces discriminative structure: liked books must outscore disliked books. At 30k steps BPR produced correct horror/sci-fi shelf probes and correct adult mystery/fantasy genre probes for the first time.

**How to apply:** Set `'loss': 'bpr'` in `get_config()`. Switch to `'mse'` to compare. Dataset must be regenerated after adding BPR pairs support (already done). BPR is ~3x slower than MSE due to 2 forward passes + 4 pad_history_batch calls per step.

## Current Embedding Sizes

```python
item_id_embedding_size    = 40   # shared: user history + item tower
user_genre_embedding_size = 50
timestamp_embedding_size  = 10
item_genre_embedding_size = 10
shelf_embedding_size      = 25
author_embedding_size     = 15
item_year_embedding_size  = 10
# user: 40 + 50 + 10 = 100
# item: 10 + 25 + 40 + 15 + 10 = 100 ✓
```
