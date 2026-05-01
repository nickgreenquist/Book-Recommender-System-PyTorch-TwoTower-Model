# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the Goodreads dataset. The model predicts ratings via dot product of user and item embeddings.

This is a sibling project to the MovieLens Two-Tower model at:
`/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model`

The architecture follows the same two-tower design but adds an **author embedding tower** not present in the movie model. Preprocessing differs significantly (different schema, two-step streaming pipeline). When in doubt about shared architecture decisions, refer to the movie repo's CLAUDE.md.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: read history (quadruple sum pools over item ID embeddings), genre affinity, shelf affinity, and timestamp. Any user can be represented at inference time with just a few books they liked — no retraining required.

**Proven design choice: user tower is intentionally simple.** Shelf and author pooling were removed from the user side after experimentation showed they degraded probe_similar quality — shared towers pulled item embeddings in too many directions during training. Shelf and author signals live only on the item side.

## Running the Code

```bash
python main.py preprocess books         # Step 1: filter books → data/base_books.parquet
python main.py preprocess interactions  # Step 2: stream interactions → remaining parquets
python main.py preprocess               # Run both steps in order
python main.py explore                  # Explore user/book threshold distributions (fast, CSV-based)
python main.py features                 # Stage 2: base parquets → data/features_*.parquet
python main.py dataset                  # Stage 3: features → data/dataset_*_v1.pt
python main.py train                    # Stage 4: train, save checkpoints
python main.py canary                   # Canary user recommendations (most recent checkpoint)
python main.py canary <path>            # Canary user recommendations (specific checkpoint)
python main.py probe                    # Embedding probes (most recent checkpoint)
python main.py probe <path>             # Embedding probes (specific checkpoint)
python main.py eval                     # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py eval <path>              # Same, specific checkpoint
python main.py                          # Run all stages in order
```

## Dataset

Raw data lives in `data/` (not in git). Required files:
- `goodreads_interactions_dedup.json.gz` — JSONL (gzipped): `user_id, book_id, rating, read_at, date_updated, date_added, is_read, is_reviewed`
- `goodreads_books.json` — JSONL: `book_id, title, authors, popular_shelves, average_rating, ratings_count, publication_year, language_code, description`
- `goodreads_book_genres_initial.json` — JSONL: `book_id, genres` (dict of curated genre label → vote count, e.g. `{'fantasy, paranormal': 31, 'fiction': 8}`)
- `book_id_map.csv` — maps anonymized `book_id` → original Goodreads book_id
- `user_id_map.csv` — maps anonymized `user_id` → original Goodreads user_id

### Filtering thresholds

```python
MIN_RATINGS_PER_BOOK = 7_500   # based on goodreads_books.json ratings_count → ~14.7k books
MIN_RATINGS_PER_USER = 15      # corpus ratings only
MAX_RATINGS_PER_USER = 1_000
MIN_NUM_SHELVES      = 2_000   # shelf must appear this many times across corpus books
```

Books below `MIN_RATINGS_PER_BOOK` are filtered out entirely — not in training and not in the recommendation corpus. Users outside `MIN_RATINGS_PER_USER` / `MAX_RATINGS_PER_USER` are dropped.

### Preprocessing pipeline

Two separate steps — run `preprocess books` first, inspect the corpus size, then run `preprocess interactions`.

**Step 1 (`preprocess books`)** — streams `goodreads_books.json`, filters by `ratings_count >= MIN_RATINGS_PER_BOOK`, joins genres from `goodreads_book_genres_initial.json`. Writes `base_books.parquet`. Fast (~30s).

**Step 2 (`preprocess interactions`)** — two streaming passes over `goodreads_interactions_dedup.json.gz` (~11GB compressed). RAM stays under ~2GB.
- Only keeps `rating > 0` (explicit feedback only)
- **Pass 1** — count ratings per user against corpus books only → build `valid_users`
- **Pass 2** — filter to `valid_users` + corpus books, parse timestamps, collect rows
- No k-core filtering — book corpus is fixed from metadata, users are independently filtered

**Timestamp field priority:** `read_at` → `date_updated` → `date_added` → skip record. `read_at` is when the book was finished (directly analogous to MovieLens rating timestamp). `date_updated` is when the rating was submitted. Skip only if all three are missing.

**Note:** Timestamp fields are in the dedup JSON but **not** in `goodreads_interactions.csv` — this is why we use the JSON version.

### Threshold exploration

`python main.py explore` — uses the fast CSV (`goodreads_interactions.csv`) to show how many users survive various `MIN_RATINGS_PER_USER` thresholds against the corpus. Run this before `preprocess interactions` to pick thresholds without streaming the 11GB file.

## Key differences from MovieLens

| Concept        | MovieLens                                 | Goodreads                                                     |
|----------------|-------------------------------------------|---------------------------------------------------------------|
| Item ID column | movieId                                   | book_id                                                       |
| User ID column | userId                                    | user_id                                                       |
| Rating scale   | 0.5–5.0 (half stars)                      | 1–5 (integers)                                                |
| Timestamp      | Unix timestamp int                        | `read_at` → `date_updated` → `date_added` — parsed to unix   |
| Year           | Parsed from title (YYYY) regex            | `publication_year` field in books JSON                        |
| Genres         | Pipe-separated string in movies.csv       | `goodreads_book_genres_initial.json` — curated labels         |
| Tags           | tags.csv (free-form user-applied)         | `popular_shelves` from goodreads_books.json                   |
| Genome scores  | genome-scores.csv (pre-built ML scores)   | Computed from `popular_shelves` shelf counts (normalized 0–1) |
| Authors        | Not used                                  | `authors` field — embedded as new tower                       |

## Genre and shelf signals

**Genres** (`goodreads_book_genres_initial.json`) — curated high-level labels (e.g. `fiction`, `fantasy, paranormal`, `mystery, thriller, crime`, `romance`) with vote counts per book. Used for the `item_genre_tower`. Weight by vote count when building the genre vector.

**Shelf scores** (`popular_shelves` in `goodreads_books.json`) — list of `{name, count}` objects per book. Granular user-applied labels (e.g. `cozy-mystery`, `dark`, `epic-fantasy`). Used for the `item_shelf_tower`:
- Shelf-relevance score = TF-IDF: `(shelf_count / total_vocab_shelf_count_for_book) * log(N / df)`
  - TF normalized over vocab shelves only (intentional — measures relevance within the signal that matters)
  - IDF suppresses universal shelves like `to-read` (df ≈ N → score ≈ 0), amplifies specific ones like `cozy-mystery`
- Only shelves appearing `>= MIN_NUM_SHELVES` times across corpus books are kept
- Stored in `base_book_shelves.parquet`

## Model Architecture (V2)

Two-tower design with Full Softmax over the entire corpus (~11k books).

```
User Tower (Quadruple History Sum Pooling):
  sum_pool(item_id_embeddings[history_full])     → 32-dim + LayerNorm
  sum_pool(item_id_embeddings[history_liked])    → 32-dim + LayerNorm
  sum_pool(item_id_embeddings[history_disliked]) → 32-dim + LayerNorm
  sum_pool(item_id_embeddings[history_weighted]) → 32-dim + LayerNorm
  user_genre_tower(rollback_genre_affinity)      → 16-dim
  user_shelf_affinity_tower(pooled_shelf_tfidf)  → 64-dim
  timestamp_embedding_tower(read_month)          → 8-dim
  concat (216-dim) → projection MLP (256) → 128-dim → L2 Norm

Item Tower:
  item_genre_tower(genre_weighted)     → 10-dim
  item_shelf_tower(tfidf_shelf_scores) → 40-dim
  item_embedding_tower(book_id)        → 32-dim
  author_tower(primary_author_idx)     → 10-dim
  year_embedding_tower(pub_year)       → 8-dim
  concat (100-dim) → projection MLP (256) → 128-dim → L2 Norm

Prediction: dot_product(user_embedding, item_embedding)
```

### Key V2 Improvements:
1. **Full Softmax**: Scores against all ~11k items instead of in-batch negatives.
2. **ReLU Activations**: Replaced all Tanh with ReLU.
3. **L2 Normalization**: Final embeddings are L2 normalized before dot product.
4. **Quadruple History**: History partitioned into Liked, Disliked, Full, and Weighted pools.
5. **Shallow Sum Pooling**: Pools directly on ID embeddings (32-dim) instead of deep item tower outputs.
6. **LayerNorm**: Stabilizes sum-pooled history magnitudes.
7. **User Genome Context**: On-the-fly shelf affinity pooling in the user tower.
8. **Apple Silicon GPU**: Training and Eval use `mps` device when available.
9. **No Weight Decay**: Relying on architecture compression for regularization.
10. **Gradient Clipping**: Norm clipped at 1.0.
11. **Config Sidecar**: `.json` config saved alongside each `.pth` checkpoint.

### Preprocessing & Dataset
- **No per-user history split in preprocess**: `preprocess.py` writes only `base_interactions_raw.parquet`. No `base_ratings_read` / `base_ratings_labels`.
- **User-level train/val split in features.py**: `build_user_features()` assigns `split='train'/'val'` (90/10, `VAL_SPLIT_SEED=42`). Stored in `features_users_v1.parquet` as `user_id, split, avg_rating` only.
- **Rollback for both train and val**: `build_softmax_dataset()` reads `base_interactions_raw.parquet` directly and generates rollback examples. `make_softmax_splits()` uses `fs.train_users` / `fs.val_users` from FeatureStore.
- **Partitioned History**: `dataset.py` builds 4 separate history indices per example (full, liked, disliked, weighted).
- **Dynamic Genre/Shelf**: All user-side affinity signals are built from the rollback context slice (no future leakage).

## Canary Users for Eval

```python
USER_TYPE_TO_FAVORITE_GENRES = {
    'Mystery Lover':   ['Mystery', 'Thriller', 'Crime'],
    'Fantasy Lover':   ['Fantasy'],
}
USER_TYPE_TO_FAVORITE_BOOKS = {
    'Mystery Lover':   ['Gone Girl', 'The Girl with the Dragon Tattoo', 'Big Little Lies'],
    'Fantasy Lover':   ['The Name of the Wind', 'The Way of Kings', 'A Game of Thrones'],
}
USER_TYPE_TO_SHELF_TAGS = {
    'Mystery Lover':   ['mystery', 'suspense', 'crime', 'thriller'],
    'Fantasy Lover':   ['fantasy', 'magic', 'epic-fantasy', 'world-building'],
}
```

Canary users are synthetic — no real read timestamps. All receive `ts_max_bin`.

## Relationship to Movie Repo

| File | Status |
|------|--------|
| `preprocess.py` | Rewritten — two-step streaming pipeline (books + interactions only, no split), Goodreads schema |
| `features.py` | Adapted — user-level 90/10 train/val split; outputs `user_id, split, avg_rating` only (no history/label columns) |
| `dataset.py` | Adapted — `FeatureStore` has `train_users`/`val_users`; `make_softmax_splits()` uses them directly; rollback reads `base_interactions_raw.parquet` |
| `model.py` | Extended — author tower on item side; V2 quadruple shallow pools + user shelf affinity tower; non-persistent buffers for `book_genre_matrix`, `book_year_idx` |
| `train.py` | Adapted — V2 config; full softmax (U @ V_all.T); checkpoint naming encodes architecture |
| `evaluate.py` | Adapted — canary dicts use book titles and shelf tags; shelf embedding probe table added |
| `export.py` | Adapted — resolves all architecture variants from checkpoint basename; stores non-persistent buffers in `feature_store.pt` |

## Current Production Model

**Serving checkpoint (deployed):** `saved_models/best_proj_softmax_ipool_20260426_093432.pth`
Serving artifacts are in `serving/` and deployed to the Streamlit app. This uses the V1 ipool architecture (item tower pooling in user history — superseded by V2).

**V2 architecture (in training):** Quadruple shallow ID-embedding pools + user shelf affinity tower. See Model Architecture section above. Retrain required before exporting to serving.

**Architecture history (all ✅ implemented and validated):**

1. **Projection MLP** — both towers end with `concat → Linear(256) → ReLU → Linear(128)`. Learns cross-feature interactions (genre × history) that plain concat + dot product cannot express. Hit Rate@10: 10.7% → 13.0% (+21%).
   - Initialization fix (critical): sub-tower linear layers `gain=0.01 → 0.1`, projection layers initialized separately to `gain=1.0`. Without this, dot products collapse to zero at step 0 and never recover.

2. **Item tower pooling** (superseded by V2) — user history pooled over full 128-dim projected item embeddings. Captured shelf+genre+author signals but ~8× slower to train due to shelf MLP being called B×H times per step. Hit Rate@10: 13.0% → 14.0% (+8%). Replaced in V2 by shallow ID-embedding pools + dedicated user shelf affinity tower.

## Future Improvements

### Training objective

**~~Sampled softmax~~** — ✅ Implemented and validated. Softmax is now the primary training path (`python main.py train softmax`). ID embeddings gained semantic structure vs BPR (King/horror cluster, LOTR/fantasy cluster confirmed via probe and canary). Canary results are strong for Literary, NonFiction, Fantasy, Sci-Fi, History, YA. Known weak spots: Horror (no horror genre in vocab — relies purely on shelf signals) and Romance (model conflates literary women's fiction with romance).

**Year feature:** `preprocess books` uses original publication year. Current checkpoint (`best_proj_softmax_ipool_20260426_093432.pth`) was trained with correct year vocab — no retraining needed.

**Next training improvements:**
1. **~~LR schedule~~** — ✅ Implemented. CosineAnnealingLR from 0.001→0 over training steps. Eliminated the early plateau seen in the first softmax run.
2. **~~Larger batch~~** — ✅ 512 (511 in-batch negatives). Confirmed better drop-from-baseline than batch=256.
3. **~~In-batch negative debiasing (log-frequency correction)~~** — ❌ Tried and removed. Subtracting `log(p_i)` from negative logits (Yi et al., Google RecSys 2019) did not converge on this dataset. Root causes: (a) with ~11k books, the item frequency distribution is very compressed — corrections ranged from 4.91 to 10.0 with ~87% of books clustered near 9-10, so the correction added almost no useful signal. (b) The correction destabilized training even with L2 normalization and diagonal zeroing. Do not re-attempt without a much larger, more skewed item distribution.
4. **~~Remove F.normalize from training (match YouTube paper)~~** — ✅ Done and re-added for V2. V2 adds L2 norm back at the output of both towers (as `F.normalize` in `user_embedding()` and `item_embedding()`). This stabilizes training with shallow sum pooling and ensures dot product = cosine similarity between normalized 128-dim vectors.
5. **~~Item tower pooling~~** — ✅ Implemented and validated (+8%). Superseded by V2 which achieves the content signal via a dedicated `user_shelf_affinity_tower` on the user side, while keeping user history pools shallow (ID embeddings only). This avoids the 8× training slowdown.

**Implicit vs explicit feedback tradeoff** — explicit ratings (BPR) give clean preference signal but are sparse. Implicit feedback (reads via `is_read`) is abundant but noisy. Consider a hybrid: softmax for candidate generation, explicit ratings for a separate ranking stage.

### Offline Evaluation Framework ✅ Implemented

`python main.py eval [checkpoint_path]` — implemented in `src/offline_eval.py`.

**Protocol:** Rollback examples from val users (same generation logic as training). For each val user, rollback examples are generated from `base_interactions_raw.parquet` — context = all reads before position i, target = read at position i. Metrics are per-example (single target per example, not multi-target per user).

Val users are fixed in `features.py` (`VAL_FRACTION=0.10`, `VAL_SPLIT_SEED=42`), stored in `fs.val_users`. 5,000 val users sampled with `random.Random(42)` for reproducibility.

**Metrics:** Recall@K, Hit Rate@K, NDCG@K, MRR at K = 1, 5, 10, 20, 50.

**Results (5,000 val users, corpus ~11k books, random Hit Rate@10 baseline ≈ 0.88%):**

| Metric | MSE | BPR | Softmax | Softmax + Projection | + Item Tower Pool | **V2 (pending)** |
|---|---|---|---|---|---|---|
| Hit Rate@10 | 4.7% | 3.5% | 10.7% | 13.0% | 14.0% | **TBD** |
| Hit Rate@50 | 17.6% | 14.4% | 28.9% | 33.0% | 36.3% | **TBD** |
| Recall@10 | 0.0069 | 0.0041 | 0.0164 | 0.0241 | 0.0258 | **TBD** |
| NDCG@10 | 0.0073 | 0.0042 | 0.0189 | 0.0255 | 0.0274 | **TBD** |
| MRR | 0.024 | 0.016 | 0.053 | 0.064 | 0.067 | **TBD** |

V2 baseline to beat: Hit Rate@10 > 14.0%, MRR > 0.067. Update this table after first V2 eval run.

**Canary quality highlights (ipool vs proj_softmax):**
- **Sci-Fi**: Significantly more specific hard SF — Revelation Space, Accelerando, The Mote in God's Eye appear vs generic Honor Harrington. Richer content signal in the user pool captures subgenre distinctions.
- **Poetry**: All 10 recommendations are actual poetry collections (vs art books in older checkpoints). Item tower pool captures the shelf+genre signal of poetry books in the user's taste representation.
- **Economics**: Clean — all 10 are finance/economics titles.
- **Horror**: Stronger — tighter horror cluster (The Rats, Hell House, Swan Song, Cabal) vs diluted results from older checkpoints.
- **Nick (personal canary)**: Better historical/WWII focus (Battle Cry of Freedom, Flags of Our Fathers, With the Old Breed) vs financial noise from older checkpoint.

**Checkpoint naming convention:** all checkpoints use the `softmax_4pool` prefix (V2 architecture — quadruple shallow ID-embedding pools):
- `best_softmax_4pool_*.pth` — best val loss checkpoint
- `softmax_4pool_*_step_NNNNNN.pth` — periodic checkpoint

`_resolve_checkpoint` in `evaluate.py` and `run_export` in `export.py` both look for these prefixes, newest first.

### YouTube DNN implementation details (Covington et al., 2016)

Key design decisions from the paper that directly apply to our softmax implementation:

**Training example construction ("rollback"):**
- For each read event, use only the user's history *before* that event as context. Predict the *future* read, not a randomly held-out one. This captures asymmetric co-read behavior and prevents leakage of future information.
- Generate a **fixed number of examples per user** (cap per user). This prevents highly active users from dominating the loss — effectively weights all users equally.
- **Why rollback for both train and val (not "last read holdout" for val):** Rollback produces examples with varying context lengths (short to long). If val used only each user's last read (full history as context), val examples would always have long contexts while training has short+long — a distribution mismatch that makes val loss unreliable. Using rollback for both keeps context length distribution consistent.

**Negative sampling:**
- The paper recommends sampling several thousand negatives per step from the background distribution (popularity-proportional) with importance weighting to correct for sampling bias.
- **Our implementation uses in-batch negatives instead** — the other 511 items in each batch of 512 serve as negatives. Simpler and effective for our ~11k book corpus. Dedicated sampled negatives would be more important at larger corpus sizes.

**"Example age" feature (skip for books):**
- The paper feeds `age = t_max - t_read` to correct bias toward older popular videos. For books this doesn't apply — *Crime and Punishment* is as recommendable as a book published last month, and the corpus is essentially static.
- User recency is already captured by the timestamp embedding (read month bin), which signals how recently the user's taste context was formed.

**Serving (inference):**
- At serving time, rank by raw dot product `u · v` — no softmax normalisation, no L2 normalisation of embeddings. The paper uses inner product search throughout (training and inference).
- YouTube uses ANN (approximate nearest neighbor search via MIPS) because their corpus is billions of videos. **Our corpus is ~11k books**, so we compute all 11k dot products exactly. No ANN needed.
- **Note:** V2 applies `F.normalize` (L2 norm) at the output of both towers before the dot product. This makes dot product = cosine similarity on the unit sphere and stabilizes training with shallow sum pooling. Both training and inference use the same normalized embeddings.

**Shared item ID embedding:**
- The item ID embedding table is shared between the item tower output and the user history avg pool (we already do this). The paper does the same — a single global video embedding used for both the "what video is this" tower and the "what has the user watched" history representation.

**Architecture notes:**
- Hidden layers use **ReLU** (not Tanh). The paper found depth 3 (1024 → 512 → 256 ReLU) best for their scale; for our ~11k book corpus, shallower is fine.
- The final user embedding dimension = the item embedding dimension (they must match for dot product). The softmax weight matrix is shape `(n_books, embedding_dim)` — each row is an item embedding.
- Out-of-vocabulary items map to a zero embedding (already our behavior for unknown authors).

**Dataset format (V2):** 8-tuple — `(X_genre, X_hist_full, X_hist_liked, X_hist_disliked, X_hist_weighted, X_rats_weighted, timestamp, target_book_idx)`. All history tensors pre-padded to `max_hist` — no per-batch padding overhead. Item features (`genre`, `year`, `author`) looked up from non-persistent registered buffers at training time via `target_book_idx` — not stored in the dataset tensors.

**Known approximations in the softmax dataset (dataset.py `build_softmax_dataset`):**
- Genre context is computed correctly from only the rollback context (no future leakage). ✓
- `avg_rat` (used for rating debiasing in the history pool) uses the user's full-history average, not the rollback-slice average. Technically it should be recomputed from only the rollback books to avoid any future leakage. Low priority since the average changes slowly and has minor impact.

### Item features

- **Shelf tower: EmbeddingBag instead of Linear** — v1 uses `nn.Linear(n_shelves, shelf_dim)` which treats the shelf context as a dense vector. But shelf vectors are sparse: preprocessing caps each book at 100 shelves, so median non-zero entries = 92/3032 (~3% density). This is architecturally different from MovieLens genome tags, which are pre-computed ML scores for *every* movie-tag pair (dense). For sparse inputs, `nn.EmbeddingBag` (or `nn.Embedding` + weighted avg pool) is more natural: each shelf gets a learned embedding, and only the book's actual shelves activate and receive gradient. Same parameter count, cleaner gradient flow, more analogous to how book history and author pooling already work.

- **~~Better shelf-relevance scoring (TF-IDF style)~~** — ✅ Implemented. See shelf scoring section above.

- **Book description embeddings** — `goodreads_books.json` includes a `description` field. Encoding with a sentence transformer (e.g. `all-MiniLM-L6-v2`) would add dense semantic signal. Skipped in v1 to keep a simple baseline and avoid heavy text encoder dependency.

- **Multi-author avg-pool** — v1 uses only the primary author (80.7% of books have one author). Future: avg-pool embeddings across all authors per book, using a padded `nn.Embedding` with `padding_idx`.

- **Consistent label/history indexing** — `read_history` stores `book_idx` integers (pre-mapped in features.py) but `label_bookIds` stores raw string IDs (mapped in dataset.py). Mirrors the movie model pattern but could be unified in features.py for consistency.

See the movie repo's CLAUDE.md "Future User Tower Improvements" and "Richer Cross-Signal Features" sections for additional ideas that apply equally here.

## Serving / Export Notes

`book_shelf_matrix` (registered buffer, 11124 × 3032 × float32 ≈ 135MB) cannot be saved inside `model.pth` — it would exceed GitHub's 100MB file limit.

In `export.py`, exclude the buffers from the saved state_dict:
```python
state_dict = {k: v for k, v in model.state_dict().items()
              if k not in ('book_shelf_matrix', 'book_author_idx')}
torch.save(state_dict, model_path)
```

Store `book_shelf_matrix` and `book_author_idx` separately in `feature_store.pt` instead. The Streamlit app rebuilds the model via `build_model(config, fs)` which reconstructs the buffers from the FeatureStore — same as training. Only the learned weights need to come from the checkpoint.

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

For changes that require retraining to validate (hyperparameters, optimizer, scheduler, loss, dataset logic): write the code, then stop. Do not commit until the user has run training and confirmed the results look better.
