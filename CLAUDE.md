# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender trained on the Goodreads dataset. Predicts ratings via dot product of L2-normalized user and item embeddings.

Sibling project to the MovieLens Two-Tower model at `/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model`. Same two-tower design, but adds an **author embedding tower** (item side) and a different schema / two-step streaming preprocess. For shared architecture decisions, refer to the movie repo's CLAUDE.md.

**No user ID embedding (critical).** Users are represented entirely by taste signals: read history (quadruple sum pools over item ID embeddings), genre affinity, shelf affinity, timestamp. Any user can be represented at inference from just a few books they liked — no retraining.

**User tower is intentionally simple (proven).** Shelf and author pooling were removed from the user side — they degraded probe_similar quality by pulling item embeddings in too many directions during training. Shelf and author signals live only on the item side.

## Running the Code

```bash
python main.py preprocess books         # Step 1: filter books → data/base_books.parquet
python main.py preprocess interactions  # Step 2: stream interactions → remaining parquets
python main.py preprocess               # Both steps in order
python main.py explore                  # User/book threshold distributions (fast, CSV-based)
python main.py features                 # Stage 2: base parquets → features_*.parquet
python main.py dataset                  # Stage 3: features → dataset_*_v1.pt
python main.py train                    # Stage 4: train, save checkpoints
python main.py canary [<path>]          # Canary user recs (most recent checkpoint, or specific)
python main.py probe [<path>]           # Embedding probes
python main.py eval [<path>]            # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py                          # All stages in order
```

## Dataset

Raw data in `data/` (not in git). Required files:
- `goodreads_interactions_dedup.json.gz` — JSONL.gz: `user_id, book_id, rating, read_at, date_updated, date_added, is_read, is_reviewed`
- `goodreads_books.json` — JSONL: `book_id, title, authors, popular_shelves, average_rating, ratings_count, publication_year, language_code, description, image_url`
- `goodreads_book_genres_initial.json` — JSONL: `book_id, genres` (dict of curated genre label → vote count)
- `book_id_map.csv` / `user_id_map.csv` — anonymized ID → original Goodreads ID

### Filtering thresholds

```python
MIN_RATINGS_PER_BOOK = 7_500   # by ratings_count → ~14.7k books
MIN_RATINGS_PER_USER = 15      # corpus ratings only
MAX_RATINGS_PER_USER = 1_000
MIN_NUM_SHELVES      = 2_000   # shelf must appear this many times across corpus books
```

Books below `MIN_RATINGS_PER_BOOK` are filtered out entirely (not in training or corpus). Users outside the user bounds are dropped. No k-core filtering.

### Preprocessing pipeline

Two steps — run `preprocess books` first, inspect corpus size, then `preprocess interactions`.

- **Step 1 (`preprocess books`)** — streams `goodreads_books.json`, filters by `ratings_count`, joins genres. Writes `base_books.parquet`. Fast (~30s).
- **Step 2 (`preprocess interactions`)** — two streaming passes over the ~11GB gz (RAM < ~2GB). Keeps only `rating > 0`. Pass 1 counts ratings per user against corpus books → `valid_users`; Pass 2 filters + parses timestamps + collects rows. Writes `base_interactions_raw.parquet` only (no history/label split).

**Timestamp priority:** `read_at` → `date_updated` → `date_added` → skip. (Timestamp fields exist in the dedup JSON but **not** in `goodreads_interactions.csv` — why we use the JSON.)

`python main.py explore` uses the fast CSV to preview how many users survive a given `MIN_RATINGS_PER_USER` without streaming the 11GB file.

## Genre and shelf signals

- **Genres** (`goodreads_book_genres_initial.json`) — curated labels (e.g. `fiction`, `fantasy, paranormal`) with vote counts. Drives `item_genre_tower`. Weight by vote count.
- **Shelf scores** (`popular_shelves` in `goodreads_books.json`) — `{name, count}` per book. Drives `item_shelf_tower`. Relevance = TF-IDF: `(shelf_count / total_vocab_shelf_count_for_book) * log(N / df)`. TF normalized over vocab shelves only; IDF suppresses universal shelves (`to-read`), amplifies specific ones (`cozy-mystery`). Only shelves with df `>= MIN_NUM_SHELVES` kept. Stored in `base_book_shelves.parquet`.

## Model Architecture (V2)

Two-tower design, Full Softmax over the entire corpus (~14.7k books).

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

Key V2 properties: full softmax (U @ V_all.T, not in-batch negatives); ReLU everywhere; L2 norm on both tower outputs; quadruple history (liked/disliked/full/weighted shallow sum pools over raw 32-dim ID embeddings + LayerNorm); on-the-fly user shelf affinity pooling; `mps` device on Apple Silicon; no weight decay; grad clip at 1.0; `.json` config sidecar per `.pth`.

**Init fix (critical):** sub-tower linear layers use `gain=0.1`, projection layers `gain=1.0`. Without separate projection init, dot products collapse to zero at step 0 and never recover.

**Popularity logit adjustment (Menon et al. 2021):** training adds `alpha * log1p(count_i)` to each item logit before softmax (current PROD `alpha=0.2`). Self-debiases embeddings; applied to training logits only — val loss and all inference use raw dot products.

**Temperature = 0.1** for full softmax (one negative per corpus item). The in-batch formula `0.5/batch_size ≈ 0.001` is wrong here — it collapses softmax to near-argmax and overfits popularity. `use_item_pool_for_history=True` in `get_softmax_config()` re-enables the older ipool behavior.

### Dataset details
- **Splits:** `features.py:build_user_features()` assigns `split='train'/'val'` (90/10, `VAL_SPLIT_SEED=42`) → `features_users_v1.parquet` (`user_id, split, avg_rating` only).
- **Rollback for both train and val:** `build_softmax_dataset()` reads `base_interactions_raw.parquet` and generates rollback examples (context = reads before position i, target = read at i). Rollback for both keeps context-length distribution consistent → reliable val loss.
- **8-tuple format:** `(X_genre, X_hist_full, X_hist_liked, X_hist_disliked, X_hist_weighted, X_rats_weighted, timestamp, target_book_idx)`. History tensors pre-padded to `max_hist`. Item features (genre, year, author) looked up from non-persistent registered buffers via `target_book_idx` — not stored in dataset tensors (~3× RAM saving).
- **No future leakage:** genre/shelf affinity built from the rollback context slice. (Known minor approximation: `avg_rat` uses full-history average, not the rollback slice — low impact.)

## Current Production Model

**Deployed:** `saved_models/best_full_softmax_4pool_alpha_02_20260503_111805.pth` — V2 (quadruple shallow pools + user shelf affinity tower + full softmax + alpha=0.2). Serving artifacts in `serving/`, deployed to the Streamlit app.

**Checkpoint naming:** `best_softmax_4pool_*.pth` (best val loss) and `softmax_4pool_*_step_NNNNNN.pth` (periodic). `_resolve_checkpoint` (evaluate.py) and `run_export` (export.py) match these prefixes, newest first.

**Current offline eval (5,000 val users, 14.7k corpus):** Hit Rate@10 16.0%, Hit Rate@50 36.0%, NDCG@10 0.088, MRR 0.079. Recall@K = Hit Rate@K here (single target per example).

## Canary Users for Eval

`src/evaluate.py`: `USER_TYPE_TO_FAVORITE_BOOKS`, `USER_TYPE_TO_LIKED_BOOKS`, `USER_TYPE_TO_SHELF_TAGS`. 16 synthetic user types (Mystery, Fantasy, Romance, YA, History, Classic, Horror, Sci-Fi, NonFiction, Economics, Philosophy, Graphic Novel, Manga, Christianity, Poetry, Children's) + Nick's personal canary. Genre context derived entirely from book history; shelf tags pull anchor books via `_get_anchor_titles()`. All receive `ts_max_bin` (no real timestamps). Known weak spots: Horror (no horror genre in vocab) and Romance (conflates literary women's fiction with romance).

## Offline Evaluation

`python main.py eval [checkpoint_path]` — `src/offline_eval.py`. Rollback examples from val users (same logic as training), per-example metrics (single target). Val users fixed in `features.py` (`VAL_FRACTION=0.10`, `VAL_SPLIT_SEED=42`); 5,000 sampled with `random.Random(42)`. Metrics: Recall@K, Hit Rate@K, NDCG@K, MRR at K ∈ {1,5,10,20,50}.

## Serving / Export Notes

`book_shelf_matrix` (≈135MB) and `book_author_idx` cannot be saved inside `model.pth` (GitHub 100MB limit). In `export.py`, exclude them from the saved state_dict and store them in `feature_store.pt` instead:
```python
state_dict = {k: v for k, v in model.state_dict().items()
              if k not in ('book_shelf_matrix', 'book_author_idx')}
```
The Streamlit app rebuilds the model via `build_model(config, fs)`, which reconstructs the buffers from the FeatureStore — only learned weights come from the checkpoint.

**Covers (Goodreads `image_url`):** ~10% of corpus books lack ISBNs (popular work-level entries especially — 59/60 of Twilight's neighbors), so the Open Library path showed placeholders for popular seeds. Export streams `goodreads_books.json`, builds `bookId_to_image_url` (`images.gr-assets.com` CDN, `nophoto` → `''`), rewrites the `m` size suffix → `l` for retina sharpness, and stores it in `feature_store.pt`. `streamlit_app.py:_cover_url` tries Goodreads → OL by ISBN → placeholder. Coverage 67.4% (9,937/14,753). Risk: the CDN has no SLA; degradation is graceful (OL fallback). Plan B: rehost the ~10k JPEGs (~50MB).

## Future Improvements

- **Shelf tower as `nn.EmbeddingBag`** — shelf vectors are sparse (~3% density); current `nn.Linear(n_shelves, shelf_dim)` treats them as dense. EmbeddingBag gives cleaner gradient flow at the same param count.
- **Book description embeddings** — `description` field via a sentence transformer (`all-MiniLM-L6-v2`) for dense semantic signal. Skipped to avoid a heavy text-encoder dependency.
- **Multi-author avg-pool** — currently primary author only (80.7% single-author). Pad with `nn.Embedding(padding_idx)` and avg-pool all authors.
- **Hybrid feedback** — softmax for candidate generation, explicit ratings for a separate ranking stage.

Do **not** re-attempt in-batch log-frequency debiasing (Yi et al. 2019): the ~14.7k item distribution is too compressed (corrections cluster at 9–10) and it destabilized training. Needs a much larger, more skewed corpus.

See the movie repo's CLAUDE.md "Future User Tower Improvements" and "Richer Cross-Signal Features" for more ideas.

## Working Style and Guidelines

### Git workflow

Never commit and push in the same command — commit first, then ask before pushing.

For changes requiring retraining to validate (hyperparameters, optimizer, scheduler, loss, dataset logic, architecture): write the code, then stop. **Wait for the user to run `python main.py train`, then `canary`, then `eval`, and confirm results before committing.** Don't update results tables in CLAUDE.md until the user reports numbers back.

### Behavioral guidelines

Supplement (not replace) the system prompt. Project-specific points worth re-stating:

- **Match the house style.** Long docstring headers on utils, NamedTuple bundles for related buffers, comment banners on training-loop functions, named slice offsets (not magic numbers), vertically-aligned parquet column comments. New ranker buckets mirror the previous bucket — same naming, util shape, docstring conventions.
- **Surgical changes.** Touch only what the task requires; don't "improve" adjacent code. Mention refactor opportunities — don't act on them.
- **Verification belongs to the user for model changes.** You verify imports/shapes/smoke-test; the user verifies metrics. Don't claim success on a model/dataset change from smoke tests alone.
- **Surface tradeoffs early, in 1–2 sentences.** Name competing interpretations briefly, pick one with a stated assumption — don't silently choose, don't open AskUserQuestion for routine calls.
- **Use TaskCreate for multi-step work, not text-form plans.** Track 3+ step tasks in the UI; don't also write the plan as inline prose or a separate `.md`.
