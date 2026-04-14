# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the Goodreads dataset. The model predicts ratings via dot product of user and item embeddings.

This is a sibling project to the MovieLens Two-Tower model at:
`/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model`

The architecture follows the same two-tower design but adds an **author embedding tower** not present in the movie model. Preprocessing differs significantly (different schema, two-step streaming pipeline). When in doubt about shared architecture decisions, refer to the movie repo's CLAUDE.md.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: read history (rating-weighted avg pooling of book embeddings), genre affinity, and timestamp. Any user can be represented at inference time with just a few books they liked — no retraining required.

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
MIN_RATINGS_PER_BOOK = 10_000  # based on goodreads_books.json ratings_count → ~11k books
MIN_RATINGS_PER_USER = 20      # corpus ratings only
MAX_RATINGS_PER_USER = 500
MIN_NUM_SHELVES      = 500     # shelf must appear this many times across corpus books
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

## Model Architecture

Two-tower design with dot product prediction. Extends the movie model with an **author tower**.

```
User Tower (intentionally simple — no shelf/author pooling):
  rating_weighted_avg_pool(item_embeddings[read_history])  → history_emb  (item_id_embedding_size)
  user_genre_tower([avg_rating_per_genre | read_frac])     → genre_emb    (user_genre_embedding_size)
  timestamp_embedding_tower(read_month)                    → ts_emb       (timestamp_embedding_size)
  concat → user_combined

Item Tower (all content signals):
  item_genre_tower(genre_weighted)     → item_genre_emb    (item_genre_embedding_size)
  item_shelf_tower(tfidf_shelf_scores) → item_shelf_emb    (shelf_embedding_size)
  item_embedding_tower(book_id)        → item_emb          (item_id_embedding_size)   [shared with user history pool]
  author_tower(primary_author_idx)     → item_author_emb   (author_embedding_size)
  year_embedding_tower(pub_year)       → year_emb          (item_year_embedding_size)
  concat → item_combined

Prediction: dot_product(user_combined, item_combined)
```

`len(user_combined) == len(item_combined)` must hold. Model raises `ValueError` at construction if violated.

### Shared towers

- `item_embedding_tower` — shared between item side and user history avg pool

### Tower depths

- `item_shelf_tower`: 2-layer MLP (3032 → 128 → 25) — deep because input is sparse (3% density)
- `user_genre_tower`: 2-layer MLP (2*n_genres → 64 → 50) — deep because combines two signals per genre
- All other towers: single Linear + Tanh — input dims are small enough that depth doesn't help

### Author tower details

- **v1: primary author only** — 80.7% of corpus books have exactly one author; multi-author avg-pool is a future improvement
- `nn.Embedding(n_authors + 1, author_embedding_size)` with padding index = `n_authors`
- Author vocab index 0 = `__unknown__` (books with no author metadata)
- User author affinity = rating-weighted avg pool of primary author embeddings over read history
- 5,857 unique author vocab entries (including `__unknown__` at index 0) in the 11k-book corpus

### Current embedding sizes

```python
item_id_embedding_size    = 40   # shared: user history pool + item tower
user_genre_embedding_size = 50
timestamp_embedding_size  = 10
item_genre_embedding_size = 10
shelf_embedding_size      = 25
author_embedding_size     = 15
item_year_embedding_size  = 10

# user: 40 + 50 + 10 = 100
# item: 10 + 25 + 40 + 15 + 10 = 100 ✓
```

## Training Details

**Primary: In-batch negatives softmax** (`python main.py train softmax`)
- **Loss**: cross-entropy over in-batch negatives. Each step: B×B score matrix, diagonal = correct targets.
- **Dataset**: rollback examples — for each read event, context = all prior reads. Up to 10 examples per user sampled randomly. 4.7M train / 526k val examples.
- **Optimizer**: Adam, `lr=0.001`, `weight_decay=1e-5`
- **Batch size**: 256 (255 in-batch negatives per example)
- **Temperature**: 0.05
- **Steps**: 150,000
- **Random baseline loss**: `log(batch_size)` = `log(256)` ≈ 5.545 — confirmed at step 0
- **Best achieved val loss**: 4.4225 (step 120k of 150k)
- Val loss has high variance with in-batch negatives (different negatives each eval batch) — ±0.2 oscillation is normal

**Legacy: BPR** (`python main.py train`) — kept for reference, not the primary path
- Pairwise loss on (liked, disliked) pairs. `weight_decay=1e-4`.
- Produces semantically structured content embeddings but ID embeddings are noise (no cross-book gradient signal).
- Softmax supersedes BPR: ID embeddings gain structure (King cluster, LOTR cluster confirmed via probe).

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
| `preprocess.py` | Rewritten — two-step streaming pipeline, Goodreads schema |
| `features.py` | Adapted — same logic, book column names, author feature added |
| `dataset.py` | Adapted — same logic, `top_books` / `bookId_to_title` field names |
| `model.py` | Extended — author tower added to both item side and user side |
| `train.py` | Identical |
| `evaluate.py` | Adapted — canary dicts use book titles, shelf tags instead of genome tags |
| `export.py` | Identical |

## Future Improvements

### Training objective

**~~Sampled softmax~~** — ✅ Implemented and validated. Softmax is now the primary training path (`python main.py train softmax`). ID embeddings gained semantic structure vs BPR (King/horror cluster, LOTR/fantasy cluster confirmed via probe and canary). Canary results are strong for Literary, NonFiction, Fantasy, Sci-Fi, History, YA. Known weak spots: Horror (no horror genre in vocab — relies purely on shelf signals) and Romance (model conflates literary women's fiction with romance).

**⚠️ Retraining required:** The year feature vocab changed — `preprocess books` was updated to use original publication year (from `goodreads_book_works.json`) instead of edition year. Year vocab indices are now different from the current checkpoint. The serving app displays correct years (from `bookId_to_year` in the feature store, not the model), but year embedding quality will be degraded until the model is retrained from scratch (`python main.py features && python main.py dataset softmax && python main.py train softmax`).

**Next training improvements:**
1. **~~LR schedule~~** — ✅ Implemented. CosineAnnealingLR from 0.001→0 over training steps. Eliminated the early plateau seen in the first softmax run.
2. **~~Larger batch~~** — ✅ 512 (511 in-batch negatives). Confirmed better drop-from-baseline than batch=256.
3. **~~In-batch negative debiasing (log-frequency correction)~~** — ❌ Tried and removed. Subtracting `log(p_i)` from negative logits (Yi et al., Google RecSys 2019) did not converge on this dataset. Root causes: (a) with ~11k books, the item frequency distribution is very compressed — corrections ranged from 4.91 to 10.0 with ~87% of books clustered near 9-10, so the correction added almost no useful signal. (b) The correction destabilized training even with L2 normalization and diagonal zeroing. Do not re-attempt without a much larger, more skewed item distribution.

**Implicit vs explicit feedback tradeoff** — explicit ratings (BPR) give clean preference signal but are sparse. Implicit feedback (reads via `is_read`) is abundant but noisy. Consider a hybrid: softmax for candidate generation, explicit ratings for a separate ranking stage.

### YouTube DNN implementation details (Covington et al., 2016)

Key design decisions from the paper that directly apply to our softmax implementation:

**Training example construction ("rollback"):**
- For each read event, use only the user's history *before* that event as context. Predict the *future* read, not a randomly held-out one. This captures asymmetric co-read behavior and prevents leakage of future information.
- Generate a **fixed number of examples per user** (cap per user). This prevents highly active users from dominating the loss — effectively weights all users equally.
- **Why rollback for both train and val (not "last read holdout" for val):** Rollback produces examples with varying context lengths (short to long). If val used only each user's last read (full history as context), val examples would always have long contexts while training has short+long — a distribution mismatch that makes val loss unreliable. Using rollback for both keeps context length distribution consistent.

**Negative sampling:**
- Sample several thousand negatives per step from the background distribution (popularity-proportional, i.e. log-uniform over item frequency). Correct for sampling bias via importance weighting.
- This gives ~100x speedup over full softmax while maintaining accuracy. Hierarchical softmax is an alternative but Google found it inferior in practice.

**"Example age" feature (skip for books):**
- The paper feeds `age = t_max - t_read` to correct bias toward older popular videos. For books this doesn't apply — *Crime and Punishment* is as recommendable as a book published last month, and the corpus is essentially static.
- User recency is already captured by the timestamp embedding (read month bin), which signals how recently the user's taste context was formed.

**Serving (inference):**
- At serving time, the softmax output layer is not needed. Prediction reduces to **nearest neighbor search in dot product space**: compute the user embedding from their read history, then find the closest item embeddings via ANN (approximate nearest neighbor).
- Item embeddings (the softmax output weight matrix) are indexed offline. User embeddings are computed on the fly from their history.
- This is identical to what our current model does — the softmax change doesn't affect the serving architecture.

**Shared item ID embedding:**
- The item ID embedding table is shared between the item tower output and the user history avg pool (we already do this). The paper does the same — a single global video embedding used for both the "what video is this" tower and the "what has the user watched" history representation.

**Architecture notes:**
- Hidden layers use **ReLU** (not Tanh). The paper found depth 3 (1024 → 512 → 256 ReLU) best for their scale; for our ~11k book corpus, shallower is fine.
- The final user embedding dimension = the item embedding dimension (they must match for dot product). The softmax weight matrix is shape `(n_books, embedding_dim)` — each row is an item embedding.
- Out-of-vocabulary items map to a zero embedding (already our behavior for unknown authors).

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
