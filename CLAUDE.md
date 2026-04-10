# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the Goodreads dataset. The model predicts ratings via dot product of user and item embeddings.

This is a sibling project to the MovieLens Two-Tower model at:
`/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model`

The architecture follows the same two-tower design but adds an **author embedding tower** not present in the movie model. Preprocessing differs significantly (different schema, two-step streaming pipeline). When in doubt about shared architecture decisions, refer to the movie repo's CLAUDE.md.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: read history (rating-weighted avg pooling of book embeddings), author affinity (rating-weighted avg pooling of author embeddings), genre affinity, shelf content pooling, and timestamp. Any user can be represented at inference time with just a few books they liked — no retraining required.

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

**Shelf scores** (`popular_shelves` in `goodreads_books.json`) — list of `{name, count}` objects per book. Granular user-applied labels (e.g. `cozy-mystery`, `dark`, `epic-fantasy`). Used for the `item_shelf_tower` (analogous to movie genome tag tower):
- Shelf-relevance score = `shelf_count / total_shelf_count_for_book` (0–1)
- Only shelves appearing `>= MIN_NUM_SHELVES` times across corpus books are kept
- Stored in `base_book_shelves.parquet`

## Model Architecture

Two-tower design with dot product prediction. Extends the movie model with an **author tower**.

```
User Tower:
  rating_weighted_avg_pool(item_embeddings[read_history])       → history_emb   (item_id_embedding_size)
  rating_weighted_avg_pool(author_tower(authors[history]))      → author_emb    (author_embedding_size)  [shared tower]
  rating_weighted_avg_pool(shelf_tower(shelf_ctx[history]))     → shelf_emb     (shelf_embedding_size)   [shared tower]
  user_genre_tower([avg_rating_per_genre | read_frac])          → genre_emb     (user_genre_embedding_size)
  timestamp_embedding_tower(read_month)                         → ts_emb        (timestamp_embedding_size)
  concat → user_combined

Item Tower:
  item_genre_tower(genre_onehot)       → item_genre_emb    (item_genre_embedding_size)
  item_shelf_tower(shelf_scores)       → item_shelf_emb    (shelf_embedding_size)          [shared with user shelf pool]
  item_embedding_tower(book_id)        → item_emb          (item_id_embedding_size)        [shared with user history pool]
  author_tower(author_ids → avg_pool)  → item_author_emb   (author_embedding_size)         [shared with user author pool]
  year_embedding_tower(pub_year)       → year_emb          (item_year_embedding_size)
  concat → item_combined

Prediction: dot_product(user_combined, item_combined)
```

`len(user_combined) == len(item_combined)` must hold. Model raises `ValueError` at construction if violated.

### Shared towers

- `item_embedding_tower` — shared between item side and user history avg pool
- `item_shelf_tower` — shared between item side and user shelf context pool
- `author_tower` — shared between item side (target book's authors) and user author pool (authors of read history)

### Author tower details

- **v1: primary author only** — 80.7% of corpus books have exactly one author; multi-author avg-pool is a future improvement
- `nn.Embedding(n_authors + 1, author_embedding_size)` with padding index = `n_authors`
- Author vocab index 0 = `__unknown__` (books with no author metadata)
- User author affinity = rating-weighted avg pool of primary author embeddings over read history
- 5,857 unique author vocab entries (including `__unknown__` at index 0) in the 11k-book corpus

### Current embedding sizes (TBD — starting point)

```python
item_id_embedding_size       = 40   # shared: user history pool + item tower
author_embedding_size        = 20   # shared: item author + user author pool
item_year_embedding_size     = 10
timestamp_embedding_size     = 10
shelf_embedding_size         = 30   # shared: item shelf tower + user shelf pool
user_genre_embedding_size    = 30
item_genre_embedding_size    = 20

# user:  40 + 20 + 30 + 30 + 10  = 130
# item:  20 + 30 + 40 + 20 + 10  = 120  ✗ — needs rebalancing before first run
```

Adjust before first training run to satisfy `user_dim == item_dim`.

## Training Details

Same as movie model to start:
- Loss: MSE on debiased ratings (raw rating − user mean)
- Optimizer: SGD, `lr=0.005`, `momentum=0.9`
- Batch size: 64
- Steps: 150,000
- Train/val split: 90% of each user's history as context, 10% as labels (chronological)

Note: Goodreads ratings skew higher than MovieLens (users tend to rate 3–4 on average).
Debiasing handles this automatically but val loss may differ from movie model baseline.

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

1. **Sampled softmax (implicit feedback)** — replace MSE regression on ratings with sampled softmax classification over the book corpus (YouTube DNN, 2016). Frame recommendation as "predict the next book read" rather than "predict a rating". Use implicit feedback (every read = positive example) with sampled negatives and cross-entropy loss. Yields orders of magnitude more training signal since ratings are sparse but reads are plentiful — especially relevant for Goodreads where many users read without rating. At serving time reduces to the same nearest neighbor search in dot product space.

2. **Implicit vs explicit feedback tradeoff** — explicit ratings (current) give clean preference signal but are sparse. Implicit feedback (reads via `is_read`) is abundant but noisy. Consider a hybrid: use implicit feedback for candidate generation (softmax) and explicit ratings for a separate ranking stage.

### Item features

- **Better shelf-relevance scoring (TF-IDF style)** — v1 uses `shelf_count / total_shelf_count_for_book`. Generic shelves like "to-read" and "fiction" score high on nearly every book and carry little information. A better score: `(shelf_count / total_for_book) * log(total_books / books_with_this_shelf)` — suppresses common shelves, amplifies specific ones like "cozy-mystery". Requires per-shelf document frequency computed during `preprocess books`.

- **Book description embeddings** — `goodreads_books.json` includes a `description` field. Encoding with a sentence transformer (e.g. `all-MiniLM-L6-v2`) would add dense semantic signal. Skipped in v1 to keep a simple baseline and avoid heavy text encoder dependency.

- **Multi-author avg-pool** — v1 uses only the primary author (80.7% of books have one author). Future: avg-pool embeddings across all authors per book, using a padded `nn.Embedding` with `padding_idx`.

- **Consistent label/history indexing** — `read_history` stores `book_idx` integers (pre-mapped in features.py) but `label_bookIds` stores raw string IDs (mapped in dataset.py). Mirrors the movie model pattern but could be unified in features.py for consistency.

See the movie repo's CLAUDE.md "Future User Tower Improvements" and "Richer Cross-Signal Features" sections for additional ideas that apply equally here.

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.
