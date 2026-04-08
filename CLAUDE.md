# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the Goodreads dataset. The model predicts ratings via dot product of user and item embeddings.

This is a sibling project to the MovieLens Two-Tower model at:
`/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model`

The architecture is identical. Only the preprocessing (different CSV/JSON schema) and canary users (book titles instead of movie titles) differ. When in doubt about architecture decisions, refer to the movie repo's CLAUDE.md and src/ as the reference implementation.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: read history (rating-weighted avg pooling of book embeddings), genre affinity, shelf content pooling, and timestamp. Any user can be represented at inference time with just a few books they liked — no retraining required.

## Running the Code

```bash
python main.py preprocess      # Stage 1: raw JSONs/CSVs → data/base_*.parquet
python main.py features        # Stage 2: base parquets → data/features_*.parquet
python main.py dataset         # Stage 3: features → data/dataset_*_v1.pt
python main.py train           # Stage 4: train, save checkpoints
python main.py canary          # Canary user recommendations (most recent checkpoint)
python main.py canary <path>   # Canary user recommendations (specific checkpoint)
python main.py probe           # Embedding probes (most recent checkpoint)
python main.py probe           # Embedding probes (specific checkpoint)
python main.py                 # Run all stages in order
```

## Dataset

Raw data lives in `data/` (not in git). Required files:
- `goodreads_interactions_dedup.json.gz` — JSONL (gzipped): `user_id, book_id, rating, date_added, is_read, is_reviewed`
- `goodreads_books.json` — JSONL: `book_id, title, authors, popular_shelves, average_rating, ratings_count, publication_year, language_code, description`
- `book_id_map.csv` — maps anonymized `book_id` → original Goodreads book_id
- `user_id_map.csv` — maps anonymized `user_id` → original Goodreads user_id

**Note:** No timestamp column in interactions. `popular_shelves` in `goodreads_books.json` serves as both the genre and tag/genome signal.

Thresholds (TBD after EDA): Do not assume the same values as the movie model.
`MIN_RATINGS_PER_BOOK`, `MIN_RATINGS_PER_USER`, `MAX_RATINGS_PER_USER`, `MIN_NUM_TAGS`
all need to be determined by analyzing the Goodreads rating distribution first.

Books below `MIN_RATINGS_PER_BOOK` are filtered out entirely — they are not in the training set and not in the recommendation corpus. Users outside the `MIN_RATINGS_PER_USER` / `MAX_RATINGS_PER_USER` range are filtered out and not used for training.

### Interactions filtering: two-pass streaming

The raw interactions file is `goodreads_interactions_dedup.json.gz` (~11GB compressed, ~40-50GB uncompressed). **Do not load it fully into memory.** Use two streaming passes over the gzip file — RAM stays under ~2GB regardless of input size.

- Only keep records where `rating > 0` — explicit feedback only, directly comparable to MovieLens. Records with only `date_added` (no rating) are "want to read" shelf additions and must be dropped. `read_at` is used as the timestamp but is not required for inclusion — a rated book without `read_at` is still kept (timestamp falls back to `date_updated` then `date_added` for those records only).
- Starting thresholds: `MIN_USER_RATINGS = 20`, `MIN_BOOK_RATINGS = 50` — adjust based on post-filter size; expect a significant reduction in data volume from the `read_at` + rating requirement
- After the two passes, apply k-core filtering iteratively until stable (alternately drop under-threshold users and books until no more are removed)
- Expected output: ~229M raw → ~104M after `rating > 0` → ~10-20M after k-core

**Pass 1** — stream the full file, count interactions per user and per book. Build `valid_users` and `valid_books` sets. These fit in RAM (~few hundred MB).

**Pass 2** — stream again, keep only records where both user and book are in the valid sets. Parse timestamp to unix int here (same format for all date fields: `'Mon Aug 01 13:41:57 -0700 2011'`). Skip records where no timestamp can be parsed. Save result to `data/goodreads_interactions_filtered.parquet`.

**Timestamp field priority:** `read_at` → `date_updated` → `date_added` → skip record. `read_at` is the moment the user finished the book (directly analogous to a MovieLens rating timestamp) and is preferred. Fall back to `date_updated` (usually when the rating was submitted), then `date_added` as last resort. Skip the record only if all three are missing or unparseable.

**Important:** These timestamp fields are present in the dedup JSON but **not** in `goodreads_interactions.csv` — this is why we use the JSON version for preprocessing. The parsed timestamp becomes the timestamp feature (binned, same as movie model).

If the filtered dataset is still too large, increase thresholds and re-run Pass 2 only — no need to redo Pass 1, just tighten the threshold on existing counts.

For faster iteration on a subset, genre-specific files (e.g. `goodreads_interactions_mystery_thriller_crime.json.gz`) use the same format and fields.

## Key differences from MovieLens

| Concept        | MovieLens                              | Goodreads                                          |
|----------------|----------------------------------------|----------------------------------------------------|
| Item ID column | movieId                                | book_id                                            |
| User ID column | userId                                 | user_id                                            |
| Rating scale   | 0.5–5.0 (half stars)                   | 1–5 (integers)                                     |
| Timestamp      | Unix timestamp int                     | `read_at` → `date_updated` → `date_added` (priority order) — parsed to unix timestamp during Pass 2 |
| Year           | Parsed from title (YYYY) regex         | `publication_year` field in books JSON             |
| Genres         | Pipe-separated string in movies.csv    | `popular_shelves` field in goodreads_books.json    |
| User tags      | tags.csv (free-form text)              | `popular_shelves` counts in goodreads_books.json   |
| Genome scores  | genome-scores.csv (pre-built, 1,128 tags) | Computed from popular_shelves shelf counts      |
| Genome tags    | genome-tags.csv (pre-built vocab)      | Derived from shelf vocabulary                      |

## Shelf data as genome equivalent

Goodreads has no pre-built genome scores. The equivalent is computed during preprocessing:
- `popular_shelves` is a list of `{name, count}` objects per book in `goodreads_books.json`
- Shelf-relevance score = fraction of shelving users who applied a given shelf name to a book
- Analogous to MovieLens genome scores (0–1, how strongly a shelf characterizes a book)
- Only shelves applied above `MIN_NUM_TAGS` threshold across all books are kept

## Model Architecture

Identical to the movie model. See the movie repo's CLAUDE.md for full architecture docs.
Variable names use generic terms: `book_id` / `bookId` instead of `movieId`, `top_books` instead of `top_movies`, etc. — but the tensor shapes, tower structure, and dimension math are the same.

### Current embedding sizes

TBD after EDA and first training run. Start with the same 120-dim layout as movies:
```python
item_id_embedding_size           = 40
item_year_embedding_size         = 10
timestamp_feature_embedding_size = 10
item_tag_embedding_size          = 15
item_genome_tag_embedding_size   = 35
user_genre_embedding_size        = 35
item_genre_embedding_size        = 20
# user: 40+35+35+10 = 120
# item: 20+15+35+40+10 = 120 ✓
```

## Training Details

Same as movie model to start:
- Loss: MSE on debiased ratings (raw rating − user mean)
- Optimizer: SGD, `lr=0.005`, `momentum=0.9`
- Batch size: 64
- Steps: 150,000
- Train/val split: 90% of each user's history as context, 10% as labels (chronological)

Note: Goodreads ratings skew higher than MovieLens (users tend to rate 3–4 on average).
Debiasing handles this automatically but be aware val loss may differ from movie model baseline.

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
USER_TYPE_TO_GENOME_TAGS = {
    'Mystery Lover':   ['mystery', 'suspense', 'crime', 'thriller'],
    'Fantasy Lover':   ['fantasy', 'magic', 'epic-fantasy', 'world-building'],
}
```

Canary users are synthetic — no real read timestamps. All receive `ts_max_bin`.

## Relationship to Movie Repo

The `src/` files in this repo are ported from the movie repo with the following changes:
- `preprocess.py` — rewritten for Goodreads JSON/CSV schema
- `evaluate.py` — canary dicts use book titles; inference logic identical
- `model.py` — identical (imported directly or copied with minor renaming)
- `train.py` — identical
- `export.py` — identical
- `features.py` — identical logic, books column names
- `dataset.py` — identical logic, `top_books` / `bookId_to_title` field names

## Future Improvements

- **Book description embeddings** — `goodreads_books.json` includes a `description` field with free-text summaries. Encoding these with a sentence transformer (e.g. `all-MiniLM-L6-v2`) would add dense semantic signal capturing subgenre, tone, and themes that shelf names miss. Skipped in v1 to keep the architecture directly comparable to the movie model and avoid a heavy text encoder dependency. Natural next step once the baseline is working.
- See the movie repo's CLAUDE.md "Future User Tower Improvements" and "Richer Cross-Signal Features" sections for additional ideas that apply equally here.

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.
