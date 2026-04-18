# Book Recommender System — PyTorch Two-Tower Model

**Live demo:** [book-recommender-system-two-tower-model.streamlit.app](https://book-recommender-system-two-tower-model.streamlit.app/)

## Introduction
A PyTorch Two-Tower neural network trained on the [UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (~11k books, ~4.7M training examples). The model learns to match users to books via dot product of user and item embeddings.

This is a sibling project to the [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model).

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: rating-weighted avg pooling of book embeddings, genre affinity, and read timestamp. Any user can get recommendations from just a few books they liked, with no retraining required.
- **In-batch negatives softmax** — trained with cross-entropy over in-batch negatives (batch size 512), following the YouTube DNN approach (Covington et al., 2016).

## Model architecture

```
User Tower:
  rating_weighted_avg_pool(item_embeddings[read_history])  → 40-dim
  user_genre_tower([avg_rating_per_genre | read_frac])     → 50-dim
  timestamp_embedding_tower(read_month)                    → 10-dim
  concat → 100-dim user embedding

Item Tower:
  item_genre_tower(genre_weighted)      → 10-dim
  item_shelf_tower(tfidf_shelf_scores)  → 25-dim
  item_embedding_tower(book_id)         → 40-dim  ← shared with user history pool
  author_tower(primary_author_idx)      → 15-dim
  year_embedding_tower(pub_year)        → 10-dim
  concat → 100-dim item embedding

Prediction: dot_product(user_embedding, item_embedding)
```

## Usage

```bash
python main.py preprocess   # Build base_books.parquet + interaction parquets
python main.py features     # Build features_*.parquet
python main.py dataset      # Build dataset_*_v1.pt
python main.py train        # Train model, save checkpoint
python main.py export       # Generate serving/ artifacts
python main.py canary       # Evaluate canary users
streamlit run streamlit_app.py
```
