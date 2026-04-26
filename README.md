# Book Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [book-recommender-system-two-tower-model.streamlit.app](https://book-recommender-system-two-tower-model.streamlit.app/)

**Sibling projects:** [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model) · [Game Recommender System](https://github.com/nickgreenquist/Game-Recommender-System-PyTorch-TwoTower-Model)

## Introduction
A PyTorch Two-Tower neural network trained on the [UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (~11k books, ~4.7M training examples).

Trained with in-batch negatives softmax loss, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant books.

## Results

Offline evaluation on 5,000 held-out val users (leave-label-out protocol, corpus of ~11k books).
Random baseline Hit Rate@10 ≈ 0.87% (avg ~10 label books per user).

| Metric | MSE | BPR | Softmax | Softmax + Projection | **+ Item Tower Pool** |
|---|---|---|---|---|---|
| Hit Rate@10 | 4.7% | 3.5% | 10.7% | 13.0% | **14.0%** |
| Hit Rate@50 | 17.6% | 14.4% | 28.9% | 33.0% | **36.3%** |
| Recall@10 | 0.0069 | 0.0041 | 0.0164 | 0.0241 | **0.0258** |
| NDCG@10 | 0.0073 | 0.0042 | 0.0189 | 0.0255 | **0.0274** |
| MRR | 0.024 | 0.016 | 0.053 | 0.064 | **0.067** |

Switching from MSE to softmax improved Hit Rate@10 by **127%**. Adding projection MLPs improved it a further **21%** (10.7% → 13.0%). Replacing the 32-dim id pool in the user tower with a rating-weighted avg pool over the full 128-dim item embeddings added another **8%** (13.0% → 14.0%).

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: rating-weighted avg pooling of item embeddings, genre affinity, and read timestamp. Any user can get recommendations from just a few books they liked, with no retraining required.
- **Item tower pooling** — the user tower pools the full projected item embedding (genre + shelf + ID + author + year → 128-dim) over read history, capturing richer cross-feature signals than pooling raw IDs alone.
- **In-batch negatives softmax** — trained with cross-entropy over in-batch negatives (batch size 512), following the YouTube DNN approach (Covington et al., 2016).

## Model architecture

```
User Tower:
  rating_weighted_avg_pool(full_item_embeddings[read_history])  → 128-dim  ← full 128-dim item tower output
  user_genre_tower([avg_rating_per_genre | read_frac])          → 32-dim
  timestamp_embedding_tower(read_month)                         → 8-dim
  concat (168-dim) → projection MLP (256) → 128-dim user embedding

Item Tower:
  item_genre_tower(genre_weighted)      → 10-dim
  item_shelf_tower(tfidf_shelf_scores)  → 40-dim  ← richest signal: 3032-dim TF-IDF, 2-layer MLP
  item_embedding_tower(book_id)         → 32-dim  ← shared lookup table
  author_tower(primary_author_idx)      → 10-dim
  year_embedding_tower(pub_year)        → 8-dim
  concat (100-dim) → projection MLP (256) → 128-dim item embedding

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
