# Book Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [book-recommender-system-two-tower-model.streamlit.app](https://book-recommender-system-two-tower-model.streamlit.app/)

## Introduction
A PyTorch Two-Tower neural network trained on the [UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (~11k books, ~4.7M training examples).

Trained with in-batch negatives softmax loss, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant books.

This is a sibling project to the [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model).

## Results

Offline evaluation on 5,000 held-out users (leave-label-out protocol, corpus of ~11k books).
Random baseline Hit Rate@10 ≈ 0.09%.

| Metric | MSE | BPR | **Softmax** |
|---|---|---|---|
| Hit Rate@10 | 4.3% | 2.8% | **11.7%** |
| Hit Rate@50 | 16.8% | 14.4% | **31.3%** |
| Recall@10 | 0.0061 | 0.0033 | **0.0194** |
| NDCG@10 | 0.0062 | 0.0036 | **0.0213** |
| MRR | 0.021 | 0.015 | **0.057** |

Switching from MSE to in-batch negatives softmax improved Hit Rate@10 by **171%** and MRR by **171%**.

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
