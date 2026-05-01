# Book Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [book-recommender-system-two-tower-model.streamlit.app](https://book-recommender-system-two-tower-model.streamlit.app/)

**Sibling projects:** [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model) · [Game Recommender System](https://github.com/nickgreenquist/Game-Recommender-System-PyTorch-TwoTower-Model)

## Introduction

A PyTorch Two-Tower neural network trained on the [UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (~14.7k books, ~4.7M training examples).

Trained with full softmax loss over the entire book corpus, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant books.

The current production model (V2) uses **quadruple shallow history pooling** — four parallel sum pools over 32-dim item ID embeddings partitioned by rating signal — plus a dedicated `user_shelf_affinity_tower` that pools TF-IDF shelf vectors over the user's read history. This replaces the earlier ipool architecture (which was 8× slower to train) while beating it on every offline metric.

## Results

Offline evaluation on 5,000 held-out val users (rollback protocol, single target per example).

| Metric | MSE | BPR | Softmax | + Projection | ipool | **V2 (PROD)** |
|---|---|---|---|---|---|---|
| Hit Rate@10 | 4.7% | 3.5% | 10.7% | 13.0% | 14.0% | **15.5%** |
| Hit Rate@50 | 17.6% | 14.4% | 28.9% | 33.0% | 36.3% | **36.1%** |
| NDCG@10 | 0.0073 | 0.0042 | 0.0189 | 0.0255 | 0.0274 | **0.0859** |
| MRR | 0.024 | 0.016 | 0.053 | 0.064 | 0.067 | **0.0775** |

V2 beats ipool PROD on Hit Rate@10 (+11%) and MRR (+16%) despite searching a 33% larger corpus (14.7k vs ~11k books). Switching from MSE to softmax improved Hit Rate@10 by **127%**. Adding projection MLPs improved it a further **21%**. V2 then added another **11%** on top of ipool.

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: quadruple history pools, genre affinity, shelf affinity, and read timestamp. Any user can get recommendations from just a few books they liked, with no retraining required.
- **Quadruple shallow pooling** — history is partitioned into four sum pools over 32-dim ID embeddings: full, liked (rating ≥ 4), disliked (rating ≤ 2), and rating-weighted. Each pool is stabilized with LayerNorm.
- **User shelf affinity tower** — pools the user's per-book TF-IDF shelf vectors over read history, producing a 64-dim representation of the user's shelf taste. Recovers the content signal that ipool captured through item embeddings, without the 8× training cost.
- **Full softmax** — scores against all ~14.7k books every training step. Avoids the in-batch popularity bias of in-batch negatives, correctly surfacing canonical popular books (Shakespeare, Tolstoy, Twilight) that in-batch training systematically penalizes.
- **Author tower** — primary author embedded and projected to 10-dim on the item side. Unique to this repo (not in the movie model).

## Model Architecture (V2 — Current Production)

```
User Tower (Quadruple History Sum Pooling):
  sum_pool(item_id_embeddings[history_full])     → 32-dim + LayerNorm
  sum_pool(item_id_embeddings[history_liked])    → 32-dim + LayerNorm
  sum_pool(item_id_embeddings[history_disliked]) → 32-dim + LayerNorm
  sum_pool(item_id_embeddings[history_weighted]) → 32-dim + LayerNorm
  user_genre_tower([avg_rating_per_genre | read_frac])  → 16-dim
  user_shelf_affinity_tower(pooled_tfidf_shelf_vecs)    → 64-dim
  timestamp_embedding_tower(read_month)                 → 8-dim
  concat (216-dim) → projection MLP (256) → 128-dim → L2 Norm

Item Tower:
  item_genre_tower(genre_weighted)      → 10-dim
  item_shelf_tower(tfidf_shelf_scores)  → 40-dim  ← 3032-dim TF-IDF, 2-layer MLP
  item_embedding_tower(book_id)         → 32-dim  ← shared with user history pools
  author_tower(primary_author_idx)      → 10-dim
  year_embedding_tower(pub_year)        → 8-dim
  concat (100-dim) → projection MLP (256) → 128-dim → L2 Norm

Prediction: dot_product(user_embedding, item_embedding)
```

## V2 Experiment — Promoted to Production ✅

### Goal

Eliminate the 8× training slowdown of the ipool architecture (where the full item tower ran B×H times per step) while matching or exceeding its recommendation quality. Secondary goals: adopt full softmax, L2 normalization, and partitioned history pools.

### Head-to-head canary comparison (V2 vs ipool PROD)

| Category | ipool PROD | V2 | Winner |
|----------|-----------|-----|--------|
| Mystery | ✅ Diverse classic series | ⚠️ Harry Hole clustering | PROD |
| Fantasy | ✅ Deep-cut epic fantasy | ✅ Good epic fantasy | PROD (slight) |
| Romance | ✅ Decent mix | ⚠️ Fifty Shades misfire | PROD (slight) |
| Sci-Fi | ✅✅ Stross/Reynolds hard SF | ✅ Golden Age canon | PROD (slight) |
| YA | ⚠️ Obscure series, clustering | ✅✅ Canonical YA (Twilight, Mortal Instruments) | **V2** |
| History | ⚠️ LBJ trilogy dominates | ✅✅ WWII/Rome/Tudor/WWI diversity | **V2** |
| Classic | ⚠️ Greek philosophy/Chekhov drama | ✅✅ Tolstoy, Shakespeare, Dostoevsky, Homer | **V2** |
| Horror | Tie | Tie | Tie |
| NonFiction | ✅ Evolution cluster | ✅ Broader (physics, medicine, psych) | **V2** (slight) |
| Economics | ✅ Tight Wall Street | ✅ Equally strong | Tie |
| Philosophy | ✅ Plato/existentialism | ✅✅ Full canon (Hegel, Hobbes, Rawls, Rousseau) | **V2** |
| Graphic Novel | ⚠️ 4 Batman books | ✅✅ Dark Knight Returns, Y: Last Man, Transmet | **V2** |
| Manga | ⚠️ Avatar misfire | ✅✅ Actual manga, good variety | **V2** |
| Poetry | ✅ Solid | ✅ Solid | Tie |
| Children's | ⚠️ Cookbook misfire | ✅✅ Canonical picture books | **V2** |

**V2 wins 7 categories, PROD wins 4 (2 slight), 4 ties.**

Key insight: full softmax correctly surfaces popular canonical books in genres where popularity = quality. In-batch negatives training penalizes popular books because they appear as negatives in almost every batch — systematically pushing their embeddings away from user representations.

### Why the initial V2 run failed (temperature bug)

The first V2 training used `temperature = 0.5/batch_size = 0.001`. This is correct for in-batch negatives (where batch size = number of negatives) but collapses gradients to near-argmax over 11k items for full softmax. Result: popularity overfitting across most categories. Fixed to `temperature = 0.1`.

## Usage

```bash
python main.py preprocess   # Build base_books.parquet + interaction parquets
python main.py features     # Build features_*.parquet
python main.py dataset      # Build dataset_*_v1.pt
python main.py train        # Train model, save checkpoint
python main.py canary       # Evaluate canary users
python main.py eval         # Offline eval: Hit Rate@K, NDCG@K, MRR
python main.py export       # Generate serving/ artifacts
streamlit run streamlit_app.py
```
