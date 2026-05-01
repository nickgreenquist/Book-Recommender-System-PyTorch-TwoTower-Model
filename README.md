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

## V2 Experiment — Industry-Standard Architecture (Not Promoted)

The goal of the V2 experiment was to make the user tower more industry-standard while eliminating the 8× training slowdown of ipool. In the ipool (current PROD) architecture, the user tower pools the full 128-dim projected item embedding over every book in read history — this means the entire item tower (shelf MLP + genre MLP + author MLP) runs once per history item per training step.

**What V2 changed:**

- Replaced the single ipool with four parallel shallow sum pools over raw 32-dim ID embeddings: `history_full`, `history_liked`, `history_disliked`, `history_weighted`, each stabilized with LayerNorm
- Added a dedicated `user_shelf_affinity_tower` — pools the user's per-book TF-IDF shelf vectors over read history to produce a 64-dim taste representation, intended to recover the shelf signal lost by dropping ipool
- Adopted full softmax (score against all ~11k books per step) with popularity debiasing (`alpha * log1p(interaction_count)`)
- L2 normalization on both tower outputs, CosineAnnealingLR, gradient clipping
- Dataset RAM reduced ~3× by looking up item features from registered buffers at training time rather than storing them in dataset tensors

**Result: V2 was trained to 150k steps but not promoted to production.**

The canary comparison against PROD (ipool) showed PROD substantially better across most categories — especially the hardest ones to get right:

- **Nick (personal canary)**: PROD recommended Battle Cry of Freedom, Founding Brothers, Flags of Our Fathers, An Army at Dawn. V2 recommended the same core but with Checklist Manifesto, Mindset, Getting Things Done bleeding in.
- **Sci-Fi**: PROD recommended Singularity Sky, Revelation Space, Accelerando, Spin, Mote in God's Eye — hard SF deep cuts across Stross/Reynolds/Vinge/Niven. V2 clustered around Asimov and Hyperion.
- **Classic Lit**: PROD recommended Chekhov, Molière, Balzac, Aristotle — broad international coverage. V2 clustered on Dickens (4/10 Dickens titles).

**Why ipool wins:** Pooling full projected item embeddings (128-dim, including shelf MLP output) directly into the user representation gives the user tower rich, learned content signal for every book they have read. V2's `user_shelf_affinity_tower` operates on raw TF-IDF vectors rather than learned embeddings, and the shallow 32-dim ID pools don't carry enough content information to compensate. The shelf signal flowing implicitly through item embeddings into the user representation is the core strength of ipool — and it is difficult to replicate with a separate user-side tower.

PROD wins 10 of 13 comparable canary categories. V2 code is preserved on branch `v2`.

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
