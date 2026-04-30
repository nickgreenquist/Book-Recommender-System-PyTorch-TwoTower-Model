# Recommender System V2 Plan

This file is used to plan a major next version of the recommender system. It is based on industry proven recommender systems.

## Plan Overview
- **Adam optimizer** use this with epsilon 1e-6
- **Full softmax** instead of in-batch negatives (Corpus is small enough at ~9k items).
- **ReLU** for all activations (Replace TanH).
- **Remove weight decay** (Compression via layer sizes provides sufficient regularization).
- **L2 Norm Layer** Applied after the user and item towers
- **Gradient Clipping** add gradient clipping 1.0
- **Sum pooling** for all history aggregation (Replace average pooling).
- **Shallow History Pooling:** Pool directly on item ID embeddings without passing them through the full item tower first.
- **User Genome Context:** Add a genome affinity tower similar to the existing genre affinity tower.
- **Update temperarute** Should be 0.5 / batch_size
- **Save model config** Save model config alongside model pth as sidecar
- **Use Apple Silicon GPU** use GPU for training as full softmax is large now
- **LayerNorm on each history pool** — required when you switch to sum pooling. Variable-length sums have magnitudes that scale with history length; LayerNorm stabilizes this before the projection MLP. Not mentioned in the plan.            
- **Cosine LR schedule** — Steam uses CosineAnnealingLR with eta_min=1e-4. Worth including explicitly.                  
- **[BIG CHANGE] Separate Quadruple-History Inputs:** Partition user history into specific behavior-based pools (Liked, Disliked, and Full history) and then have a rating weighted full history pool (still a sum pool). Liked is rating. this also means remove the full item tower pooling.