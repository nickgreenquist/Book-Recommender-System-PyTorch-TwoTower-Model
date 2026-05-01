# Recommender System V2 Plan

This file is used to plan a major next version of the recommender system. It is based on industry proven recommender systems.

## Plan Overview
- **Adam optimizer** use this with epsilon 1e-6 ✅
- **Full softmax** instead of in-batch negatives ✅
- **ReLU** for all activations ✅
- **Remove weight decay** ✅
- **L2 Norm Layer** ✅
- **Gradient Clipping** ✅
- **Sum pooling** ✅
- **Shallow History Pooling:** ✅
- **User Genome Context:** ✅
- **Check and if needed fix User Genre Context:** ✅
- **Update temperature** ✅
- **Save model config** ✅
- **Use Apple Silicon GPU** ✅
- **Remove LayerNorm after each history pool** — industry standard (YouTube DNN, TikTok); projection MLP learns the right scale. Book model still has LayerNorm here — remove it and retrain. If gradients explode in the first 500 steps, add it back.
- **LayerNorm on each history pool** ✅ ← remove this (see above)
- **Cosine LR schedule** ✅               
- **Fixed val eval indices** ✅   
- **Static user avg rating** ✅
- **[BIG CHANGE] Separate Quadruple-History Inputs:** ✅

## Popularity Bias (Add Only If Needed)

The book model has not shown popularity bias — Harry Potter and other mega-popular books are not polluting canary results. Do not add this unless canary users start getting dominated by the same few blockbusters regardless of their taste profile.

**If that happens, here is how to add it:**

Training — subtract a log-popularity penalty from logits after dividing by temperature:
```python
# scores shape: (B, n_items)
pop_bias = alpha * torch.log1p(item_interaction_counts)   # (n_items,)
scores = (U @ V_all.T) / temperature - pop_bias           # bias AFTER dividing by temp
```
`alpha=0.4` is a reasonable starting point. The bias must come after `/temperature` — otherwise it lives in the wrong scale and becomes 1000× too large at inference.

Inference — multiply by temperature to convert back to dot-product space, then optionally apply a 2× multiplier for stronger suppression at serving time:
```python
pop_bias_inference = temperature * alpha * 2.0 * torch.log1p(counts)
scores = (user_emb @ item_embs.T) - pop_bias_inference
```
Use 1× in offline eval (matches training) and 2× in the Streamlit app / canary (stronger serving-time suppression). Read `alpha` and `temperature` from the checkpoint config sidecar — never hardcode.

---

## Performance Optimization (Remaining)
- **Pre-Allocation Refactor:** Refactor `dataset.py` to pre-allocate fixed-size NumPy arrays for history buffers rather than using Python lists.
- **Zero-Padding Loop:** Update `train.py` training loop to leverage direct tensor slicing on the pre-allocated, zero-padded tensors.
- **Remove Batch Padding Overhead:** Remove all calls to `pad_history_batch` and `pad_history_ratings_batch` in `train.py` as they are now redundant.