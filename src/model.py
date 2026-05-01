"""
Two-Tower BookRecommender model.

Registered buffers:
  book_shelf_matrix  — (n_books+1, n_shelves) shelf TF-IDF scores; last row = zeros  [persistent]
  book_author_idx    — (n_books+1,) primary author vocab index; last = padding idx    [persistent]
  book_genre_matrix  — (n_books+1, n_genres) genre context; last row = zeros          [non-persistent]
  book_year_idx      — (n_books+1,) year vocab index; last entry = 0                  [non-persistent]

User tower: 4×sum_pool(32) + genre(16) + shelf_affinity(64) + ts(8) = 216 → projection MLP → output_dim
Item tower: genre(10) + shelf(40) + book_id(32) + author(10) + year(8) = 100 → projection MLP → output_dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BookRecommender(nn.Module):
    def __init__(self,
                 n_genres,
                 n_shelves,
                 n_books,
                 n_years,
                 n_authors,
                 n_timestamp_bins,
                 user_context_size,
                 book_shelf_matrix,
                 book_author_idx,
                 book_genre_matrix,
                 book_year_idx,
                 item_id_embedding_size=32,
                 author_embedding_size=10,
                 item_year_embedding_size=8,
                 timestamp_embedding_size=8,
                 shelf_embedding_size=40,
                 user_genre_embedding_size=16,
                 user_shelf_affinity_embedding_size=64,
                 item_genre_embedding_size=10,
                 proj_hidden=256,
                 output_dim=128,
                 ):
        super().__init__()

        self.book_pad_idx   = n_books
        self.author_pad_idx = n_authors
        self.output_dim     = output_dim

        # Persistent buffers — saved in state_dict (excluded at export via key filter)
        self.register_buffer('book_shelf_matrix', book_shelf_matrix)
        self.register_buffer('book_author_idx',   book_author_idx)
        # Non-persistent — NOT saved in state_dict; rebuilt from FeatureStore on every load
        self.register_buffer('book_genre_matrix', book_genre_matrix, persistent=False)
        self.register_buffer('book_year_idx',     book_year_idx,     persistent=False)

        # ── Shared item embedding ─────────────────────────────────────────────
        self.item_embedding_lookup = nn.Embedding(
            n_books + 1, item_id_embedding_size, padding_idx=n_books
        )
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_id_embedding_size, item_id_embedding_size),
            nn.ReLU()
        )

        # ── Item-only author tower ────────────────────────────────────────────
        self.author_embedding_lookup = nn.Embedding(
            n_authors + 1, author_embedding_size, padding_idx=n_authors
        )
        self.author_tower = nn.Sequential(
            nn.Linear(author_embedding_size, author_embedding_size),
            nn.ReLU()
        )

        # ── Item-only shelf tower ─────────────────────────────────────────────
        shelf_hidden = 128
        self.item_shelf_tower = nn.Sequential(
            nn.Linear(n_shelves, shelf_hidden),
            nn.ReLU(),
            nn.Linear(shelf_hidden, shelf_embedding_size),
            nn.ReLU()
        )

        # ── Item-only towers ──────────────────────────────────────────────────
        self.item_genre_tower = nn.Sequential(
            nn.Linear(n_genres, item_genre_embedding_size),
            nn.ReLU()
        )
        self.year_embedding_lookup = nn.Embedding(n_years, item_year_embedding_size)
        self.year_embedding_tower = nn.Sequential(
            nn.Linear(item_year_embedding_size, item_year_embedding_size),
            nn.ReLU()
        )

        # ── User-only towers ──────────────────────────────────────────────────
        genre_hidden = 64
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, genre_hidden),
            nn.ReLU(),
            nn.Linear(genre_hidden, user_genre_embedding_size),
            nn.ReLU()
        )
        self.timestamp_embedding_lookup = nn.Embedding(
            n_timestamp_bins, timestamp_embedding_size
        )
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_embedding_size, timestamp_embedding_size),
            nn.ReLU()
        )

        # ── Quadruple history sum pools with LayerNorm ────────────────────────
        self.hist_full_norm     = nn.LayerNorm(item_id_embedding_size)
        self.hist_liked_norm    = nn.LayerNorm(item_id_embedding_size)
        self.hist_disliked_norm = nn.LayerNorm(item_id_embedding_size)
        self.hist_weighted_norm = nn.LayerNorm(item_id_embedding_size)

        # ── User-side shelf affinity tower ────────────────────────────────────
        shelf_aff_hidden = 128
        self.user_shelf_affinity_tower = nn.Sequential(
            nn.Linear(n_shelves, shelf_aff_hidden),
            nn.ReLU(),
            nn.Linear(shelf_aff_hidden, user_shelf_affinity_embedding_size),
            nn.ReLU()
        )

        # ── Projection MLPs (learn cross-feature interactions) ────────────────
        # 4 sum pools × item_id_embedding_size + genre + shelf_affinity + ts
        user_concat_dim = (4 * item_id_embedding_size + user_genre_embedding_size
                           + user_shelf_affinity_embedding_size + timestamp_embedding_size)
        item_concat_dim = (item_genre_embedding_size + shelf_embedding_size
                           + item_id_embedding_size + author_embedding_size
                           + item_year_embedding_size)

        # No activation on the final linear — feeds directly into dot product.
        self.user_projection = nn.Sequential(
            nn.Linear(user_concat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, output_dim),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(item_concat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, output_dim),
        )

        self.apply(self._init_weights)
        # Projection layers need standard gain — gain=0.1 compounds across multiple
        # layers and causes vanishing gradients when sub-tower outputs are also small.
        for proj in [self.user_projection, self.item_projection]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def _sum_pool(self, X_history, norm_layer, weights=None):
        """Shallow sum pooling on item ID embeddings with LayerNorm."""
        embs = self.item_embedding_lookup(X_history)
        if weights is not None:
            embs = embs * weights.unsqueeze(-1)
        pooled = embs.sum(dim=1)
        return norm_layer(pooled)

    def user_embedding(self, X_genre, X_hist_full, X_hist_liked, X_hist_disliked, 
                       X_hist_weighted, X_rats_weighted, timestamps):
        """User tower. Returns (batch, output_dim)."""
        genre_emb   = self.user_genre_tower(X_genre)
        ts_emb      = self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps))

        # 1. Sum pool history: Full, Liked, Disliked (unweighted sum)
        pool_full     = self._sum_pool(X_hist_full,     self.hist_full_norm)
        pool_liked    = self._sum_pool(X_hist_liked,    self.hist_liked_norm)
        pool_disliked = self._sum_pool(X_hist_disliked, self.hist_disliked_norm)

        # 2. Sum pool history: Rating-weighted
        pool_weighted = self._sum_pool(X_hist_weighted, self.hist_weighted_norm, 
                                       weights=X_rats_weighted)
        
        history_emb = torch.cat([pool_full, pool_liked, pool_disliked, pool_weighted], dim=1)

        # ── User Genome (Shelf Affinity) ──
        pad_mask       = (X_hist_weighted != self.book_pad_idx).float().unsqueeze(-1)
        rating_weights = X_rats_weighted.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)
        
        history_shelves = self.book_shelf_matrix[X_hist_weighted]
        shelf_affinity  = (history_shelves * rating_weights).sum(dim=1) / weight_sum
        shelf_affinity_emb = self.user_shelf_affinity_tower(shelf_affinity)

        concat = torch.cat([history_emb, genre_emb, shelf_affinity_emb, ts_emb], dim=1)
        out = self.user_projection(concat)
        return F.normalize(out, p=2, dim=1)

    def item_embedding(self, target_book_idx):
        """Item tower. Looks up all features from registered buffers. Returns (batch, output_dim)."""
        item_genre_emb  = self.item_genre_tower(self.book_genre_matrix[target_book_idx])
        item_shelf_emb  = self.item_shelf_tower(self.book_shelf_matrix[target_book_idx])
        item_emb        = self.item_embedding_tower(self.item_embedding_lookup(target_book_idx))
        item_author_emb = self.author_tower(
                              self.author_embedding_lookup(self.book_author_idx[target_book_idx]))
        year_emb        = self.year_embedding_tower(
                              self.year_embedding_lookup(self.book_year_idx[target_book_idx]))
        concat = torch.cat([item_genre_emb, item_shelf_emb, item_emb,
                            item_author_emb, year_emb], dim=1)
        out = self.item_projection(concat)
        return F.normalize(out, p=2, dim=1)

    def full_item_embedding(self):
        """Returns all item embeddings in the corpus (n_books, output_dim)."""
        n_books = self.book_pad_idx # book_pad_idx is n_books
        all_idxs = torch.arange(n_books, device=self.book_shelf_matrix.device)
        return self.item_embedding(all_idxs)

    def forward(self, X_genre, X_hist_full, X_hist_liked, X_hist_disliked,
                X_hist_weighted, X_rats_weighted, timestamps, target_book_idx):
        """Dot-product score for a (user, item) pair."""
        user_emb = self.user_embedding(X_genre, X_hist_full, X_hist_liked, X_hist_disliked,
                                       X_hist_weighted, X_rats_weighted, timestamps)
        item_emb = self.item_embedding(target_book_idx)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
