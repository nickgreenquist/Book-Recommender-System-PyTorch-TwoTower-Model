"""
Two-Tower BookRecommender model.

Registered buffers:
  book_shelf_matrix  — (n_books+1, n_shelves) shelf scores; last row = padding  [persistent]
  book_author_idx    — (n_books+1,) primary author vocab index; last = padding   [persistent]
  book_genre_matrix  — (n_books+1, n_genres) genre context; last row = zeros     [non-persistent]
  book_year_idx      — (n_books+1,) year vocab index; last entry = 0             [non-persistent]

User tower modes:
  use_item_pool_for_history=False (prod):
    concat: id_pool(32) + genre(32) + ts(8) = 72 → proj MLP → output_dim
  use_item_pool_for_history=True (experiment):
    concat: item_pool(output_dim) + genre(32) + ts(8) = 168 → proj MLP → output_dim

Item tower: genre + shelf + book_id + author + year → projection MLP
  concat: 10 + 40 + 32 + 10 + 8 = 100 → proj MLP → output_dim

proj_hidden=None → no projection (legacy flat model).
"""
import torch
import torch.nn as nn


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
                 book_genre_matrix=None,
                 book_year_idx=None,
                 use_item_pool_for_history=False,
                 item_id_embedding_size=32,
                 author_embedding_size=10,
                 item_year_embedding_size=8,
                 timestamp_embedding_size=8,
                 shelf_embedding_size=40,
                 user_genre_embedding_size=32,
                 item_genre_embedding_size=10,
                 proj_hidden=256,
                 output_dim=128,
                 ):
        super().__init__()

        self.book_pad_idx              = n_books
        self.author_pad_idx            = n_authors
        self.output_dim                = output_dim
        self.use_item_pool_for_history = use_item_pool_for_history

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
            nn.Tanh()
        )

        # ── Item-only author tower ────────────────────────────────────────────
        self.author_embedding_lookup = nn.Embedding(
            n_authors + 1, author_embedding_size, padding_idx=n_authors
        )
        self.author_tower = nn.Sequential(
            nn.Linear(author_embedding_size, author_embedding_size),
            nn.Tanh()
        )

        # ── Item-only shelf tower ─────────────────────────────────────────────
        shelf_hidden = 128
        self.item_shelf_tower = nn.Sequential(
            nn.Linear(n_shelves, shelf_hidden),
            nn.Tanh(),
            nn.Linear(shelf_hidden, shelf_embedding_size),
            nn.Tanh()
        )

        # ── Item-only towers ──────────────────────────────────────────────────
        self.item_genre_tower = nn.Sequential(
            nn.Linear(n_genres, item_genre_embedding_size),
            nn.Tanh()
        )
        self.year_embedding_lookup = nn.Embedding(n_years, item_year_embedding_size)
        self.year_embedding_tower = nn.Sequential(
            nn.Linear(item_year_embedding_size, item_year_embedding_size),
            nn.Tanh()
        )

        # ── User-only towers ──────────────────────────────────────────────────
        genre_hidden = 64
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, genre_hidden),
            nn.Tanh(),
            nn.Linear(genre_hidden, user_genre_embedding_size),
            nn.Tanh()
        )
        self.timestamp_embedding_lookup = nn.Embedding(
            n_timestamp_bins, timestamp_embedding_size
        )
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_embedding_size, timestamp_embedding_size),
            nn.Tanh()
        )

        # ── Projection MLPs (learn cross-feature interactions) ────────────────
        # proj_hidden=None → no projection; towers output raw concat (legacy flat model).
        if proj_hidden is not None:
            if use_item_pool_for_history:
                # item pool produces output_dim-dim vector in place of the id pool
                user_concat_dim = (output_dim + user_genre_embedding_size
                                   + timestamp_embedding_size)
            else:
                user_concat_dim = (item_id_embedding_size + user_genre_embedding_size
                                   + timestamp_embedding_size)
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
        else:
            self.user_projection = None
            self.item_projection = None

        self.apply(self._init_weights)
        # Projection layers need standard gain — gain=0.1 compounds across multiple
        # layers and causes vanishing gradients when sub-tower outputs are also small.
        if self.user_projection is not None:
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

    def _item_pool(self, X_history, rating_weights, weight_sum):
        """Pool full projected item embeddings over read history. Returns (B, output_dim)."""
        B, H = X_history.shape
        flat = X_history.view(-1)                       # (B*H,)
        embs = self.item_embedding(flat).view(B, H, -1) # (B, H, output_dim)
        return (embs * rating_weights).sum(dim=1) / weight_sum

    def user_embedding(self, X_genre, X_history, X_history_ratings, timestamps):
        """User tower. Returns (batch, output_dim)."""
        pad_mask       = (X_history != self.book_pad_idx).float().unsqueeze(-1)
        rating_weights = X_history_ratings.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)

        genre_emb = self.user_genre_tower(X_genre)
        ts_emb    = self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps))

        if self.use_item_pool_for_history:
            history_emb = self._item_pool(X_history, rating_weights, weight_sum)
        else:
            history_embs = self.item_embedding_lookup(X_history)
            history_emb  = (history_embs * rating_weights).sum(dim=1) / weight_sum

        concat = torch.cat([history_emb, genre_emb, ts_emb], dim=1)
        return self.user_projection(concat) if self.user_projection is not None else concat

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
        return self.item_projection(concat) if self.item_projection is not None else concat

    def forward(self, X_genre, X_history, X_history_ratings, timestamps, target_book_idx):
        """Dot-product score for a (user, item) pair. Used by BPR and MSE training."""
        user_emb = self.user_embedding(X_genre, X_history, X_history_ratings, timestamps)
        item_emb = self.item_embedding(target_book_idx)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
