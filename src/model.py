"""
Two-Tower BookRecommender model.

Two registered buffers:
  book_shelf_matrix  — (n_books+1, n_shelves) shelf scores per book; last row = padding zeros
  book_author_idx    — (n_books+1,) primary author vocab index per book; last = author padding

User tower: read history avg pool + genre context + timestamp (no shelf/author pooling)
Item tower: genre + shelf + book_id + author + year  (all content signals on item side)

Embedding sizes (must satisfy user_dim == item_dim):
  item_id_embedding_size    = 40   shared: user history pool + item book tower
  user_genre_embedding_size = 50
  timestamp_embedding_size  = 10
  item_genre_embedding_size = 10   item-only
  shelf_embedding_size      = 25   item-only
  author_embedding_size     = 15   item-only
  item_year_embedding_size  = 10   item-only

  user: 40 + 50 + 10 = 100
  item: 10 + 25 + 40 + 15 + 10 = 100  ✓
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
                 item_id_embedding_size=40,
                 author_embedding_size=20,
                 item_year_embedding_size=10,
                 timestamp_embedding_size=10,
                 shelf_embedding_size=30,
                 user_genre_embedding_size=30,
                 item_genre_embedding_size=30,
                 ):
        """
        book_shelf_matrix : float32 tensor (n_books+1, n_shelves)
            Row i = shelf scores for book at embedding index i.
            Last row (index n_books) = zeros, used as padding.
        book_author_idx   : int64 tensor (n_books+1,)
            Entry i = primary author vocab index for book at embedding index i.
            Last entry (index n_books) = n_authors (author padding index).
        """
        super().__init__()

        self.book_pad_idx   = n_books
        self.author_pad_idx = n_authors

        # Registered buffers — non-trainable, device-portable, saved in state_dict
        self.register_buffer('book_shelf_matrix', book_shelf_matrix)
        self.register_buffer('book_author_idx',   book_author_idx)

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
        self.item_shelf_tower = nn.Sequential(
            nn.Linear(n_shelves, shelf_embedding_size),
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
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, user_genre_embedding_size),
            nn.Tanh()
        )
        self.timestamp_embedding_lookup = nn.Embedding(
            n_timestamp_bins, timestamp_embedding_size
        )
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_embedding_size, timestamp_embedding_size),
            nn.Tanh()
        )

        # ── Dimension check ───────────────────────────────────────────────────
        user_dim = (item_id_embedding_size + user_genre_embedding_size
                    + timestamp_embedding_size)
        item_dim = (item_genre_embedding_size + shelf_embedding_size
                    + item_id_embedding_size + author_embedding_size
                    + item_year_embedding_size)
        if user_dim != item_dim:
            raise ValueError(
                f"User dim ({user_dim}) != item dim ({item_dim}). "
                f"user: history {item_id_embedding_size} + genre {user_genre_embedding_size} "
                f"+ ts {timestamp_embedding_size}. "
                f"item: genre {item_genre_embedding_size} + shelf {shelf_embedding_size} "
                f"+ book {item_id_embedding_size} + author {author_embedding_size} "
                f"+ year {item_year_embedding_size}."
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def forward(self, X_genre, X_history, X_history_ratings, timestamps,
                target_genre, target_year, target_book_idx, target_author_idx):
        """
        Args:
            X_genre              (Tensor): (batch, user_context_size)
            X_history            (Tensor): (batch, max_hist_len)   padded book_idx
            X_history_ratings    (Tensor): (batch, max_hist_len)   debiased ratings; 0 at padding
            timestamps           (Tensor): (batch,)
            target_genre         (Tensor): (batch, n_genres)
            target_year          (Tensor): (batch,)
            target_book_idx      (Tensor): (batch,)
            target_author_idx    (Tensor): (batch,)
        """
        pad_mask       = (X_history != self.book_pad_idx).float().unsqueeze(-1)  # (batch, hist, 1)
        rating_weights = X_history_ratings.unsqueeze(-1) * pad_mask              # (batch, hist, 1)
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)         # (batch, 1)

        # ── User: book embedding pool ─────────────────────────────────────────
        history_embs = self.item_embedding_lookup(X_history)                     # (batch, hist, D)
        history_emb  = (history_embs * rating_weights).sum(dim=1) / weight_sum   # (batch, D)

        # ── User: genre and timestamp ─────────────────────────────────────────
        genre_emb     = self.user_genre_tower(X_genre)
        ts_emb        = self.timestamp_embedding_tower(
                            self.timestamp_embedding_lookup(timestamps))
        user_combined = torch.cat([history_emb, genre_emb, ts_emb], dim=1)

        # ── Item towers ───────────────────────────────────────────────────────
        item_genre_emb  = self.item_genre_tower(target_genre)
        item_shelf_vec  = self.book_shelf_matrix[target_book_idx]                # (batch, n_shelves)
        item_shelf_emb  = self.item_shelf_tower(item_shelf_vec)
        item_emb        = self.item_embedding_tower(
                              self.item_embedding_lookup(target_book_idx))
        item_author_emb = self.author_tower(
                              self.author_embedding_lookup(target_author_idx))
        year_emb        = self.year_embedding_tower(
                              self.year_embedding_lookup(target_year))
        item_combined   = torch.cat([item_genre_emb, item_shelf_emb, item_emb,
                                     item_author_emb, year_emb], dim=1)

        # ── Dot product prediction ────────────────────────────────────────────
        return torch.einsum('ij,ij->i', user_combined, item_combined)
