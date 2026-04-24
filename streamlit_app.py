"""
Book Recommender — Streamlit app.

Run locally:  streamlit run streamlit_app.py
Requires:     serving/model.pth
              serving/book_embeddings.pt
              serving/feature_store.pt

Generate serving/ with: python main.py export
"""
import importlib

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

import src.evaluate
importlib.reload(src.evaluate)
from src.evaluate import (
    USER_TYPE_TO_FAVORITE_GENRES,
    USER_TYPE_TO_WORST_GENRES,
    USER_TYPE_TO_FAVORITE_BOOKS,
    USER_TYPE_TO_LIKED_BOOKS,
    USER_TYPE_TO_SHELF_TAGS,
    VALUE_FAVORITE_BOOK_RATING,
    VALUE_ANCHOR_BOOK_RATING,
    VALUE_FAVORITE_GENRE_RATING,
)
from src.model import BookRecommender

_LIKED_BOOK  = VALUE_FAVORITE_BOOK_RATING   # 2.0
_ANCHOR_BOOK = VALUE_ANCHOR_BOOK_RATING     # 1.0
_LIKED_GENRE = VALUE_FAVORITE_GENRE_RATING  # 4.0
_ANCHORS_PER_TAG = 5
_COVER_WIDTH = 100
_COVER_ROW_HEIGHT = 200



# ── Startup ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    fs  = torch.load('serving/feature_store.pt', weights_only=False)
    be  = torch.load('serving/book_embeddings.pt', weights_only=False)
    cfg = fs['model_config']

    # Stored as float16 to stay under GitHub's 100MB limit; cast back for inference
    fs['book_shelf_matrix'] = fs['book_shelf_matrix'].float()

    model = BookRecommender(
        n_genres=len(fs['genres_ordered']),
        n_shelves=len(fs['shelves_ordered']),
        n_books=len(fs['top_books']),
        n_years=len(fs['years_ordered']),
        n_authors=len(fs['authors_ordered']),
        n_timestamp_bins=fs['timestamp_num_bins'],
        user_context_size=fs['user_context_size'],
        book_shelf_matrix=fs['book_shelf_matrix'],
        book_author_idx=fs['book_author_idx'],
        item_id_embedding_size=cfg['item_id_embedding_size'],
        author_embedding_size=cfg['author_embedding_size'],
        shelf_embedding_size=cfg['shelf_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        timestamp_embedding_size=cfg['timestamp_embedding_size'],
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        proj_hidden=cfg.get('proj_hidden', 256),
        output_dim=cfg.get('output_dim', 128),
    )
    # strict=False: model.pth excludes buffers (book_shelf_matrix, book_author_idx)
    # which are already set via the constructor above.
    model.load_state_dict(torch.load('serving/model.pth', weights_only=True), strict=False)
    model.eval()

    all_ids  = list(be.keys())
    all_embs = torch.cat([be[b]['BOOK_EMBEDDING_COMBINED'] for b in all_ids], dim=0)
    all_norm = F.normalize(all_embs, dim=1)

    ts_inference = torch.bucketize(
        torch.tensor([float(fs['timestamp_bins'][-1].item())]),
        fs['timestamp_bins'], right=False,
    )

    return model, fs, be, all_ids, all_embs, all_norm, ts_inference


# ── Inference helpers ─────────────────────────────────────────────────────────

def _get_shelf_anchors(fs, shelf_tags, exclude):
    """Return up to _ANCHORS_PER_TAG top books per shelf tag, skipping excluded titles."""
    anchor_titles = []
    seen = set(exclude)
    for tag in shelf_tags:
        if not tag or tag not in fs['shelf_to_i']:
            continue
        tag_idx    = fs['shelf_to_i'][tag]
        sorted_bids = sorted(
            fs['top_books'],
            key=lambda bid: float(fs['book_shelf_matrix'][fs['bookId_to_idx'][bid]][tag_idx]),
            reverse=True,
        )
        count = 0
        for bid in sorted_bids:
            if count >= _ANCHORS_PER_TAG:
                break
            title = fs['bookId_to_title'][bid]
            if title not in seen:
                anchor_titles.append(title)
                seen.add(title)
                count += 1
    return anchor_titles


def _build_user_embedding(model, fs, liked_titles_with_weights, liked_genres, ts_inference):
    """
    Build a combined user embedding from book and genre signals.
    liked_titles_with_weights: list of (title, weight) — explicit likes use _LIKED_BOOK,
      shelf anchors use _ANCHOR_BOOK.
    liked_genres: list of genre name strings (from genres_ordered).
    """
    n_genres = len(fs['genres_ordered'])
    ctx = [0.0] * (2 * n_genres)

    # Derive genre context from liked books — ensures ctx is informative even
    # when the user skips the explicit genre selector.
    genre_rating_sum  = {}
    genre_book_count  = {}
    total_books = 0
    for title, w in liked_titles_with_weights:
        bid = fs['title_to_bookId'].get(title)
        if bid is None:
            continue
        total_books += 1
        for g in fs['bookId_to_genres'].get(bid, []):
            genre_rating_sum[g]  = genre_rating_sum.get(g, 0.0) + w
            genre_book_count[g]  = genre_book_count.get(g, 0)   + 1

    for g, rsum in genre_rating_sum.items():
        avg_r = rsum / genre_book_count[g]
        frac  = genre_book_count[g] / max(total_books, 1)
        if g in fs['genre_to_i']:
            ctx[fs['genre_to_i'][g]]            = avg_r
            ctx[n_genres + fs['genre_to_i'][g]] = frac

    # Explicit genre overrides — stronger signal than book-derived estimates
    for g in liked_genres:
        if g in fs['genre_to_i']:
            ctx[fs['genre_to_i'][g]]            = _LIKED_GENRE
            ctx[n_genres + fs['genre_to_i'][g]] = 1.0 / max(len(liked_genres), 1)

    # History pool — rating-weighted avg of book embeddings
    history = [
        (fs['bookId_to_idx'][fs['title_to_bookId'][t]], w)
        for t, w in liked_titles_with_weights
        if t in fs['title_to_bookId'] and fs['title_to_bookId'][t] in fs['bookId_to_idx']
    ]

    if history:
        hist_ids = torch.tensor([h[0] for h in history], dtype=torch.long).unsqueeze(0)
        hist_wts = torch.tensor([[h[1] for h in history]], dtype=torch.float)
        hist_embs   = model.item_embedding_lookup(hist_ids)
        wt_sum      = hist_wts.unsqueeze(-1).abs().sum(dim=1).clamp(min=1e-6)
        history_emb = (hist_embs * hist_wts.unsqueeze(-1)).sum(dim=1) / wt_sum
    else:
        history_emb = torch.zeros(1, model.item_embedding_lookup.embedding_dim)

    genre_emb = model.user_genre_tower(torch.tensor([ctx]))
    ts_emb    = model.timestamp_embedding_tower(
                    model.timestamp_embedding_lookup(ts_inference))
    concat = torch.cat([history_emb, genre_emb, ts_emb], dim=1)
    return model.user_projection(concat) if model.user_projection is not None else concat


def _cover_url(bid, fs):
    """Return Open Library cover URL for a book, or None if no ISBN."""
    isbn = fs.get('bookId_to_isbn', {}).get(bid, '')
    if not isbn:
        return ''
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"


def _top_shelves(bid, fs, n=3):
    """Return the top-n shelf tag names for a book by raw TF-IDF score."""
    shelf_ctx = fs['book_shelf_matrix'][fs['bookId_to_idx'][bid]]
    top_idx   = sorted(range(len(shelf_ctx)), key=lambda i: -shelf_ctx[i].item())[:n]
    return ', '.join(fs['shelves_ordered'][i] for i in top_idx if shelf_ctx[i] > 0)


def _score_books(user_emb, all_ids, all_embs, fs, exclude_titles, top_n=20):
    """Dot-product score all books, filter seeds, return top-n as a DataFrame."""
    raw_scores = (all_embs @ user_emb.T).squeeze(-1)
    exclude    = set(exclude_titles)
    rows = []
    for i in raw_scores.argsort(descending=True).tolist():
        bid   = all_ids[i]
        title = fs['bookId_to_title'][bid]
        if title in exclude:
            continue
        rows.append({
            'Cover':      _cover_url(bid, fs),
            'Title':      title,
            'Author':     fs['bookId_to_author'].get(bid, ''),
            'Year':       fs['bookId_to_year'].get(bid, '') if fs['bookId_to_year'].get(bid, '') != '-1' else '',
            'Top Shelves': _top_shelves(bid, fs),
        })
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def _render_results(df: pd.DataFrame) -> None:
    """Render a results DataFrame as a dataframe with cover images."""
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        row_height=_COVER_ROW_HEIGHT,
        height=_COVER_ROW_HEIGHT * (len(df) // 2) + 38,  # 38px for header row
        column_config={
            'Cover': st.column_config.ImageColumn('Cover'),
        },
    )


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(model, fs, all_ids, all_embs, ts_inference):
    st.caption(
        "Select books you've enjoyed and the model will infer your taste. "
        "The more books you add, the sharper the recommendations."
    )
    all_titles = fs['popularity_ordered_titles']

    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_shelf_tags'):
            st.session_state[key] = []

    profile = st.session_state.pop('_load_profile', None)
    if profile:
        st.session_state['rec_liked'] = USER_TYPE_TO_FAVORITE_BOOKS[profile]

    liked_titles = st.multiselect("Favorite Books", all_titles, key='rec_liked', max_selections=20)
    liked_genres = []  # derived from book history in _build_user_embedding

    with st.expander("Refine by Shelf Tags (optional)"):
        st.caption(
            "Select shelf tags to describe what you're looking for — subgenres, moods, settings "
            "(e.g. 'epic-fantasy', 'cozy-mystery', 'coming-of-age'). "
            "The 5 most representative books for each tag will be added as implicit likes."
        )
        shelf_tags     = sorted(t for t in fs['shelves_ordered'] if t)
        selected_shelves = st.multiselect("Shelf tags", shelf_tags, key='rec_shelf_tags', max_selections=10)

    st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="secondary"] {
            display: block;
            margin: 1rem auto;
            padding: 0.75rem 3rem;
            font-size: 1.2rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    _, btn_col, clear_col = st.columns([2, 1, 2])
    if clear_col.button("Clear All", use_container_width=False):
        st.session_state['_clear_rec'] = True
        st.rerun()
    if btn_col.button("Get Recommendations", use_container_width=True):
        if not liked_titles and not selected_shelves:
            st.warning("Select at least one liked book, genre, or shelf tag.")
            return

        # Compute per-tag anchor pairs so we can display them accurately
        anchor_tag_book_pairs = []
        seen_anchors = set(liked_titles)
        for tag in selected_shelves:
            for t in _get_shelf_anchors(fs, [tag], exclude=seen_anchors):
                anchor_tag_book_pairs.append((tag, t))
                seen_anchors.add(t)

        anchor_titles = [t for _, t in anchor_tag_book_pairs]
        liked_with_weights = (
            [(t, _LIKED_BOOK)  for t in liked_titles] +
            [(t, _ANCHOR_BOOK) for t in anchor_titles]
        )

        with torch.no_grad():
            user_emb = _build_user_embedding(
                model, fs, liked_with_weights, liked_genres, ts_inference,
            )

        if anchor_tag_book_pairs:
            st.caption("Shelf anchors — " + " · ".join(
                f"{tag}: {t}" for tag, t in anchor_tag_book_pairs
            ))

        df = _score_books(user_emb, all_ids, all_embs, fs,
                          exclude_titles=liked_titles + anchor_titles)
        _render_results(df)


# ── Tab: Recommend (Examples) ─────────────────────────────────────────────────

def tab_recommend_examples(model, fs, all_ids, all_embs, ts_inference):
    st.caption("Select a pre-built user profile to see what the model recommends for that taste.")
    profiles = list(USER_TYPE_TO_FAVORITE_GENRES.keys())
    selected_profile = st.selectbox(
        "Profile",
        options=[None] + profiles,
        format_func=lambda x: "Choose a profile..." if x is None else x,
        label_visibility="collapsed",
    )

    if selected_profile:
        fav_books    = USER_TYPE_TO_FAVORITE_BOOKS[selected_profile]
        liked_books  = USER_TYPE_TO_LIKED_BOOKS.get(selected_profile, [])
        fav_genres   = USER_TYPE_TO_FAVORITE_GENRES[selected_profile]
        shelf_tags   = USER_TYPE_TO_SHELF_TAGS.get(selected_profile, [])

        anchor_titles = _get_shelf_anchors(
            fs, shelf_tags, exclude=set(fav_books) | set(liked_books))

        missing = [t for t in fav_books if t not in fs['title_to_bookId']]
        if missing:
            st.warning("Not found in corpus (check title format): " + ", ".join(missing))

        liked_with_weights = (
            [(t, _LIKED_BOOK)  for t in fav_books]   +
            [(t, _ANCHOR_BOOK) for t in liked_books] +
            [(t, _ANCHOR_BOOK) for t in anchor_titles]
        )

        with torch.no_grad():
            user_emb = _build_user_embedding(
                model, fs, liked_with_weights, fav_genres, ts_inference,
            )

        df = _score_books(user_emb, all_ids, all_embs, fs,
                          exclude_titles=fav_books + liked_books + anchor_titles)

        display_name = selected_profile.removesuffix("'s Recommendations") if selected_profile.endswith("'s Recommendations") else selected_profile
        st.subheader(f"Recommendations for: {display_name}")
        if fav_books:
            st.caption("Because you like: " + ", ".join(fav_books))
        if fav_genres:
            st.caption("Favorite genres: " + ", ".join(fav_genres))
        if anchor_titles:
            st.caption("Shelf anchors: " + ", ".join(anchor_titles))
        _render_results(df)


# ── Tab: Similar ──────────────────────────────────────────────────────────────

def tab_similar(be, fs, all_ids, all_norm):
    st.caption(
        "Each book is represented by a single combined embedding — the concatenation of "
        "its genre, shelf, book ID, author, and year towers. "
        "This tab finds books whose combined embedding is most similar (cosine) to the selected seed."
    )
    all_titles  = fs['popularity_ordered_titles']
    selections  = st.multiselect("Book", all_titles, key='sim_title', max_selections=10)

    if st.button("Find Similar Books"):
        if not selections:
            st.warning("Select a book.")
            return
        for title in selections:
            bid = fs['title_to_bookId'].get(title)
            if bid not in be:
                st.error(f"'{title}' not in corpus.")
                continue

            with torch.no_grad():
                seed_norm = F.normalize(be[bid]['BOOK_EMBEDDING_COMBINED'], dim=1)
                sims      = (all_norm @ seed_norm.T).squeeze(-1)

            rows = []
            for idx in sims.argsort(descending=True).tolist():
                candidate = all_ids[idx]
                if candidate == bid:
                    continue
                rows.append({
                    'Cover':      _cover_url(candidate, fs),
                    'Title':      fs['bookId_to_title'][candidate],
                    'Author':     fs['bookId_to_author'].get(candidate, ''),
                    'Year':       fs['bookId_to_year'].get(candidate, '') if fs['bookId_to_year'].get(candidate, '') != '-1' else '',
                    'Top Shelves': _top_shelves(candidate, fs),
                    'Score':      f"{sims[idx].item():.3f}",
                })
                if len(rows) >= 20:
                    break
            st.subheader(f"Similar to: {title}")
            _render_results(pd.DataFrame(rows))


# ── Tab: Explore Genres ───────────────────────────────────────────────────────

def tab_explore_genres(model, be, fs):
    st.subheader("Explore Genre Item Tower Embeddings")
    st.caption(
        "Queries the item genre embedding space directly — finds books whose "
        "genre embedding best matches the selected genres."
    )
    genres          = fs['genres_ordered']
    selected_genres = st.multiselect("Genres", genres, key='explore_genre')
    if st.button("Explore", key='btn_genre'):
        if not selected_genres:
            st.warning("Select at least one genre.")
            return
        ctx = [0.0] * len(genres)
        for g in selected_genres:
            ctx[fs['genre_to_i'][g]] = 1.0
        with torch.no_grad():
            query = model.item_genre_tower(torch.tensor([ctx])).view(-1)
        sims = {
            bid: F.cosine_similarity(
                query.unsqueeze(0),
                be[bid]['BOOK_GENRE_EMBEDDING'].view(-1).unsqueeze(0),
            ).item()
            for bid in fs['top_books']
        }
        rows = [
            {
                'Cover':      _cover_url(bid, fs),
                'Title':      fs['bookId_to_title'][bid],
                'Author':     fs['bookId_to_author'].get(bid, ''),
                'Year':       fs['bookId_to_year'].get(bid, '') if fs['bookId_to_year'].get(bid, '') != '-1' else '',
                'Top Shelves': _top_shelves(bid, fs),
                'Score':      f'{s:.4f}',
            }
            for bid, s in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        _render_results(pd.DataFrame(rows))


# ── Tab: Explore Shelves ──────────────────────────────────────────────────────

def tab_explore_shelves(model, be, fs):
    st.subheader("Explore Shelf Tag Item Tower Embeddings")
    st.caption(
        "Select shelf tags to describe what you're looking for — subgenres, moods, settings, "
        "tropes (e.g. 'epic-fantasy', 'cozy-mystery', 'unreliable-narrator', 'dark'). "
        "The 5 most representative books for those tags are averaged into a query embedding, "
        "then similar books are found in the shelf embedding space."
    )
    shelf_tags    = sorted(t for t in fs['shelves_ordered'] if t)
    selected_tags = st.multiselect("Shelf tags", shelf_tags, key='explore_shelf', max_selections=10)
    if st.button("Explore", key='btn_shelf'):
        if not selected_tags:
            st.warning("Select at least one shelf tag.")
            return

        anchor_tag_book_pairs = []
        seen_titles = set()
        for tag in selected_tags:
            if tag not in fs['shelf_to_i']:
                continue
            tag_idx     = fs['shelf_to_i'][tag]
            sorted_bids = sorted(
                fs['top_books'],
                key=lambda bid: float(fs['book_shelf_matrix'][fs['bookId_to_idx'][bid]][tag_idx]),
                reverse=True,
            )
            count = 0
            for bid in sorted_bids:
                if count >= _ANCHORS_PER_TAG:
                    break
                title = fs['bookId_to_title'][bid]
                if title not in seen_titles:
                    anchor_tag_book_pairs.append((tag, bid, title))
                    seen_titles.add(title)
                    count += 1

        if not anchor_tag_book_pairs:
            st.warning("None of the selected tags found in shelf vocabulary.")
            return

        anchor_bids = [bid for _, bid, _ in anchor_tag_book_pairs]
        anchor_set  = set(anchor_bids)
        query_emb   = torch.stack([
            be[bid]['BOOK_SHELF_EMBEDDING'].view(-1) for bid in anchor_bids
        ]).mean(dim=0)

        sims = {
            bid: F.cosine_similarity(
                query_emb.unsqueeze(0),
                be[bid]['BOOK_SHELF_EMBEDDING'].view(-1).unsqueeze(0),
            ).item()
            for bid in fs['top_books']
        }
        rows = [
            {
                'Cover':      _cover_url(bid, fs),
                'Title':      fs['bookId_to_title'][bid] + ('  ◀ ANCHOR' if bid in anchor_set else ''),
                'Author':     fs['bookId_to_author'].get(bid, ''),
                'Year':       fs['bookId_to_year'].get(bid, '') if fs['bookId_to_year'].get(bid, '') != '-1' else '',
                'Top Shelves': _top_shelves(bid, fs),
                'Score':      f'{s:.4f}',
            }
            for bid, s in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        st.caption(
            "Shelf anchors — "
            + " · ".join(f"{tag}: {title}" for tag, _, title in anchor_tag_book_pairs)
        )
        _render_results(pd.DataFrame(rows))


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    col, _ = st.columns([1, 1])
    with col:
        st.header("What is this?")
        st.markdown(
            "A PyTorch two-tower neural network trained on the "
            "[UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) "
            "(~11k books, ~4.7M training examples)."
        )
        st.markdown(
            "Trained with in-batch negatives softmax loss, following the YouTube DNN retrieval "
            "approach (Covington et al., 2016)."
        )
        st.markdown(
            "At inference, a dot product of the user and item embeddings retrieves the most relevant books."
        )

        st.subheader("The core design choice: no user ID")
        st.markdown("Most recommender systems embed a unique ID for every user in the training set.")
        st.markdown("This works, but has a fundamental limitation: **inference is only possible for users the model has already seen.**")
        st.markdown("If a new user signs up, you have no embedding for them. Your options are:")
        st.markdown("""
- Retrain the entire model
- Partially fine-tune the new user in with a few gradient steps
- Find an existing user who seems similar and use their embedding as a proxy
""")
        st.markdown("This model takes a different approach. **There is no user ID embedding.**")
        st.markdown("Instead, every user is represented as a function of their taste signals — read history, genre affinity, and timestamp.")
        st.markdown("The model learns to embed *features of the user*, not the user themselves.")
        st.markdown("This means the model can generate recommendations for **any user** as long as you can provide even a small amount of signal: a few books they liked, some genres they prefer.")
        st.markdown("No retraining required. No cold-start problem at the user level. The same trained model works for users who never existed when the model was trained.")

    col, _ = st.columns([1, 1])
    with col:
        st.header("User Tower")
        st.markdown(
            "Each component encodes a different aspect of taste into a fixed-size vector. "
            "All three are concatenated into a single 100-dim user embedding."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| Rating-Weighted Avg Pool | Read history — book IDs weighted by ratings | Collaborative taste — liked books pull the user toward similar items in embedding space |
| user_genre_tower | Avg rating per genre + read fraction per genre | Genre affinity — how strongly you lean toward each of the broad genre categories |
| timestamp_embedding_tower | Month bin of most recent read activity | Temporal context — captures when in time the user's taste signal was formed |
""", unsafe_allow_html=True)

        st.header("Item Tower")
        st.markdown(
            "Each book is encoded from five independent signals into a single 100-dim item embedding."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| item_embedding_tower | Book ID (shared lookup with user history pool) | Collaborative identity — a learned fingerprint for each book based on who reads it together |
| item_genre_tower | Genre vote-weighted vector | Broad genre positioning |
| item_shelf_tower | TF-IDF shelf tag scores (3,032 tags) | Crowd-sourced descriptors — granular signals like "cozy-mystery", "epic-fantasy", "dark" |
| author_tower | Primary author index | Author identity — clusters books by the same author and stylistically similar authors |
| year_embedding_tower | Original publication year | Era — captures stylistic and cultural shifts across decades |
""", unsafe_allow_html=True)

        st.header("Shared Embeddings")
        st.markdown("""
**item_embedding_lookup** — The same embedding table is used for the target book's ID
*and* for each book in the user's read history pool.

This forces the user's history representation and the item's identity into the same
space: a book you liked pulls your user embedding directly toward that book's embedding.
""")

        st.header("Training")
        st.markdown("""
- **Dataset:** UCSD Goodreads — ~228M interactions across ~2.4M users and ~2.4M books
- **Corpus filtering:** Books with fewer than 10,000 ratings are excluded (~11k books retained). Users with fewer than 20 or more than 500 corpus ratings are excluded. This ensures every book and user in the training set has enough signal to learn meaningful embeddings. It also keeps the serving artifacts small enough to deploy on Streamlit Cloud without needing paid infrastructure.
- **Loss:** Cross-entropy over in-batch negatives (softmax) — each step produces a B×B score matrix; diagonal entries are the correct targets
- **Optimizer:** Adam, lr=0.001, weight_decay=1e-5, CosineAnnealingLR
- **Batch size:** 512 (511 in-batch negatives per example)
- **Steps:** 150,000
- **Training examples:** Rollback construction — for each read event, context = all prior reads. Up to 10 examples per user sampled randomly (~4.7M train / 526k val)
""")

        st.header("Offline Evaluation Results")
        st.markdown(
            "Leave-label-out protocol on 5,000 held-out val users (~11k book corpus). "
            "Random baseline Hit Rate@10 ≈ 0.87% (avg ~10 label books per user)."
        )
        st.markdown("""
| Metric | MSE | BPR | Softmax | **Softmax + Projection** |
|---|---|---|---|---|
| Hit Rate@10 | 4.7% | 3.5% | 10.7% | **13.0%** |
| Hit Rate@50 | 17.6% | 14.4% | 28.9% | **33.0%** |
| Recall@10 | 0.0069 | 0.0041 | 0.0164 | **0.0241** |
| NDCG@10 | 0.0073 | 0.0042 | 0.0189 | **0.0255** |
| MRR | 0.024 | 0.016 | 0.053 | **0.064** |
""")
        st.markdown(
            "Switching from MSE to softmax improved Hit Rate@10 by **127%**. "
            "Adding projection MLPs to both towers improved it a further **21%** (10.7% → 13.0%)."
        )

        st.header("Limitations")
        st.markdown("""
- ~11k-book corpus — books with fewer than 10,000 ratings are not included
- Romance and literary women's fiction can bleed into each other due to overlapping genre/shelf signals
- The timestamp tower is a weak signal in the app — all inference users receive the most recent timestamp bin
""")


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Book Recommender", layout="wide")
st.markdown("""
    <style>
    div[data-testid="stTabs"] > div:first-child {
        overflow-x: auto;
        white-space: nowrap;
        flex-wrap: nowrap;
    }
    .main .block-container {
        overflow-x: hidden;
        max-width: 100%;
    }
    table {
        display: block;
        overflow-x: auto;
        max-width: 100%;
    }
    div[data-testid="stDataFrame"] {
        overflow-x: auto;
        max-width: 100%;
    }
    div[data-testid="stCaptionContainer"] p {
        word-break: break-word;
        white-space: normal;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Book Recommender")
model, fs, be, all_ids, all_embs, all_norm, ts_inference = load_artifacts()

st.markdown(
    "<small>Two-Tower neural network · Built with "
    "<a href='https://mengtingwan.github.io/data/goodreads.html' target='_blank'>Goodreads</a>"
    " and <a href='https://pytorch.org' target='_blank'>PyTorch</a><br>"
    "Code: <a href='https://github.com/nickgreenquist/Book-Recommender-System-PyTorch-TwoTower-Model' target='_blank'>GitHub</a></small>",
    unsafe_allow_html=True,
)

recommend_tab, examples_tab, similar_tab, genres_tab, shelves_tab, about_tab = st.tabs(
    ["Recommend", "Examples", "Similar", "Genres", "Shelves", "About"]
)

with recommend_tab:
    tab_recommend(model, fs, all_ids, all_embs, ts_inference)

with examples_tab:
    tab_recommend_examples(model, fs, all_ids, all_embs, ts_inference)

with similar_tab:
    tab_similar(be, fs, all_ids, all_norm)

with genres_tab:
    tab_explore_genres(model, be, fs)

with shelves_tab:
    tab_explore_shelves(model, be, fs)

with about_tab:
    tab_about()
