"""
Inference, canary user evaluation, and embedding probes.

Usage:
    python main.py canary [checkpoint_path]
    python main.py probe  [checkpoint_path]
"""
import glob
import os
from itertools import zip_longest

import numpy as np
import torch
import torch.nn.functional as F
from src.dataset import FeatureStore
from src.model import BookRecommender
from src.train import build_model, get_config, print_model_summary


# ── Canary user definitions ───────────────────────────────────────────────────

# Genre names must match base_vocab.parquet exactly.
# Available genres: children, comics, graphic, fantasy, paranormal,
#   fiction, history, historical fiction, biography,
#   mystery, thriller, crime, non-fiction, poetry, romance, young-adult

USER_TYPE_TO_FAVORITE_GENRES = {
    'Mystery Lover':   ['mystery, thriller, crime'],
    'Fantasy Lover':   ['fantasy, paranormal'],
    'Romance Lover':   ['romance'],
    'YA Lover':        ['young-adult'],
    'History Lover':   ['history, historical fiction, biography'],
    'Literary Lover':  ['fiction'],
    'Horror Lover':    [],  # no horror genre in vocab — relies on shelf tags + books
    'Sci-Fi Lover':    [],  # no sci-fi genre in vocab — relies on shelf tags + books
    'NonFiction Lover': ['non-fiction'],
}

USER_TYPE_TO_WORST_GENRES = {
    'Mystery Lover':   ['romance', 'young-adult'],
    'Fantasy Lover':   ['romance', 'non-fiction'],
    'Romance Lover':   ['mystery, thriller, crime', 'non-fiction'],
    'YA Lover':        ['non-fiction'],
    'History Lover':   ['fantasy, paranormal', 'romance'],
    'Literary Lover':  ['romance', 'young-adult'],
    'Horror Lover':    ['romance', 'young-adult'],
    'Sci-Fi Lover':    ['romance', 'young-adult'],
    'NonFiction Lover': ['fantasy, paranormal', 'romance'],
}

USER_TYPE_TO_FAVORITE_BOOKS = {
    'Mystery Lover': [
        'Gone Girl',
        'The Girl with the Dragon Tattoo (Millennium, #1)',
        'Big Little Lies',
    ],
    'Fantasy Lover': [
        'The Name of the Wind (The Kingkiller Chronicle, #1)',
        'The Way of Kings (The Stormlight Archive, #1)',
        'A Game of Thrones (A Song of Ice and Fire, #1)',
    ],
    'Romance Lover': [
        'Pride and Prejudice',
        'Me Before You (Me Before You, #1)',
        'The Notebook (The Notebook, #1)',
    ],
    'YA Lover': [
        "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",
        'The Hunger Games (The Hunger Games, #1)',
        'Divergent (Divergent, #1)',
    ],
    'History Lover': [
        'The Pillars of the Earth (Kingsbridge, #1)',
        'Wolf Hall (Thomas Cromwell, #1)',
    ],
    'Literary Lover': [
        'To Kill a Mockingbird',
        '1984',
        'The Great Gatsby',
    ],
    'Horror Lover': [
        'It',
        'The Shining',
        'Pet Sematary',
    ],
    'Sci-Fi Lover': [
        "Ender's Game (Ender's Saga, #1)",
        "The Hitchhiker's Guide to the Galaxy (Hitchhiker's Guide, #1)",
        'Fahrenheit 451',
    ],
    'NonFiction Lover': [
        'Sapiens: A Brief History of Humankind',
        'Thinking, Fast and Slow',
        'The Power of Habit: Why We Do What We Do in Life and Business',
    ],
}

USER_TYPE_TO_SHELF_TAGS = {
    'Mystery Lover':   ['mystery', 'thriller', 'suspense', 'crime'],
    'Fantasy Lover':   ['fantasy', 'magic', 'epic-fantasy', 'world-building'],
    'Romance Lover':   ['romance', 'love-story', 'chick-lit'],
    'YA Lover':        ['young-adult', 'ya', 'coming-of-age'],
    'History Lover':   ['historical-fiction', 'history', 'historical'],
    'Literary Lover':  ['literary-fiction', 'classics', 'literary'],
    'Horror Lover':    ['horror', 'scary', 'dark', 'creepy'],
    'Sci-Fi Lover':    ['science-fiction', 'sci-fi', 'dystopia', 'space'],
    'NonFiction Lover': ['non-fiction', 'nonfiction', 'science', 'psychology'],
}

VALUE_FAVORITE_GENRE_RATING = 4.0
VALUE_DISLIKED_GENRE_RATING = -2.0
VALUE_FAVORITE_BOOK_RATING  = 2.0
VALUE_ANCHOR_BOOK_RATING    = 1.0
ANCHORS_PER_TAG             = 5


# ── Book embedding cache ──────────────────────────────────────────────────────

def build_book_embeddings(model: BookRecommender, fs: FeatureStore) -> dict:
    """
    Pre-compute all book combined embeddings for fast recommendation scoring.
    Returns book_id → {'BOOK_EMBEDDING_COMBINED': Tensor, ...}
    Batched for efficiency.
    """
    model.eval()
    n_books    = len(fs.top_books)
    batch_size = 512

    all_book_idxs    = torch.tensor(list(range(n_books)), dtype=torch.long)
    all_genre_ctx    = torch.tensor(
        [fs.bookId_to_genre_context[bid] for bid in fs.top_books], dtype=torch.float32
    )
    all_year_idxs    = torch.tensor(
        [fs.year_to_i.get(fs.bookId_to_year[bid], 0) for bid in fs.top_books], dtype=torch.long
    )

    genre_embs  = []
    shelf_embs  = []
    book_embs   = []
    author_embs = []
    year_embs   = []

    with torch.no_grad():
        for start in range(0, n_books, batch_size):
            end      = min(start + batch_size, n_books)
            bidxs    = all_book_idxs[start:end]
            gctx     = all_genre_ctx[start:end]
            yidxs    = all_year_idxs[start:end]
            aidxs    = model.book_author_idx[bidxs]

            genre_embs.append(model.item_genre_tower(gctx))
            shelf_embs.append(model.item_shelf_tower(model.book_shelf_matrix[bidxs]))
            book_embs.append(model.item_embedding_tower(model.item_embedding_lookup(bidxs)))
            author_embs.append(model.author_tower(model.author_embedding_lookup(aidxs)))
            year_embs.append(model.year_embedding_tower(model.year_embedding_lookup(yidxs)))

    genre_all  = torch.cat(genre_embs,  dim=0)
    shelf_all  = torch.cat(shelf_embs,  dim=0)
    book_all   = torch.cat(book_embs,   dim=0)
    author_all = torch.cat(author_embs, dim=0)
    year_all   = torch.cat(year_embs,   dim=0)
    combined   = torch.cat([genre_all, shelf_all, book_all, author_all, year_all], dim=1)

    bookId_to_embedding = {}
    for i, bid in enumerate(fs.top_books):
        bookId_to_embedding[bid] = {
            'BOOK_GENRE_EMBEDDING':    genre_all[i].unsqueeze(0),
            'BOOK_SHELF_EMBEDDING':    shelf_all[i].unsqueeze(0),
            'BOOK_ID_EMBEDDING':       book_all[i].unsqueeze(0),
            'BOOK_AUTHOR_EMBEDDING':   author_all[i].unsqueeze(0),
            'BOOK_YEAR_EMBEDDING':     year_all[i].unsqueeze(0),
            'BOOK_EMBEDDING_COMBINED': combined[i].unsqueeze(0),
        }

    return bookId_to_embedding


# ── Canary user inference ─────────────────────────────────────────────────────

def _get_anchor_titles(fs: FeatureStore, shelf_tags: list, exclude: set) -> list:
    """Return up to ANCHORS_PER_TAG top books per shelf tag, skipping titles in exclude."""
    anchor_titles = []
    seen = set(exclude)
    for tag in shelf_tags:
        if tag not in fs.shelf_to_i:
            continue
        tag_idx = fs.shelf_to_i[tag]
        sorted_bids = sorted(
            fs.top_books,
            key=lambda bid: float(fs.bookId_to_shelf_context[bid][tag_idx]),
            reverse=True,
        )
        count = 0
        for bid in sorted_bids:
            if count >= ANCHORS_PER_TAG:
                break
            title = fs.bookId_to_title[bid]
            if title not in seen:
                anchor_titles.append(title)
                seen.add(title)
                count += 1
    return anchor_titles


def _build_user_embedding(model: BookRecommender, fs: FeatureStore, user_type: str,
                          ts_inference: torch.Tensor) -> torch.Tensor:
    """Build the combined user embedding for a canary user type. Mirrors forward() logic."""
    fav_genres   = USER_TYPE_TO_FAVORITE_GENRES[user_type]
    worst_genres = USER_TYPE_TO_WORST_GENRES[user_type]
    fav_books    = USER_TYPE_TO_FAVORITE_BOOKS[user_type]
    shelf_tags   = USER_TYPE_TO_SHELF_TAGS.get(user_type, [])

    anchor_titles = _get_anchor_titles(fs, shelf_tags, exclude=set(fav_books))

    liked_with_weights = (
        [(t, VALUE_FAVORITE_BOOK_RATING) for t in fav_books] +
        [(t, VALUE_ANCHOR_BOOK_RATING)   for t in anchor_titles]
    )

    # Resolve titles → book indices + ratings (skip titles not in corpus)
    history = []  # list of (book_idx, rating)
    for title, w in liked_with_weights:
        bid = fs.title_to_bookId.get(title)
        if bid is None or bid not in fs.bookId_to_idx:
            continue
        history.append((fs.bookId_to_idx[bid], w))

    # ── Genre context ─────────────────────────────────────────────────────────
    n_genres = len(fs.genres_ordered)
    ctx = [0.0] * (2 * n_genres)
    genre_rating_sum  = {}
    genre_book_count  = {}
    total_books = 0
    for title, w in liked_with_weights:
        bid = fs.title_to_bookId.get(title)
        if bid is None:
            continue
        total_books += 1
        for g in fs.bookId_to_genres.get(bid, []):
            genre_rating_sum[g]  = genre_rating_sum.get(g, 0.0)  + w
            genre_book_count[g]  = genre_book_count.get(g, 0)    + 1
    for g, rsum in genre_rating_sum.items():
        avg_r = rsum / genre_book_count[g]
        frac  = genre_book_count[g] / max(total_books, 1)
        if g in fs.genre_to_i:
            ctx[fs.genre_to_i[g]]            = avg_r
            ctx[n_genres + fs.genre_to_i[g]] = frac
    # Explicit genre overrides
    for g in fav_genres:
        if g in fs.genre_to_i:
            ctx[fs.genre_to_i[g]]            = VALUE_FAVORITE_GENRE_RATING
            ctx[n_genres + fs.genre_to_i[g]] = 1.0 / max(len(fav_genres), 1)
    for g in worst_genres:
        if g in fs.genre_to_i:
            ctx[fs.genre_to_i[g]] = VALUE_DISLIKED_GENRE_RATING

    # ── Build user towers (mirrors model.forward user side) ───────────────────
    if history:
        hist_idx_t = torch.tensor([h[0] for h in history], dtype=torch.long).unsqueeze(0)  # (1, hist)
        hist_wts_t = torch.tensor([[h[1] for h in history]], dtype=torch.float)             # (1, hist)
        pad_mask   = torch.ones_like(hist_wts_t).unsqueeze(-1)                              # (1, hist, 1)
        rat_wts    = hist_wts_t.unsqueeze(-1) * pad_mask                                    # (1, hist, 1)
        wt_sum     = rat_wts.abs().sum(dim=1).clamp(min=1e-6)                               # (1, 1)

        # Book embedding pool
        hist_embs   = model.item_embedding_lookup(hist_idx_t)                               # (1, hist, D)
        history_emb = (hist_embs * rat_wts).sum(dim=1) / wt_sum                             # (1, D)

        # Shelf pool (transform-then-pool, shared tower)
        hist_shelf_vecs = model.book_shelf_matrix[hist_idx_t]                               # (1, hist, n_shelves)
        shelf_embs_h    = model.item_shelf_tower(hist_shelf_vecs)                           # (1, hist, shelf_dim)
        user_shelf_emb  = (shelf_embs_h * rat_wts).sum(dim=1) / wt_sum                     # (1, shelf_dim)

        # Author pool (transform-then-pool, shared tower)
        hist_author_idx = model.book_author_idx[hist_idx_t]                                 # (1, hist)
        auth_embs_raw   = model.author_embedding_lookup(hist_author_idx)                    # (1, hist, author_dim)
        auth_embs       = model.author_tower(auth_embs_raw)                                 # (1, hist, author_dim)
        user_author_emb = (auth_embs * rat_wts).sum(dim=1) / wt_sum                        # (1, author_dim)
    else:
        d_hist   = model.item_embedding_lookup.embedding_dim
        d_shelf  = model.item_shelf_tower[0].out_features
        d_author = model.author_embedding_lookup.embedding_dim
        history_emb    = torch.zeros(1, d_hist)
        user_shelf_emb = torch.zeros(1, d_shelf)
        user_author_emb = torch.zeros(1, d_author)

    X_inf     = torch.tensor([ctx])
    genre_emb = model.user_genre_tower(X_inf)
    ts_emb    = model.timestamp_embedding_tower(model.timestamp_embedding_lookup(ts_inference))

    return torch.cat([history_emb, user_author_emb, user_shelf_emb, genre_emb, ts_emb], dim=1)


def run_canary_eval(model: BookRecommender, fs: FeatureStore,
                    book_embeddings: dict, all_ids: list, all_embs: torch.Tensor,
                    top_n: int = 10) -> None:
    """Run all canary users and print recommendation tables."""
    model.eval()

    ts_max_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False
    )

    with torch.no_grad():
        for user_type in USER_TYPE_TO_FAVORITE_GENRES:
            user_emb   = _build_user_embedding(model, fs, user_type, ts_max_bin)
            fav_books  = USER_TYPE_TO_FAVORITE_BOOKS[user_type]
            shelf_tags = USER_TYPE_TO_SHELF_TAGS.get(user_type, [])
            anchor_titles = _get_anchor_titles(fs, shelf_tags, exclude=set(fav_books))
            exclude_set   = set(fav_books) | set(anchor_titles)

            raw_scores = (all_embs @ user_emb.T).squeeze(-1)
            scores     = {all_ids[i]: raw_scores[i].item() for i in range(len(all_ids))}

            recs       = []
            seen_titles = set(exclude_set)
            for bid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if len(recs) >= top_n:
                    break
                title = fs.bookId_to_title[bid]
                if title not in seen_titles:
                    seen_titles.add(title)
                    recs.append(title)

            fav_genres      = ', '.join(USER_TYPE_TO_FAVORITE_GENRES[user_type]) or '—'
            disliked_genres = ', '.join(USER_TYPE_TO_WORST_GENRES[user_type])    or '—'

            col_w      = min(50, max((len(t) for t in fav_books), default=20))
            rec_w      = min(50, max((len(r) for r in recs), default=20))
            title_line = f"{user_type}  |  Likes: {fav_genres}  |  Dislikes: {disliked_genres}"
            if shelf_tags:
                title_line += f"  |  Shelves: {', '.join(shelf_tags[:4])}"
            bar_w      = max(col_w + rec_w + 4, len(title_line))

            print(f"\n{'═' * bar_w}")
            print(title_line)
            print(f"{'═' * bar_w}")
            header = f"{'Favorite Books':<{col_w}}  Recommendations"
            print(header)
            print('─' * bar_w)
            for a, b in zip_longest(fav_books, recs, fillvalue=''):
                print(f"{a:<{col_w}}  {b}")


# ── Embedding probes ──────────────────────────────────────────────────────────

def probe_genre(model: BookRecommender, genre: str, book_embeddings: dict,
                fs: FeatureStore, top_n: int = 10) -> None:
    """
    Find the most representative books for a genre in item genre embedding space.
    Passes a one-hot genre vector through item_genre_tower, compares via cosine similarity.
    """
    if genre not in fs.genre_to_i:
        print(f"Genre '{genre}' not in vocabulary. Available: {fs.genres_ordered}")
        return

    ctx = [0.0] * len(fs.genres_ordered)
    ctx[fs.genre_to_i[genre]] = 1.0

    with torch.no_grad():
        query_emb = model.item_genre_tower(torch.tensor([ctx])).view(-1)

    sims = {
        bid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            book_embeddings[bid]['BOOK_GENRE_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for bid in fs.top_books
    }

    print(f"\nTop-{top_n} books for genre '{genre}':")
    seen = set()
    for bid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen) >= top_n:
            break
        title = fs.bookId_to_title[bid]
        if title not in seen:
            seen.add(title)
            print(f"  {sim:.4f}  {title}")


def probe_shelf(model: BookRecommender, shelf_tags: list, book_embeddings: dict,
                fs: FeatureStore, top_n: int = 10, k_anchors: int = 3) -> None:
    """
    Find books most similar to a shelf tag query in the item shelf embedding space.
    Finds the top-k_anchors books by raw shelf score, averages their BOOK_SHELF_EMBEDDING
    as the query, then compares via cosine similarity against all books.
    """
    raw_scores = {}
    valid_tags = [t for t in shelf_tags if t in fs.shelf_to_i]
    if not valid_tags:
        print(f"No shelf tags from {shelf_tags} found in vocabulary.")
        return

    for bid in fs.top_books:
        shelf_ctx  = fs.bookId_to_shelf_context[bid]
        raw_scores[bid] = sum(shelf_ctx[fs.shelf_to_i[t]] for t in valid_tags)

    anchors   = sorted(raw_scores, key=raw_scores.get, reverse=True)[:k_anchors]
    query_emb = torch.stack([
        book_embeddings[bid]['BOOK_SHELF_EMBEDDING'].view(-1) for bid in anchors
    ]).mean(dim=0)

    anchor_titles = [fs.bookId_to_title[bid] for bid in anchors]
    print(f"\nShelf anchors for {shelf_tags}: {anchor_titles}")

    sims = {
        bid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            book_embeddings[bid]['BOOK_SHELF_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for bid in fs.top_books
    }

    anchor_set  = set(anchors)
    seen_titles = set()
    print(f"Top-{top_n} books:")
    for bid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen_titles) >= top_n:
            break
        title = fs.bookId_to_title[bid]
        if title not in seen_titles:
            seen_titles.add(title)
            marker = " [seed]" if bid in anchor_set else ""
            print(f"  {sim:.4f}  {title}{marker}")


def probe_similar(book_embeddings: dict, fs: FeatureStore,
                  all_ids: list, all_norm: torch.Tensor,
                  titles: list, top_n: int = 5) -> None:
    """
    For each query title, find the top-N most similar books by cosine similarity
    on BOOK_EMBEDDING_COMBINED. Uses pre-normalized all_norm matrix from _setup.
    """
    TRUNC = 30

    def trunc(s: str) -> str:
        return s if len(s) <= TRUNC else s[:TRUNC - 1] + '…'

    rows = []
    for title in titles:
        bid = fs.title_to_bookId.get(title)
        if bid is None:
            rows.append((title, []))
            continue
        query   = F.normalize(book_embeddings[bid]['BOOK_EMBEDDING_COMBINED'], dim=1)
        sims    = (all_norm @ query.T).squeeze(-1)
        top_idx = sims.argsort(descending=True)
        results = []
        seen_titles = {title}  # exclude seed title and all its duplicate editions
        for idx in top_idx:
            candidate       = all_ids[idx.item()]
            candidate_title = fs.bookId_to_title[candidate]
            if candidate_title in seen_titles:
                continue
            seen_titles.add(candidate_title)
            results.append(candidate_title)
            if len(results) >= top_n:
                break
        rows.append((title, results))

    seed_w = max(len(trunc(t)) for t, _ in rows)
    col_w  = TRUNC
    header = f"{'Seed':<{seed_w}}" + "".join(f"  {'#'+str(i+1):<{col_w}}" for i in range(top_n))
    print(f"\n── Most similar books ──")
    print(header)
    print('─' * len(header))
    for title, results in rows:
        if not results:
            print(f"{trunc(title):<{seed_w}}  (not in corpus)")
            continue
        row = f"{trunc(title):<{seed_w}}"
        for t in results:
            row += f"  {trunc(t):<{col_w}}"
        print(row)


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint_path: str, checkpoint_dir: str):
    if checkpoint_path is not None:
        return checkpoint_path
    pattern    = os.path.join(checkpoint_dir, 'best_checkpoint_*.pth')
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        print("No checkpoint found in saved_models/. Train a model first.")
        return None
    return candidates[0]


def _load_model_and_embeddings(checkpoint_path: str, fs):
    """Build model, load weights, pre-compute book embeddings."""
    config = get_config()
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()
    print_model_summary(model)

    print("\nBuilding book embeddings ...")
    book_embeddings = build_book_embeddings(model, fs)

    print("Precomputing embedding matrix ...")
    all_ids  = list(book_embeddings.keys())
    all_embs = torch.cat([book_embeddings[bid]['BOOK_EMBEDDING_COMBINED'] for bid in all_ids], dim=0)
    all_norm = F.normalize(all_embs, dim=1)
    return model, book_embeddings, all_ids, all_embs, all_norm


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_canary(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    from src.dataset import load_features
    config = get_config()
    cp = _resolve_checkpoint(checkpoint_path, config['checkpoint_dir'])
    if cp is None:
        return
    print("Loading features ...")
    fs = load_features(data_dir, version)
    model, book_embeddings, all_ids, all_embs, all_norm = _load_model_and_embeddings(cp, fs)
    print("\n── Canary user evaluation ──")
    run_canary_eval(model, fs, book_embeddings, all_ids, all_embs)


PROBE_SIMILAR_TITLES = [
    'A Game of Thrones (A Song of Ice and Fire, #1)',
    "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",
    'Gone Girl',
    'The Hunger Games (The Hunger Games, #1)',
    'To Kill a Mockingbird',
    'Sapiens: A Brief History of Humankind',
    'It',
    "Ender's Game (Ender's Saga, #1)",
]


def run_probes(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    from src.dataset import load_book_features
    config = get_config()
    cp = _resolve_checkpoint(checkpoint_path, config['checkpoint_dir'])
    if cp is None:
        return
    print("Loading book features ...")
    fs = load_book_features(data_dir, version)
    model, book_embeddings, all_ids, all_embs, all_norm = _load_model_and_embeddings(cp, fs)
    print("\n── Embedding probes ──")
    probe_genre(model, 'mystery, thriller, crime', book_embeddings, fs)
    probe_genre(model, 'fantasy, paranormal',      book_embeddings, fs)
    probe_genre(model, 'romance',                  book_embeddings, fs)
    probe_shelf(model, ['horror', 'scary', 'dark'],          book_embeddings, fs)
    probe_shelf(model, ['science-fiction', 'sci-fi', 'space'], book_embeddings, fs)
    probe_shelf(model, ['epic-fantasy', 'magic', 'world-building'], book_embeddings, fs)
    probe_similar(book_embeddings, fs, all_ids, all_norm, PROBE_SIMILAR_TITLES)
