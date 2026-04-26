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
from src.train import (build_model, get_config, get_softmax_config,
                        get_softmax_config_legacy, print_model_summary)


# ── Canary user definitions ───────────────────────────────────────────────────

# Genre names must match base_vocab.parquet exactly.
# Available genres: children, comics, graphic, fantasy, paranormal,
#   fiction, history, historical fiction, biography,
#   mystery, thriller, crime, non-fiction, poetry, romance, young-adult

USER_TYPE_TO_FAVORITE_GENRES = {
    "Nick's Recommendations": [''],  # already have extremly rich read history
    'Mystery Lover':   ['mystery, crime'],
    'Fantasy Lover':   ['fantasy'],
    'Romance Lover':   ['romance'],
    'YA Lover':        ['young-adult'],
    'History Lover':   ['history, biography'],
    'Classic Lover':     [''],
    'Horror Lover':       [],  # no horror genre in vocab — relies on shelf tags + books
    'Sci-Fi Lover':       [],  # no sci-fi genre in vocab — relies on shelf tags + books
    'NonFiction Lover':   ['non-fiction'],
    'Economics Lover':  [''],
    'Manga Lover':      [],  # no manga genre in vocab — relies on shelf tags + books
    'Poetry Lover':     ['poetry'],
    "Children's Book Lover": ['children'],
}

USER_TYPE_TO_WORST_GENRES = {
    "Nick's Recommendations": [],
    'Mystery Lover':     [],
    'Fantasy Lover':     [],
    'Romance Lover':     [],
    'YA Lover':          [],
    'History Lover':     [],
    'Classic Lover':    [],
    'Horror Lover':      [],
    'Sci-Fi Lover':      [],
    'NonFiction Lover':  [],
    'Economics Lover': [],
    'Manga Lover':     [],
    'Poetry Lover':    [],
    "Children's Book Lover": [],
}

USER_TYPE_TO_FAVORITE_BOOKS = {
    "Nick's Recommendations": [
        'Tales of the South Pacific',
        'Martin Eden',
        'The Silmarillion',
        'Where the Red Fern Grows',
        'Animal Farm',
        'White Fang',
        'The Call of the Wild',
        'Meditations',
        'The Return of the King (The Lord of the Rings, #3)',
        'The Two Towers (The Lord of the Rings, #2)',
        'The Big Short: Inside the Doomsday Machine',
        'The Fellowship of the Ring (The Lord of the Rings, #1)',
        'Flowers for Algernon',
        'Killers of the Flower Moon: The Osage Murders and the Birth of the FBI',
        'Einstein: His Life and Universe',
        "Man's Search for Meaning",
        'Empire of the Summer Moon: Quanah Parker and the Rise and Fall of the Comanches, the Most Powerful Indian Tribe in American History',
        'Genghis Khan and the Making of the Modern World',
        'The Martian',
        'When Things Fall Apart: Heart Advice for Difficult Times',
        'Stumbling on Happiness',
        'Travels with Charley: In Search of America',
        'Stoner',
        'Centennial',
        'All Quiet on the Western Front',
        'The Things They Carried',
        'Elon Musk: Tesla, SpaceX, and the Quest for a Fantastic Future',
        'Hawaii',
        'Zero to One: Notes on Startups, or How to Build the Future',
        'Bridge to Terabithia',
        'Angle of Repose',
        'Nothing to Envy: Ordinary Lives in North Korea',
        'Noble House (Asian Saga, #5)',
        'Shantaram',
        'First They Killed My Father: A Daughter of Cambodia Remembers',
        'King Rat (Asian Saga, #4)',
        'Tai-Pan (Asian Saga, #2)',
        'Shōgun (Asian Saga, #1)',
        'The Beach',
        'The Count of Monte Cristo',
        'One Up On Wall Street: How to Use What You Already Know to Make Money in the Market',
        'The Everything Store: Jeff Bezos and the Age of Amazon',
        'Thinking, Fast and Slow',
        'The Selfish Gene',
        "The River of Doubt: Theodore Roosevelt's Darkest Journey",
        'Steve Jobs',
        'There are No Children Here: The Story of Two Boys Growing Up in the Other America',
        'The Subtle Knife (His Dark Materials, #2)',
        'Talent is Overrated: What Really Separates World-Class Performers from Everybody Else',
        'East of Eden',
        'The Perks of Being a Wallflower',
        'A Short History of Nearly Everything',
        'The Glass Castle',
        'From Here to Eternity',
        'The Book Thief',
        'The Lord of the Rings (The Lord of the Rings, #1-3)',
        'Fight Club',
        'The BFG',
        'A Storm of Swords (A Song of Ice and Fire, #3)',
        'A Game of Thrones (A Song of Ice and Fire, #1)',
        'In Cold Blood',
        'Siddhartha',
        'A Farewell to Arms',
        'Of Mice and Men',
        'The Kite Runner',
        'The Catcher in the Rye',
        'A Clockwork Orange',
        'The Hunger Games (The Hunger Games, #1)',
        'The Lightning Thief (Percy Jackson and the Olympians, #1)',
        "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",
    ],
    'Mystery Lover': [
        'Gone Girl',
        'The Girl with the Dragon Tattoo (Millennium, #1)',
        'Big Little Lies',
        'The Silence of the Lambs  (Hannibal Lecter, #2)',
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
    'Classic Lover': [
        'Anna Karenina',
        'Crime and Punishment',
        'Great Expectations',
        'Moby-Dick or, The Whale',
    ],
    'Horror Lover': [
        'The Shining',
        'Pet Sematary',
        'The Exorcist',
        'House of Leaves',
        'The Haunting of Hill House',
    ],
    'Sci-Fi Lover': [
        'Hyperion (Hyperion Cantos, #1)',
        "The Hitchhiker's Guide to the Galaxy (Hitchhiker's Guide, #1)",
    ],
    'NonFiction Lover': [
        'Sapiens: A Brief History of Humankind',
        'Thinking, Fast and Slow',
    ],
    'Economics Lover': [
        'The Intelligent Investor',
        'The Wealth of Nations',
        'Capital in the Twenty-First Century',
    ],
    'Manga Lover': [
        'Fullmetal Alchemist, Vol. 1 (Fullmetal Alchemist, #1)',
        'Death Note, Vol. 2: Confluence (Death Note, #2)',
        'Naruto, Vol. 01: The Tests of the Ninja (Naruto, #1)',
        'Bleach, Volume 01',
    ],
    'Poetry Lover': [
        'Leaves of Grass',
        'The Complete Poems of Emily Dickinson',
        'The Essential Rumi',
        'Milk and Honey',
    ],
    "Children's Book Lover": [
        'Where the Wild Things Are',
        'The Giving Tree',
        'Where the Sidewalk Ends',
        'Matilda',
    ],
}

USER_TYPE_TO_LIKED_BOOKS = {
    "Nick's Recommendations": [
        'To Have and Have Not',
        'Tortilla Flat',
        'Crime and Punishment',
        'A Christmas Carol',
        'Show Your Work!: 10 Ways to Share Your Creativity and Get Discovered',
        'The Untethered Soul: The Journey Beyond Yourself',
        'The Wright Brothers',
        'Deep Work: Rules for Focused Success in a Distracted World',
        'Hiroshima',
        'And Then There Were None',
        'Station Eleven',
        'At Home: A Short History of Private Life',
        'Rich Dad, Poor Dad',
        'The Picture of Dorian Gray',
        'Behind the Beautiful Forevers: Life, Death, and Hope in a Mumbai Undercity',
        "Endurance: Shackleton's Incredible Voyage",
        'Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel',
        'The Poisonwood Bible',
        'The 48 Laws of Power',
        'The Information: A History, a Theory, a Flood',
        'On the Road',
        'Astrophysics for People in a Hurry',
        'The Idiot',
        'Casino Royale (James Bond, #1)',
        'In a Sunburned Country',
        'Breakfast of Champions',
        'The English Patient',
        'Norse Mythology',
        'Down and Out in Paris and London',
        'The Sun Also Rises',
        'Delivering Happiness: A Path to Profits, Passion, and Purpose',
        'In the Heart of the Sea: The Tragedy of the Whaleship Essex',
        'The Joy Luck Club',
        'Benjamin Franklin: An American Life',
        'The Curious Case of Benjamin Button',
        'A Gentleman in Moscow',
        'The Grapes of Wrath',
        'The Subtle Art of Not Giving a F*ck: A Counterintuitive Approach to Living a Good Life',
        'Unaccustomed Earth',
        'In the Plex: How Google Thinks, Works, and Shapes Our Lives',
        'Creativity, Inc.: Overcoming the Unseen Forces That Stand in the Way of True Inspiration',
        'Charlie and the Great Glass Elevator (Charlie Bucket, #2)',
        'Sapiens: A Brief History of Humankind',
        'Hillbilly Elegy: A Memoir of a Family and Culture in Crisis',
        'Fahrenheit 451',
        'Watchmen',
        'The One Thing: The Surprisingly Simple Truth Behind Extraordinary Results',
        'The Old Man and the Sea',
        'The Undoing Project: A Friendship That Changed Our Minds',
        'Watership Down (Watership Down, #1)',
        'The Design of Everyday Things',
        'No Country for Old Men',
        'Middlesex',
        "The Girl in the Spider's Web (Millennium, #4)",
        'The Art of War',
        'Anna Karenina',
        'The Children of Húrin',
        'The Amber Spyglass (His Dark Materials, #3)',
        'The Golden Compass (His Dark Materials, #1)',
        'Looking for Alaska',
        'The Road to Character',
        'The Road',
        'Murder on the Orient Express (Hercule Poirot, #10)',
        'The Alchemist',
        'Catching Fire (The Hunger Games, #2)',
        "Hatchet (Brian's Saga, #1)",
        'Charlie and the Chocolate Factory (Charlie Bucket, #1)',
        'A Thousand Splendid Suns',
        'A Long Way Gone: Memoirs of a Boy Soldier',
        'Eleanor & Park',
        'Divergent (Divergent, #1)',
        'Son (The Giver, #4)',
        'Messenger (The Giver, #3)',
        'Gathering Blue (The Giver, #2)',
        'The Giver (The Giver, #1)',
        'Fifty Shades of Grey (Fifty Shades, #1)',
        'The Fault in Our Stars',
        'The Brothers Karamazov',
        'The Adventures of Captain Underpants (Captain Underpants, #1)',
        'A Brief History of Time',
        'The Bad Beginning (A Series of Unfortunate Events, #1)',
        'The Thief Lord',
        'To Kill a Mockingbird',
        'The Great Gatsby',
        'Harry Potter and the Chamber of Secrets (Harry Potter, #2)',
        'Harry Potter and the Deathly Hallows (Harry Potter, #7)',
        'Shiloh (Shiloh, #1)',
        'Holes (Holes, #1)',
        'The Outsiders',
        'Captain Underpants and the Attack of the Talking Toilets (Captain Underpants, #2)',
        'The War of the Worlds',
        "The Innovator's Dilemma: The Revolutionary Book that Will Change the Way You Do Business",
        'Lone Survivor: The Eyewitness Account of Operation Redwing and the Lost Heroes of SEAL Team 10',
        'Killing Lincoln: The Shocking Assassination that Changed America Forever',
        "A Connecticut Yankee in King Arthur's Court",
        'The Prince and the Pauper',
        'Redwall (Redwall, #1)',
        'Twenty Thousand Leagues Under the Sea',
        'A Dance with Dragons (A Song of Ice and Fire, #5)',
        'A Clash of Kings  (A Song of Ice and Fire, #2)',
        'Outliers: The Story of Success',
        'Blink: The Power of Thinking Without Thinking',
        'SuperFreakonomics: Global Cooling, Patriotic Prostitutes And Why Suicide Bombers Should Buy Life Insurance',
        'The Tipping Point: How Little Things Can Make a Big Difference',
        'Great Expectations',
        'Frankenstein',
        'Animal Farm',
        'Hamlet: Screenplay, Introduction And Film Diary',
        'Romeo and Juliet',
        'Inkheart (Inkworld, #1)',
        'Artemis Fowl (Artemis Fowl, #1)',
        'Mossflower (Redwall, #2)',
        'The Son of Neptune (The Heroes of Olympus, #2)',
        'The Lost Hero (The Heroes of Olympus, #1)',
        'The Battle of the Labyrinth (Percy Jackson and the Olympians, #4)',
    ],
    'Mystery Lover':          [],
    'Fantasy Lover':          [],
    'Romance Lover':          [],
    'YA Lover':               [],
    'History Lover':          [],
    'Classic Lover':          [],
    'Horror Lover':           [],
    'Sci-Fi Lover':           [],
    'NonFiction Lover':       [],
    'Economics Lover':        [],
    'Manga Lover':            [],
    'Poetry Lover':           [],
    "Children's Book Lover":  [],
}

USER_TYPE_TO_SHELF_TAGS = {
    "Nick's Recommendations": [''],  # read history is rich enough
    'Mystery Lover':   ['mystery', 'crime'],
    'Fantasy Lover':   ['epic-fantasy', 'world-building'],
    'Romance Lover':   ['romance', 'love-story', 'chick-lit'],
    'YA Lover':        ['young-adult', 'ya', 'coming-of-age'],
    'History Lover':   ['history', 'historical'],
    'Classic Lover':  ['classics'],
    'Horror Lover':    ['horror'],
    'Sci-Fi Lover':    ['science-fiction', 'sci-fi'],
    'NonFiction Lover':  ['non-fiction'],
    'Economics Lover': ['economics'],
    'Manga Lover':     [''],
    'Poetry Lover':    ['poetry'],
    "Children's Book Lover": ['childrens', 'children-s', 'picture-books'],
}

VALUE_FAVORITE_GENRE_RATING = 4.0
VALUE_DISLIKED_GENRE_RATING = -2.0
VALUE_FAVORITE_BOOK_RATING  = 2.0
VALUE_ANCHOR_BOOK_RATING    = 1.0
ANCHORS_PER_TAG             = 5


# ── Book embedding cache ──────────────────────────────────────────────────────

def build_book_embeddings(model: BookRecommender, fs: FeatureStore) -> dict:
    """
    Pre-compute all book embeddings for recommendation scoring and probes.
    Returns book_id → {'BOOK_EMBEDDING_COMBINED': Tensor, ...}
    """
    model.eval()
    n_books    = len(fs.top_books)
    batch_size = 512
    all_book_idxs = torch.tensor(list(range(n_books)), dtype=torch.long)

    genre_embs    = []
    shelf_embs    = []
    book_embs     = []
    author_embs   = []
    year_embs     = []
    combined_embs = []

    with torch.no_grad():
        for start in range(0, n_books, batch_size):
            end   = min(start + batch_size, n_books)
            bidxs = all_book_idxs[start:end]

            genre_embs.append(model.item_genre_tower(model.book_genre_matrix[bidxs]))
            shelf_embs.append(model.item_shelf_tower(model.book_shelf_matrix[bidxs]))
            book_embs.append(model.item_embedding_tower(model.item_embedding_lookup(bidxs)))
            author_embs.append(model.author_tower(
                model.author_embedding_lookup(model.book_author_idx[bidxs])))
            year_embs.append(model.year_embedding_tower(
                model.year_embedding_lookup(model.book_year_idx[bidxs])))
            combined_embs.append(model.item_embedding(bidxs))

    genre_all    = torch.cat(genre_embs,    dim=0)
    shelf_all    = torch.cat(shelf_embs,    dim=0)
    book_all     = torch.cat(book_embs,     dim=0)
    author_all   = torch.cat(author_embs,   dim=0)
    year_all     = torch.cat(year_embs,     dim=0)
    combined_all = torch.cat(combined_embs, dim=0)

    bookId_to_embedding = {}
    for i, bid in enumerate(fs.top_books):
        bookId_to_embedding[bid] = {
            'BOOK_GENRE_EMBEDDING':    genre_all[i].unsqueeze(0),
            'BOOK_SHELF_EMBEDDING':    shelf_all[i].unsqueeze(0),
            'BOOK_ID_EMBEDDING':       book_all[i].unsqueeze(0),
            'BOOK_AUTHOR_EMBEDDING':   author_all[i].unsqueeze(0),
            'BOOK_YEAR_EMBEDDING':     year_all[i].unsqueeze(0),
            'BOOK_EMBEDDING_COMBINED': combined_all[i].unsqueeze(0),
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

    liked_books   = USER_TYPE_TO_LIKED_BOOKS.get(user_type, [])
    anchor_titles = _get_anchor_titles(fs, shelf_tags,
                                       exclude=set(fav_books) | set(liked_books))

    liked_with_weights = (
        [(t, VALUE_FAVORITE_BOOK_RATING) for t in fav_books]   +
        [(t, VALUE_ANCHOR_BOOK_RATING)   for t in liked_books] +
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

    # ── Build user embedding via model.user_embedding() ──────────────────────
    if history:
        hist_idx_t = torch.tensor([[h[0] for h in history]], dtype=torch.long)  # (1, hist)
        hist_wts_t = torch.tensor([[h[1] for h in history]], dtype=torch.float)  # (1, hist)
    else:
        hist_idx_t = torch.full((1, 1), model.book_pad_idx, dtype=torch.long)
        hist_wts_t = torch.zeros(1, 1)

    X_inf = torch.tensor([ctx])
    return model.user_embedding(X_inf, hist_idx_t, hist_wts_t, ts_inference)


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
            user_emb    = _build_user_embedding(model, fs, user_type, ts_max_bin)
            fav_books   = USER_TYPE_TO_FAVORITE_BOOKS[user_type]
            liked_books = USER_TYPE_TO_LIKED_BOOKS.get(user_type, [])
            shelf_tags  = USER_TYPE_TO_SHELF_TAGS.get(user_type, [])
            anchor_titles = _get_anchor_titles(
                fs, shelf_tags, exclude=set(fav_books) | set(liked_books))
            exclude_set   = set(fav_books) | set(liked_books) | set(anchor_titles)

            raw_scores = (all_embs @ user_emb.T).squeeze(-1)
            scores     = {all_ids[i]: raw_scores[i].item() for i in range(len(all_ids))}

            n = 20 if user_type == "Nick's Recommendations" else top_n
            recs       = []
            seen_titles = set(exclude_set)
            for bid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if len(recs) >= n:
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
            if anchor_titles:
                print(f"Shelf anchors (rating={VALUE_ANCHOR_BOOK_RATING}):")
                for t in anchor_titles:
                    print(f"  + {t}")
                print('─' * bar_w)
            header = f"{'Favorite Books':<{col_w}}  Recommendations"
            print(header)
            print('─' * bar_w)
            for a, b in zip_longest(fav_books, recs, fillvalue=''):
                print(f"{a:<{col_w}}  {b}")


# ── Embedding probes ──────────────────────────────────────────────────────────

def probe_genre(model: BookRecommender, genre, book_embeddings: dict,
                fs: FeatureStore, top_n: int = 10) -> None:
    """
    Find the most representative books for a genre in item genre embedding space.
    Passes a one-hot (or multi-hot) genre vector through item_genre_tower, compares via cosine similarity.
    genre may be a single string or a list of strings.
    """
    genres = [genre] if isinstance(genre, str) else genre
    for g in genres:
        if g not in fs.genre_to_i:
            print(f"Genre '{g}' not in vocabulary. Available: {fs.genres_ordered}")
            return

    ctx = [0.0] * len(fs.genres_ordered)
    for g in genres:
        ctx[fs.genre_to_i[g]] = 1.0

    with torch.no_grad():
        query_emb = model.item_genre_tower(torch.tensor([ctx])).view(-1)

    sims = {
        bid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            book_embeddings[bid]['BOOK_GENRE_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for bid in fs.top_books
    }

    label = ' + '.join(genres)
    print(f"\nTop-{top_n} books for genre '{label}':")
    seen = set()
    for bid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen) >= top_n:
            break
        title = fs.bookId_to_title[bid]
        if title not in seen:
            seen.add(title)
            book_genres = ', '.join(fs.bookId_to_genres.get(bid, []))
            print(f"  {sim:.4f}  {title}  [{book_genres}]")


def probe_shelf(model: BookRecommender, shelf_tags: list, book_embeddings: dict,
                fs: FeatureStore, top_n: int = 10, k_anchors: int = 5) -> None:
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
            book_genres = ', '.join(fs.bookId_to_genres.get(bid, []))
            print(f"  {sim:.4f}  {title}{marker}  [{book_genres}]")


def probe_similar(book_embeddings: dict, fs: FeatureStore,
                  all_ids: list, all_norm: torch.Tensor,
                  titles: list, top_n: int = 5,
                  all_norm_id: torch.Tensor = None,
                  all_norm_shelf: torch.Tensor = None) -> None:
    """
    For each query title, find the top-N most similar books by cosine similarity.
    Shows results for BOOK_EMBEDDING_COMBINED, BOOK_ID_EMBEDDING, and BOOK_SHELF_EMBEDDING.
    Uses pre-normalized matrices from _load_model_and_embeddings.
    """
    TRUNC = 30

    def trunc(s: str) -> str:
        return s if len(s) <= TRUNC else s[:TRUNC - 1] + '…'

    def get_top_n(norm_matrix: torch.Tensor, emb_key: str, title: str) -> list:
        bid = fs.title_to_bookId.get(title)
        if bid is None:
            return []
        query   = F.normalize(book_embeddings[bid][emb_key], dim=1)
        sims    = (norm_matrix @ query.T).squeeze(-1)
        top_idx = sims.argsort(descending=True)
        results = []
        seen_titles = {title}
        for idx in top_idx:
            candidate_title = fs.bookId_to_title[all_ids[idx.item()]]
            if candidate_title in seen_titles:
                continue
            seen_titles.add(candidate_title)
            results.append(candidate_title)
            if len(results) >= top_n:
                break
        return results

    def print_table(label: str, rows: list) -> None:
        seed_w = max(len(trunc(t)) for t, _ in rows)
        col_w  = TRUNC
        header = f"{'Seed':<{seed_w}}" + "".join(f"  {'#'+str(i+1):<{col_w}}" for i in range(top_n))
        print(f"\n── Most similar books ({label}) ──")
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

    combined_rows = [(t, get_top_n(all_norm,    'BOOK_EMBEDDING_COMBINED', t)) for t in titles]
    print_table('combined embedding', combined_rows)

    if all_norm_id is not None:
        id_rows = [(t, get_top_n(all_norm_id, 'BOOK_ID_EMBEDDING', t)) for t in titles]
        print_table('book ID embedding only', id_rows)

    if all_norm_shelf is not None:
        shelf_rows = [(t, get_top_n(all_norm_shelf, 'BOOK_SHELF_EMBEDDING', t)) for t in titles]
        print_table('shelf embedding only', shelf_rows)


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint_path: str, checkpoint_dir: str):
    if checkpoint_path is not None:
        return checkpoint_path
    candidates = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'best_checkpoint_*.pth'))        +
        glob.glob(os.path.join(checkpoint_dir, 'best_proj_softmax_ipool_*.pth')) +
        glob.glob(os.path.join(checkpoint_dir, 'best_proj_softmax_*.pth'))       +
        glob.glob(os.path.join(checkpoint_dir, 'best_softmax_*.pth'))            +
        glob.glob(os.path.join(checkpoint_dir, 'best_bpr_*.pth'))                +
        glob.glob(os.path.join(checkpoint_dir, 'best_mse_*.pth')),
        key=os.path.getmtime, reverse=True
    )
    if not candidates:
        print("No checkpoint found in saved_models/. Train a model first.")
        return None
    return candidates[0]


def _load_model_and_embeddings(checkpoint_path: str, fs):
    """Build model, load weights, pre-compute book embeddings."""
    basename = os.path.basename(checkpoint_path)
    if 'ipool' in basename:
        config = get_softmax_config()
        config['use_item_pool_for_history'] = True
    elif basename.startswith('best_proj_softmax_') or basename.startswith('proj_softmax_'):
        config = get_softmax_config()
    elif basename.startswith('best_softmax_') or basename.startswith('softmax_'):
        config = get_softmax_config_legacy()
    else:
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
    all_ids       = list(book_embeddings.keys())
    all_embs      = torch.cat([book_embeddings[bid]['BOOK_EMBEDDING_COMBINED'] for bid in all_ids], dim=0)
    all_norm      = F.normalize(all_embs, dim=1)
    all_id_embs   = torch.cat([book_embeddings[bid]['BOOK_ID_EMBEDDING']       for bid in all_ids], dim=0)
    all_norm_id   = F.normalize(all_id_embs, dim=1)
    all_shelf_embs = torch.cat([book_embeddings[bid]['BOOK_SHELF_EMBEDDING']   for bid in all_ids], dim=0)
    all_norm_shelf = F.normalize(all_shelf_embs, dim=1)
    return model, book_embeddings, all_ids, all_embs, all_norm, all_norm_id, all_norm_shelf


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_canary(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    from src.dataset import load_features
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        return
    print("Loading features ...")
    fs = load_features(data_dir, version)
    model, book_embeddings, all_ids, all_embs, all_norm, all_norm_id, all_norm_shelf = _load_model_and_embeddings(cp, fs)
    print("\n── Canary user evaluation ──")
    run_canary_eval(model, fs, book_embeddings, all_ids, all_embs)


PROBE_SIMILAR_TITLES = [
    'A Game of Thrones (A Song of Ice and Fire, #1)',
    'The Fellowship of the Ring (The Lord of the Rings, #1)',
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
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        return
    print("Loading book features ...")
    fs = load_book_features(data_dir, version)
    model, book_embeddings, all_ids, all_embs, all_norm, all_norm_id, all_norm_shelf = _load_model_and_embeddings(cp, fs)
    print("\n── Embedding probes ──")
    probe_genre(model, 'mystery, thriller, crime', book_embeddings, fs)
    probe_genre(model, 'fantasy, paranormal',      book_embeddings, fs)
    probe_genre(model, ['romance', 'young-adult'],  book_embeddings, fs)
    probe_shelf(model, ['horror', 'scary', 'dark'],          book_embeddings, fs)
    probe_shelf(model, ['science-fiction', 'sci-fi', 'space'], book_embeddings, fs)
    probe_shelf(model, ['epic-fantasy', 'magic', 'world-building'], book_embeddings, fs)
    probe_similar(book_embeddings, fs, all_ids, all_norm, PROBE_SIMILAR_TITLES,
                  all_norm_id=all_norm_id, all_norm_shelf=all_norm_shelf)
