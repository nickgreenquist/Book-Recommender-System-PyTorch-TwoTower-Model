"""
Threshold exploration tool — uses goodreads_interactions.csv (fast) to estimate
how many users survive various MIN_RATINGS_PER_USER thresholds against the
already-filtered book corpus (base_books.parquet).

Does NOT produce any output parquets. Run this to pick MIN_RATINGS_PER_USER
before committing to the slow two-pass JSON streaming.

Usage:
    python main.py explore
"""
import os

import pandas as pd


MIN_THRESHOLDS = [5, 10, 20, 30, 50, 100]
MAX_THRESHOLDS = [50, 100, 200, 500, 1000, 2000]


def run(data_dir: str = 'data') -> None:
    base_books_path = os.path.join(data_dir, 'base_books.parquet')
    if not os.path.exists(base_books_path):
        raise FileNotFoundError(
            f"{base_books_path} not found — run 'python main.py preprocess books' first"
        )

    # Load corpus book IDs — CSV uses anonymized integer book_ids, need the map
    print("Loading base_books.parquet ...")
    books_df = pd.read_parquet(base_books_path, columns=['book_id'])
    corpus_book_ids = set(books_df['book_id'].astype(str).tolist())
    print(f"  Corpus: {len(corpus_book_ids):,} books")

    print("\nLoading book_id_map.csv ...")
    book_id_map = pd.read_csv(os.path.join(data_dir, 'book_id_map.csv'))
    # book_id_csv (int) → book_id (original string)
    csv_to_original = dict(zip(book_id_map['book_id_csv'].astype(str),
                               book_id_map['book_id'].astype(str)))
    corpus_csv_ids = {csv_id for csv_id, orig in csv_to_original.items()
                      if orig in corpus_book_ids}
    print(f"  Corpus books found in CSV map: {len(corpus_csv_ids):,}")

    # Stream CSV in chunks, count ratings per user against corpus books only
    print("\nStreaming goodreads_interactions.csv ...")
    user_counts: dict = {}
    total_rated = 0
    chunk_size = 1_000_000

    csv_path = os.path.join(data_dir, 'goodreads_interactions.csv')
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             dtype={'user_id': str, 'book_id': str, 'rating': float}):
        rated = chunk[(chunk['rating'] > 0) & (chunk['book_id'].isin(corpus_csv_ids))]
        total_rated += len(rated)
        for uid, count in rated.groupby('user_id').size().items():
            user_counts[uid] = user_counts.get(uid, 0) + count

    total_users = len(user_counts)
    print(f"  Total rated interactions (corpus books): {total_rated:,}")
    print(f"  Total users with ≥1 corpus rating:       {total_users:,}")

    print(f"\nMin ratings threshold (users with AT LEAST N corpus ratings):")
    print(f"{'Min':>10}  {'Users kept':>12}  {'%':>6}")
    print("-" * 34)
    for t in MIN_THRESHOLDS:
        kept = sum(1 for c in user_counts.values() if c >= t)
        print(f"{t:>10,}  {kept:>12,}  {100*kept/total_users:>5.1f}%")

    print(f"\nMax ratings threshold (users with AT MOST N corpus ratings):")
    print(f"{'Max':>10}  {'Users kept':>12}  {'%':>6}")
    print("-" * 34)
    for t in MAX_THRESHOLDS:
        kept = sum(1 for c in user_counts.values() if c <= t)
        print(f"{t:>10,}  {kept:>12,}  {100*kept/total_users:>5.1f}%")
