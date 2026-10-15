import os
import pickle
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "1b33ce5cd91092b0df91f9c02f063bd5"  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def fetch_modern_tmdb_dataset(num_pages=40):
    print("Connecting to TMDB API...")

    genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    try:
        genres_res = session.get(genre_url, headers=headers, timeout=10).json()
        genre_dict = {g['id']: g['name'] for g in genres_res.get('genres', [])}
    except Exception as e:
        print(f"Failed to fetch genres: {e}")
        genre_dict = {}

    all_movies = []
    seen_ids = set()
    endpoints = ['popular', 'top_rated']

    for ep in endpoints:
        print(f"Fetching {ep} movies...")
        for page in range(1, (num_pages // 2) + 1):
            url = f"https://api.themoviedb.org/3/movie/{ep}?api_key={TMDB_API_KEY}&language=en-US&page={page}"
            try:
                res = session.get(url, headers=headers, timeout=10).json()
                results = res.get('results', [])

                for m in results:
                    m_id = m.get('id')
                    if m_id in seen_ids or not m.get('overview') or not m.get('title'):
                        continue
                    seen_ids.add(m_id)

                    genre_names = [
                        genre_dict[gid]
                        for gid in m.get('genre_ids', [])
                        if gid in genre_dict
                    ]

                    all_movies.append({
                        'movie_id': m_id,
                        'title': m.get('title'),
                        'overview': m.get('overview'),
                        'genre_list': genre_names,
                        'vote_average': m.get('vote_average', 0.0),
                        'vote_count': m.get('vote_count', 0),
                        'release_date': m.get('release_date', '')
                    })
                time.sleep(0.2) 
            except Exception as e:
                print(f"Error on page {page}: {e}")
                time.sleep(1)
                continue

    df = pd.DataFrame(all_movies)
    print(f"Successfully collected {len(df)} modern movies.")
    return df

def build_pipeline():
    movies_df = fetch_modern_tmdb_dataset(num_pages=40)
    if movies_df.empty:
        print("Error: No data retrieved. Check your network or API key.")
        return

    print("Constructing tags for vectorization...")
    movies_df['tags'] = (
        movies_df['genre_list'].apply(lambda x: " ".join(x).lower())
        + " "
        + movies_df['overview'].str.lower()
    )

    print("Computing TF-IDF Matrix...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    output_path = os.path.join(BASE_DIR, 'movie_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump((movies_df, cosine_sim, tfidf, tfidf_matrix), f)

    print(f"Pipeline complete! Artifact saved at: {output_path}")

if __name__ == '__main__':
    build_pipeline()