import ast
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Absolute directory path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_list(obj):
    """Extracts 'name' values from JSON string representation."""
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except (ValueError, SyntaxError):
        return []


def parse_cast(obj):
    """Extracts top 3 cast members."""
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except (ValueError, SyntaxError):
        return []


def parse_crew(obj):
    """Extracts only director names."""
    try:
        return [
            i['name'] for i in ast.literal_eval(obj) if i.get('job') == 'Director'
        ]
    except (ValueError, SyntaxError):
        return []


def build_recommendation_artifacts():
    print("Loading datasets...")
    credits_path = os.path.join(BASE_DIR, 'tmdb_5000_credits.csv')
    movies_path = os.path.join(BASE_DIR, 'tmdb_5000_movies.csv')

    credits = pd.read_csv(credits_path)
    movies = pd.read_csv(movies_path)

    # Merge on title
    movies = movies.merge(credits, on='title')

    # Retain necessary fields including vote metrics
    cols_to_keep = [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew',
        'vote_average',
        'vote_count',
    ]
    movies = movies[cols_to_keep].dropna(subset=['overview', 'title'])

    print("Extracting metadata...")
    movies['genre_list'] = movies['genres'].apply(parse_list)
    movies['keywords_list'] = movies['keywords'].apply(parse_list)
    movies['cast_list'] = movies['cast'].apply(parse_cast)
    movies['crew_list'] = movies['crew'].apply(parse_crew)
    movies['overview_list'] = movies['overview'].apply(lambda x: str(x).split())

    # Build semantic tags
    movies['tags'] = (
        movies['genre_list']
        + movies['keywords_list']
        + movies['cast_list']
        + movies['crew_list']
        + movies['overview_list']
    )
    movies['tags'] = movies['tags'].apply(
        lambda x: " ".join(x).lower()
    )

    # Keep clean dataset for the app
    final_movies = movies[
        ['movie_id', 'title', 'overview', 'genre_list', 'vote_average', 'vote_count', 'tags']
    ].reset_index(drop=True)

    print("Computing TF-IDF Matrix & Cosine Similarity...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(final_movies['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print("Saving to movie_data.pkl...")
    pkl_path = os.path.join(BASE_DIR, 'movie_data.pkl')
    with open(pkl_path, 'wb') as f:
        # Saving 4-tuple so app.py can perform live query transformations
        pickle.dump((final_movies, cosine_sim, tfidf, tfidf_matrix), f)

    print("Artifacts generated successfully!")


if __name__ == '__main__':
    build_recommendation_artifacts()

