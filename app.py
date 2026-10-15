import os
import pickle
import re
import requests
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="CineVibe | Smart Movie Discovery", layout="wide")

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
LOGO_BASE_URL = "https://image.tmdb.org/t/p/w92"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    pkl_path = os.path.join(BASE_DIR, 'movie_data.pkl')
    with open(pkl_path, 'rb') as f:
        movies_df, sim_matrix, vectorizer, matrix = pickle.load(f)
    return movies_df, sim_matrix, vectorizer, matrix


movies, cosine_sim, tfidf, tfidf_matrix = load_data()


@st.cache_data(ttl=86400)
def fetch_movie_details(movie_id, region="IN"):
    details = {
        "poster": "https://via.placeholder.com/500x750?text=No+Poster",
        "trailer_url": None,
        "providers": []
    }
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        
        info_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        r_info = requests.get(info_url, headers=headers, timeout=5).json()
        if r_info.get("poster_path"):
            details["poster"] = POSTER_BASE_URL + r_info["poster_path"]
            
       
        vid_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
        vids = requests.get(vid_url, headers=headers, timeout=5).json().get("results", [])
        
        
        for v in vids:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                details["trailer_url"] = f"https://www.youtube.com/watch?v={v['key']}"
                break
        if not details["trailer_url"]:
            for v in vids:
                if v.get("site") == "YouTube":
                    details["trailer_url"] = f"https://www.youtube.com/watch?v={v['key']}"
                    break

        
        prov_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={TMDB_API_KEY}"
        prov_data = requests.get(prov_url, headers=headers, timeout=5).json().get("results", {})
        
        target_region = prov_data.get(region) or prov_data.get("US", {})
        
       
        offerings = target_region.get("flatrate") or target_region.get("buy") or []
        for p in offerings[:3]:
            details["providers"].append({
                "name": p.get("provider_name"),
                "logo": LOGO_BASE_URL + p["logo_path"] if p.get("logo_path") else None
            })

    except Exception as e:
        print(f"Error fetching TMDB data for ID {movie_id}: {e}")

    return details


import re
def clean_title_tokens(title: str) -> set:
    """Tokenizes title to detect and filter sequels/franchises."""
    words = re.findall(r'\b\w+\b', str(title).lower())
    stop_words = {
        'the', 'a', 'an', 'of', 'and', 'in', 'on', 'part', 'vol', 'ii',
        'iii', 'iv', 'v', '2', '3', '4', 'rises', 'begins', 'returns', 'movie'
    }
    return {w for w in words if w not in stop_words and len(w) > 2}


def get_genre_recommendations(
    title: str,
    top_n: int = 5,
    min_genre_overlap: int = 1,
    exclude_franchise: bool = False,
    min_rating: float = 6.0
):
    matches = movies[movies['title'].str.lower() == title.lower()]
    if matches.empty:
        return []

    target_idx = matches.index[0]
    target_genres = set(movies.loc[target_idx, 'genre_list'])
    target_tokens = clean_title_tokens(title)

    sim_scores = list(enumerate(cosine_sim[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for cand_idx, score in sim_scores:
        if cand_idx == target_idx:
            continue

        cand = movies.iloc[cand_idx]
        cand_genres = set(cand['genre_list'])

        required_overlap = min(min_genre_overlap, len(target_genres))
        if len(target_genres.intersection(cand_genres)) < required_overlap:
            continue

        if exclude_franchise:
            cand_tokens = clean_title_tokens(cand['title'])
            if len(target_tokens.intersection(cand_tokens)) > 0:
                continue

        if cand['vote_average'] < min_rating:
            continue

        recommendations.append(cand)
        if len(recommendations) == top_n:
            break

    return recommendations


def get_vibe_recommendations(vibe_query: str, top_n: int = 5, min_rating: float = 5.5):
    query_vec = tfidf.transform([vibe_query.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = scores.argsort()[::-1]

    recommendations = []
    for idx in ranked_indices:
        if scores[idx] < 0.05:
            break
        cand = movies.iloc[idx]
        if cand['vote_average'] >= min_rating:
            recommendations.append(cand)
        if len(recommendations) == top_n:
            break

    return recommendations


st.sidebar.title("⚙️ Engine Controls")
mode = st.sidebar.radio(
    "Discovery Mode",
    ["Similar Movies (Intra-Genre)", "Natural Language Vibe Search"]
)

region = st.sidebar.selectbox("OTT Region", ["IN", "US", "GB", "CA", "AU"], index=0)
num_recs = st.sidebar.slider("Recommendations Count", 3, 10, 5)
min_rating = st.sidebar.slider("Min TMDB Rating", 0.0, 8.5, 6.0, 0.5)

if mode == "Similar Movies (Intra-Genre)":
    exclude_franchise = st.sidebar.checkbox("Filter Sequels & Franchises", value=False)
    min_genre_overlap = st.sidebar.slider("Min Matching Genres", 1, 4, 2)

st.title("🎬 CineVibe Recommender")

if mode == "Similar Movies (Intra-Genre)":
    st.caption("Discover movies with similar plots and genres.")
    selected_movie = st.selectbox(
        "Choose a Reference Film:",
        options=movies['title'].values,
        index=int(movies[movies['title'] == 'Interstellar'].index[0]) if 'Interstellar' in movies['title'].values else 0
    )
    
    genres = movies[movies['title'] == selected_movie]['genre_list'].values[0]
    st.markdown(f"**Genres:** `{', '.join(genres)}`")
    run_search = st.button("Find Similar Movies", type="primary")

else:
    st.caption("Search by natural language prompts, plot elements, or aesthetic vibes.")
    vibe_query = st.text_input(
        "Describe what you want to watch:",
        placeholder="e.g. mind-bending space travel with psychological dread and time dilation"
    )
    run_search = st.button("Search Vibe", type="primary")

if run_search:
    with st.spinner("Analyzing semantics and fetching live providers..."):
        if mode == "Similar Movies (Intra-Genre)":
            recs = get_genre_recommendations(
                selected_movie,
                top_n=num_recs,
                min_genre_overlap=min_genre_overlap,
                exclude_franchise=exclude_franchise,
                min_rating=min_rating
            )
        else:
            if not vibe_query.strip():
                st.warning("Please enter a vibe or description first.")
                recs = []
            else:
                recs = get_vibe_recommendations(vibe_query, top_n=num_recs, min_rating=min_rating)

    if recs:
        cols = st.columns(len(recs))
        for i, cand in enumerate(recs):
            with cols[i]:
                details = fetch_movie_details(cand['movie_id'], region=region)
                st.image(details["poster"], width="stretch")
                st.markdown(f"#### {cand['title']}")
                st.caption(f"⭐ **{cand['vote_average']:.1f}** | {', '.join(cand['genre_list'][:2])}")

                if details["providers"]:
                    st.markdown("**Available on:**")
                    prov_cols = st.columns(len(details["providers"]))
                    for p_idx, prov in enumerate(details["providers"]):
                        with prov_cols[p_idx]:
                            if prov["logo"]:
                                st.image(prov["logo"], width="stretch")
                            else:
                                st.caption(prov["name"])
                else:
                    st.caption("No flatrate streaming found")

                if details["trailer_url"]:
                    with st.expander("▶ Watch Trailer"):
                        st.video(details["trailer_url"])
                else:
                    st.caption("No trailer available")
    elif run_search and not recs:
        st.info("No matching movies found. Try relaxing the rating or genre filters.")