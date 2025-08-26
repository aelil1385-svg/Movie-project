
import os
import ast
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config & Globals
# ----------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

DATA_FILENAMES = {
    "movies": "tmdb_5000_movies.csv",
    "credits": "tmdb_5000_credits.csv",
}

ALT_DATA_DIR = "/mnt/data"  # fallback location in this environment

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

# Very simple demo user store (replace with a DB in real apps)
DEFAULT_USERS = {"demo": "demo", "admin": "admin"}


# ----------------------------
# Utility functions
# ----------------------------
def find_data_file(name: str) -> Optional[str]:
    """
    Find the CSV either in current directory or in ALT_DATA_DIR.
    Returns a filepath or None.
    """
    if os.path.exists(name):
        return name
    alt_path = os.path.join(ALT_DATA_DIR, name)
    if os.path.exists(alt_path):
        return alt_path
    return None


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_path = find_data_file(DATA_FILENAMES["movies"])
    credits_path = find_data_file(DATA_FILENAMES["credits"])
    if not movies_path or not credits_path:
        st.error("Could not find input CSVs. Please place 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' in the app folder.")
        st.stop()
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    return movies, credits


def parse_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Parse JSON-like columns (e.g., genres, keywords, cast, crew) into Python objects.
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x))
    return df


def get_director(crew_list: List[dict]) -> str:
    for p in crew_list:
        if p.get("job") == "Director":
            return p.get("name", "")
    return ""


def clean_name(name: str) -> str:
    return name.replace(" ", "").lower()


def build_soup(row: pd.Series) -> str:
    # Combine genres, keywords, top 3 cast, and director into a single "soup" string
    genres = [clean_name(x["name"]) for x in row.get("genres", []) if "name" in x]
    keywords = [clean_name(x["name"]) for x in row.get("keywords", []) if "name" in x]
    cast = [clean_name(x["name"]) for x in row.get("cast", [])[:3] if "name" in x]
    director = clean_name(row.get("director", "")) if row.get("director", "") else ""

    components = genres + keywords + cast + ([director] if director else [])
    return " ".join(components)


def fetch_poster(tmdb_id: int) -> str:
    """
    Fetch poster URL from TMDB. Requires TMDB_API_KEY.
    Fallback to placeholder if not available.
    """
    if not TMDB_API_KEY or pd.isna(tmdb_id):
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            p = data.get("poster_path")
            if p:
                return f"{TMDB_IMAGE_BASE}{p}"
    except Exception:
        pass
    return PLACEHOLDER_POSTER


def popularity_score(row: pd.Series) -> float:
    # Weighted score to rank movies when searching by actor/crew
    v = row.get("vote_count", 0) or 0
    r = row.get("vote_average", 0.0) or 0.0
    return float(r) * np.log1p(float(v))


@st.cache_data(show_spinner=False)
def prepare_data() -> Tuple[pd.DataFrame, np.ndarray]:
    movies, credits = load_data()

    # Parse JSON-like columns
    movies = parse_features(movies, ["genres", "keywords"])
    credits = parse_features(credits, ["cast", "crew"])

    # Merge on 'id'
    credits_renamed = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits_renamed[["id", "cast", "crew"]], on="id", how="left")

    # Extract director
    df["director"] = df["crew"].apply(get_director)

    # Build searchable fields
    df["title_lower"] = df["title"].str.lower()
    df["soup"] = df.apply(build_soup, axis=1)

    # Text vectors for content similarity: overview (tf-idf) + soup (count)
    # TF-IDF on overview
    overview = df["overview"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(overview)

    # CountVectorizer on soup
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df["soup"])

    # Combine by simple concatenation (hstack-like using numpy)
    from scipy.sparse import hstack
    combined = hstack([tfidf_matrix, count_matrix])

    # Precompute cosine similarity
    sim = cosine_similarity(combined, dense_output=False)

    # Precompute a few helper columns
    df["actor_names"] = df["cast"].apply(lambda lst: [x.get("name", "") for x in lst] if isinstance(lst, list) else [])
    df["crew_names"] = df["crew"].apply(lambda lst: [x.get("name", "") for x in lst] if isinstance(lst, list) else [])
    df["score"] = df.apply(popularity_score, axis=1)

    return df, sim


def search_movies(df: pd.DataFrame, query: str, limit: int = 20) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return df.head(0)
    mask = (
        df["title_lower"].str.contains(q, na=False)
        | df["actor_names"].apply(lambda names: any(q in (n or "").lower() for n in names))
        | df["crew_names"].apply(lambda names: any(q in (n or "").lower() for n in names))
        | df["director"].apply(lambda d: q in (d or "").lower())
    )
    res = df[mask].copy()
    res["rank"] = np.arange(1, len(res) + 1)
    return res.sort_values(["rank"]).head(limit)


def top_by_actor(df: pd.DataFrame, actor_name: str, limit: int = 12) -> pd.DataFrame:
    a = actor_name.strip().lower()
    if not a:
        return df.head(0)
    mask = df["actor_names"].apply(lambda names: any(a in (n or "").lower() for n in names))
    res = df[mask].copy()
    return res.sort_values("score", ascending=False).head(limit)


def top_by_crew(df: pd.DataFrame, crew_name: str, limit: int = 12) -> pd.DataFrame:
    c = crew_name.strip().lower()
    if not c:
        return df.head(0)
    mask = df["crew_names"].apply(lambda names: any(c in (n or "").lower() for n in names))
    res = df[mask].copy()
    return res.sort_values("score", ascending=False).head(limit)


def recommend_similar(df: pd.DataFrame, sim, title: str, limit: int = 12) -> pd.DataFrame:
    t = title.strip().lower()
    idxs = df.index[df["title_lower"] == t].tolist()
    if not idxs:
        # try partial match
        idxs = df.index[df["title_lower"].str.contains(t, na=False)].tolist()
    if not idxs:
        return df.head(0)
    idx = idxs[0]
    # get similarity row
    row = sim[idx].toarray().ravel()
    # get top similar indices excluding itself
    similar_idx = np.argsort(-row)[1: limit + 1]
    res = df.iloc[similar_idx].copy()
    res["sim_score"] = row[similar_idx]
    return res.sort_values("sim_score", ascending=False)


def poster_grid(items: pd.DataFrame, caption_col: str = "title") -> None:
    n = len(items)
    if n == 0:
        st.info("No results.")
        return
    cols = st.columns(6)  # 6 posters per row
    for i, (_, row) in enumerate(items.iterrows()):
        with cols[i % 6]:
            poster = fetch_poster(row.get("id"))
            st.image(poster, use_container_width=True)
            st.caption(f"{row.get(caption_col, '')}")


# ----------------------------
# Login
# ----------------------------
def login_view():
    st.markdown("### 🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", type="password", value="demo")
        submitted = st.form_submit_button("Login")
        if submitted:
            # In a real app, verify against DB/hashed passwords
            if username in DEFAULT_USERS and DEFAULT_USERS[username] == password:
                st.session_state["auth"] = True
                st.session_state["user"] = username
                st.success("Logged in!")
                time.sleep(0.3)
                st.rerun()
            else:
                st.error("Invalid username or password.")


# ----------------------------
# Main App UI
# ----------------------------
def main_app():
    st.markdown("# 🎬 Movie Recommender")
    st.caption("Search movies, find similar titles, and browse by actor or crew. Posters via TMDB.")

    df, sim = prepare_data()

    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Search", "🧠 Similar Movies", "🧑‍🎤 By Actor", "🎬 By Crew"])

    with tab1:
        st.subheader("Search movies / people")
        q = st.text_input("Type a movie title, actor, or crew name", "")
        if q:
            results = search_movies(df, q, limit=30)
            poster_grid(results)

    with tab2:
        st.subheader("Find similar movies")
        c1, c2 = st.columns([2, 1])
        with c1:
            movie_title = st.text_input("Enter a movie title", "")
        with c2:
            n = st.number_input("How many recommendations?", min_value=6, max_value=30, value=12, step=1)
        if movie_title:
            recs = recommend_similar(df, sim, movie_title, limit=int(n))
            poster_grid(recs)

    with tab3:
        st.subheader("Top movies by actor")
        actor = st.text_input("Actor name", "")
        if actor:
            res = top_by_actor(df, actor, limit=18)
            poster_grid(res)

    with tab4:
        st.subheader("Top movies by crew (e.g., directors, writers)")
        crew = st.text_input("Crew name", "")
        if crew:
            res = top_by_crew(df, crew, limit=18)
            poster_grid(res)

    st.divider()
    st.markdown("**Tip:** Set your TMDB API key in the environment variable `TMDB_API_KEY` to see real posters.")

    # Footer / logout
    with st.sidebar:
        st.markdown("## Account")
        user = st.session_state.get("user", "guest")
        st.write(f"Signed in as **{user}**")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()


def run():
    # Auth gate
    if "auth" not in st.session_state or not st.session_state.get("auth", False):
        login_view()
        return
    main_app()


if __name__ == "__main__":
    run()
