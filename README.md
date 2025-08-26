
# 🎬 Movie Recommender (Streamlit)

A ready-to-run Streamlit app that offers:

- 🔐 **Login page** (demo credentials: `demo` / `demo`)
- 🔎 **Search bar** for movies, actors, or crew
- 🧠 **Similar movie recommendations** based on content
- 🧑‍🎤 **Top movies by actor**
- 🎬 **Top movies by crew** (e.g., directors, writers)
- 🖼️ **Posters** fetched from TMDB (requires API key)

## 1) Files needed

Place these two datasets in the same folder as `app.py`:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

(When running here, the app can also read them from `/mnt/data/`.)

## 2) Install & run

Create a virtual environment (optional but recommended), then:

```bash
pip install -r requirements.txt
# On macOS/Linux:
export TMDB_API_KEY="YOUR_TMDB_API_KEY"
# On Windows PowerShell:
$env:TMDB_API_KEY="YOUR_TMDB_API_KEY"

streamlit run app.py
```

Open the shown local URL in your browser.

Login with `demo` / `demo` to explore.

## 3) How it works

- Merges `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on the TMDB `id`.
- Builds a "soup" from **genres, keywords, top 3 cast, and director**.
- Uses **TF-IDF** on the overview and **CountVectorizer** on the soup, then combines them.
- Computes cosine similarity to recommend similar movies.
- Actor/Crew pages rank films by a weighted popularity score `vote_average * log(1 + vote_count)`.

## 4) Posters

Set `TMDB_API_KEY` to fetch real posters from TMDB. Without the key, you'll see placeholders.

## 5) Customize

- Replace the simple in-memory `DEFAULT_USERS` with a database or OAuth.
- Tweak the `poster_grid` layout, the similarity pipeline, or add more filters (year, genre).
- Add a "watchlist" by storing user selections in a small DB (SQLite) keyed by `st.session_state["user"]`.
