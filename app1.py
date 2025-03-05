import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from scipy.sparse import hstack

# TMDb API Key
api_key = "48b555118cb07476a8e7f103af2e7e7c"
base_image_url = "https://image.tmdb.org/t/p/w500"  # TMDb base URL for images

# fetch movie details from TMDb by searching its title
def search_movie_by_title(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(search_url)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            return results[0]  # Return the first matching result
    return None

# fetch keywords for a movie by its ID
def fetch_movie_keywords(movie_id):
    keyword_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords?api_key={api_key}"
    response = requests.get(keyword_url)
    if response.status_code == 200:
        keywords_data = response.json().get("keywords", [])
        return " ".join([kw["name"] for kw in keywords_data])  # Return keywords as a string
    return ""

# fetch genres for a movie by its ID
def fetch_movie_genres(movie_id):
    genre_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    response = requests.get(genre_url)
    if response.status_code == 200:
        genres_data = response.json().get("genres", [])
        return " ".join([genre["name"] for genre in genres_data])  # Return genres as a string
    return ""

# fetch movie plot (overview) by ID
def fetch_movie_plot(movie_id):
    plot_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(plot_url)
    if response.status_code == 200:
        return response.json().get("overview", "No plot available.")
    return "No plot available."

# fetch popular movies (for initial dataset)
@st.cache_data
def fetch_popular_movies(num_pages=20):
    movies = []
    base_url = "https://api.themoviedb.org/3/movie/popular"
    
    for page in range(1, num_pages + 1):
        response = requests.get(f"{base_url}?api_key={api_key}&language=en-US&page={page}")
        if response.status_code == 200:
            data = response.json()
            movies.extend(data["results"])
        time.sleep(0.2)  # Avoid rate limits

    return pd.DataFrame(movies)[["id", "title", "poster_path"]]

# fetch and preprocess movies
df = fetch_popular_movies(num_pages=20)
df["keywords_str"] = df["id"].apply(fetch_movie_keywords)
df["genres_str"] = df["id"].apply(fetch_movie_genres)

# TF-IDF Vectorization for genres and keywords
tfidf_keywords = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_genres = TfidfVectorizer(stop_words="english", max_features=100, binary=True) 


tfidf_keywords_matrix = tfidf_keywords.fit_transform(df["keywords_str"])
tfidf_genres_matrix = tfidf_genres.fit_transform(df["genres_str"])

# apply different weights to genres and keywords
keywords_weight = 1.5  # keywords are more specific
genres_weight = 0.7    # genres are broader categories

# combine weighted matrices
combined_tfidf_matrix = hstack([tfidf_keywords_matrix * keywords_weight, tfidf_genres_matrix * genres_weight])

# cosine similarity on the weighted combination
cosine_sim = cosine_similarity(combined_tfidf_matrix, combined_tfidf_matrix)

# get similar movies based on genres + keywords
def recommend_similar_movies(movie_title):
    # check if the movie exists in TMDb
    movie_data = search_movie_by_title(movie_title)
    
    if not movie_data:
        return f"No results found for '{movie_title}'. Try another title."

    # get the movie ID and fetch genres + keywords
    movie_id = movie_data["id"]
    keywords = fetch_movie_keywords(movie_id)
    genres = fetch_movie_genres(movie_id)

    if not keywords and not genres:
        return f"'{movie_title}' has no associated keywords or genres, so we can't recommend similar movies."

    # new dataframe row for the input movie
    new_movie = pd.DataFrame({
        "title": [movie_data["title"]],
        "keywords_str": [keywords],
        "genres_str": [genres],
        "poster_path": [movie_data.get("poster_path")]
    })
    
    # transform new movie using the same TF-IDF vectorizers
    new_keywords_tfidf = tfidf_keywords.transform(new_movie["keywords_str"])
    new_genres_tfidf = tfidf_genres.transform(new_movie["genres_str"])

    # apply the same weights
    new_combined_tfidf = hstack([new_keywords_tfidf * keywords_weight, new_genres_tfidf * genres_weight])

    # compute similarity with existing movies
    new_cosine_sim = cosine_similarity(new_combined_tfidf, combined_tfidf_matrix)

    # get similarity scores
    sim_scores = list(enumerate(new_cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity

    # removes the input movie itself
    sim_scores = [s for s in sim_scores if df.iloc[s[0]]["title"].lower() != movie_title.lower()]

    # gets top 5 recommended movies (excluding the input movie)
    sim_scores = sim_scores[:5]

    # get recommended movie titles, poster paths, and IDs
    recommendations = df.iloc[[i[0] for i in sim_scores]][["id", "title", "poster_path"]]

    # removes duplicates
    recommendations = recommendations.drop_duplicates(subset="title")

    # adds image URLs
    recommendations["poster_url"] = recommendations["poster_path"].apply(lambda x: f"{base_image_url}{x}" if x else None)

    # fetch plot summaries for each recommended movie
    recommendations["plot"] = recommendations["id"].apply(fetch_movie_plot)

    return recommendations
#test
# Streamlit UI
st.title("❤️‍🔥 If you liked _____, try these!🦩")

user_input = st.text_input("Type a movie name to get recommendations:")

if st.button("Get Recommendations"):
    recommendations = recommend_similar_movies(user_input)

    if isinstance(recommendations, str):  
        st.warning(recommendations)
    elif not recommendations.empty:
        st.subheader(f"Movies similar to {user_input}:")
        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])  
            with col1:
                if row["poster_url"]:
                    st.image(row["poster_url"], width=120)
                else:
                    st.write("No image available")
            with col2:
                st.write(f"🎥 **{row['title']}**")
                st.write(f"📝 *{row['plot']}*")  
                st.markdown("---")
    else:
        st.warning("No similar movies found.")