import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Flavor Finder 🍴", page_icon="🍕", layout="wide")

st.title("🍽️ Flavor Finder")
st.write("Where hungry hearts find their perfect match!🍜")
st.subheader("Ready to fall in love with your next meal? Let's go!🍔")

# Load your dataset directly (assumes certain column names)
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# 👇 replace with your dataset path or use uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Pick known fields if they exist
    food_col = "recipename" if "recipename" in df.columns else df.columns[0]
    rest_col = "restaurantname" if "restaurantname" in df.columns else None
    price_col = "price" if "price" in df.columns else None
    rating_col = "rating" if "rating" in df.columns else None
    url_col = "url" if "url" in df.columns else None

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df[food_col].astype(str))

    # Search box
    query = st.text_input("🍟 What are you craving today?")
    if query:
        query_vec = vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        df["similarity"] = sim_scores
        results = df.sort_values("similarity", ascending=False).head(5)

        st.subheader("🧉 Top Picks for You")

        for _, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### 🍳 {row[food_col]}")
                    if rest_col: st.markdown(f"🏠 {row[rest_col]}")
                    if rating_col: st.markdown(f"⭐ {row[rating_col]}")
                    if price_col: st.markdown(f"💰 {row[price_col]}")

                with col2:
                    if url_col:
                        st.link_button("Get Recipe🥗", row[url_col])

                st.divider()
