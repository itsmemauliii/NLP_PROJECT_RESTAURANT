import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import random
from PIL import Image

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Flavor Finder: A Recipe Recommendation App",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Flavor Finder")
st.markdown("## Find Your Next Favorite Recipe")

# --- Sidebar Controls ---
st.sidebar.header("Data & Mode Selection")
uploaded_file = st.sidebar.file_uploader(
    "Upload your cleaned CSV file (with 'title', 'ingredients', and 'cuisine_type' columns)",
    type="csv"
)

mode = st.sidebar.radio(
    "Select a mode:",
    ("Text-based Recommendation (NLP)", "Image-based Recommendation (CV)")
)

# Initialize dataframes and models
recipes_df = None
tfidf = None
tfidf_matrix = None

if uploaded_file is not None:
    try:
        recipes_df = pd.read_csv(uploaded_file)
        if not all(col in recipes_df.columns for col in ['title', 'ingredients', 'cuisine_type']):
            st.error("The uploaded CSV must contain 'title', 'ingredients', and 'cuisine_type' columns.")
            recipes_df = None
        else:
            st.sidebar.success("File uploaded successfully!")
            # Vectorize the ingredient data after upload
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(recipes_df['ingredients'])
    except Exception as e:
        st.error(f"Error loading file: {e}")
        recipes_df = None

# --- Recommendation Section (shared function) ---
def get_recommendations(user_query, tfidf_matrix, df, tfidf_vectorizer):
    """Generates recipe recommendations based on a user's query."""
    if not user_query or df is None:
        return pd.DataFrame()

    try:
        # Vectorize user input
        user_tfidf = tfidf_vectorizer.transform([user_query])
    except ValueError as e:
        st.warning(f"Could not vectorize your input. Please try a different combination of words. Error: {e}")
        return pd.DataFrame()

    # Compute similarity scores
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Get the top 5 most similar recipes
    top_indices = cosine_similarities.argsort()[:-6:-1]
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity_score'] = [cosine_similarities[i] for i in top_indices]

    return recommendations

# --- Mode-specific UI and Logic ---
if mode == "Text-based Recommendation (NLP)":
    st.header("Text-based Recommendation Engine")
    st.markdown("Enter ingredients, and the app will find recipes with a similar flavor profile.")
    
    user_input = st.text_area(
        "Enter a few ingredients, flavors, or a dish you like:",
        "chicken, garlic, and fresh herbs"
    )

    if st.button("Find Recipes"):
        if recipes_df is not None:
            recommendations = get_recommendations(user_input, tfidf_matrix, recipes_df, tfidf)
            if not recommendations.empty:
                st.subheader("Top Recipe Recommendations:")
                st.dataframe(recommendations[['title', 'ingredients', 'similarity_score']])
            else:
                st.warning("No recommendations found. Please try different keywords or check your data.")
        else:
            st.warning("Please upload a CSV file first.")

    st.markdown("---")
    
    # --- Hyperparameter Tuning Section ---
    st.header("Hyperparameter Tuning Demonstration with Cross-Validation")
    st.markdown(
        "This section demonstrates how to find the best hyperparameters for a model "
        "using `GridSearchCV` and a `Pipeline` with 5-fold cross-validation. "
        "We will tune a Support Vector Classifier (SVC) to classify recipes as 'Sweet' or 'Savory'."
    )
    
    if st.button("Start Hyperparameter Tuning"):
        if recipes_df is not None:
            with st.spinner('Tuning in progress... This may take a moment.'):
                # Prepare data for the tuning example
                X = recipes_df['ingredients']
                y = recipes_df['cuisine_type']
    
                # Create a pipeline
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', SVC())
                ])
    
                # Define the parameter grid to search
                param_grid = {
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'clf__C': [0.1, 1, 10, 100],
                    'clf__kernel': ['linear', 'rbf']
                }
    
                # Setup GridSearchCV with 5-fold cross-validation
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=5,
                    verbose=1,
                    n_jobs=-1  # Use all available cores
                )
    
                # Split data and fit the grid search model
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                grid_search.fit(X_train, y_train)
    
                # Display results
                st.success("Tuning complete!")
                st.markdown(f"**Best Cross-Validation Score:** `{grid_search.best_score_:.4f}`")
                st.markdown(f"**Best Hyperparameters:** `{grid_search.best_params_}`")
                
                # Optionally, show some predictions from the best model
                st.subheader("Example Predictions from the Best Model")
                best_model = grid_search.best_estimator_
                example_data = X_test.sample(n=min(5, len(X_test)), random_state=42)
                predictions = best_model.predict(example_data)
                
                results_df = pd.DataFrame({
                    'Recipe Ingredients': example_data,
                    'Predicted Type': predictions
                })
                st.dataframe(results_df)
    
        else:
            st.warning("Please upload a CSV file first.")

elif mode == "Image-based Recommendation (CV)":
    st.header("Image-based Recommendation Engine")
    st.markdown("Upload an image of a food item, and the app will try to identify key ingredients to find recipes.")
    
    uploaded_image = st.file_uploader(
        "Upload an image of a food item",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_image is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # --- Simulated Computer Vision Model ---
            # In a real app, you would use a pre-trained model to get
            # a list of predicted ingredients. Here, we'll use a placeholder.
            st.markdown("Simulating computer vision analysis...")
            
            # Simple simulation: assume certain images contain certain ingredients
            simulated_ingredients = "chicken, onions, tomatoes, peppers"
            
            st.markdown(f"**Simulated CV Result:** The model identified the following ingredients: `{simulated_ingredients}`")

            # Use the simulated ingredients to get recommendations
            if st.button("Find Recipes Based on Image"):
                if recipes_df is not None:
                    recommendations = get_recommendations(simulated_ingredients, tfidf_matrix, recipes_df, tfidf)
                    if not recommendations.empty:
                        st.subheader("Top Recipe Recommendations:")
                        st.dataframe(recommendations[['title', 'ingredients', 'similarity_score']])
                    else:
                        st.warning("No recommendations found based on the image's ingredients. Please try another image or check your data.")
                else:
                    st.warning("Please upload a CSV file first.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
