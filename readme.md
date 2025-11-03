# FlavorLens: Recipes at Your Fingertips

### AI-Powered Recipe Discovery, Ingredient Detection & ML Tuning 

**Author:** Mauli Patel
**Course:** MSc Big Data Analytics (Semester 3 NLP Project)

---

## Project Overview

**FlavorLens** is an AI-driven recipe recommender that enables users to explore and generate recipes through both **text** and **image-based inputs**.
Users can either type in ingredients to discover new dishes or upload a food image to identify its ingredients automatically.

The project integrates **text mining**, **image classification**, and **machine learning model tuning** into a unified **Streamlit** interface.
This system is part of a research initiative studying how **multimodal AI models** improve personalization and recommendation systems in the culinary domain.

---

## Key Features

* **Text-Based Recipe Search**
  Type in ingredients or flavor profiles (e.g., “chicken, garlic, herb”) to retrieve similar recipes using **TF-IDF vectorization** and **cosine similarity**.

* **Image-Based Ingredient Detection (Snap2Recipe)**
  Upload a food image and let a pre-trained **ResNet-18 CNN model** detect the most probable ingredients or dish type.

* **AI Recipe Generator**
  Generates new recipe ideas dynamically from an ingredient list using simple rule-based and NLP-enhanced logic.

* **Machine Learning Model Tuning**
  Demonstrates **SVM hyperparameter tuning** using **GridSearchCV** to classify cuisines and optimize model accuracy.

* **Streamlit-Powered Dashboard**
  Intuitive interface for visualization, exploration, and experimentation with caching for faster data handling.

---

## Technologies Used

| Category         | Tools and Libraries                      |
| ---------------- | ---------------------------------------- |
| Framework        | Streamlit                                |
| Data Handling    | Pandas                                   |
| NLP & ML         | Scikit-learn (TF-IDF, SVM, GridSearchCV) |
| Computer Vision  | PyTorch, torchvision (ResNet-18)         |
| Image Processing | Pillow                                   |
| Language         | Python 3.10+                             |

---

## Repository Structure

```
FlavorLens/
│
├── app.py                  # Main Streamlit application
├── models/
│   ├── resnet_model.pth    # Pre-trained CNN model
│   └── vectorizer.pkl      # Saved TF-IDF vectorizer
├── data/
│   ├── recipes.csv         # Recipe dataset
│   └── sample_images/      # Demo food images
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── utils/
    ├── nlp_utils.py        # Text vectorization functions
    ├── vision_utils.py     # Image detection utilities
    └── model_tuning.py     # ML optimization scripts
```

---

## Setup Instructions

1. **Clone this repository**

   ```bash
   git clone https://github.com/itsmemauliii/nlp-project-restaurant.git
   cd nlp-project-restaurant
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Optional:**
   Replace the dataset (`data/recipes.csv`) with your own recipe dataset for custom results.

---

## Future Enhancements

* Integration of a **Generative AI recipe writer (LLM-based)**
* Expansion of the dataset for multilingual and cultural cuisines
* Deployment on **Streamlit Cloud** or **Hugging Face Spaces**
* User profile personalization and ingredient substitution recommendations

---

## Acknowledgements

Developed by **Mauli Patel** as part of an academic project under the MSc Big Data Analytics program.
Inspired by the intersection of AI, creativity, and everyday cooking experiences.

---
