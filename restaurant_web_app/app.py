import streamlit as st, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import torch
from PIL import Image
from torchvision import transforms, models

st.set_page_config(page_title="FlavorLens: Recipes at Your Fingertips", layout="wide")
st.title("👁️‍🗨️ FlavorLens: Recipes at Your Fingertips")
st.caption("Find, create, and explore recipes by text... or snap a food photo for suggestions!")

@st.cache_data
def load_data():
    df = pd.read_csv("recipes_dataset.csv")
    df = df[df['Ingredients_CleanedAuto_CleanedAuto'].notna() & df['RecipeName_CleanedAuto_CleanedAuto'].notna()]
    rec = df[['RecipeName_CleanedAuto_CleanedAuto','Ingredients_CleanedAuto_CleanedAuto','Cuisine_CleanedAuto_CleanedAuto','Course_CleanedAuto_CleanedAuto']].fillna("")
    rec.columns=['title','ingredients','cuisine','course']
    return rec

def get_recommendations(user_query, tfidf_matrix, tfidf, rec_df):
    try:
        vec = tfidf.transform([user_query])
        sims = cosine_similarity(vec, tfidf_matrix).flatten()
        top = sims.argsort()[-5:][::-1]
        res = rec_df.iloc[top].copy()
        res['similarity_score'] = sims[top]
        return res
    except:
        return pd.DataFrame()

rec_df = load_data()
stopwords = set(ENGLISH_STOP_WORDS)
tfidf = TfidfVectorizer(stop_words=stopwords)
tfidf_matrix = tfidf.fit_transform(rec_df['ingredients'])

col1, col2 = st.columns([2,1])
with col1:
    user_input = st.text_area("🔍 Ingredients, flavors, or dish:", "chicken, garlic, herb")
    if st.button("Find Recipes"):
        if user_input.strip():
            r = get_recommendations(user_input, tfidf_matrix, tfidf, rec_df)
            if not r.empty:
                st.dataframe(r[["title","ingredients","cuisine","similarity_score"]],use_container_width=True,hide_index=True)
            else: st.warning("No good matches found. Try more or different words.")
        else: st.warning("Type some ingredients or a recipe idea.")
with col2:
    st.image("https://placehold.co/400x320/e0f2fe/1f2937?text=Snap+or+Type+for+Recipes",use_column_width=True)

st.markdown("---")
st.subheader("📷 Snap2Recipe: AI-powered food photo ingredient detector")
with st.expander("Upload a food photo to suggest ingredients & recipes!"):
    f = st.file_uploader("Choose a food photo...",type=["jpg","jpeg","png"])
    if f:
        img = Image.open(f).convert("RGB")
        st.image(img,caption="Your uploaded image",use_column_width=True)
        model = models.resnet18(weights="IMAGENET1K_V1"); model.eval()
        trf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        timg = trf(img).unsqueeze(0)
        with torch.no_grad(): out = model(timg)
        _, idx = torch.topk(out,5)
        @st.cache_data
        def get_labels():
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            return pd.read_csv(url,header=None)[0].tolist()
        labels = get_labels(); preds = [labels[i] for i in idx[0]]
        foodwords=['dish','pizza','salad','meat','sandwich','rice','noodle','bread','vegetable','cake','chocolate','soup','egg','burger']
        fp = [p for p in preds if any(x in p for x in foodwords)]
        uq = ', '.join(fp) if fp else ', '.join(preds[:3])
        st.markdown("#### Detected: "+", ".join(fp if fp else preds[:3]))
        if st.button("🔍 Find Recipes from Image Ingredients"):
            r = get_recommendations(uq,tfidf_matrix,tfidf,rec_df)
            if not r.empty: st.dataframe(r[["title","ingredients","cuisine","similarity_score"]],use_container_width=True,hide_index=True)
            else: st.warning("No matching recipes found.")

st.markdown("---")
st.subheader("🏢 Simple Automated Recipe Generator")
with st.expander("Give ingredients, get a creative AI recipe!"):
    ci = st.text_area("Ingredients for new recipe:", "tofu, spinach, chili flakes")
    if st.button("Generate New Recipe"):
        ing = [x.strip().capitalize() for x in ci.split(',') if x.strip()]
        if ing:
            title="Fusion "+" and ".join(ing)+" Surprise"
            st.success("Your AI-generated recipe:")
            st.subheader(title)
            st.markdown(f"**Ingredients:** {', '.join(ing)}")
            st.text("1. Combine all ingredients.\n2. Cook and season to taste.\n3. Serve and enjoy!")
        else: st.warning("Please add some ingredients.")

st.markdown("---")
st.subheader("🔧 Model Hyperparameter Tuning Demo")
with st.expander("See ML tuning using real recipe data:"):
    if st.button("Start Hyperparameter Tuning"):
        with st.spinner('Tuning ML model...'):
            s=rec_df.sample(n=100,random_state=42) if len(rec_df)>100 else rec_df
            X,y=s['ingredients'],s['cuisine']
            pipeline=Pipeline([('tfidf',TfidfVectorizer(stop_words=stopwords)),('clf',SVC())])
            grid=GridSearchCV(pipeline,{'tfidf__ngram_range':[(1,1),(1,2)],'clf__C':[0.1,1,10,100],'clf__kernel':['linear','rbf']},cv=2,verbose=1,n_jobs=-1)
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)
            grid.fit(X_tr,y_tr)
            st.success("Tuning complete!");st.write(f"**Best CV Score:** `{grid.best_score_:.4f}`");st.write(f"**Best Hyperparameters:** `{grid.best_params_}`")
            e=X_te.sample(n=min(3,len(X_te)),random_state=42)
            preds=grid.best_estimator_.predict(e)
            df=pd.DataFrame({'Ingredients':e.values,'Predicted Cuisine':preds})
            st.dataframe(df,use_container_width=True,hide_index=True)
