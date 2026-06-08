import streamlit as st
import joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import pandas as pd
import plotly.express as px

model = joblib.load("news_classifier.pkl") 
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS]
    return " ".join(tokens)


st.markdown("""
<style>
/* Supprimer padding Streamlit par défaut et fond gris clair */
.css-18e3th9 {
    padding-top: 0rem;
    background-color: #f8f9fa;
}

/* Police */
h1, h2, h3, h4 {
    font-family: 'Arial', sans-serif;
}

/* Bouton personnalisé + hover */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    height: 50px;
    width: 100%;
    transition: background-color 0.3s ease, transform 0.2s ease;
}
.stButton>button:hover {
    background-color: #FF1E1E;
    transform: scale(1.05);
}

/* Cartes des probabilités 3D effet + icônes */
.card {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    margin: 8px;
    text-align: center;
    font-weight: bold;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}
.card h3 {
    margin-bottom: 12px;
    font-size: 22px;
}
.card p {
    font-size: 20px;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="📰 News Category Classifier", page_icon="📰", layout="wide")
st.title("📰 News Category Classifier")
st.markdown("**Entrez le titre d'un article pour prédire sa catégorie automatiquement.**")

user_input = st.text_area(
    "✏️ Titre de l'article :", 
    height=100, 
    placeholder="Ex : UN announces new climate agreement to combat global warming"
)

if st.button("🔍 Classifier l'article"):
    if not user_input.strip():
        st.warning("⚠️ Veuillez entrer un titre d'article.")
    else:
        cleaned_text = preprocess(user_input)

        proba = model.predict_proba([cleaned_text])[0]
        predicted_index = model.predict([cleaned_text])[0]
        categories = ["World", "Sports", "Business", "Sci/Tech"]
        icons = ["🌍", "⚽", "💼", "🧪"]
        colors = ["#FF5733", "#33C3FF", "#FFC300", "#33FF57"]
        proba_dict = {cat: round(p*100, 2) for cat, p in zip(categories, proba)}


        st.subheader("📊 Probabilités par catégorie")
        cols = st.columns(len(categories))
        for col, cat, color, icon in zip(cols, categories, colors, icons):
            col.markdown(f"""
            <div class="card" style="background-color:{color}; color:white;">
                <h3>{icon} {cat}</h3>
                <p>{proba_dict[cat]}%</p>
            </div>
            """, unsafe_allow_html=True)


        st.subheader("📈 Visualisation interactive")
        df_plot = pd.DataFrame({"Catégorie": list(proba_dict.keys()), "Probabilité": list(proba_dict.values())})
        fig = px.bar(df_plot, x="Catégorie", y="Probabilité",
                     text="Probabilité", color="Catégorie",
                     color_discrete_sequence=colors, range_y=[0,100])
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', 
                          xaxis_title="", yaxis_title="%", margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("🧠 Catégorie prédite")
        st.markdown(f"<h2 style='color:#FF4B4B'>👉 {categories[predicted_index-1]}</h2>", unsafe_allow_html=True)

        st.subheader("💡 Texte prétraité")
        st.code(cleaned_text, language="text")


















