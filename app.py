import re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import spacy
import streamlit as st
from spacy.lang.en.stop_words import STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "news_classifier_lr.pkl"
CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
CLASS_ICONS = {"World": "🌍", "Sports": "⚽", "Business": "💼", "Sci/Tech": "🧪"}
CLASS_COLORS = {
    "World": "#FF5733",
    "Sports": "#33C3FF",
    "Business": "#FFC300",
    "Sci/Tech": "#33AA57",
}

st.set_page_config(page_title="News Category Classifier", page_icon="📰", layout="wide")


@st.cache_resource
def load_resources():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modèle requis introuvable : {MODEL_PATH.name}")
    classifier = joblib.load(MODEL_PATH)
    expected_classes = set(CLASS_NAMES)
    actual_classes = {int(label) for label in classifier.classes_}
    if actual_classes != expected_classes:
        raise ValueError(f"Classes du modèle inattendues : {sorted(actual_classes)}")
    language_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return classifier, language_model


def preprocess(text):
    text = re.sub(r"[^a-z\s]", "", str(text).casefold())
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if token.text not in STOP_WORDS)


try:
    model, nlp = load_resources()
except Exception as exc:
    st.error("Le classificateur n'a pas pu être chargé. Vérifiez les dépendances, le modèle et le modèle spaCy anglais.")
    st.exception(exc)
    st.stop()


st.title("📰 Classificateur de catégories d’actualités")
st.write("Saisissez un titre d’article en anglais pour estimer sa catégorie.")

user_input = st.text_area(
    "Titre de l’article",
    height=100,
    max_chars=500,
    placeholder="Ex. UN announces new climate agreement to combat global warming",
)

if st.button("Classifier le titre", type="primary", width="stretch"):
    if not user_input.strip():
        st.warning("Veuillez entrer un titre.")
    else:
        try:
            cleaned_text = preprocess(user_input)
            if not cleaned_text:
                st.warning("Le titre ne contient aucun mot exploitable après le nettoyage.")
            else:
                probabilities = model.predict_proba([cleaned_text])[0]
                model_classes = [int(label) for label in model.classes_]
                probability_by_category = {
                    CLASS_NAMES[label]: float(probability)
                    for label, probability in zip(model_classes, probabilities)
                }
                best_position = max(range(len(probabilities)), key=probabilities.__getitem__)
                predicted_category = CLASS_NAMES[model_classes[best_position]]
                categories = [CLASS_NAMES[label] for label in model_classes]

                st.subheader("Probabilités estimées")
                columns = st.columns(len(categories))
                for column, category in zip(columns, categories):
                    column.metric(
                        f"{CLASS_ICONS[category]} {category}",
                        f"{probability_by_category[category]:.2%}",
                    )

                chart_data = pd.DataFrame(
                    {
                        "Category": categories,
                        "Probability (%)": [100 * probability_by_category[name] for name in categories],
                    }
                )
                figure = px.bar(
                    chart_data,
                    x="Category",
                    y="Probability (%)",
                    text="Probability (%)",
                    color="Category",
                    color_discrete_map=CLASS_COLORS,
                    category_orders={"Category": categories},
                    range_y=[0, 100],
                )
                figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                figure.update_layout(
                    showlegend=False,
                    yaxis_title="Probabilité estimée (%)",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(figure, width="stretch")

                st.subheader("Catégorie prédite")
                st.success(f"{CLASS_ICONS[predicted_category]} {predicted_category}")
                with st.expander("Voir le texte prétraité"):
                    st.code(cleaned_text, language="text")
        except Exception as exc:
            st.error("La classification de ce titre a échoué.")
            st.exception(exc)

st.divider()
st.caption(
    "Le modèle Logistic Regression a obtenu 85,1 % d’exactitude sur les 7 600 titres de test fournis. "
    "Les probabilités affichées ne sont pas calibrées. Le modèle est entraîné sur des titres en anglais."
)
