# News Headline Category Classifier

This project is a Natural Language Processing application that classifies news headlines into thematic categories.

The Streamlit interface allows the user to enter a news title, preprocess it with spaCy, classify it with a trained machine learning model and visualize class probabilities.

## Categories

The model predicts one of the following categories:

- `World`
- `Sports`
- `Business`
- `Sci/Tech`

## Features

- text preprocessing with lowercasing, cleaning, lemmatization and stop-word removal;
- headline classification using a trained ML pipeline;
- probability display for each category;
- interactive Plotly bar chart;
- Streamlit user interface.

## Project Structure

```text
app.py                    Streamlit application
model.ipynb               Notebook used for model training and experiments
train.csv                 Training dataset
test.csv                  Test dataset
news_classifier.pkl       Trained classifier
news_classifier_lr.pkl    Logistic Regression classifier
tfidf_vectorizer.pkl      TF-IDF vectorizer
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the Application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Technologies

```text
Python | Streamlit | spaCy | scikit-learn | TF-IDF | Plotly | pandas
```

