# News Headline Category Classifier

This project is a Natural Language Processing application that classifies news headlines into thematic categories.

The Streamlit interface accepts an English news headline, preprocesses it with spaCy, classifies it with the saved Logistic Regression pipeline, and displays class scores.

## Categories

The model predicts one of the following categories:

- `World`
- `Sports`
- `Business`
- `Sci/Tech`

## Features

- text preprocessing with lowercasing, cleaning, lemmatization and stop-word removal;
- headline classification using TF-IDF and Logistic Regression;
- probability display for each category;
- interactive Plotly bar chart;
- Streamlit user interface.

## Project Structure

```text
app.py                    Streamlit application
model.ipynb               Notebook used for model training and experiments
train.csv                 Training dataset
test.csv                  Test dataset
news_classifier_lr.pkl    Active TF-IDF + Logistic Regression pipeline
news_classifier.pkl       Naive Bayes baseline for comparison
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

The app loads `news_classifier_lr.pkl` from its own project directory. Its class
mapping is checked at startup. The model is a scikit-learn pipeline, so its
TF-IDF vectorizer is included in the model file; no separate vectorizer file is
needed.

## Model Evaluation

The supplied `test.csv` contains 7,600 headlines, balanced across the four
classes. In the audit run, Logistic Regression reached about 85.1% accuracy,
compared with about 83.5% for the Naive Bayes baseline. The app uses Logistic
Regression. These scores describe the supplied test set; the displayed
per-class scores are not calibrated probabilities.

## Technologies

```text
Python | Streamlit | spaCy | scikit-learn | TF-IDF | Plotly | pandas
```

