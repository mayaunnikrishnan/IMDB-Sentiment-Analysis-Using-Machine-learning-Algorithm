import streamlit as st
import spacy
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = {
        "ADJ": wordnet.ADJ,
        "NOUN": wordnet.NOUN,
        "VERB": wordnet.VERB,
        "ADV": wordnet.ADV
    }.get(word.pos_, wordnet.NOUN)
    return tag

def lemmatize_tokens(tokens):
    lemmatized_tokens = [word.lemma_ for word in tokens]
    return lemmatized_tokens

def load_model_and_vectorizer(model_path='sentiment_app/logistic_regression_model.pkl', vectorizer_path='sentiment_app/tfidf_vectorizer.pkl', encoder_path='sentiment_app/label_encoder.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)
    return model, vectorizer, label_encoder

def predict_sentiment(review, model, vectorizer, label_encoder):
    # Tokenize and lemmatize the review using spaCy and NLTK
    doc = nlp(review)
    tokens = [token for token in doc]
    lemmatized_tokens = lemmatize_tokens(tokens)
    lemmatized_review = ' '.join(lemmatized_tokens)

    # Vectorize the lemmatized review
    review_vector = vectorizer.transform([lemmatized_review])

    # Predict sentiment
    prediction = model.predict(review_vector)
    sentiment = label_encoder.inverse_transform(prediction)
    return sentiment[0]

# Load the saved model, vectorizer, and label encoder
model, vectorizer, label_encoder = load_model_and_vectorizer()

# Streamlit interface
st.title("Sentiment Analysis Prediction App")
st.write("Enter a movie review and get the predicted sentiment.")

# Input text
review = st.text_area("Review:")

if st.button("Predict"):
    if review:
        sentiment = predict_sentiment(review, model, vectorizer, label_encoder)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review.")
