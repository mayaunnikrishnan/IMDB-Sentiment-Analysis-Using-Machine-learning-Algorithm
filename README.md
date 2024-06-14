# IMDB-Sentiment-Analysis-Using-Machine-learning-Algorithm




## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Text Preprocessing Techniques](#text-preprocessing-techniques)
- [Tokenization and Lemmatization](#tokenization-and-lemmatization)
- [Data Transformation](#data-transformation)
- [Model Training](#model-training)
- [Deployment](#deployment)
        - You can access the live demo of the application [here](https://sentiment-maqh5pxtrod9omv5gf9dhy.streamlit.app/).
- [Files](#files)
- [Usage](#usage)

## Overview
This project implements sentiment analysis using a machine learning model trained on the IMDb dataset. The goal is to predict the sentiment (positive or negative) of movie reviews.

## Dataset
The IMDb dataset consists of movie reviews labeled with sentiment (positive or negative).

## Text Preprocessing Techniques
- **Lowercasing**: Convert text to lowercase to ensure consistency.
- **Removal of HTML tags and URLs**: Clean text by removing HTML tags and URLs.
- **Removal of punctuation**: Eliminate punctuation marks from the text.
- **Stop words removal**: Exclude common stop words to focus on meaningful words.

## Tokenization and Lemmatization
- **Tokenization**: Utilized spaCy for tokenization of text data.
- **Lemmatization**: Perform lemmatization using spaCy with part-of-speech tagging to reduce words to their base form.

## Data Transformation
- **Label Encoding**: Encode sentiment labels ('positive' and 'negative') into numeric format for model training.
- **TF-IDF Vectorization**: Convert text data into numerical TF-IDF (Term Frequency-Inverse Document Frequency) vectors to represent the reviews.

## Model Training
- **Model**: Logistic Regression
- **Accuracy**: Achieved 87% accuracy on the test dataset.
- **Files Saved**: Using Joblib, the following files were saved:
  - `logistic_regression_model.pkl`: Trained logistic regression model.
  - `tfidf_vectorizer.pkl`: TF-IDF vectorizer used for text vectorization.
  - `label_encoder.pkl`: Label encoder for sentiment classes.

## Deployment
A sentiment prediction app for movie reviews was created and deployed using Streamlit. You can access the app using the following link: [Sentiment Prediction App](https://sentiment-maqh5pxtrod9omv5gf9dhy.streamlit.app/).
## Files
- `logistic_regression_model.pkl`: Trained logistic regression model.
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
- `label_encoder.pkl`: Encoded labels for sentiment classes.

## Usage
To use the sentiment prediction app:
1. Ensure Python and necessary libraries are installed (`pip install -r requirements.txt`).
2. Run the Streamlit app: `streamlit run app.py`.
3. Input a movie review text and get the predicted sentiment.

