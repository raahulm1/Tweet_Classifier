from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import re

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)
CORS(app)

# Sentiment analysis and tag counting logic
def count_tags(tweet_c):
    space_pattern = r'\s+'
    giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = r'@[\w\-]+'
    hashtag_regex = r'#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweet_c)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))

sentiment_analyzer = VS()

def sentiment_analysis(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    twitter_objs = count_tags(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[0], twitter_objs[1], twitter_objs[2]]
    return features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data.get('tweet', '')

    # Process sentiment analysis
    sentiment_features = sentiment_analysis(tweet)

    # Vectorize the tweet using the loaded TF-IDF vectorizer
    tweet_tfidf = tfidf_vectorizer.transform([tweet])

    # Only use TF-IDF features for prediction, without combining sentiment features
    tweet_features = tweet_tfidf.toarray()  # Use only the TF-IDF features

    # Make prediction
    prediction = model.predict(tweet_features)

    # Map prediction to class label
    class_labels = {0: "Offensive", 1: "Hate", 2: "Neutral"}
    result = class_labels[prediction[0]]

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(port=8000)
