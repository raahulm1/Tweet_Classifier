import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords
stopwords = nltk.corpus.stopwords.words("english")

# Extending the stopwords
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

def preprocess(tweets):  
    processed_tweets = []
    
    for tweet in tweets:
        # Step 1: Remove mentions (@name)
        tweet = re.sub(r'@[\w\-]+', '', tweet)

        # Step 2: Remove links (http[s]://...)
        tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                        r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)

        # Step 3: Remove punctuation and numbers
        tweet = re.sub(r'[^a-zA-Z\s]', ' ', tweet)

        # Step 4: Remove extra spaces
        tweet = re.sub(r'\s+', ' ', tweet).strip()

        # Step 5: Convert to lowercase
        tweet = tweet.lower()

        # Step 6: Tokenizing
        tokenized_tweet = tweet.split()

        # Step 7: Removal of stopwords
        tokenized_tweet = [word for word in tokenized_tweet if word not in stopwords]

        # Step 8: Stemming
        tokenized_tweet = [stemmer.stem(word) for word in tokenized_tweet]

        # Join tokens back to string
        processed_tweets.append(' '.join(tokenized_tweet))

    return processed_tweets

# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with your dataset file path

# Preprocess the tweets
df['processed_tweets'] = preprocess(df['tweet'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
tfidf = tfidf_vectorizer.fit_transform(df['processed_tweets'])

# Labels (classes) for training
X = tfidf
y = df['class'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer have been saved.")
