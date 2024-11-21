from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time

# Download stopwords if not already available
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load Logistic Regression model and TF-IDF vectorizer
model_logistic = joblib.load(r'D:\Computer Vision\Positive_Negative_Comments\sentiment_model.pkl')
tfidf = joblib.load(r'D:\Computer Vision\Positive_Negative_Comments\tfidf_vectorizer.pkl')

# Load CNN model and tokenizer
model_cnn = tf.keras.models.load_model(r'D:\Computer Vision\Positive_Negative_Comments\sentiment_cnn_model.h5')
tokenizer = joblib.load(r'D:\Computer Vision\Positive_Negative_Comments\tokenizer.pkl')

# Initialize stop words and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Path to ChromeDriver
driver_path = r'D:\Computer Vision\chromedriver-win64\chromedriver-win64\chromedriver.exe'

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

# Function to get reviews from URL using Selenium
def get_reviews_from_url(url):
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)
    driver.get(url)
    time.sleep(10)  # Wait to ensure the page is fully loaded

    reviews = []
    review_elements = driver.find_elements(By.CLASS_NAME, 'ipc-html-content-inner-div')
    for review_element in review_elements:
        reviews.append(review_element.text)
        if len(reviews) >= 20:  # Limit to 20 reviews
            break

    driver.quit()
    return reviews

# Home route to display HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '')
    file = request.files.get('file')

    reviews = []

    if url:
        # Get reviews from URL
        reviews = get_reviews_from_url(url)
        if not reviews:
            return jsonify({"error": "No reviews found at the provided URL."})
    elif file:
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Read CSV file into DataFrame
        df = pd.read_csv(file, encoding='utf-8')

        # Check if 'review' column exists in the CSV file
        if 'review' not in df.columns:
            return jsonify({"error": "CSV file must contain a 'review' column"})

        reviews = df['review'].tolist()

    if not reviews:
        return jsonify({"error": "No reviews to analyze."})

    # Clean the reviews
    cleaned_reviews = [clean_text(review) for review in reviews]

    positive_reviews = []
    negative_reviews = []
    model_type = request.form.get('model_type', 'logistic')

    if model_type == 'logistic':
        transformed_text = tfidf.transform(cleaned_reviews)
        probs = model_logistic.predict_proba(transformed_text)

        for review, prob in zip(reviews, probs):
            sentiment = "Positive" if prob[1] > 0.5 else "Negative"
            if sentiment == "Positive":
                positive_reviews.append(review)
            else:
                negative_reviews.append(review)

    elif model_type == 'cnn':
        sequences = tokenizer.texts_to_sequences(cleaned_reviews)
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        predictions = model_cnn.predict(padded_sequences)

        for review, pred in zip(reviews, predictions):
            if pred[0] > 0.5:
                positive_reviews.append(review)
            else:
                negative_reviews.append(review)

    # Limit to 10 reviews for display
    positive_reviews = positive_reviews[:10]
    negative_reviews = negative_reviews[:10]

    return jsonify({
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews
    })

if __name__ == "__main__":
    app.run(debug=True)
