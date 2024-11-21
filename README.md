# Sentiment Analysis Tool

## 📜 Overview
The **Sentiment Analysis Tool** is a machine learning-based application designed to classify text into **Positive** or **Negative** sentiment categories. It supports real-time sentiment analysis from user inputs, uploaded files, or reviews scraped from URLs. This project demonstrates:
- **Logistic Regression and CNN Models**
- **Flexible Input Handling**
- **Automated Review Scraping**

## 🎥 Demo Video:
...

## 📂 Project Structure
Positive_Negative_Comments/
├── app.py                    # Your main Python file
├── sentiment_model.pkl       # Pre-trained logistic regression model
├── sentiment_cnn_model.h5    # Pre-trained CNN model
├── tokenizer.pkl             # Tokenizer for CNN model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt          # Dependencies for the project
├── templates/                # Folder for HTML files
│   ├── index.html
├── README.md                 # Documentation for the project (to be created)

## 🛠️ Features
1. **Sentiment Classification**:
   - Logistic Regression for fast and interpretable results.
   - CNN for higher accuracy using deep learning.
2. **Flexible Input**:
   - Analyze custom text input.
   - Upload files (e.g., CSV with reviews).
   - Scrape reviews from URLs.
3. **Review Scraper**:
   - Automatically scrape reviews from websites using Selenium.
4. **Text Preprocessing**:
   - Removes stopwords, applies stemming, and cleans text.

## 💻 Requirements
Ensure the following are installed:
- Python 3.8 or higher
- Required libraries (in `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```

### Libraries Included:
- `Flask`
- `joblib`
- `pandas`
- `nltk`
- `tensorflow`
- `selenium`

## 🚀 How to Run
1. Clone this repository:
    ```bash
    git clone https://github.com/LeTuyen2002/Sentiment-Analysis-Tool.git
    cd Sentiment-Analysis-Tool
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask app:
    ```bash
    python app.py
    ```

4. Access the application:
   - Open your browser and navigate to `http://127.0.0.1:5000`.

## 📊 Output
When you run the project:
- The system classifies text or reviews into positive and negative categories.
- Displays up to 10 positive and 10 negative reviews.
- Outputs the results in JSON format for easy integration.

## 📧 Contact
- **Author**: Le Tuyen
- **Email**: [trungtuyenlevn@gmail.com](mailto:trungtuyenlevn@gmail.com)
- **GitHub**: [LeTuyen2002](https://github.com/LeTuyen2002)

## 🤝 Contributions
Contributions are welcome! Feel free to:
- Fork this repository
- Open an issue
- Submit a pull request
