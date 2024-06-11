from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Sample data for recommendation system
items = [
    "Item 1 is a great product",
    "Item 2 provides excellent value",
    "Item 3 has received positive reviews",
    "Item 4 is highly recommended",
    "Item 5 is popular among customers"
]

def content_based_recommendation(user_preferences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(items)
    user_vector = tfidf_vectorizer.transform([user_preferences])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    similar_items_indices = similarities.argsort()[0][::-1]
    recommended_items = [items[i] for i in similar_items_indices]
    return recommended_items

def train_predictive_model(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions.tolist()

@app.route('/')
def home():
    return render_template('newindex.html')

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(filtered_text)
    sentiment = 'neutral'
    if ss['compound'] > 0.5:
        sentiment = 'positive'
    elif ss['compound'] < -0.5:
        sentiment = 'negative'
    return {'filtered_words': filtered_words, 'scores': ss, 'sentiment': sentiment}
    

@app.route('/remove-punctuation', methods=['POST'])
def remove_punctuation():
    translator = str.maketrans('', '', string.punctuation)
    filtered_text = text.translate(translator)
    return filtered_text


@app.route('/recommendation', methods=['POST'])
def recommendation():
    data = request.json
    user_preferences = data['preferences']
    recommended_items = content_based_recommendation(user_preferences)
    return jsonify({'recommended_items': recommended_items})

@app.route('/predictive-modeling', methods=['POST'])
def predictive_modeling():
    data = request.json
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    predictions = train_predictive_model(X_train, y_train, X_test)
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
