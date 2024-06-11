from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string

app = Flask(__name__)

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

def analysis_sentiment(text):
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

def remove_punc(text):
    translator = str.maketrans('', '', string.punctuation)
    filtered_text = text.translate(translator)
    return filtered_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    data = request.json
    text = data['text']
    result = analysis_sentiment(text)
    return jsonify(result)

@app.route('/remove-punctuation', methods=['POST'])
def remove_punctuation():
    data = request.json
    text = data['text']
    result = remove_punc(text)
    return jsonify({'filtered_text': result})

if __name__ == '__main__':
    app.run(debug=True)
