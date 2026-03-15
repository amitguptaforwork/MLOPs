from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download only once (optional in production)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

stopwords_set = set(stopwords.words('english')  )
def clean_tokenized_sentence(s):
    """Performs basic cleaning of a sentence"""
    cleaned_s = ""
    words = word_tokenize(s)

    for word in words:
        c_word = word.lower()
        c_word = re.sub(r'[^\w\s]', '', c_word)
        if c_word != '' and c_word not in stopwords_set:
            cleaned_s = cleaned_s + " " + c_word

    return cleaned_s.strip()

# ---- ADD THIS ----
#To avoid error AttributeError: Can't get attribute 'clean_tokenized_sentence'
import __main__
__main__.clean_tokenized_sentence = clean_tokenized_sentence
# ------------------

# Load saved model pipeline 
with open("spam_pipeline.pkl", "rb") as f:
    model = pickle.load(f)
    
@app.route("/")
def home():
    return "Spam Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Please provide 'message'"}), 400

    message = data["message"]

    prediction = model.predict([message])[0]

    result = "spam" if prediction == 1 else "ham"

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)