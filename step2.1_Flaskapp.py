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

# Load saved model and vectorizer
with open("spam_model.pkl", "rb") as f_model:
    model = pickle.load(f_model)

with open("vectorizer.pkl", "rb") as f_vec:
    vectorizer = pickle.load(f_vec)

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


@app.route("/")
def home():
    return "Spam Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Please provide 'message' in JSON"}), 400

    message = data["message"]

    # Step 1: Clean
    cleaned = clean_tokenized_sentence(message)

    # Step 2: Vectorize
    vectorized = vectorizer.transform([cleaned])

    # Step 3: Predict
    prediction = model.predict(vectorized)[0]

    result = "spam" if prediction == 1 else "ham"

    return jsonify({
        "original_message": message,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)