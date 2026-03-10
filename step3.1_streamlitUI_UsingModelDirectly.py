import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------------------
# Download NLTK resources (only if needed)
# -----------------------------------------
# For a deployed app (like Streamlit), you should avoid downloading NLTK resources 
# every time the script runs. Instead, check if they exist and download only if missing.
resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords")
]

for path, name in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)



# -----------------------------------------
# Cache Stopwords (Light Data)
# -----------------------------------------
@st.cache_data
def load_stopwords():
    return set(stopwords.words('english'))


stopwords_set = load_stopwords()


# -----------------------------------------
# Text Cleaning Function
# -----------------------------------------
def clean_tokenized_sentence(s):
    words = word_tokenize(s)
    cleaned_words = []

    for word in words:
        c_word = word.lower()
        c_word = re.sub(r'[^\w\s]', '', c_word)

        if c_word != '' and c_word not in stopwords_set:
            cleaned_words.append(c_word)

    return " ".join(cleaned_words)


# -----------------------------------------
# Cache Model (Heavy Resource)
# -----------------------------------------
@st.cache_resource
def load_model():
    with open("spam_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.title("📧 Spam Detection App")

user_input = st.text_area("Enter your message")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        result = "Spam 🚨" if prediction == 1 else "Ham ✅"

        st.subheader("Prediction:")
        st.success(result)