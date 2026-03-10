from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df = pd.read_csv('spam_clean.csv', encoding='latin-1')
df["type"] = df["type"].map({'spam':1,'ham':0})

from sklearn.model_selection import train_test_split

df_X_train, df_X_test, y_train, y_test = train_test_split(df['message'], df['type'], test_size=0.25, random_state=42)


def clean_tokenized_sentence(s):
    words = word_tokenize(s)
    cleaned_words = []

    for word in words:
        c_word = word.lower()
        c_word = re.sub(r'[^\w\s]', '', c_word)

        if c_word != '' and c_word not in stop_words:
            cleaned_words.append(c_word)

    return " ".join(cleaned_words)


pipeline = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=clean_tokenized_sentence)),
    ('classifier', MultinomialNB(alpha=1))
])

# 🔥 IMPORTANT: use RAW df_X_train (not cleaned_message column)
pipeline.fit(df_X_train, y_train)

# Predict on RAW df_X_test
y_pred = pipeline.predict(df_X_test)

# Print F1 score
print("F1 Score:", f1_score(y_test, y_pred))

# Save entire pipeline
with open("spam_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline saved successfully.")