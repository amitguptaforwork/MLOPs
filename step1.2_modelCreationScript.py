# %%
# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# %%
df = pd.read_csv('spam_clean.csv', encoding='latin-1')
df.head()

# %% [markdown]
# Data preprocessing

# %%
# Libraries for text processing
import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def clean_tokenized_sentence(s):
    """Performs basic cleaning of a tokenized sentence"""
    cleaned_s = ""  # Create empty string to store processed sentence.
    words = nltk.word_tokenize(s)
    for word in words:
        # Convert to lowercase #
        c_word = word.lower()
        # Remove punctuations #
        c_word = re.sub(r'[^\w\s]', '', c_word)
        # Remove stopwords #
        if c_word != '' and c_word not in stopwords.words('english'):
            cleaned_s = cleaned_s + " " + c_word    # Append processed words to new list.
    return(cleaned_s.strip())

# %% [markdown]
# Calling the clean_tokenized_sentence(s) function onto each text in our dataset using the apply method, and storing them in a new column, cleaned_message

# %%
df["cleaned_message"] = df["message"].apply(clean_tokenized_sentence)
df.head(10)

# %%
df["type"] = df["type"].map({'spam':1,'ham':0})

from sklearn.model_selection import train_test_split

df_X_train, df_X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['type'], test_size=0.25, random_state=42)
print([np.shape(df_X_train), np.shape(df_X_test)])

# %%
from sklearn import feature_extraction, naive_bayes, metrics

#Count Vectorizer
f = feature_extraction.text.CountVectorizer()

X_train = f.fit_transform(df_X_train)
X_test = f.transform(df_X_test)

print(X_train.shape,X_test.shape)

# %% [markdown]
# Taking different values of Laplace Smoothing constant

# %%
params = {
        'alpha':[0.01, 0.1, 1, 10]
        }

# %% [markdown]
# We plug in the following values into our `GridSearchCV()` function to get the results:-
# - Multinomial NB classifier,
# - dictionary containing the range of values we wish to try for our hyperparameter,
# - scoring metric
# - number of folds for the cross validation set
# 
# 
# Since data is Imbalanced, we use F-1 score as evaluation metrics

# %%
# Multinomial NB

from sklearn.model_selection import GridSearchCV

mnb = naive_bayes.MultinomialNB()
clf = GridSearchCV(mnb, params, scoring = "f1", cv=3)

clf.fit(X_train, y_train)

res = clf.cv_results_

for i in range(len(res["params"])):
  print(f"Parameters:{res['params'][i]} Mean_score: {res['mean_test_score'][i]} Rank: {res['rank_test_score'][i]}")


# %% [markdown]
# As you can see, we get the best performance when  α=1 , with F1- score of 0.9,
# 
# Now implementing this Naive Bayes on test Data

# %%
mnb = naive_bayes.MultinomialNB(alpha=1)
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

print(metrics.f1_score(y_test,y_pred))

# %% [markdown]
# We see how Multinomial Naive Bayes achieved f-1 score of 0.92 even when data is imbalanced.
# 
# showing how Mulitnomial Naive Bayes is not much effected by the class priors

# %%
import pickle

# Save model
with open("spam_model.pkl", "wb") as f_model:
    pickle.dump(mnb, f_model)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f_vec:
    pickle.dump(f, f_vec)

print("Model and vectorizer saved successfully.")

# %%
message1 = "I have a free ticket for you, claim now!"
message2 = "Hi Amit, how are you doing?"
message = message1
# Step 1: Clean
cleaned = clean_tokenized_sentence(message)

# Step 2: Vectorize
vectorized = f.transform([cleaned])

# Step 3: Predict
prediction = mnb.predict(vectorized)[0]

result = "spam" if prediction == 1 else "ham"

print(f"original_message:{message},\nprediction: {result}")




