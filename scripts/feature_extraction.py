import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

df = pd.read_csv('../data/imdb_cleaned.csv')
X_text = df['cleaned_review'].values
y = df['sentiment'].map({'negative': 0, 'positive': 1}).values

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, stratify=y, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

with open('../data_processed/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

np.save('../data_processed/X_train.npy', X_train.toarray())
np.save('../data_processed/X_test.npy', X_test.toarray())
np.save('../data_processed/y_train.npy', y_train)
np.save('../data_processed/y_test.npy', y_test)

print('Feature extraction complete and saved in data_processed folder.')




