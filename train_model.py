# train_model_bq.py
import pandas as pd
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Make sure your GOOGLE_APPLICATION_CREDENTIALS env variable is set
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "path/to/your/service_account.json"

PROJECT_ID = "encoded-pointer-464118-f6"
DATASET = "news_dataset"
TABLE = "news_data"  # BigQuery table with label column

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

query = f"""
SELECT title, subject, date, label
FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
"""

df = client.query(query).to_dataframe()

# Combine title + subject as feature text
df['full_text'] = df['title'] + " " + df['subject']

X = df['full_text']
y = df['label']  # 'fake' or 'real'

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vect = vectorizer.fit_transform(X)

# Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_vect, y)

# Save model and vectorizer
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training complete. Models saved as 'logreg_model.pkl' and 'vectorizer.pkl'")