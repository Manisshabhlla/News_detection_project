from google.cloud import bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------
# 1. Load data from BigQuery
# ---------------------------
PROJECT_ID = "encoded-pointer-464118-f6"
DATASET = "news_dataset"

client = bigquery.Client()

query = f"""
SELECT * FROM `{PROJECT_ID}.{DATASET}.news_data`
"""
df = client.query(query).to_dataframe()

print("Dataset loaded. Rows:", len(df))
print(df.head())

# ---------------------------
# 2. Preprocess text
# ---------------------------
# Combine title + text for more context
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Optional: lowercase
df['full_text'] = df['full_text'].str.lower().str.strip()

# ---------------------------
# 3. Split dataset
# ---------------------------
X = df['full_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train rows:", len(X_train), "Test rows:", len(X_test))

# ---------------------------
# 4. Vectorize text
# ---------------------------
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# 5. Train classifier
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------------------
# 6. Evaluate
# ---------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 7. Save predictions back to BigQuery (optional)
# ---------------------------
df_test = pd.DataFrame({'full_text': X_test, 'true_label': y_test, 'predicted_label': y_pred})
df_test.to_gbq(f'{DATASET}.news_predictions', project_id=PROJECT_ID, if_exists='replace')
print("✅ Predictions saved to BigQuery table: news_predictions")