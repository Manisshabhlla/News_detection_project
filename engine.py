from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from utils import load_table, preprocess_text
from lsh import build_lsh, query_lsh

# Load Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def train_model():
    df = load_table("news_data")  # has title, text, label
    df = preprocess_text(df, text_col="title")
    
    X_texts = df["title"].tolist()
    y = df["label"].apply(lambda x: 1 if x.lower()=="fake" else 0).values
    
    embeddings = embedding_model.encode(X_texts, convert_to_tensor=True)
    
    # Build LSH for fast similarity search
    lsh, minhashes = build_lsh(X_texts)
    
    # Split for evaluation
    train_idx, test_idx = train_test_split(np.arange(len(X_texts)), test_size=0.2, random_state=42)
    
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Store dataset stats
    metrics = {
        "fake_pct": np.mean(y)*100,
        "real_pct": 100 - np.mean(y)
    }
    
    return embeddings, y, lsh, minhashes, metrics, df

def predict(text, embeddings, y, lsh, minhashes, df):
    # Semantic embedding
    text_emb = embedding_model.encode([text], convert_to_tensor=True)
    
    # Cosine similarity with dataset
    sims = util.cos_sim(text_emb, embeddings)[0].cpu().numpy()
    top_idx = sims.argmax()
    
    label = df.iloc[top_idx]["label"]
    confidence = sims[top_idx]  # similarity as proxy confidence
    
    # LSH for similar articles
    similar_idx = query_lsh(lsh, minhashes, text)
    similar_texts = df.iloc[similar_idx]["title"].tolist()
    
    return label, float(confidence), similar_texts