import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from api import fetch_news
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# LOAD DATA
# -----------------------
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
TABLE = os.getenv("TABLE")

client = bigquery.Client(project=PROJECT_ID)
df = client.query(
    f"SELECT title, subject, label FROM `{PROJECT_ID}.{DATASET}.{TABLE}`"
).to_dataframe()

df["text"] = df["title"] + " " + df["subject"]

fake_texts = df[df.label == "fake"]["text"].tolist()
real_texts = df[df.label == "real"]["text"].tolist()

# -----------------------
# TF-IDF MODEL
# -----------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

clf = LogisticRegression(max_iter=500)
clf.fit(X, y)

# -----------------------
# BERT MODEL (sentiment proxy)
# -----------------------
bert = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True
)

# -----------------------
# EMBEDDINGS
# -----------------------
st_model = SentenceTransformer("all-MiniLM-L6-v2")

fake_emb = st_model.encode(fake_texts, convert_to_numpy=True)
real_emb = st_model.encode(real_texts, convert_to_numpy=True)

# -----------------------
# PIE CHART
# -----------------------
def plot_pie(fake_p, real_p):
    fig, ax = plt.subplots()
    ax.pie(
        [fake_p, real_p],
        labels=["Fake", "Real"],
        autopct="%1.1f%%",
        colors=["#ef4444", "#22c55e"]
    )
    ax.set_title("Prediction Probability")
    return fig

# -----------------------
# MAIN FUNCTION
# -----------------------
def analyze(text):

    if not text.strip():
        return "No input", "0%", "", "", None

    # ---------------- TF-IDF ----------------
    X_input = vectorizer.transform([text])
    prob = clf.predict_proba(X_input)[0]

    fake_prob = prob[0] if clf.classes_[0] == "fake" else prob[1]
    real_prob = 1 - fake_prob

    # ---------------- BERT (SAFE) ----------------
    bert_out = bert(text)[0]

    bert_fake = 0.5
    bert_real = 0.5

    for item in bert_out:
        if isinstance(item, dict):
            label = item.get("label", "").upper()
            score = item.get("score", 0)

            if label == "NEGATIVE":
                bert_fake = score
            else:
                bert_real = score

    # ---------------- EMBEDDINGS ----------------
    emb = st_model.encode([text], convert_to_numpy=True)

    sim_fake = util.cos_sim(emb, fake_emb).numpy()[0]
    sim_real = util.cos_sim(emb, real_emb).numpy()[0]

    best_fake = float(np.max(sim_fake))
    best_real = float(np.max(sim_real))

    # ---------------- NORMALIZED PROBABILITY ----------------
    raw_fake = (
        0.2 * fake_prob +
        0.1 * bert_fake +
        0.7 * best_fake
    )

    raw_real = (
        0.2 * real_prob +
        0.1 * bert_real +
        0.7 * best_real
    )
    total = raw_fake + raw_real + 1e-8

    fake_p = raw_fake / total
    real_p = raw_real / total

    if abs(real_p - fake_p) < 0.08:
        pred = "REAL" if best_real > best_fake else "FAKE"
    else:
        pred = "REAL" if real_p > fake_p else "FAKE"

    total = raw_fake + raw_real + 1e-8

    fake_p = raw_fake / total
    real_p = raw_real / total

    # ---------------- FINAL DECISION ----------------
    confidence = max(real_p, fake_p) * 100

    if real_p > fake_p:
        pred = "REAL"
    else:
        pred = "FAKE"

    conf = min(confidence, 99.99)

    # ---------------- DISPLAY ----------------
    signals = f"""
TF-IDF Fake: {fake_prob:.3f}
BERT Fake: {bert_fake:.3f}
Embedding Fake: {best_fake:.3f}
Embedding Real: {best_real:.3f}
"""

    matches = "🔴 Fake Examples:\n"
    matches += "\n".join([fake_texts[i] for i in np.argsort(-sim_fake)[:2]])

    matches += "\n\n🟢 Real Examples:\n"
    matches += "\n".join([real_texts[i] for i in np.argsort(-sim_real)[:2]])

    emb_text = f"""
Fake Similarity: {best_fake:.3f}
Real Similarity: {best_real:.3f}
"""

    pie = plot_pie(fake_p, real_p)

    return pred, f"{conf:.2f}%", signals, matches, emb_text, pie

def get_live_news(query):
    articles = fetch_news(query)
    
    if not articles:
        return "No news found."

    output = ""
    for i, art in enumerate(articles):
        output += f"{i+1}. {art['title']}\n"

    return output

def analyze_live_news(query):
    articles = fetch_news(query)

    if not articles:
        return "No news found."

    results = []

    for art in articles:
        text = art.get("title", "")

        try:
            pred, *_ = analyze(text)
        except:
            pred = "ERROR"

        results.append(f"{text} → {pred}")

    return "\n\n".join(results)
# -----------------------
# UI
# -----------------------
with gr.Blocks() as demo:

    gr.Markdown("# 🧠 Fake Vs Real News Detection System")
    gr.Markdown("""
        📰 **End-to-End News Intelligence Pipeline**

        BigQuery • Data Partitioning • NLP (TF-IDF + BERT + Embeddings) •  
        Graph Analytics (Neo4j) • Probabilistic Classification Model
        """)
    inp = gr.Textbox(lines=6, label="Enter News Text")
    btn = gr.Button("Analyze")

    pred = gr.Textbox(label="Prediction")
    conf = gr.Textbox(label="Confidence")

    signals = gr.Textbox(label="Model Signals")
    matches = gr.Textbox(label="Similar News")

    emb = gr.Textbox(label="Embedding Scores")
    pie = gr.Plot(label="Probability Pie Chart")
    gr.Markdown("## 🌐 Live News Analysis")

    query_input = gr.Textbox(label="🔎 Search Topic (e.g. Trump, Economy)")
    fetch_btn = gr.Button("🌐 Fetch News")

    live_output = gr.Textbox(label="Live News Results")
 
    fetch_btn.click(
        get_live_news,
        inputs=query_input,
        outputs=live_output
    )

    btn.click(
        analyze,
        inputs=inp,
        outputs=[pred, conf, signals, matches, emb, pie]
    )

demo.launch()