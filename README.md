📰 News Detection Pipeline

A Scalable Hybrid Framework for Detecting Duplicate, Paraphrased, and AI-Generated News Content

Manisha Bhalla (M25DE1050) · Vitthal Pandey (M25DE1060) · Sayan Chakraborty (M25DE1044)

📌 Overview

With the rapid rise of large language models like GPT-4, the creation of fake, manipulated, and AI-generated news has become easier than ever. This project presents a scalable hybrid pipeline that can detect:

1. Duplicate news articles (identical copies)
2. Paraphrased articles (semantically similar but reworded)
3. AI-generated articles (synthetic content mimicking human writing)

The system combines approximate similarity search with deep semantic analysis, achieving near-perfect detection accuracy while maintaining high computational efficiency.

🏗️ System Architecture
The pipeline consists of 6 stages:
Data Ingestion → MinHash + LSH → BERT Embeddings → AI Detector → Hybrid Fusion → Gradio UI
Stage	Component	Purpose
1	Data Ingestion & Preprocessing	Load, clean, normalize, tokenize
2	MinHash + LSH	Approximate similarity search (near-linear complexity)
3	Sentence-BERT	Deep semantic analysis for paraphrase detection
4	AI Content Detector	Classify AI-generated vs human-written text
5	Hybrid Decision Fusion	Weighted combination of all signals
6	Gradio Dashboard	Interactive UI with real-time predictions
<img width="746" height="169" alt="image" src="https://github.com/user-attachments/assets/9cb48ff2-942b-4d5c-8c46-947eb1f65d3f" />

Hybrid Fusion Formula
P_fake = w1 * P_tfidf  +  w2 * P_bert  +  w3 * S_fake
P_real = w1 * (1 - P_tfidf)  +  w2 * P_bert  +  w3 * S_real

ŷ = argmax(P_fake, P_real)

🛠️ Tech Stack
Layer	Tools
Data Storage	Google BigQuery
Preprocessing	Python, Pandas, NumPy
Lexical Model	TF-IDF + Logistic Regression (scikit-learn)
Transformer Model	DistilBERT (HuggingFace)
Semantic Similarity	Sentence-BERT (SBERT)
Approximate Search	MinHash + LSH
API Backend	FastAPI + Pydantic
Frontend / UI	Gradio
Evaluation	scikit-learn (Confusion Matrix, ROC, AUC)
<img width="161" height="745" alt="image" src="https://github.com/user-attachments/assets/cf9d721d-7479-47a5-96ca-9627642c076b" />


📁 Project Structure
news-detection-project/
│
├── engine.py              # Core prediction logic, embedding scoring, LSH integration
├── bert_module.py         # Sentence Transformer embeddings + cosine similarity
├── lsh.py                 # MinHash-based document representation + LSH indexing
├── train_model.py         # Model training pipeline (TF-IDF + Logistic Regression)
├── main.py                # FastAPI REST API (prediction endpoint)
├── app.py                 # Gradio dashboard (interactive UI)
├── api.py                 # Live news integration via external APIs
├── evaluation.py          # Confusion matrix, ROC curve, AUC computation
└── README.md

⚙️ Setup & Installation
Prerequisites

Python 3.8+
pip install -r requirements.txt
pip install transformers sentence-transformers scikit-learn \
            pandas numpy fastapi uvicorn gradio datasketch

🚀 Running the Project
1. Train the Model
   python train_model.py
This will preprocess the dataset, train the TF-IDF + Logistic Regression classifier, and serialize the model.

2. Launch the Gradio Dashboard
   python app.py
Opens an interactive UI for entering news text and viewing real-time predictions, probability distributions, and similar article comparisons.

3. Start the API Server
uvicorn main:app --reload
The REST API will be available at http://localhost:8000.
Endpoint:
POST /predict
Body: { "text": "your news article here" }
Response: { "label": "Fake" | "Real", "confidence": 0.97, "signals": {...} }

📊 Results
Evaluated on a dataset of 50,000 – 200,000 news articles (Kaggle + AI-generated).

Metric	Value
True Negatives (Real → Real)	21,413
True Positives (Fake → Fake)	23,481
False Positives	4
False Negatives	0
<img width="161" height="433" alt="image" src="https://github.com/user-attachments/assets/1ebf1aee-0e63-47dd-8413-c9693078b1c5" />

👥 Team Contributions
Manisha Bhalla — Lead Architect & Core Algorithms

Designed and implemented the hybrid detection architecture
Built the semantic similarity module (bert_module.py) using Sentence Transformers
Developed core prediction logic in engine.py including confidence estimation
Tuned similarity thresholds and model parameters

Vitthal Pandey — Data Engineer & Scalable Pipeline

Implemented data ingestion, preprocessing, and text normalization
Built the TF-IDF vectorization and n-gram feature engineering pipeline
Developed the LSH module (lsh.py) with MinHash-based indexing
Built the model training pipeline (train_model.py)

Sayan Chakraborty — API, Visualization & Evaluation

Developed the FastAPI backend (main.py) with Pydantic schemas
Built the Gradio interactive dashboard (app.py)
Implemented live news integration via external APIs (api.py)
Built the evaluation framework with Confusion Matrix, ROC Curve, and AUC
