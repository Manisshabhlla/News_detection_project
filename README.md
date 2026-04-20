📰 News Detection Pipeline

A Scalable Hybrid Framework for Detecting Duplicate, Paraphrased, and AI-Generated News Content

Manisha Bhalla (M25DE1050) · Vitthal Pandey (M25DE1060) · Sayan Chakraborty (M25DE1044)

📌 Overview

The emergence of large language models such as GPT-4 makes it easier than before to create false information, manipulated, and synthetic generated news. This work presents a scalable hybrid pipeline to determine whether news is duplicate, paraphrased, or ai generated. The proposed solution uses two methods for determining this; first, an approximate similarity method to find identical versions of the same piece of news and second, a deep semantic analysis method to identify paraphrases of the original piece of news.

Both solutions are combined together with other features to obtain a high level of precision for the determination of the type of generated news. The method proposed is scalable due to its use of efficient algorithms and architectures.

🏗️ System Architecture
The proposed pipeline consists of six steps:
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
News-detection-project/
│
├── engine.py # Core prediction logic, embedding scoring, lsh integration
├── bert_module.py # Sentence Transformer embeddings + cosine similarity
├── lsh.py # minhash-based document representation + lsh indexing
├── train_model.py # model training pipeline (tf-idf + logistic Regression)
├── main.py # FastAPI REST API (prediction endpoint)
├── app.py # Gradio dashboard (interactive ui)
├── api.py # live news integration via external APIs
├── evaluation.py # Confusion Matrix, ROC Curve, AUC computation
└── readme.md

⚙️ Setup & Installation
Prerequisites

Python 3.8+
pip install -r requirements.txt
pip install transformers sentence-transformers scikit-learn \
            pandas numpy fastapi uvicorn gradio datasketch

🚀 Running the Project
1. Train the Model
   python train_model.py
This will perform the preprocessing for your dataset and also train the tf-idf + logistic Regression classifier and save it as a model. 

2. Launch the Gradio Dashboard
   python app.py
It opens an interactive interface for you to enter your own news texts and see how they will be classified in real time along with the distribution of their respective probabilities and comparison of their similarities to each other.

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
