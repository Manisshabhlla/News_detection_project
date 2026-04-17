from fastapi import FastAPI
from pydantic import BaseModel
from engine import load_data, train_model, predict
from bert_model import similarity

app = FastAPI(title="News Detection API")

# Input model
class NewsItem(BaseModel):
    text: str

# Load your model at startup
model, vectorizer = train_model()  # or load pre-trained if saved

@app.post("/predict")
def predict_news(item: NewsItem):
    # Fast similarity + prediction
    result = predict(model, vectorizer, item.text)
    sim_scores = similarity(item.text)
    return {"prediction": result, "similarity": sim_scores}

@app.get("/")
def root():
    return {"message": "News Detection API is running"}