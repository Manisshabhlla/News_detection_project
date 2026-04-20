from fastapi import FastAPI
from pydantic import BaseModel
from engine import predict

app = FastAPI(title="News Detection API")

class NewsItem(BaseModel):
    text: str

@app.post("/predict")
def predict_news(item: NewsItem):
    return predict(item.text)

@app.get("/")
def root():
    return {"message": "News Detection API is running"}