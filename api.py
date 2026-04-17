import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWSAPI_KEY")
BASE_URL = os.getenv("BASE_URL")

def fetch_news(query, page_size=5):
    params = {
        "q": query,
        "apiKey": API_KEY,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "relevancy"
    }

    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        articles = data.get("articles", [])

        news = []
        for article in articles:
            news.append({
                "title": article.get("title", "No Title"),
                "desc": article.get("description", "No Description")
            })

        return news

    except Exception as e:
        print("Error fetching live news:", e)
        return []