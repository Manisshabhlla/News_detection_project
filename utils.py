import os
from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # path to your service account JSON

def get_bigquery_client():
    if CREDENTIALS:
        client = bigquery.Client.from_service_account_json(CREDENTIALS, project=PROJECT_ID)
    else:
        client = bigquery.Client(project=PROJECT_ID)
    return client

def load_table(table_name: str):
    client = get_bigquery_client()
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{table_name}`"
    df = client.query(query).to_dataframe()
    return df

def preprocess_text(df, text_col="title"):
    # Simple preprocessing: lowercase, strip
    df[text_col] = df[text_col].str.lower().str.strip()
    return df