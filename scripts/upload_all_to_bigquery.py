from google.cloud import bigquery
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize BigQuery client
client = bigquery.Client()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")

# ---------------------------
# 1. UPLOAD FAKE DATA
# ---------------------------
print("Uploading Fake.csv...")

# Read CSV and clean headers
df_fake = pd.read_csv("data/Fake.csv")
df_fake.columns = df_fake.columns.str.strip() 
print("Fake rows:", len(df_fake))

df_fake = df_fake.fillna("")

fake_table_id = f"{PROJECT_ID}.{DATASET}.fake_news"
job_fake = client.load_table_from_dataframe(
    df_fake,
    fake_table_id,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
)
job_fake.result()
print("✅ Fake data uploaded")

# ---------------------------
# 2. UPLOAD TRUE DATA
# ---------------------------
print("Uploading True.csv...")

df_true = pd.read_csv("data/True.csv")
df_true.columns = df_true.columns.str.strip()
print("True rows:", len(df_true))
df_true = df_true.fillna("")

true_table_id = f"{PROJECT_ID}.{DATASET}.true_news"
job_true = client.load_table_from_dataframe(
    df_true,
    true_table_id,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
)
job_true.result()
print("✅ True data uploaded")

# ---------------------------
# 3. MERGE TABLES AND ADD LABEL
# ---------------------------
print("Merging tables in BigQuery...")

merge_query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.news_data` AS
SELECT *, 'fake' AS label FROM `{PROJECT_ID}.{DATASET}.fake_news`
UNION ALL
SELECT *, 'real' AS label FROM `{PROJECT_ID}.{DATASET}.true_news`
"""

query_job = client.query(merge_query)
query_job.result()
print("✅ Merged table created: news_data with all columns + label")