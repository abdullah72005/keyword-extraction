import time
import pandas as pd
from preprocessing import preprocess_text
from israel import vectorize_documents
from advanced_alg import extract_keywords
import warnings
warnings.filterwarnings('ignore')

df = pd.read_json("datasett/test.jsonl", lines=True)

# TF-IDF Setup
docs = df["summary"].apply(preprocess_text).tolist()

# Measure TF-IDF
start_time = time.time()
for doc in docs[:10]:
    temp_df = pd.Series([doc])
    try:
        vectorize_documents(temp_df)
    except:
        pass
tfidf_time = (time.time() - start_time) / 10

# Measure KeyBERT
start_time = time.time()
for doc in docs[:10]:
    try:
        extract_keywords(doc)
    except:
        pass
keybert_time = (time.time() - start_time) / 10

print(f"TF-IDF avg runtime: {tfidf_time:.4f} sec")
print(f"KeyBERT avg runtime: {keybert_time:.4f} sec")
print(f"Avg runtime difference: {keybert_time - tfidf_time:.4f} sec")

