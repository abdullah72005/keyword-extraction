import os
import json
import pandas as pd
from preprocessing import preprocess_text
def load_dataset():
    preprocessed_data_path = "preprocessed_data.csv"
    
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data from file...")
        df = pd.read_csv(preprocessed_data_path)
        df = pd.Series(df['0'].tolist()) 
    else:
        print("Preprocessed data file not found. Loading raw data and preprocessing...")
        df = pd.read_json("datasett/train.jsonl", lines=True)

        tokens = []
        for i in range(0, len(df)):
            tokens.append(preprocess_text(df['summary'][i]))
            print(f"Processed document {i+1}/1800")

        df = pd.Series(tokens)
        df.to_csv(preprocessed_data_path)

    return df