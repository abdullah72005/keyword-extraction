import os
import pandas as pd

from preprocessing import preprocess_text


def _load_cached_series(cache_path):
    df = pd.read_csv(cache_path)
    return pd.Series(df["0"].tolist())


def load_dataset_split(split_path, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading preprocessed data from {cache_path}...")
        return _load_cached_series(cache_path)

    print(f"Preprocessed data not found. Loading raw data from {split_path}...")
    df = pd.read_json(split_path, lines=True)

    tokens = []
    for i in range(0, len(df)):
        tokens.append(preprocess_text(df["summary"][i]))
        print(f"Processed document {i + 1}/{len(df)}")

    series = pd.Series(tokens)
    series.to_csv(cache_path)
    return series


def load_dataset():
    return load_dataset_split("datasett/train.jsonl", "preprocessed_data.csv")