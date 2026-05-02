"""
Data loading and cleaning module.
Concatenates all CSV files from the CICIDS2017 dataset.
"""

import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Raw label → merged class mapping
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Web Attack � Brute Force": "Web Attack",
    "Web Attack � XSS": "Web Attack",
    "Web Attack � Sql Injection": "Web Attack",
    "FTP-Patator": "Patator",
    "SSH-Patator": "Patator",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "Heartbleed": "Heartbleed",
}

BINARY_MAP = {label: ("ATTACK" if label != "BENIGN" else "BENIGN") for label in LABEL_MAP}


def load_raw(data_dir: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """Loads and concatenates all CSV files."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    frames = []
    for f in tqdm(files, desc="Loading CSVs"):
        df = pd.read_csv(f, low_memory=False)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.str.strip()
    print(f"Loaded: {len(data):,} rows from {len(files)} files")
    return data


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Removes infinite values, NaNs, and duplicate rows."""
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna().drop_duplicates()
    print(f"Cleaning: removed {before - len(df):,} rows → {len(df):,} rows remaining")
    return df.reset_index(drop=True)


def apply_labels(df: pd.DataFrame, task: str = "multiclass") -> pd.DataFrame:
    """task='multiclass' → 9 classes | task='binary' → BENIGN / ATTACK"""
    df = df.copy()
    mapping = BINARY_MAP if task == "binary" else LABEL_MAP
    df["Label"] = df["Label"].map(mapping).fillna(df["Label"])
    print("Class distribution:\n", df["Label"].value_counts().to_string())
    return df


def load_dataset(
    data_dir: str,
    sample_frac: float = 1.0,
    task: str = "multiclass",
) -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: load → clean → label → return X, y."""
    df = load_raw(data_dir, sample_frac)
    df = clean(df)
    df = apply_labels(df, task)
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return X, y
