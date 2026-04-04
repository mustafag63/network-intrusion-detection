"""
Veri yükleme ve temizleme modülü.
CICIDS2017 veri setindeki CSV dosyalarını birleştirir.
"""

import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ham label → birleştirilmiş sınıf haritası
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Web Attack \ufffd Brute Force": "Web Attack",
    "Web Attack \ufffd XSS": "Web Attack",
    "Web Attack \ufffd Sql Injection": "Web Attack",
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
    """Tüm CSV dosyalarını yükler ve birleştirir."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV bulunamadı: {data_dir}")

    frames = []
    for f in tqdm(files, desc="CSV yükleniyor"):
        df = pd.read_csv(f, low_memory=False)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.str.strip()
    print(f"Yüklendi: {len(data):,} satır, {len(files)} dosya")
    return data


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sonsuz değerleri, NaN'ları ve tekrarları kaldırır."""
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna().drop_duplicates()
    print(f"Temizleme: {before - len(df):,} satır kaldırıldı → {len(df):,} satır kaldı")
    return df.reset_index(drop=True)


def apply_labels(df: pd.DataFrame, task: str = "multiclass") -> pd.DataFrame:
    """task='multiclass' → 9 sınıf | task='binary' → BENIGN / ATTACK"""
    df = df.copy()
    mapping = BINARY_MAP if task == "binary" else LABEL_MAP
    df["Label"] = df["Label"].map(mapping).fillna(df["Label"])
    print("Sınıf dağılımı:\n", df["Label"].value_counts().to_string())
    return df


def load_dataset(
    data_dir: str,
    sample_frac: float = 1.0,
    task: str = "multiclass",
) -> tuple[pd.DataFrame, pd.Series]:
    """Tam pipeline: yükle → temizle → etiketle → X, y döndür."""
    df = load_raw(data_dir, sample_frac)
    df = clean(df)
    df = apply_labels(df, task)
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return X, y
