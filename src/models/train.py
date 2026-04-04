"""
Model eğitimi ve çapraz doğrulama modülü.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Model fabrikası — her çağrıda taze kopya döner
MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, C=1.0, n_jobs=-1, random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, learning_rate=0.1, eval_metric="mlogloss",
        n_jobs=-1, random_state=42, verbosity=0
    ),
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=200, learning_rate=0.1, n_jobs=-1,
        random_state=42, verbose=-1
    ),
    "svm": lambda: SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
}


def get_model(name: str):
    """Model adına göre taze model örneği döndürür."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: '{name}'. Seçenekler: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


def train_cv(
    pipeline: ImbPipeline,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Pipeline + model ile StratifiedKFold çapraz doğrulama yapar.
    Her fold'un metriklerini DataFrame olarak döndürür.
    """
    if scoring is None:
        scoring = ["f1_macro", "accuracy"]

    model = get_model(model_name)
    pipeline.steps.append(("clf", model))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"\n{model_name} — {n_splits}-fold CV başlıyor...")
    t0 = time.time()
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed = time.time() - t0
    print(f"Tamamlandı ({elapsed:.1f}s)")

    # fold bazlı sonuçlar
    rows = []
    for fold in range(n_splits):
        row = {"fold": fold + 1, "model": model_name}
        for metric in scoring:
            row[metric] = cv_results[f"test_{metric}"][fold]
        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n{'Metrik':<20} {'Ortalama':>10} {'±Std':>10}")
    print("-" * 42)
    for metric in scoring:
        vals = df[metric]
        print(f"{metric:<20} {vals.mean():>10.4f} {vals.std():>10.4f}")

    # Pipeline'dan clf'i çıkar (yeniden kullanım için)
    pipeline.steps.pop()
    return df


def train_final(
    pipeline: ImbPipeline,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ImbPipeline:
    """Tüm eğitim verisi üzerinde final modeli eğitir."""
    model = get_model(model_name)
    pipeline.steps.append(("clf", model))
    print(f"\n{model_name} — final model eğitiliyor...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Tamamlandı ({time.time() - t0:.1f}s)")
    return pipeline


def save_model(pipeline: ImbPipeline, name: str, models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(pipeline, path)
    print(f"Model kaydedildi → {path}")
    return path


def load_model(name: str, models_dir: str) -> ImbPipeline:
    path = os.path.join(models_dir, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    return joblib.load(path)
