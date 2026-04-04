"""
Özellik ön işleme pipeline'ı.
sklearn Pipeline döndürür → fit/transform/predict zincirine doğrudan eklenir.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_pipeline(
    variance_threshold: float = 0.01,
    scale: bool = True,
) -> ImbPipeline:
    """
    Preprocessing pipeline döndürür.

    Adımlar:
        1. VarianceThreshold  – sabit/düşük varyanslı feature'ları atar
        2. StandardScaler     – z-score normalizasyon (isteğe bağlı)
    """
    steps = [
        ("variance", VarianceThreshold(threshold=variance_threshold)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    return ImbPipeline(steps)


def build_pipeline_with_smote(
    variance_threshold: float = 0.01,
    scale: bool = True,
) -> ImbPipeline:
    """
    SMOTE dahil pipeline (sadece eğitimde kullanılır).

    Adımlar:
        1. VarianceThreshold
        2. StandardScaler
        3. SMOTE
    """
    steps = [
        ("variance", VarianceThreshold(threshold=variance_threshold)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("smote", SMOTE(random_state=42)))

    return ImbPipeline(steps)


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Kategorik etiketleri sayıya çevirir, encoder'ı döndürür."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Sınıflar ({len(le.classes_)}): {list(le.classes_)}")
    return y_enc, le
