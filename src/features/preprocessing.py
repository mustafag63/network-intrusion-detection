"""
Feature preprocessing pipeline.
Returns an sklearn Pipeline → plugs directly into fit/transform/predict chains.
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
    Returns a preprocessing pipeline.

    Steps:
        1. VarianceThreshold  – drops constant / near-zero-variance features
        2. StandardScaler     – z-score normalisation (optional)
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
    Pipeline including SMOTE (training only).

    Steps:
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
    """Encodes categorical labels to integers; returns the fitted encoder."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    return y_enc, le
