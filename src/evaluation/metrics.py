"""
Model değerlendirme ve görselleştirme modülü.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def evaluate(
    pipeline: ImbPipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
    figures_dir: str,
) -> dict:
    """
    Test seti üzerinde tahmin yapar, metrikleri hesaplar ve grafikleri kaydeder.
    """
    y_pred = pipeline.predict(X_test)
    classes = label_encoder.classes_

    f1_macro = f1_score(y_test, y_pred, average="macro")
    accuracy = (y_test == y_pred).mean()

    # ROC-AUC (olasılık destekliyorsa)
    auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            pass

    print(f"\n{'=' * 55}")
    print(f"  Model : {model_name}")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=classes))

    plot_confusion_matrix(y_test, y_pred, classes, model_name, figures_dir)

    return {
        "model": model_name,
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4) if auc else None,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    model_name: str,
    figures_dir: str,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"Karmaşıklık Matrisi — {model_name}")
    plt.tight_layout()
    path = os.path.join(figures_dir, f"confusion_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Karmaşıklık matrisi kaydedildi → {path}")


def plot_class_distribution(y: pd.Series, figures_dir: str) -> None:
    """Sınıf dağılımı çubuk grafiği."""
    os.makedirs(figures_dir, exist_ok=True)
    counts = y.value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Örnek Sayısı")
    ax.set_title("Sınıf Dağılımı")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(figures_dir, "sinif_dagilimi.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Sınıf dağılımı kaydedildi → {path}")


def compare_models(all_metrics: list[dict], figures_dir: str) -> pd.DataFrame:
    """Modellerin F1 Macro karşılaştırma tablosu ve grafiği."""
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics).set_index("model")

    print("\n=== Model Karşılaştırması ===")
    print(df.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    df["f1_macro"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Model Karşılaştırması — F1 Macro")
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(figures_dir, "model_karsilastirma.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Karşılaştırma grafiği kaydedildi → {path}")

    return df


def save_cv_results(cv_df: pd.DataFrame, model_name: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}_cv_folds.csv")
    cv_df.to_csv(path, index=False)

    summary_rows = []
    for col in cv_df.columns:
        if col not in ("fold", "model"):
            summary_rows.append({
                "metric": col,
                "mean": cv_df[col].mean(),
                "std": cv_df[col].std(),
            })
    summary_path = os.path.join(results_dir, f"{model_name}_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"CV sonuçları kaydedildi → {path}")
