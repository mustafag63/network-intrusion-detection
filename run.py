"""
Kullanım:
  python run.py                          # config.yaml'daki model, tam veri
  python run.py --model xgboost          # farklı model
  python run.py --sample 0.1            # %10 veriyle hızlı test
  python run.py --task binary           # BENIGN / ATTACK
  python run.py --no-smote              # SMOTE olmadan
  python run.py --compare               # tüm modelleri karşılaştır
"""

import argparse
import os
import sys

import joblib
import yaml
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compare_models, evaluate, save_cv_results
from src.features.preprocessing import (
    build_pipeline,
    build_pipeline_with_smote,
    encode_labels,
)
from src.models.train import MODEL_REGISTRY, save_model, train_cv, train_final
from src.utils.data_loader import load_dataset


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Network Intrusion Detection Pipeline")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), help="Eğitilecek model")
    parser.add_argument("--sample", type=float, help="Veri oranı (örn. 0.1)")
    parser.add_argument("--task", choices=["binary", "multiclass"], help="binary / multiclass")
    parser.add_argument("--no-smote", action="store_true", help="SMOTE'u devre dışı bırak")
    parser.add_argument("--compare", action="store_true", help="Tüm modelleri karşılaştır")
    parser.add_argument("--config", default="configs/config.yaml", help="Config dosya yolu")
    return parser.parse_args()


def run(cfg: dict, model_name: str, use_smote: bool):
    data_cfg = cfg["data"]
    pp_cfg = cfg["preprocessing"]
    cv_cfg = cfg["cv"]
    out_cfg = cfg["output"]

    # 1. Veri yükle
    X, y = load_dataset(
        data_dir=data_cfg["data_dir"],
        sample_frac=data_cfg["sample_frac"],
        task=cfg["task"],
    )

    # 2. Encode + split
    y_enc, le = encode_labels(y)
    X_arr = X.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_enc,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y_enc,
    )

    # 3. CV
    pipeline_cv = build_pipeline(
        variance_threshold=pp_cfg["variance_threshold"],
        scale=pp_cfg["scale"],
    )
    cv_df = train_cv(
        pipeline=pipeline_cv,
        model_name=model_name,
        X=X_train,
        y=y_train,
        n_splits=cv_cfg["n_splits"],
        scoring=cv_cfg["scoring"],
        random_state=cv_cfg["random_state"],
    )
    save_cv_results(cv_df, model_name, out_cfg["results_dir"])

    # 4. Final model
    if use_smote:
        final_pipeline = build_pipeline_with_smote(
            variance_threshold=pp_cfg["variance_threshold"],
            scale=pp_cfg["scale"],
        )
    else:
        final_pipeline = build_pipeline(
            variance_threshold=pp_cfg["variance_threshold"],
            scale=pp_cfg["scale"],
        )

    final_pipeline = train_final(final_pipeline, model_name, X_train, y_train)

    # 5. Değerlendirme
    metrics = evaluate(
        final_pipeline, X_test, y_test,
        label_encoder=le,
        model_name=model_name,
        figures_dir=out_cfg["figures_dir"],
    )

    # 6. Kaydet
    if cfg["output"]["save_model"]:
        save_model(final_pipeline, name=f"{model_name}_baseline", models_dir=out_cfg["models_dir"])
        joblib.dump(le, os.path.join(out_cfg["models_dir"], "label_encoder.joblib"))

    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI argümanları config'i override eder
    if args.model:
        cfg["model"]["name"] = args.model
    if args.sample:
        cfg["data"]["sample_frac"] = args.sample
    if args.task:
        cfg["task"] = args.task
    if args.no_smote:
        cfg["preprocessing"]["use_smote"] = False

    use_smote = cfg["preprocessing"]["use_smote"]

    if args.compare:
        all_metrics = []
        for name in MODEL_REGISTRY:
            print(f"\n{'#'*60}")
            print(f"  {name}")
            print(f"{'#'*60}")
            metrics = run(cfg, model_name=name, use_smote=use_smote)
            all_metrics.append(metrics)
        compare_models(all_metrics, figures_dir=cfg["output"]["figures_dir"])
    else:
        run(cfg, model_name=cfg["model"]["name"], use_smote=use_smote)

    print("\nTamamlandı. Çıktılar → outputs/")


if __name__ == "__main__":
    main()
