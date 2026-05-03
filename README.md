# Network Intrusion Detection

Multi-class network attack detection using the CICIDS2017 dataset.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
network-intrusion-detection/
│
├── MachineLearningCVE/              # Raw data (8 CSVs, ~2.8M rows)
│
├── configs/
│   └── config.yaml                  # All experiment parameters
│
├── src/                             # Reusable modules
│   ├── utils/data_loader.py         # Loading, cleaning, labelling
│   ├── features/preprocessing.py   # sklearn Pipeline (Scaler, SMOTE)
│   ├── models/train.py             # Model factory, CV, saving
│   └── evaluation/metrics.py       # Metrics, plots
│
├── notebooks/                       # Development workflow
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_baseline.ipynb            # First model + CV
│   ├── 03_comparison.ipynb          # Model comparison
│   ├── 04_feature_engineering.ipynb # Feature selection
│   └── 05_hyperparameter.ipynb      # Hyperparameter optimisation
│
├── outputs/
│   ├── models/                      # Saved models (.joblib)
│   ├── results/                     # CV results (.csv)
│   └── figures/                     # Plots (.png)
│
├── run.py                           # CLI entry point (outside notebooks)
├── PLAN.md                          # Development roadmap
└── requirements.txt
```

---

## Workflow

Development is driven from the notebooks:

| Notebook | Content | Status |
|----------|---------|--------|
| `01_eda.ipynb` | Data loading, class distribution, correlation, missing values | ✅ |
| `02_baseline.ipynb` | First model, 5-fold CV, confusion matrix | ✅ |
| `03_comparison.ipynb` | LR / RF / XGBoost / LightGBM comparison | ✅ |
| `04_feature_engineering.ipynb` | Feature importance, selection, PCA | ✅ |
| `05_hyperparameter.ipynb` | RandomizedSearch / Optuna optimisation | ✅ |

```bash
# Start the notebook server
jupyter notebook notebooks/
```

---

## Dataset

**CICIDS2017** — 8 days, ~2.8M network flow records, 79 features.

| Class | Description |
|-------|-------------|
| BENIGN | Normal traffic |
| DDoS | Distributed denial of service |
| DoS | Hulk, GoldenEye, Slowloris, Slowhttptest |
| PortScan | Port scanning |
| Bot | Botnet traffic |
| Web Attack | Brute Force, XSS, SQL Injection |
| Patator | FTP / SSH brute force |
| Infiltration | Infiltration attempt |
| Heartbleed | Heartbleed exploit |

---

## Results

| Phase | Model | F1 Macro | F1 Weighted | ROC-AUC |
|-------|-------|----------|-------------|---------|
| Phase 2 — Baseline | Random Forest | 0.8449 | 0.9966 | 1.0000 |
| Phase 3 — Model Comparison | XGBoost | 0.8593 | 0.9987 | 1.0000 |
| Phase 4 — Feature Engineering | XGBoost (selected features) | 0.8605 | 0.9987 | 1.0000 |
| Phase 5 — Hyperparameter Tuning | XGBoost (tuned) | 0.7871 | 0.9986 | 1.0000 |

**Final model: `outputs/models/xgboost_feat_eng.joblib` (Phase 4) — F1 Macro=0.8605, ROC-AUC=1.0000**

> Phase 5 tuned model regressed due to RandomizedSearchCV overfitting on SMOTE-augmented CV folds.

---

## Final Model

| File | Description |
|------|-------------|
| `outputs/models/xgboost_feat_eng.joblib` | **Final model** — XGBoost, selected features |
| `outputs/models/preprocessor.joblib` | VarianceThreshold + StandardScaler pipeline |
| `outputs/models/label_encoder.joblib` | LabelEncoder for 15 attack classes |
| `outputs/results/selected_features.json` | Feature list used by the final model |

---

## Evaluation

- **Test set** — F1 Macro, Accuracy, ROC-AUC (OvR weighted)
- **Plots** — Confusion matrix, model comparison bar chart, feature importance
- **Outputs** — `outputs/results/*.csv`, `outputs/figures/*.png`

---

## config.yaml Quick Reference

```yaml
model:
  name: "random_forest"   # change this to switch the model

data:
  sample_frac: 0.1        # 0.1 for quick tests, 1.0 for full run

task: "multiclass"        # multiclass | binary
```
