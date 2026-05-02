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

| Notebook | Content |
|----------|---------|
| `01_eda.ipynb` | Data loading, class distribution, correlation, missing values |
| `02_baseline.ipynb` | First model, 5-fold CV, confusion matrix |
| `03_comparison.ipynb` | LR / RF / XGBoost / LightGBM comparison |
| `04_feature_engineering.ipynb` | Feature importance, selection, PCA |
| `05_hyperparameter.ipynb` | RandomizedSearch / Optuna optimisation |

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

## Models

| Key | Model |
|-----|-------|
| `logistic_regression` | Logistic Regression |
| `random_forest` | Random Forest |
| `gradient_boosting` | Gradient Boosting |
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `svm` | Support Vector Machine |

---

## Evaluation

- **5-Fold Stratified CV** — F1 Macro and Accuracy per fold
- **Test set** — F1 Macro, Accuracy, ROC-AUC (OvR)
- **Plots** — Confusion matrix, model comparison bar chart
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
