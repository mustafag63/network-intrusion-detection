# Network Intrusion Detection with Machine Learning

Multi-class network attack detection using the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.  
Covers the full ML pipeline: EDA → baseline → model comparison → feature engineering → hyperparameter tuning.

---

## Results

| Phase | Model | F1 Macro | F1 Weighted | ROC-AUC |
|-------|-------|:--------:|:-----------:|:-------:|
| Phase 2 — Baseline | Random Forest | 0.8449 | 0.9966 | 1.0000 |
| Phase 3 — Model Comparison | XGBoost | 0.8593 | 0.9987 | 1.0000 |
| **Phase 4 — Feature Engineering** | **XGBoost (selected features)** | **0.8605** | **0.9987** | **1.0000** |
| Phase 5 — Hyperparameter Tuning | XGBoost (tuned) | 0.7871 | 0.9986 | 1.0000 |

**Final model:** `outputs/models/xgboost_feat_eng.joblib` — XGBoost with selected features, F1 Macro = **0.8605**

> Phase 5 tuned model regressed due to RandomizedSearchCV overfitting on SMOTE-augmented CV folds.  
> Phase 4 model is retained as the final production model.

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity, 2017  
8 days of network traffic, ~2.8M flow records, 79 features.

| Class | Samples | Description |
|-------|--------:|-------------|
| BENIGN | ~2,273,097 | Normal traffic |
| DDoS | ~128,027 | Distributed denial of service |
| DoS | ~252,661 | Hulk, GoldenEye, Slowloris, Slowhttptest |
| PortScan | ~158,930 | Port scanning |
| Bot | ~1,966 | Botnet traffic |
| Web Attack | ~2,180 | Brute Force, XSS, SQL Injection |
| Patator | ~13,835 | FTP / SSH brute force |
| Infiltration | ~36 | Infiltration attempt |
| Heartbleed | ~11 | Heartbleed exploit |

> Raw data is not included in this repository. Download from the [CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html) and place CSVs under `MachineLearningCVE/`.

---

## Project Structure

```
network-intrusion-detection/
│
├── MachineLearningCVE/              # Raw data (not tracked — download separately)
│
├── configs/
│   └── config.yaml                  # All experiment parameters
│
├── src/                             # Reusable modules
│   ├── utils/data_loader.py         # Loading, cleaning, labelling
│   ├── features/preprocessing.py   # sklearn Pipeline (VarianceThreshold, Scaler, SMOTE)
│   ├── models/train.py              # Model factory, CV, saving
│   └── evaluation/metrics.py       # Metrics, plots
│
├── notebooks/                       # Step-by-step development
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_baseline.ipynb            # Random Forest baseline + 5-fold CV
│   ├── 03_comparison.ipynb          # LR / RF / XGBoost / LightGBM comparison
│   ├── 04_feature_engineering.ipynb # Feature importance & selection
│   └── 05_hyperparameter.ipynb      # RandomizedSearchCV
│
├── outputs/
│   ├── models/                      # Saved models (.joblib) — not tracked
│   ├── results/                     # CSV results per experiment
│   └── figures/                     # Plots (.png)
│
├── run.py                           # CLI entry point
├── PLAN.md                          # Development roadmap
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/<your-username>/network-intrusion-detection.git
cd network-intrusion-detection
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Usage

### Notebooks (recommended)

Notebooks are designed to run on **Google Colab with GPU** (T4 or better).  
Open any notebook and follow the cells top to bottom.

```bash
# Local
jupyter notebook notebooks/
```

Each notebook mounts Google Drive for data/model persistence:
```python
OUTPUTS = Path('/content/drive/MyDrive/nids/outputs')
```

### CLI

```bash
# Train with default config (XGBoost, full dataset)
python run.py

# Quick test with 10% of data
python run.py --sample 0.1

# Train a specific model
python run.py --model random_forest

# Compare all models
python run.py --compare

# Binary classification (BENIGN vs ATTACK)
python run.py --task binary
```

### config.yaml

```yaml
model:
  name: "xgboost"       # logistic_regression | random_forest | xgboost | lightgbm

data:
  sample_frac: 1.0      # 0.1 for quick tests, 1.0 for full dataset

task: "multiclass"      # multiclass | binary
```

---

## Pipeline

```
Raw CSVs (8 files, ~2.8M rows)
    │
    ▼
Data Cleaning          — duplicates, ±inf, NaN, negative durations, constant columns
    │
    ▼
Preprocessing          — VarianceThreshold → StandardScaler
    │
    ▼
SMOTE                  — minority classes upsampled to 5,000 samples each
    │
    ▼
Model Training         — XGBoost (n_estimators=300, max_depth=8, learning_rate=0.1)
    │
    ▼
Feature Selection      — zero-importance + high-correlation (|r|≥0.95) features dropped
    │
    ▼
Evaluation             — F1 Macro, F1 Weighted, ROC-AUC (OvR), classification report
```

---

## Notebooks

| Notebook | Description | Status |
|----------|-------------|:------:|
| `01_eda.ipynb` | Class distribution, correlation heatmap, skewness, ANOVA F-scores | ✅ |
| `02_baseline.ipynb` | Random Forest with 5-fold stratified CV, SMOTE, confusion matrix | ✅ |
| `03_comparison.ipynb` | XGBoost / LightGBM / Logistic Regression vs RF baseline | ✅ |
| `04_feature_engineering.ipynb` | XGBoost feature importance (gain + cover), feature selection | ✅ |
| `05_hyperparameter.ipynb` | RandomizedSearchCV (30 iter × 3-fold) on XGBoost | ✅ |

---

## Final Model Artifacts

| File | Description |
|------|-------------|
| `outputs/models/xgboost_feat_eng.joblib` | Final XGBoost model |
| `outputs/models/preprocessor.joblib` | Fitted VarianceThreshold + StandardScaler |
| `outputs/models/label_encoder.joblib` | LabelEncoder for 15 attack classes |
| `outputs/results/selected_features.json` | Feature list used by the final model |

---

## Key Findings

- **XGBoost** outperforms Random Forest on F1 Macro (+0.0144) with identical ROC-AUC
- **Feature selection** (dropping zero-importance + high-corr features) gives a small additional gain (+0.0012) with a leaner model
- **Minority classes** (Bot, XSS, Brute Force) remain the hardest to classify — SMOTE alone is insufficient; threshold tuning or ADASYN would be the next step
- **Hyperparameter search on SMOTE-augmented data** can overfit to synthetic samples — CV should be applied before SMOTE for reliable generalisation

---

## License

MIT
