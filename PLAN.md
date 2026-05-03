# Network Intrusion Detection — Development Plan

Each phase corresponds to one notebook.
Completed ✅, in progress 🔄, pending ⬜.

---

## Phase 1 — Exploratory Data Analysis (`01_eda.ipynb`) ✅

- [x] Load, concatenate, and clean all CSVs
  - Duplicate columns, infinite values, NaNs, negative values, constant columns, duplicate rows
  - Added `Init_Win_bytes` sentinel flag; generated `tcp_win_seen_*` columns
- [x] Visualise class distribution (`class_distribution.png`)
  - Family-level grouping (7 families), IR Binary & IR Macro computed
  - Critical minority classes identified → SMOTE k value should be reduced in Phase 2
- [x] Missing / infinite value analysis
  - Root cause of `duration=0` → `inf` documented; skewness analysis (`skewness_distribution.png`)
- [x] Feature correlation heatmap (`correlation.png`, `high_correlation.png`)
  - Pairs with `|r| ≥ 0.95` identified → `high_corr_to_drop.json` (to be applied in Phase 4)
- [x] Compare feature distributions by class
  - Ranked by ANOVA F-score (`feature_f_scores.csv`)
  - BENIGN vs ATTACK histograms + family-level box plots generated
- [x] Report class imbalance ratio
  - IR Binary and IR Macro computed; added to summary report

---

## Phase 2 — Baseline Model (`02_baseline.ipynb`) ✅

- [x] Label encoding + train/test split (80/20)
- [x] Build pipeline: VarianceThreshold → StandardScaler
  - 4 columns dropped (`Fwd URG Flags`, `RST Flag Count`, `CWE Flag Count`, `ECE Flag Count`)
  - `preprocessor.joblib` and `kept_features.json` saved
- [x] Random Forest with 5-fold stratified CV (10% sample)
  - Added `class_weight='balanced'`
  - Very rare classes excluded from CV pool (Heartbleed, Infiltration, Web Attack–SQL Injection — each < 5 samples)
    - Root cause: these classes don't appear in some validation folds → model produces no column → roc_auc NaN
  - Scorer: `roc_auc_ovr_weighted` (standard; custom scorer not needed)
  - val_f1_macro ≈ 0.83, val_roc_auc no longer NaN
- [x] Per-fold `classification_report` — identified which classes have low F1
  - Saved to `outputs/results/random_forest_cv_class_report.csv`
- [x] Save CV results (`outputs/results/random_forest_cv_folds.csv`)
- [x] Re-balance with SMOTE, train final model
  - 10 classes upsampled to 5,000 samples each (+30,879 synthetic samples)
  - `max_depth=25`, `min_samples_leaf=4` (to reduce overfitting)
- [x] Test set evaluation: F1 Macro, Accuracy, ROC-AUC
  - Test Accuracy: **0.9966** | F1 Macro: **0.845** | F1 Weighted: 0.997 | ROC-AUC: 1.000
  - Problematic classes: Bot (F1=0.41, precision=0.26 — too many FPs), XSS (0.47), Brute Force (0.72)
  - Heartbleed/SQL Injection test set < 5 samples — F1 for these classes is statistically meaningless
- [x] Save confusion matrix (`outputs/figures/confusion_random_forest.png`)
- [x] Save model (`outputs/models/random_forest_baseline.joblib`)

---

## Phase 3 — Model Comparison (`03_comparison.ipynb`) ✅

**Findings from Phase 2 and focus areas:**
- Bot class precision=0.26 → SMOTE insufficient, try ADASYN or threshold tuning
- XSS and Brute Force still weak → gradient boosting models may outperform RF
- Baseline F1 Macro 0.845 → target ≥ 0.90

- [x] Notebook structure + helper `evaluate_model` function written
- [x] Compare models with the same pipeline + SMOTE:
  - [x] Random Forest (Phase 2 baseline — reference, hardcoded result)
  - [x] XGBoost (`n_estimators=300`, `max_depth=8`, `device='cuda'`)
  - [x] LightGBM (`n_estimators=300`, `num_leaves=63`)
  - [x] Logistic Regression (`solver='qn'`, cuML GPU)
- [x] Run all cells end-to-end on Colab (30% stratified sample, ~840K rows)
- [x] Test set evaluation results produced
- [x] Build comparison table → `outputs/results/model_comparison.csv`
- [x] F1 Macro bar chart → `outputs/figures/model_comparison.png`
- [x] Save models (`xgboost_comparison.joblib`, `lightgbm_comparison.joblib`, `logreg_comparison.joblib`)
- [x] Best model selected → **XGBoost** (F1 Macro=0.8593, ROC-AUC=1.0000, train=0.8 min)

**Results:**
| Model | F1 Macro | F1 Weighted | ROC-AUC |
|-------|----------|-------------|---------|
| XGBoost | **0.8593** | 0.9987 | 1.0000 |
| RF baseline | 0.8449 | 0.9966 | 1.0000 |

- Improvement over RF baseline: **+0.0144**
- Target ≥ 0.90 not yet reached → addressed in Phase 5 (hyperparameter tuning)

---

## Phase 4 — Feature Engineering (`04_feature_engineering.ipynb`) ✅

**Context:** XGBoost selected from Phase 3 (F1 Macro=0.8593). Goal: improve weak classes (Bot, XSS, Brute Force) via better features.

- [x] XGBoost feature importance plot (gain + cover) → `outputs/figures/feature_importance_xgb.png`
- [x] Drop zero/near-zero importance features
- [x] Drop highly correlated feature pairs from `high_corr_to_drop.json`
- [x] Retrain XGBoost on reduced feature set → compare F1 Macro before/after
- [x] Save reduced feature list → `outputs/results/selected_features.json`
- [x] Save retrained model → `outputs/models/xgboost_feat_eng.joblib`

**Results:**
| Model | F1 Macro | F1 Weighted | ROC-AUC |
|-------|----------|-------------|---------|
| XGBoost Phase 3 (all features) | 0.8593 | 0.9987 | 1.0000 |
| XGBoost Phase 4 (selected features) | **0.8605** | 0.9987 | 1.0000 |

- Improvement: **+0.0012** with fewer features → carries to Phase 5

---

## Phase 5 — Hyperparameter Optimisation (`05_hyperparameter.ipynb`) ✅

**Context:** XGBoost Phase 4 (F1 Macro=0.8605) is the base model. Target: F1 Macro ≥ 0.90.

- [x] RandomizedSearchCV (30 iter × 3-fold) run on Colab GPU (~38 dk)
- [x] best_params.json, hyperopt_cv_results.csv, hyperopt_comparison.csv saved

**Results:**
| Model | F1 Macro | F1 Weighted | ROC-AUC |
|-------|----------|-------------|---------|
| XGBoost Phase 4 (selected features) | **0.8605** | 0.9987 | 1.0000 |
| XGBoost Phase 5 (tuned) | 0.7871 | 0.9986 | 1.0000 |

**Decision: Phase 4 model is the final model.**
- Tuned model regressed (-0.073) — root cause: RandomizedSearchCV was run on SMOTE-augmented data, best params overfit to synthetic samples and did not generalise to the real test set.
- `outputs/models/xgboost_feat_eng.joblib` is the production model.

---

## General Outputs

| Folder | Contents |
|--------|----------|
| `outputs/models/` | `.joblib` per model + `label_encoder.joblib` |
| `outputs/results/` | `_cv_folds.csv` + `_summary.csv` per model |
| `outputs/figures/` | Confusion matrices, comparison charts |
