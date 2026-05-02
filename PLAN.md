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

## Phase 3 — Model Comparison (`03_comparison.ipynb`) 🔄

**Findings from Phase 2 and focus areas:**
- Bot class precision=0.26 → SMOTE insufficient, try ADASYN or threshold tuning
- XSS and Brute Force still weak → gradient boosting models may outperform RF
- Baseline F1 Macro 0.845 → target ≥ 0.90

- [ ] Compare models with the same pipeline + SMOTE:
  - [ ] Random Forest (Phase 2 baseline — reference)
  - [ ] XGBoost
  - [ ] LightGBM
  - [ ] Logistic Regression (lower bound)
- [ ] 5-fold CV + test set evaluation for each model
- [ ] Compare ADASYN vs SMOTE for Bot class
- [ ] Build comparison table (F1 Macro, per-class F1, training time)
- [ ] F1 Macro bar chart (`outputs/figures/model_comparison.png`)
- [ ] Select best model → carry to Phase 4

---

## Phase 4 — Feature Engineering (`04_feature_engineering.ipynb`) ⬜

- [ ] Random Forest / XGBoost feature importance plot
- [ ] Handle low-importance features
- [ ] Drop highly correlated feature pairs
- [ ] Try PCA dimensionality reduction
- [ ] Compare model performance before and after feature selection

---

## Phase 5 — Hyperparameter Optimisation (`05_hyperparameter.ipynb`) ⬜

- [ ] Apply RandomizedSearchCV to the best model from Phase 3
- [ ] Add search space to `config.yaml`
- [ ] Report optimal parameters
- [ ] Baseline vs. optimised comparison

---

## General Outputs

| Folder | Contents |
|--------|----------|
| `outputs/models/` | `.joblib` per model + `label_encoder.joblib` |
| `outputs/results/` | `_cv_folds.csv` + `_summary.csv` per model |
| `outputs/figures/` | Confusion matrices, comparison charts |
