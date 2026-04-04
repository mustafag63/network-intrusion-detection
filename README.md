# Network Intrusion Detection

CICIDS2017 veri setiyle çok sınıflı ağ saldırısı tespiti.

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Proje Yapısı

```
network-intrusion-detection/
│
├── MachineLearningCVE/              # Ham veri (8 CSV, ~2.8M satır)
│
├── configs/
│   └── config.yaml                  # Tüm deney parametreleri
│
├── src/                             # Yeniden kullanılabilir modüller
│   ├── utils/data_loader.py         # Veri yükleme, temizleme, etiketleme
│   ├── features/preprocessing.py   # sklearn Pipeline (Scaler, SMOTE)
│   ├── models/train.py             # Model fabrikası, CV, kaydetme
│   └── evaluation/metrics.py       # Metrikler, grafikler
│
├── notebooks/                       # Geliştirme buradan yürütülür
│   ├── 01_eda.ipynb                 # Keşifçi veri analizi
│   ├── 02_baseline.ipynb            # İlk model + CV
│   ├── 03_comparison.ipynb          # Model karşılaştırması
│   ├── 04_feature_engineering.ipynb # Özellik seçimi
│   └── 05_hyperparameter.ipynb      # Hiperparametre optimizasyonu
│
├── outputs/
│   ├── models/                      # Kayıtlı modeller (.joblib)
│   ├── results/                     # CV sonuçları (.csv)
│   └── figures/                     # Grafikler (.png)
│
├── run.py                           # Notebook dışı CLI çalıştırma
├── PLAN.md                          # Geliştirme yol haritası
└── requirements.txt
```

---

## Çalışma Akışı

Geliştirme notebook'lar üzerinden yürütülür:

| Notebook | İçerik |
|----------|---------|
| `01_eda.ipynb` | Veri yükleme, sınıf dağılımı, korelasyon, eksik veri |
| `02_baseline.ipynb` | İlk model, 5-fold CV, confusion matrix |
| `03_comparison.ipynb` | LR / RF / XGBoost / LightGBM karşılaştırması |
| `04_feature_engineering.ipynb` | Feature importance, seçim, PCA |
| `05_hyperparameter.ipynb` | RandomizedSearch / Optuna optimizasyonu |

```bash
# Notebook sunucusunu başlat
jupyter notebook notebooks/
```

---

## Veri Seti

**CICIDS2017** — 8 gün, ~2.8M ağ akışı kaydı, 79 özellik.

| Sınıf | Açıklama |
|-------|----------|
| BENIGN | Normal trafik |
| DDoS | Dağıtık servis dışı bırakma |
| DoS | Hulk, GoldenEye, Slowloris, Slowhttptest |
| PortScan | Port tarama |
| Bot | Botnet trafiği |
| Web Attack | Brute Force, XSS, SQL Injection |
| Patator | FTP / SSH kaba kuvvet |
| Infiltration | Sızma denemesi |
| Heartbleed | Heartbleed açığı |

---

## Modeller

| Anahtar | Model |
|---------|-------|
| `logistic_regression` | Logistic Regression |
| `random_forest` | Random Forest |
| `gradient_boosting` | Gradient Boosting |
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `svm` | Support Vector Machine |

---

## Değerlendirme

- **5-Fold Stratified CV** — her fold için F1 Macro, Accuracy
- **Test seti** — F1 Macro, Accuracy, ROC-AUC (OvR)
- **Grafikler** — Confusion Matrix, model karşılaştırma çubuk grafiği
- **Çıktılar** — `outputs/results/*.csv`, `outputs/figures/*.png`

---

## config.yaml Hızlı Referans

```yaml
model:
  name: "random_forest"   # hangi modeli eğiteceğini buradan değiştir

data:
  sample_frac: 0.1        # hızlı test için 0.1, tam çalışma için 1.0

task: "multiclass"        # multiclass | binary
```
