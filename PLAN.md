# Network Intrusion Detection — Geliştirme Planı

Her faz bir notebook'a karşılık gelir.
Tamamlananlar ✅, devam edenler 🔄, bekleyenler ⬜.

---

## Faz 1 — Keşifçi Veri Analizi (`01_eda.ipynb`) ⬜

- [ ] Tüm CSV'leri yükle, birleştir, temizle
- [ ] Sınıf dağılımını görselleştir (`sinif_dagilimi.png`)
- [ ] Eksik değer / sonsuz değer analizi
- [ ] Feature korelasyon ısı haritası (`korelasyon.png`)
- [ ] Özellik dağılımlarını sınıfa göre karşılaştır
- [ ] Veri dengesizliği oranını raporla

---

## Faz 2 — Baseline Model (`02_baseline.ipynb`) ⬜

- [ ] Label encoding + train/test split (%80/%20)
- [ ] Pipeline kur: VarianceThreshold → StandardScaler
- [ ] Random Forest ile 5-fold stratified CV
- [ ] CV sonuçlarını kaydet (`outputs/results/random_forest_cv_folds.csv`)
- [ ] SMOTE ile yeniden dengele, final model eğit
- [ ] Test seti değerlendirmesi: F1 Macro, Accuracy, ROC-AUC
- [ ] Confusion matrix kaydet (`outputs/figures/confusion_random_forest.png`)
- [ ] Modeli kaydet (`outputs/models/random_forest_baseline.joblib`)

---

## Faz 3 — Model Karşılaştırması (`03_comparison.ipynb`) ⬜

- [ ] Aynı pipeline ile 5 model çalıştır:
  - [ ] Logistic Regression
  - [ ] Random Forest
  - [ ] XGBoost
  - [ ] LightGBM
  - [ ] Gradient Boosting
- [ ] Her model için CV sonuçlarını kaydet
- [ ] Karşılaştırma tablosu oluştur
- [ ] F1 Macro bar grafiği (`outputs/figures/model_karsilastirma.png`)

---

## Faz 4 — Özellik Mühendisliği (`04_feature_engineering.ipynb`) ⬜

- [ ] Random Forest / XGBoost feature importance grafiği
- [ ] Düşük önemli feature'ları ele
- [ ] Yüksek korelasyonlu feature çiftlerini çıkar
- [ ] PCA ile boyut indirgemeyi dene (6000 → N)
- [ ] Feature seçimi sonrası model performansını karşılaştır

---

## Faz 5 — Hiperparametre Optimizasyonu (`05_hyperparameter.ipynb`) ⬜

- [ ] Faz 3'teki en iyi model için RandomizedSearchCV uygula
- [ ] Arama uzayını `config.yaml`'a ekle
- [ ] Optimum parametreleri raporla
- [ ] Baseline vs. optimized karşılaştırması

---

## Genel Çıktılar

| Klasör | İçerik |
|--------|--------|
| `outputs/models/` | Her model için `.joblib` + `label_encoder.joblib` |
| `outputs/results/` | Her model için `_cv_folds.csv` + `_summary.csv` |
| `outputs/figures/` | Confusion matrix, karşılaştırma grafikleri |
