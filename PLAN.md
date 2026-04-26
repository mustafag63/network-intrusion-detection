# Network Intrusion Detection — Geliştirme Planı

Her faz bir notebook'a karşılık gelir.
Tamamlananlar ✅, devam edenler 🔄, bekleyenler ⬜.

---

## Faz 1 — Exploatory Data Analysis (`01_eda.ipynb`) ✅

- [x] Tüm CSV'leri yükle, birleştir, temizle
  - Kopya sütun, sonsuz değer, NaN, negatif değer, sabit sütun, kopya satır
  - `Init_Win_bytes` sentinel flag eklendi; `tcp_win_seen_*` sütunları üretildi
- [x] Sınıf dağılımını görselleştir (`sinif_dagilimi.png`)
  - Aile bazlı gruplama (7 aile), IR Binary & IR Macro hesaplandı
  - Kritik küçük sınıflar tespit edildi → Faz 2'de SMOTE k değeri küçültülmeli
- [x] Eksik değer / sonsuz değer analizi
  - `duration=0` → `inf` kökeni belgelendi; skewness analizi yapıldı (`skewness_dagilimi.png`)
- [x] Feature korelasyon ısı haritası (`korelasyon.png`, `yuksek_korelasyon.png`)
  - `|r| ≥ 0.95` çiftler tespit edildi → `high_corr_to_drop.json` (Faz 4'te uygulanacak)
- [x] Özellik dağılımlarını sınıfa göre karşılaştır
  - ANOVA F-skoru ile sıralandı (`feature_f_scores.csv`)
  - BENIGN vs ATTACK histogram + aile bazlı box plot üretildi
- [x] Veri dengesizliği oranını raporla
  - IR Binary, IR Macro hesaplandı; özet rapora eklendi

---

## Faz 2 — Baseline Model (`02_baseline.ipynb`) ✅

- [x] Label encoding + train/test split (%80/%20)
- [x] Pipeline kur: VarianceThreshold → StandardScaler
  - 4 sütun çıkarıldı (`Fwd URG Flags`, `RST Flag Count`, `CWE Flag Count`, `ECE Flag Count`)
  - `preprocessor.joblib` ve `kept_features.json` kaydedildi
- [x] Random Forest ile 5-fold stratified CV (%10 örnekleme)
  - `class_weight='balanced'` eklendi
  - Çok nadir sınıflar CV havuzundan çıkarıldı (Heartbleed, Infiltration, Web Attack–SQL Injection — her biri < 5 örnek)
    - Kök neden: bu sınıflar bazı fold'larda val'a düşmüyor → model sütun üretmiyor → roc_auc NaN
  - Scorer: `roc_auc_ovr_weighted` (standard, custom scorer gereksiz)
  - val_f1_macro ≈ 0.83, val_roc_auc artık NaN değil
- [x] Fold başına `classification_report` — hangi sınıfların F1'i düşük tespit edildi
  - `outputs/results/random_forest_cv_class_report.csv` kaydedildi
- [x] CV sonuçlarını kaydet (`outputs/results/random_forest_cv_folds.csv`)
- [x] SMOTE ile yeniden dengele, final model eğit
  - 10 sınıf 5.000 örneğe çıkarıldı (+30.879 sentetik örnek)
  - `max_depth=25`, `min_samples_leaf=4` (overfitting azaltmak için)
- [x] Test seti değerlendirmesi: F1 Macro, Accuracy, ROC-AUC
  - Test Accuracy: **0.9966** | F1 Macro: **0.845** | F1 Weighted: 0.997 | ROC-AUC: 1.000
  - Sorunlu sınıflar: Bot (F1=0.41, precision=0.26 — çok fazla FP), XSS (0.47), Brute Force (0.72)
  - Heartbleed/SQL Injection test seti < 5 örnek — bu sınıfların F1'i istatistiksel olarak anlamsız
- [x] Confusion matrix kaydet (`outputs/figures/confusion_random_forest.png`)
- [x] Modeli kaydet (`outputs/models/random_forest_baseline.joblib`)

---

## Faz 3 — Model Karşılaştırması (`03_comparison.ipynb`) 🔄

**Faz 2'den gelen bulgular ve odak noktaları:**
- Bot sınıfında precision=0.26 → SMOTE yetersiz, ADASYN veya threshold tuning denenecek
- XSS ve Brute Force hâlâ zayıf → gradient boosting tabanlı modeller RF'e göre üstün olabilir
- Baseline F1 Macro 0.845 → hedef ≥ 0.90

- [ ] Aynı pipeline + SMOTE ile modelleri karşılaştır:
  - [ ] Random Forest (Faz 2 baseline — referans)
  - [ ] XGBoost
  - [ ] LightGBM
  - [ ] Logistic Regression (alt tavan için)
- [ ] Her model için 5-fold CV + test seti değerlendirmesi
- [ ] Bot sınıfı için ADASYN'ı SMOTE ile karşılaştır
- [ ] Karşılaştırma tablosu oluştur (F1 Macro, F1 per-class, eğitim süresi)
- [ ] F1 Macro bar grafiği (`outputs/figures/model_karsilastirma.png`)
- [ ] En iyi modeli seç → Faz 4'e taşı

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
