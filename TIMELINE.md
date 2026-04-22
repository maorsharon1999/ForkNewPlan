# Development Timeline — Fork ET Pipeline

## Stage 1 — Architecture Design

**Goal:** Design a modular pipeline for analyzing IMU sensor data during fork-based eating tasks.

**Result:** A 7-stage architecture:
1. Scan participants → 2. Load CRF → 3. Preprocess → 4. Extract features → 5. Regression → 6. Classification → 7. Visualization

**Files:** `config.py`, `utils.py` — base configuration and utilities.

---

## Stage 2 — Data Loading & Preparation

**Goal:** Implement automatic scanning of patient folders, CSV parsing, CRF Excel reading.

**Result:**
- Recursive scanning of `New Data/משתתפים/` with `ET-XXX` and `Control-XXX` support
- Hand determination from filename prefix (`Fork1`=Right, `Fork2`=Left)
- CRF Excel parsing with Hebrew label support
- Found 95 CSV files from 24 patients

**Files:** `data_loader.py`

---

## Stage 3 — Signal Preprocessing

**Goal:** Filtering, magnitude computation, eating activity segment detection.

**Result:**
- 4th-order Butterworth bandpass filter at 0.5–20 Hz (zero-phase)
- Euclidean accelerometer magnitude
- Threshold-based activity detection (|mag − 1g| > 0.3g) with gap filling and minimum duration

**Files:** `preprocessing.py`

---

## Stage 4 — Basic Feature Extraction

**Goal:** Implement initial feature set: time-domain and frequency-domain.

**Result:**
- 66 features: time-domain (mean, std, rms, range, IQR, kurtosis, skewness) + frequency-domain (dominant frequency, ET-band 4–12 Hz power)
- Per-segment extraction mode

**Files:** `feature_extraction.py` (v1)

---

## Stage 5 — Basic ML Pipeline

**Goal:** Implement regression (score prediction) and classification (ET vs Control).

**Result:**
- 3 models: LinearRegression, RandomForest, XGBoost
- KFold cross-validation
- Basic metrics: MAE, RMSE, R², Accuracy, F1, AUC
- **Results:** R²=0.86 (RF, KFold) — inflated due to data leakage

**Files:** `ml_pipeline.py` (v1), `main.py` (v1)

---

## Stage 6 — Visualization

**Goal:** Automatic plot generation.

**Result:**
- PCA 2D (ET vs Control), Feature boxplots, Scatter (pred vs true), ROC, Activity segments

**Files:** `visualization.py` (v1)

---

## Stage 7 — Advanced Features

**Goal:** Add advanced feature groups for improved prediction.

**Result (66 → 159 features):**
- Jerk (acceleration derivative) — 18 features
- Cross-axis correlation — 9 features
- Magnitude features — 10 features
- Spectral shape (entropy, flatness, centroid, rolloff) — 24 features
- Wavelet CWT energy (4–12 Hz, Morlet) — 18 features
- Manual NumPy CWT implementation (replacing deprecated scipy functions)

**Files:** `feature_extraction.py` (v2)

---

## Stage 8 — Advanced ML

**Goal:** Implement LOSO CV, GridSearchCV, stacking ensemble, per-segment predictions.

**Result:**
- **LOSO CV** — eliminates data leakage (one patient's segments never split across train/test)
- **GridSearchCV** — automatic RF and XGBoost hyperparameter tuning
- **StackingEnsemble** — stacking with RidgeCV meta-learner
- **Per-segment** — predictions per segment → median aggregation per patient

**Honest results (LOSO):** R²=0.664 (LR), AUC=0.613 — expected drop from correcting leakage

**Files:** `ml_pipeline.py` (v2), `config.py` (v2)

---

## Stage 9 — Including Fork_ Files

**Goal:** Add previously skipped `Fork_*.csv` files (no hand digit).

**Result:**
- Configurable default hand (`FORK_DEFAULT_HAND = "Right"`)
- `INCLUDE_AMBIGUOUS_FORK` flag
- **95 → 105 CSV** files

**Files:** `data_loader.py` (v2), `config.py` (v3)

---

## Stage 10 — Transferring Cup/Brush Best Practices

**Goal:** Study Cup (`ML.py`, `featureExtraction.py`) and Brush (`ml_pipeline.py`, `feature_extraction.py`) pipelines and transfer best approaches.

**Result (159 → 184 features):**

### From Cup:
- **Weighted spectral features** — weighted mean/median/max frequency, std, skewness, FFT amplitudes
- **Peak-to-peak features** — peak intervals, peak frequency, signal duration
- **Data augmentation** — noise injection + mixup (`augment_and_balance()`)
- **GradientBoosting, Ridge, Lasso** — new regression models
- **SVC with class_weight** — balanced classification

### From Brush:
- **RF importance selection** — feature selection via RandomForest importance
- **Regression-to-ROC** — binarization for regression evaluation

### New metrics and visualizations:
- Sensitivity, Specificity, PPV, NPV, Pearson r, Spearman ρ
- Confusion Matrix, Bland-Altman plot

**Results:**
- Local R²: 0.664 → **0.793** (XGBoost)
- Global R²: 0.694 → **0.841** (XGBoost)
- AUC: 0.564 → **0.799** (RandomForest)

**Files:** all 8 files updated

---

## Stage 11 — Improving Classification Specificity

**Goal:** Address low Specificity (0.519) by optimizing classification thresholds and implementing regress-then-classify.

**Problem:** Default threshold of 0.5 causes ~50% of healthy subjects to be flagged as ET — clinically unacceptable.

**Two solutions implemented:**

### 1. Youden's J Threshold Optimization
Finds the probability cutoff that maximizes `Sensitivity + Specificity − 1`. Applied automatically to every classifier.

### 2. Regress-then-Classify
Predicts clinical tremor score for ALL patients (Controls get score=0), then classifies ET if score > threshold. Leverages the strong regression model (R²=0.82).

**Results:**

| Approach | Sensitivity | Specificity | Balance |
|----------|-------------|-------------|---------|
| Default (threshold=0.5) | 0.858 | 0.500 | ❌ poor |
| **Youden optimal (threshold=0.86)** | **0.727** | **0.708** | **✅ balanced** |
| SVC (default) | 0.786 | 0.689 | ✅ good |

**Files:** `ml_pipeline.py` (v3), `main.py` (v3)

---

## Stage 12 — Advanced Features & Interpretability

**Goal:** Add tremor-specific features, multi-resolution analysis, dual bandpass, temporal features, SHAP interpretability, and calibrated classifiers.

**Result (184 → 214 features):**

### New feature groups:
- **Tremor-specific** — Tremor Stability Index (TSI), Tremor Power Ratio, Harmonic-to-Noise Ratio (HNR)
- **Multi-resolution CWT** — CWT energy variance/CV/trend across 2-sec windows (tremor temporal stability)
- **Dual bandpass (3–15 Hz)** — narrow tremor-band RMS, std, max, energy ratio
- **Temporal** — autocorrelation at ET-band lags, sample entropy (regularity)

### New pipeline stages:
- **Stage 6c: CalibratedClassifierCV** — Platt scaling for reliable probability calibration
- **Stage 6d: SHAP analysis** — TreeExplainer bar + beeswarm plots for interpretability

### SHAP Top Features:
**Regression:** `gyro_p2p_mean` (0.40), `gyro_y_dom_freq` (0.14), `gyro_z_jerk_max` (0.14)
**Classification:** `gyro_x_spec_rolloff` (0.99), `acc_x_power_4_12hz` (0.71), `gyro_sample_entropy` (0.53)

### Results:

| Model | Sensitivity | Specificity | Youden J |
|-------|-------------|-------------|----------|
| LR_Youden | 0.745 | **0.755** | 0.500 |
| RF_calibrated_Youden | 0.932 | **0.575** | **0.507** |
| RF (default) | 0.935 | 0.547 | — |
| RF_Youden | 0.935 | 0.566 | 0.501 |

**Files:** `feature_extraction.py` (v3), `ml_pipeline.py` (v4), `main.py` (v4), `requirements.txt`

---

## Stage 13 — Next-Level Improvements (Handling Sparsity and Imbalance)

**Goal:** Implement Dimensionality Reduction (RFE), Dataset Multiplication via Sliding Windows, synthetic minority oversampling (SMOTE) with CV integration, and Demographic Features (Age, Gender).

**Result (216 features):**

### Capabilities
- **RFE Feature Selection** — dynamically selects top 15 most predictive features to combat the 'curse of dimensionality'
- **SMOTE Class Balancing** — synthetically generates minority samples *inside* the KFold loop to prevent leakage
- **Demographic Features** — extracted Age and Gender from CRF parsing and automatically imputed missing data

### Results:

| Model | Sensitivity | Specificity | Youden J | AUC |
|-------|-------------|-------------|----------|-----|
| LR_Youden | 0.792 | **0.487** | 0.279 | 0.627 |
| RF_calibrated_Youden | **0.842** | 0.357 | 0.199 | 0.616 |
| RF (default) | 0.788 | 0.374 | —     | 0.636 |

**Files:** `config.py`, `preprocessing.py`, `ml_pipeline.py`, `data_loader.py`, `main.py`, `feature_extraction.py`, `visualization.py`

---

## Stage 14 — Automated Clinical PDF Reporting

**Goal:** Generate professional PDF reports with all metrics and visualizations for clinicians.

**Result:**
- PDF generation using `fpdf2`
- Integration of all CV metrics, SHAP top features, and RFE stability
- Embedded plots (Confusion Matrix, ROC, etc.)

**Files:** `report_generator.py`, `main.py`

---

## Stage 15 - Code Quality & Bug Fixes

**Goal:** Fix data loss from combined folders, clean unused imports, improve visualisation accuracy.

**Changes:**
- **`_parse_folder_name`**: switched from `re.match` (first match only) to `re.findall` (all matches) - recovered patient **ET-020** from `Control-003 and ET-020` folder
- **`scan_participants`**: inner `os.walk` loop now iterates per-patient inside `for group, patient_id in parsed`
- **`main.py` imports**: removed unused `GroupKFold`, `LeaveOneGroupOut`, `LinearRegression`
- **Scatter/Bland-Altman plots**: switched from `LinearRegression` to `RandomForestRegressor` for the best visual representation

**Impact:** 87 recordings / 461 segments (was 85/443), +1 ET patient

**Files:** `data_loader.py`, `main.py`

---

## Stage 16 - Data Leakage Fix

**Goal:** Eliminate data leakage by moving StandardScaler + RFE inside each CV fold.

**Problem:** StandardScaler and `select_features(RFE)` were applied to the **entire** dataset before cross-validation, meaning the test patient was "seen" during feature scaling and selection.

**Fix:**
- `run_regression`: wrapped scaler + RFE + model in an `sklearn.pipeline.Pipeline`, passed to `cross_val_predict` so each fold trains its own scaler and RFE
- `run_classification`: manual CV loop now applies `StandardScaler().fit_transform()` and `RFE.fit()` on train fold only, then `.transform()` on test fold, before SMOTE
- Stacking and Youden's J also use Pipeline wrappers
- Hyperparameter tuning removed (was also leaky via global X_sel)
- `run_calibrated_classification` and `run_shap_analysis` NOT touched

**Honest Results (no leakage):**

| Task | Best Model | R2 / AUC | Pearson r |
|------|-----------|----------|----------|
| Local Regression | RF | 0.770 | 0.886 |
| Global Regression | GradBoost | 0.848 | 0.922 |
| Classification | XGB | AUC=0.625 | - |

**Files:** `ml_pipeline.py`

---

## Stage 17 - Complete Leakage Elimination

**Goal:** Eliminate remaining data leakage in all remaining functions and visualizations.

**Remaining Leakage Sources Fixed:**
- `run_regress_then_classify`: wrapped scaler + RFE + model in Pipeline
- `run_calibrated_classification`: moved scaler + RFE inside each CV fold (before SMOTE)
- `main.py` Stage 7 scatter/Bland-Altman: Pipeline(scaler→RFE→RF) for regression viz
- `main.py` Stage 7 confusion matrix/ROC: Pipeline(scaler→RFE→LR) for classification viz

**Also:**
- Added `rtc_results` to `report_metrics` dict → now included in PDF
- Removed unused imports (`select_features`, `regression_to_roc`) from `main.py`

**Files:** `ml_pipeline.py`, `main.py`, `README.md`

---

## Stage 18 - Stabilizing CV with VarianceThreshold

**Goal:** Fix `RuntimeWarning: invalid value encountered in divide` caused by features dropping to zero variance within specific cross-validation splits.

**Problem:** Standardizing a feature with zero variance causes a division by zero, leading to NaNs. This happened because certain features (like specific peak intervals or entropy measures) became completely constant for all samples remaining in a particular training fold after a subject was left out.

**Fix:**
- Injected `VarianceThreshold()` as the absolute first step in all `Pipeline` wrappers (`_make_pipeline`, `_make_cls_pipeline`, `pipe_j`, `_make_rtc_pipeline`).
- Added manual `VarianceThreshold.fit_transform()` on training data and `transform()` on test data inside the explicit cross-validation loops in `run_classification` and `run_calibrated_classification` BEFORE applying `StandardScaler`.
- Added a warning filter in `main.py` to suppress non-critical `UserWarning`s (like those from KNNImputer regarding missing feature names), while strictly preserving `RuntimeWarning`s to ensure any future mathematical anomalies remain visible.

**Impact:** Pipeline executes cleanly without divide-by-zero warnings, ensuring math stability and preventing NaNs from propagating through models or RFE.

**Files:** `ml_pipeline.py`, `main.py`

---

## Stage 19 - RFE Feature Stability Tracking

**Goal:** Track how consistently features are selected by recursive feature elimination (RFE) across Leave-One-Subject-Out (LOSO) cross-validation folds.

**Changes:**
- Initialized a `defaultdict` counter at the start of `run_classification` to track feature selection.
- Injected logic inside the cross-validation loop to find the exact features surviving both `VarianceThreshold` and `RFE` for the `LogisticRegression` model.
- Appended `__rfe_stability__` (percentage a feature was selected in N folds) to `cls_results`.
- Updated `main.py` to extract the stability tracking metric before logging.
- Modified `report_generator.py` to render **4b. RFE Feature Stability** displaying the top consistently selected features explicitly.

**Impact:** Provides quantitative insight into which of the 216 features are most robust across different patients, improving clinical explainability.

**Files:** `ml_pipeline.py`, `main.py`, `report_generator.py`

---

## Stage 20 - Porting Spoon Improvements Back to Fork

**Goal:** Re-sync Fork with improvements developed during the Spoon pipeline experiment.

**Changes:**

### `config.py`
- `ACTIVITY_THRESHOLD`: 0.3 → **0.25** (lower threshold detects more subtle movements)
- `USE_AUGMENTATION`: True → **False** (disabled — caused overfitting and long runtimes)
- `TUNE_HYPERPARAMS`: True → **False** (disabled — caused overfitting and long runtimes)
- `USE_STACKING`: True → **False** (disabled — caused overfitting and long runtimes)

### `preprocessing.py`
- Added `infer_hand_from_signal(df)` — infers hand from gyroscope signal physics (pronation/supination direction). Used as fallback when file name does not specify a hand.
- Added `from scipy.stats import skew` import.

### `visualization.py`
- Added `plot_and_save_patient_signal()` — saves a 2-panel (accelerometer + gyroscope) signal plot per recording to `output/figures/patient_signals/`.

### `main.py`
- Added `import os`
- **Stage 2**: added raw CRF data table logging (`_get_crf_data()` → DataFrame → logger) for debug visibility
- **Stage 4**: renamed to "Feature Extraction & Patient Signal Plotting"; added `file_counts` dict and `viz.plot_and_save_patient_signal()` call per recording
- Removed `age` / `gender` from `extract_all_features()` call (Fork's `feature_extraction.py` does not accept these as positional params)
- Added `"imu": imu` to `processed` list so Stage 4 can access raw filtered DataFrame for plotting

### `ml_pipeline.py`
- `compute_patient_metrics()`: patient-level score aggregation changed from **mean** → **75th percentile** (`np.percentile(x, 75)`) — takes the upper quartile of segment scores, which is more robust against low-activity segments

**Files:** `config.py`, `preprocessing.py`, `visualization.py`, `main.py`, `ml_pipeline.py`, `README.md`, `TIMELINE.md`

---

## Stage 21 - Advanced Pipeline Dynamics and Noise Elimination

**Goal:** Automate hand inference per test, handle append-mode CSV recording, and eliminate non-feeding noise.

**Changes:**
- **CSV Deduplication:** Modified `data_loader.py` to identify device-appended data (only keeping the largest file per physical device label) and returning a unified, continuous stream.
- **Behavioral Test Separation:** Re-orchestrated `main.py` Stages 2-4. Single bulk recordings are organically severed into multiple, isolated test segments whenever a physical pause of >10,000ms occurs between *actual eating patterns* (reach-pierce-bring).
- **Dynamic Hand Inference:** Hand side (left/right) is no longer assumed from the filename or patient defaults. `infer_hand_from_signal()` runs dynamically on *every slice test* via majority voting (`Counter`) of movement physics inside that specific test.
- **Eating Phase Recognition (Noise Filter):** Added `filter_eating_cycles(segments, raw_df, fs)` to `preprocessing.py`. This forces all generalized movement segments to pass a strict physics heuristic (Reach pitch change > 0.2g and Pierce impact jerk > 1.5x mean) before feature extraction.

**Impact:** Dramatically improves feature matrix purity by guaranteeing CRF scores match the exact hand used during a sequence, isolating genuine feeding tasks from conversation or rest.

**Files:** `data_loader.py`, `preprocessing.py`, `main.py`, `README.md`, `TIMELINE.md`

---

## Stage 22 - Phase 1 Research: Cup & Toothbrush Knowledge Extraction

**Goal:** Establish empirical grounding for the next-generation segmentation and movement-classification pipeline by interrogating prior work on sister IMU projects (smart cup, smart toothbrush) before writing any new code.

**Method:**
- Drafted ~60 highly targeted questions across 10 sections (sensor setup, preprocessing, activity detection, cycle segmentation, gravity/tilt, handedness, movement classification, tremor handling, feature engineering, evaluation).
- Queried both source documents via NotebookLM.
- Consolidated raw answers into a structured reference file.

**Key findings (relevant to Fork redesign):**
- Cup/Brush use **fixed bandpass filters** (1–15 Hz IMU, 2–15 Hz Cup, 4–12 Hz video) — no adaptive filtering, no sensor fusion, gyroscope and accelerometer treated as independent magnitudes.
- Activity detection is **magnitude-based and tilt-invariant** — gravity removed via ideal HPF + Euclidean magnitude `|a|`. Cup rule: `|a'| + |ω'| + ||a|−1| ≤ τ` AND `|ω − ω_min| ≤ τ₂`.
- Cycle separation is **stringent**: consecutive cycles without idle gap are merged ("total miss") — accepted because it does not significantly affect tremor features. No DTW, no template matching, no sub-phase decomposition.
- Feature extraction uses **10-second sliding windows, 80% overlap** — not per-cycle. No inter-cycle features.
- Handedness was **never algorithmically detected** in either study — always known from protocol. Inter-hand variability ≈ inter-subject variability.
- Movement type (scoop vs. stab vs. drink) was **never classified** — only tremor severity. The Fork project's per-gesture classification is genuinely novel work.
- Top features across studies: gyroscope frequency skewness, max gyroscope magnitude amplitude (`A_ω_max`), video path length XY (`path_len_xy`).
- Hard thresholds were heuristically tuned to dataset → **will require recalibration for Fork hardware and cohort**.
- ET subjects had 2.24 s mean boundary detection error vs 0.99 s for controls — tremor interference directly degrades segmentation quality.

**Output:** `research_knowledge_base.md` — single structured reference, organized by sections A–J matching the original question set.

**Files:** `research_knowledge_base.md` (new), `README.md`, `TIMELINE.md`

**Next:** Phase 3 — Algorithmic design synthesis: new segmentation, handedness inference, and movement-type classification pipeline for Fork, informed by these findings.
