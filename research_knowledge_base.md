# Research Knowledge Base — Cup Thesis & Toothbrush Study
## Phase 1: Interrogation Results (NotebookLM Query, April 2026)

This document consolidates answers extracted from the Cup Master's Thesis and Toothbrush Study.
It serves as the empirical foundation for Phase 3 algorithmic design of the new Fork segmentation pipeline.

---

## A. Sensor Setup & Physical Orientation

| Question | Answer |
|----------|--------|
| Sensor mounting location | Bottom/base of handle (toothbrush). Cup: embedded in utensil. Exact orientation relative to object axes not specified. |
| Coordinate frame | Cartesian X, Y, Z. Gravity treated as a DC component across axes. No specific rest-frame convention documented. |
| Sampling frequency | Variable raw capture → linearly interpolated to **fixed 50 Hz** (median inter-sample interval used to estimate effective rate). |
| Hardware / IMU chip | Not specified in sources. MPU-6050 mentioned as a commonly used 6-axis IMU in similar prototypes. |
| Gravity at rest (axis values) | Not documented. Gravity handled analytically as a DC component — removed via high-pass filter. |
| Known sensor artifacts | Three categories: (1) extreme outliers from noise/tracking errors, (2) transient sharp spikes from mechanical perturbations, (3) prolonged inactivity periods. No grip-pressure or vibration-coupling effects documented. |

---

## B. Preprocessing & Noise Removal

### Filters used (by project and stage)

| Project / Stage | Filter Type | Parameters |
|-----------------|-------------|------------|
| Cup — gravity removal | Ideal Fourier high-pass filter | Removes DC/low-frequency components |
| Cup — tremor isolation | 6th-order Butterworth band-pass | **2–15 Hz** (feature extraction only) |
| Cup — smoothing (before segmentation) | Median filter | Window N=5 (≈100 ms) |
| Cup — smoothing (after median) | Savitzky-Golay filter | Removes discontinuities without flattening |
| Toothbrush IMU | Ideal Fourier band-pass | **1–15 Hz** |
| Toothbrush video | 4th-order zero-phase Butterworth band-pass | **4–12 Hz** |
| Both — DC drift | Ideal Fourier high-pass | Removes baseline drift |

### Key preprocessing decisions

- **Gravity removal:** Ideal high-pass filter in frequency domain — removes DC component. Euclidean magnitude `|a| = √(Ax²+Ay²+Az²)` used as tilt-invariant signal. No explicit gravity vector subtraction.
- **Sensor fusion:** None. Accelerometer and gyroscope treated as **independent signals** throughout.
- **Gyroscope usage:** Same preprocessing pipeline as accelerometer from the very first step (outlier removal → interpolation → filtering → magnitude computation).
- **Normalization:** Z-score normalization (`X_scaled = (X − μ) / σ`) applied **after** feature extraction, not to raw signals. Signal amplitude normalization only used for visualization.
- **Baseline drift:** Corrected by ideal high-pass filter (same as gravity removal step).

---

## C. Activity Detection & Idle/Noise Segmentation

### Detection method
- **Cup project:** Combined criterion on magnitudes and first derivatives:
  - A point is *active* if: `|a'| + |ω'| + ||a| − 1| ≤ threshold_1` AND `|ω − ω_min| ≤ threshold_2`
  - Activity start `T_s`: beginning of cluster where active point ratio in surrounding window ≥ minimum proportion
  - Activity end `T_f`: window where inactive point ratio exceeds `p(w−1)`
- **Toothbrush project:** Standard deviation / variance of acceleration magnitude. Low-variance regions excluded.
- **Video:** Rolling std of velocity magnitude over 1-second window.

### Threshold determination
- **Empirical / heuristic** — chosen via heuristic search minimizing detection error on their specific dataset. Not analytically derived.
- Video thresholds: 5th-percentile-based (velocity magnitude samples below 5th percentile dropped).

### Minimum segment duration
- IMU: **5 seconds minimum** (shorter sessions discarded)
- Video: **2 seconds minimum**
- Maximum: **not specified**

### Spike suppression
1. Compute first derivative; flag any slope > **99.5th percentile** as a spike → replace with NaN
2. Linear interpolation over NaN gaps
3. Median filter (N=5) + Savitzky-Golay for residual smoothing

### Gap / refractory period handling
- IMU: adjacent sessions with gap < **5 seconds** → merged into one continuous session
- Video: new session only if inactivity gap > **3 seconds**

### Outlier removal
- Percentile-based: values outside **0.05th–99.5th percentile** range stripped → NaN → linear interpolation

### Intentional vs. incidental movement
- Fixed bandpass filter separates tremor frequency (1–15 Hz IMU, 4–12 Hz video) from lower-frequency voluntary motion
- Ground-truth validation: sensor data synchronized with independent video evaluation by trained clinician

---

## D. Cycle Segmentation

| Aspect | Answer |
|--------|--------|
| Start detection anchor | Magnitudes `\|a\|` and `\|ω\|` + their first derivatives |
| End detection | Same sliding-window; inactive point ratio exceeds threshold |
| Consecutive cycles without idle gap | Often merged into a single phase ("total miss") — authors accepted this as it does not significantly affect tremor features |
| Fragmented / short movements | Discarded via duration thresholds (< 5s IMU, < 2s video). Intra-session gaps < 5s merged. |
| DTW / template matching | **Not used** |
| Complete vs. incomplete cycle definition | Not explicitly defined. Duration threshold acts as proxy — too-short = discarded, not flagged. |
| Sub-phase decomposition (reach/lift/drink) | **Not done.** Algorithm identifies only global `T_s` and `T_f`. No internal phase boundaries. |
| Brush stroke counting | **Not done.** Continuous sliding 10-second windows with 80% overlap used instead. |

---

## E. Gravity & Tilt Handling

- **No dynamic gravity compensation.** Gravity not tracked or subtracted in the spatial domain.
- **Two-step approach:**
  1. Ideal high-pass filter removes DC component (including static gravity projection)
  2. Euclidean magnitude `|a| = √(Ax²+Ay²+Az²)` — inherently tilt-invariant (magnitude is invariant to rotation)
- **No orientation computation.** Quaternion integration, complementary filters, and Kalman filters were not used.
- **No adaptive threshold.** Detection thresholds applied to magnitudes → naturally tilt-invariant by design.
- Gyroscope angular velocity converted to `|ω|` magnitude — used for features, not orientation reconstruction.

---

## F. Handedness Detection

| Aspect | Answer |
|--------|--------|
| How determined | **Known in advance** — experimentally labeled as part of protocol |
| Algorithmic detection | Not performed |
| Axis-based discriminators | Not studied |
| Calibration gesture | None — unconstrained natural monitoring was an explicit design goal |
| Inter-hand variability | Hand-to-hand variability within a subject ≈ inter-subject variability. Features capture task-level tremor robustly, not individual grip signatures. |
| Systematic orientation differences (L vs R) | Not documented — independent statistical observations for each hand |

---

## G. Movement Type Classification

- **No movement type classification was attempted.** Tasks (fork scooping, stabbing, drinking) were performed under protocol but ML models targeted only *tremor severity*, not *movement type*.
- **No features** were defined or selected to discriminate scooping vs. stabbing vs. drinking.
- **Classifiers used (for tremor severity):** SVC with RBF kernel (best), Random Forest, Logistic Regression.
- **Training labels:** Manual annotation by trained clinician using **CRST+ clinical rating scale**, with video recordings as ground truth.
- **Multi-class severity accuracy (0–4 scale):** 51% strict; **99% "acceptable"** (errors of ±1 class ignored). Most confused classes: **0 and 1** (mild/absent tremor boundary highly ambiguous). Class 2 also problematic.
- **Toothbrush:** No brushing technique classification. Severity scoring only. Sliding-window approach throughout.

---

## H. Tremor Handling

### Tremor signal model (Cup thesis)
Total displacement modeled as three components:
```
b_l(t) = b_l,s (very slow) + b_l,g (normal daily activity) + b_l,f (fast = tremor)
```
Filters applied to **isolate** `b_l,f` — tremor is explicitly extracted, not discarded as noise.

### Frequency bands

| Project | Band | Notes |
|---------|------|-------|
| Toothbrush (IMU) | 1–15 Hz | Wider to capture subtle ET dynamics |
| Toothbrush (video) | 4–12 Hz | Matches standard ET range |
| Cup (feature extraction) | 2–15 Hz | 6th-order Butterworth BPF |
| ET tremor literature | 3–12 Hz | ET typically 4–8 Hz |

### Impact on segmentation
- ET subjects: mean boundary detection error = **2.24 seconds**
- Control subjects: mean boundary detection error = **0.99 seconds**
- Cause: Fragmented motion boundaries from tremor interference → occasional false triggers or merged consecutive actions
- Handling: Same pipeline for all subjects; "total misses" accepted under stringent evaluation metric

### Adaptive filtering
- **Not used.** All filtering is fixed-frequency bandpass. No dynamic separation of voluntary motion from tremor.

---

## I. Feature Engineering

### Feature extraction window
- **NOT per-cycle.** Fixed sliding window: **10 seconds, 80% overlap** (applied continuously to full session).
- No inter-cycle features (cycle-to-cycle variability, rhythm regularity) computed.

### Features extracted

**Time-domain (IMU):**
- Median, mean, std of amplitude
- Activity duration
- Min / mean / median peak-to-peak times
- Relative peak occurrence frequency

**Frequency-domain (IMU):**
- Max, min, median, weighted mean frequency
- Std of frequency distribution
- Skewness of frequency distribution

**Time-domain (Video):**
- Mean velocity magnitude, std, peak-to-peak range, RMS
- Zero-crossing count
- Number of local velocity peaks

**Frequency-domain (Video):**
- Tremor band power (4–12 Hz)
- Tremor power / total power ratio
- Tremor power / low-frequency power ratio
- Spectral entropy, spectral centroid, dominant frequency, peak PSD

**Spatial / Trajectory (Video):**
- Path length in X-Y plane (`path_len_xy`)
- Radial amplitude range
- Overall spatial spread of fingertip trajectory
- Movement anisotropy

**Combined / Kinematic:**
- RMS of acceleration
- RMS of jerk (first derivative of acceleration)
- Correlation between horizontal and vertical velocity components

### Frequency transforms
- **FFT** applied to `|a|` and `|ω|` magnitudes (not raw 3D axes)
- **Welch's method** (PSD) applied to filtered video velocity magnitude

### Dimensionality reduction
- **PCA and t-SNE:** visualization only — not used as preprocessing for model training
- **Feature selection:** Random Forest intrinsic importance (Mean Decrease in Impurity)

### Top discriminative features

| Context | Feature | Importance |
|---------|---------|-----------|
| Brushing score (IMU) | Gyroscope frequency skewness | 0.796 |
| Cup — Subtotal B extended | Max gyroscope magnitude amplitude `A_ω_max` | 0.444 |
| Cup — binary classification | `A_ω_max` | 7.69% (normal), 10.74% (merged classes) |
| Brushing score (video) | Path length XY (`path_len_xy`) | 15.25% |
| Subtotal B extended (video) | Path length XY (`path_len_xy`) | 14.12% |

---

## J. Evaluation & Practical Findings

### Dataset size

| Study | Subjects | Sessions |
|-------|---------|---------|
| Toothbrush | 23 (16 ET, 7 controls), age 58–82 | 46 (both hands separately) |
| Cup thesis | 23 ET + 11 controls | Not broken into discrete cycles |

### Cross-validation strategy
- Stratified **train/test split** (80/20 or 75/25, unseen subjects in test set)
- **5-fold CV** within training data across time observations

### Final metrics

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Binary classification (Cup) | SVC (RBF) | Accuracy | 0.95 |
| Binary classification (Cup) | SVC (RBF) | F1 | 0.80 |
| Binary classification (Cup) | SVC (RBF) | AUC | 0.94 |
| Binary classification (Toothbrush IMU) | Best model | AUC | 0.87–0.88 |
| Multi-class severity (0–4) | SVC | Strict accuracy | 51% |
| Multi-class severity (0–4) | SVC | Acceptable accuracy (±1) | 99% |
| Regression — Brushing score | RF / XGBoost | R² | 0.69 |
| Regression — Subtotal B extended | RF / XGBoost | R² | 0.64 |
| Regression — Cup (specific drinking score) | RF | R² | 0.35–0.62 |

### Biggest challenges
1. **Activity phase detection in ET subjects** — tremor causes fragmented boundaries → 2.24s mean error vs 0.99s for controls
2. **Clinical score ambiguity** — classes 0 (absent) and 1 (mild) overlap heavily in feature space; forced class merging for reliable binary classification

### Recommendations from authors (future work)
- Expand dataset with larger, more diverse ET cohorts
- Longitudinal home monitoring (days/weeks) for symptom fluctuation and medication tracking
- Transition to deep learning: temporal CNNs or transformer-based architectures
- Multi-modal fusion: pressure/grip force sensors + video-based gesture segmentation
- Personalization of utensil physical properties to individual motor profiles

### Hard-coded thresholds (dataset-specific)
- Activity detection thresholds, window sizes, and active point ratios were all chosen via **heuristic search** against their specific error data
- These parameters are **tightly coupled to their hardware and participant group** and will require recalibration for a new device or population

---

*Phase 2: Data gathered. Ready for Phase 3 — Algorithmic Design for the Fork Pipeline.*
