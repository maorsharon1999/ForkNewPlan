"""
Central configuration for the Fork ET Detection Pipeline.

All paths, thresholds, column mappings, and frequency parameters live here
so that every other module imports from a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "New Data", "משתתפים")
CRF_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "New Data",
    "HIT Study CRF - No personal Data.xlsx",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "figures")

# ---------------------------------------------------------------------------
# CRF Excel — sheet names
# ---------------------------------------------------------------------------
CRF_SHEET_ET = "ET Rater 1-F2F Dr.Lassman"
CRF_SHEET_CONTROL = "Control Rater 1-F2F Dr.Lassman"

# ---------------------------------------------------------------------------
# CRF Excel — column indices (0-based)
# ---------------------------------------------------------------------------
CRF_COL_SUBJECT = 1
CRF_COL_GENDER = 2
CRF_COL_AGE = 3
CRF_COL_TREMOR_HAND = 7
CRF_COL_RT_FORK_SCOOP = 53
CRF_COL_LF_FORK_SCOOP = 54
CRF_COL_RT_FORK_STAB = 55
CRF_COL_LF_FORK_STAB = 56
CRF_COL_SUBTOTAL_B_EXT = 62

# ---------------------------------------------------------------------------
# Fork hand mapping
# ---------------------------------------------------------------------------
FORK_HAND_MAP = {"Fork1": "Right", "Fork2": "Left"}
FORK_DEFAULT_HAND = "Right"
INCLUDE_AMBIGUOUS_FORK = True

# ---------------------------------------------------------------------------
# IMU / signal parameters
# ---------------------------------------------------------------------------
FS = 100
LOWPASS_HZ = 20.0
HIGHPASS_HZ = 0.5
BUTTER_ORDER = 4

# ---------------------------------------------------------------------------
# Activity detection
# ---------------------------------------------------------------------------
ACTIVITY_THRESHOLD = 0.25
GAP_FILL_SEC = 2.0
MIN_SEGMENT_SEC = 3.0

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
ET_FREQ_LOW = 4.0
ET_FREQ_HIGH = 12.0
WAVELET_NAME = "morl"
WAVELET_SCALES_HZ = (4, 5, 6, 7, 8, 9, 10, 11, 12)

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
FEATURE_SELECTION_METHOD = "rfe"   # "rfe", "rf_importance", or "select_k_best"
TOP_K_FEATURES = 25
RF_IMPORTANCE_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
USE_AUGMENTATION = False #Took too long and can be a reason for overfitting
NOISE_SCALE = 0.05
MIXUP_ALPHA = 0.01

# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_LOSO = True
TUNE_HYPERPARAMS = False #Took too long and can be a reason for overfitting
USE_STACKING = False #Took too long and can be a reason for overfitting
PER_SEGMENT = True
TARGET_SPECIFICITY = 0.80
