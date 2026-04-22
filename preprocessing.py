"""
Signal preprocessing: filtering, magnitude computation, robust cleaning,
and activity detection (Cup combined criterion).
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, savgol_filter, sosfiltfilt
from scipy.stats import skew

import config as cfg

logger = logging.getLogger("fork_pipeline.preprocessing")

_IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ── Stage A: Robust preprocessing ─────────────────────────────────────────


def reject_outliers(
    df: pd.DataFrame,
    pct_low: float = cfg.OUTLIER_PERCENTILE_LOW,
    pct_high: float = cfg.OUTLIER_PERCENTILE_HIGH,
) -> pd.DataFrame:
    """Clip per-axis values outside [pct_low, pct_high] percentiles, then interpolate.

    Only operates on the 6 IMU columns; other columns (e.g. timestamp) are untouched.
    """
    df = df.copy()
    for col in [c for c in _IMU_COLS if c in df.columns]:
        lo = np.percentile(df[col].values, pct_low)
        hi = np.percentile(df[col].values, pct_high)
        mask = (df[col] < lo) | (df[col] > hi)
        df.loc[mask, col] = np.nan
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
    return df


def reject_spikes(
    df: pd.DataFrame,
    pct_deriv: float = cfg.SPIKE_DERIV_PERCENTILE,
) -> pd.DataFrame:
    """Flag first-derivative spikes above the pct_deriv-th percentile and interpolate.

    Only operates on the 6 IMU columns.
    """
    df = df.copy()
    for col in [c for c in _IMU_COLS if c in df.columns]:
        vals = df[col].values.astype(np.float64)
        deriv = np.abs(np.diff(vals, prepend=vals[0]))
        thresh = np.percentile(deriv, pct_deriv)
        mask = deriv > thresh
        df.loc[mask, col] = np.nan
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
    return df


def smooth_signal(
    df: pd.DataFrame,
    median_n: int = cfg.SMOOTH_MEDIAN_N,
    sg_window: int = cfg.SMOOTH_SG_WINDOW,
    sg_poly: int = cfg.SMOOTH_SG_POLY,
) -> pd.DataFrame:
    """Apply median filter then Savitzky-Golay to each IMU axis.

    For segmentation use only — not applied before tremor feature extraction.
    """
    df = df.copy()
    for col in [c for c in _IMU_COLS if c in df.columns]:
        s = medfilt(df[col].values.astype(np.float64), kernel_size=median_n)
        if len(s) > sg_window:
            s = savgol_filter(s, window_length=sg_window, polyorder=sg_poly)
        df[col] = s
    return df


# ── Magnitude helpers ──────────────────────────────────────────────────────


def compute_magnitude(acc_df: pd.DataFrame) -> pd.Series:
    """Euclidean magnitude of acc_x/y/z (in *g*)."""
    return np.sqrt(
        acc_df["acc_x"] ** 2 + acc_df["acc_y"] ** 2 + acc_df["acc_z"] ** 2
    )


def compute_gyro_magnitude(df: pd.DataFrame) -> pd.Series:
    """Euclidean magnitude of gyro_x/y/z (in rad/s)."""
    return np.sqrt(
        df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
    )


# ── Bandpass filter ────────────────────────────────────────────────────────


def bandpass_filter(
    signal: np.ndarray,
    fs: float = cfg.FS,
    low: float = cfg.HIGHPASS_HZ,
    high: float = cfg.LOWPASS_HZ,
    order: int = cfg.BUTTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        signal: 1-D array.
        fs: Sampling frequency (Hz).
        low: Low cutoff (Hz).
        high: High cutoff (Hz).
        order: Filter order.
    """
    nyq = 0.5 * fs
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float64)


# ── Stage B: Activity detection (Cup combined criterion) ──────────────────


def detect_activity(
    df: pd.DataFrame,
    raw_acc_mag: np.ndarray,
    fs: float = cfg.FS,
    tau_1: float = cfg.TAU_1,
    tau_2: float = cfg.TAU_2,
    p_active: float = cfg.P_ACTIVE,
    p_inactive: float = cfg.P_INACTIVE,
    gap_fill_sec: float = cfg.INTRA_CYCLE_GAP_FILL_SEC,
    min_segment_sec: float = cfg.MIN_SEGMENT_SEC,
    max_segment_sec: float = cfg.MAX_SEGMENT_SEC,
) -> List[Tuple[int, int]]:
    """Detect eating-activity segments using the Cup combined criterion.

    Criterion (per sample):
        combined(t) = |a'(t)| + |ω'(t)| + ||a(t)| - 1|
        gyro_above_rest(t) = |ω(t)| - ω_min

        active(t)  iff  combined(t) >= tau_1  AND  gyro_above_rest(t) >= tau_2

    Start T_s: first window where active-sample ratio >= p_active.
    End   T_f: first subsequent window where inactive-sample ratio >= p_inactive.
    Short intra-cycle gaps (< gap_fill_sec) are merged.
    Segments outside [min_segment_sec, max_segment_sec] are discarded.

    Args:
        df: Narrowband-filtered (2-15 Hz) IMU DataFrame with all 6 axes.
        raw_acc_mag: Smoothed raw acc magnitude array (len == len(df)),
                     used for the ||a(t)| - 1g| tilt term.
        fs: Sampling frequency.
        tau_1, tau_2, p_active, p_inactive: Thresholds (see above).
        gap_fill_sec: Max intra-cycle gap to merge.
        min_segment_sec, max_segment_sec: Duration bounds.

    Returns:
        List of (start_idx, end_idx) tuples (inclusive, both ends).
    """
    n = len(df)
    if n == 0:
        return []

    # Compute per-sample activity signal
    acc_mag_filt = compute_magnitude(df).values.astype(np.float64)
    gyro_mag_filt = compute_gyro_magnitude(df).values.astype(np.float64)

    acc_deriv = np.abs(np.gradient(acc_mag_filt, 1.0 / fs))
    gyro_deriv = np.abs(np.gradient(gyro_mag_filt, 1.0 / fs))

    static_tilt = np.abs(raw_acc_mag.astype(np.float64) - 1.0)

    combined = acc_deriv + gyro_deriv + static_tilt

    gyro_min = float(np.min(gyro_mag_filt))
    gyro_above_rest = gyro_mag_filt - gyro_min

    active = (combined >= tau_1) & (gyro_above_rest >= tau_2)

    # Rolling window to smooth the active flag
    window_samples = max(1, int(cfg.ACTIVITY_WINDOW_SEC * fs))
    kernel = np.ones(window_samples) / window_samples
    roll_active = np.convolve(active.astype(float), kernel, mode="same")
    active_window = roll_active >= p_active

    # Fill short intra-cycle gaps
    gap_samples = int(gap_fill_sec * fs)
    filled = active_window.copy()
    i = 0
    while i < n:
        if not filled[i]:
            gap_start = i
            while i < n and not filled[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= gap_samples and gap_start > 0 and i < n:
                filled[gap_start:i] = True
        else:
            i += 1

    # Extract contiguous active segments
    min_samples = int(min_segment_sec * fs)
    max_samples = int(max_segment_sec * fs)
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if filled[i]:
            start = i
            while i < n and filled[i]:
                i += 1
            end = i - 1
            dur = end - start + 1
            if dur < min_samples:
                continue
            if dur <= max_samples:
                segments.append((start, end))
            else:
                # Split fused segments into max-length chunks
                for sub_start in range(start, end + 1, max_samples):
                    sub_end = min(sub_start + max_samples - 1, end)
                    if (sub_end - sub_start + 1) >= min_samples:
                        segments.append((sub_start, sub_end))
        else:
            i += 1

    logger.debug(
        "Cup detect_activity: %d segments (tau1=%.2f, tau2=%.2f, min=%.1fs, max=%.1fs)",
        len(segments), tau_1, tau_2, min_segment_sec, max_segment_sec,
    )
    return segments


def classify_cycle_quality(
    segments: List[Tuple[int, int]],
    raw_df: pd.DataFrame,
    fs: float = cfg.FS,
) -> List[str]:
    """Classify each segment as 'cycle' or 'fragment'.

    A segment is 'cycle' if it has measurable tilt (acc_y or acc_z range
    > FRAGMENT_TILT_MIN) AND a sharp jerk peak (peak/mean > FRAGMENT_JERK_RATIO).
    Everything else is 'fragment'.
    """
    labels: List[str] = []
    for start, end in segments:
        seg = raw_df.iloc[start : end + 1]
        if len(seg) < 10:
            labels.append("fragment")
            continue

        acc_y_range = float(seg["acc_y"].max() - seg["acc_y"].min())
        acc_z_range = float(seg["acc_z"].max() - seg["acc_z"].min())
        tilt = max(acc_y_range, acc_z_range)

        jerk_y = np.abs(np.diff(seg["acc_y"].values) * fs)
        jerk_z = np.abs(np.diff(seg["acc_z"].values) * fs)
        jerk = np.maximum(jerk_y, jerk_z) if len(jerk_y) > 0 else np.array([0.0])

        peak_jerk = float(jerk.max()) if len(jerk) > 0 else 0.0
        mean_jerk = float(jerk.mean()) if len(jerk) > 0 else 0.0

        if (
            tilt > cfg.FRAGMENT_TILT_MIN
            and mean_jerk > 1e-10
            and peak_jerk / mean_jerk >= cfg.FRAGMENT_JERK_RATIO
        ):
            labels.append("cycle")
        else:
            labels.append("fragment")

    logger.debug(
        "classify_cycle_quality: %d/%d segments classified as 'cycle'",
        labels.count("cycle"), len(labels),
    )
    return labels


# ── Segment slicer ─────────────────────────────────────────────────────────


def segment_signal(
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
) -> List[pd.DataFrame]:
    """Slice the IMU DataFrame according to detected activity segments.

    Args:
        df: Full IMU DataFrame.
        segments: List of (start_idx, end_idx) tuples.

    Returns:
        List of DataFrame slices (reset index).
    """
    return [df.iloc[s : e + 1].reset_index(drop=True) for s, e in segments]


# ── Deprecated helpers (kept for fallback validation) ─────────────────────


def filter_eating_cycles(
    segments: List[Tuple[int, int]],
    raw_df: pd.DataFrame,
    fs: float = cfg.FS,
) -> List[Tuple[int, int]]:
    """[DEPRECATED] Filter general activity segments into true eating cycles.

    Replaced by classify_cycle_quality in the main pipeline.
    Kept for diagnostic comparison during threshold calibration.
    """
    valid_segments = []
    for start, end in segments:
        seg_df = raw_df.iloc[start : end + 1]
        if len(seg_df) < int(cfg.MIN_SEGMENT_SEC * fs):
            continue
        acc_y_range = seg_df["acc_y"].max() - seg_df["acc_y"].min()
        acc_z_range = seg_df["acc_z"].max() - seg_df["acc_z"].min()
        tilt_magnitude = max(acc_y_range, acc_z_range)
        jerk_z = np.diff(seg_df["acc_z"].values) * fs
        jerk_y = np.diff(seg_df["acc_y"].values) * fs
        peak_jerk = max(np.max(np.abs(jerk_z)), np.max(np.abs(jerk_y)))
        mean_jerk = (np.mean(np.abs(jerk_z)) + np.mean(np.abs(jerk_y))) / 2.0
        if tilt_magnitude > 0.2 and peak_jerk > (mean_jerk * 1.5) and peak_jerk > 0.5:
            valid_segments.append((start, end))
    logger.debug(
        "filter_eating_cycles (deprecated): %d → %d segments",
        len(segments), len(valid_segments),
    )
    return valid_segments


def infer_hand_from_signal(df: pd.DataFrame) -> str:
    """[DEPRECATED] Heuristic hand inference from gyro_y asymmetry.

    Replaced by HandednessClassifier in handedness.py.
    Kept as a fallback if the classifier is not fitted.
    """
    if "gyro_y" not in df.columns:
        return "Right"
    y_sig = df["gyro_y"].values.astype(np.float64)
    window = max(3, int(len(y_sig) * 0.01))
    smoothed = np.convolve(y_sig, np.ones(window) / window, mode="valid")
    if len(smoothed) == 0:
        return "Right"
    p95 = np.percentile(smoothed, 95)
    p05 = np.percentile(smoothed, 5)
    return "Right" if abs(p95) >= abs(p05) else "Left"
