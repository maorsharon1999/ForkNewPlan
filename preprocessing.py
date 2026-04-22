"""
Signal preprocessing: filtering, magnitude computation, activity detection.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew

import config as cfg

logger = logging.getLogger("fork_pipeline.preprocessing")

def filter_eating_cycles(
    segments: List[Tuple[int, int]],
    raw_df: pd.DataFrame,
    fs: float = cfg.FS
) -> List[Tuple[int, int]]:
    """Filter general activity segments into true eating cycles.
    
    Checks for the 3-phase eating pattern (reach down -> pierce/scoop -> bring to mouth).
    This separates targeted eating movements from general noise/talking.
    
    Args:
        segments: List of (start, end) indices representing activity segments.
        raw_df: Raw DataFrame containing acc_y, acc_z (with gravity) for tilt analysis.
        fs: Sampling frequency.
        
    Returns:
        List of filtered (start, end) indices.
    """
    valid_segments = []
    
    for start, end in segments:
        seg_df = raw_df.iloc[start:end + 1]
        if len(seg_df) < int(cfg.MIN_SEGMENT_SEC * fs):
            continue
            
        # Phase 1 & 3: Check for physical tilt/orientation change
        # Reaching down and back up changes the projection of gravity on the axes.
        # We must use raw_df because bandpass filtering removes the DC gravity component!
        acc_y_range = seg_df["acc_y"].max() - seg_df["acc_y"].min()
        acc_z_range = seg_df["acc_z"].max() - seg_df["acc_z"].min()
        tilt_magnitude = max(acc_y_range, acc_z_range)
        
        # Phase 2: Check for piercing/scooping impact
        # A fork interaction usually creates a concentrated spike in jerk (derivative of acceleration).
        jerk_z = np.diff(seg_df["acc_z"].values) * fs
        jerk_y = np.diff(seg_df["acc_y"].values) * fs
        
        # Take the maximum peak
        peak_jerk = max(np.max(np.abs(jerk_z)), np.max(np.abs(jerk_y)))
        mean_jerk = (np.mean(np.abs(jerk_z)) + np.mean(np.abs(jerk_y))) / 2.0
        
        # Heuristics for a valid eating cycle pattern:
        # - Noticeable tilt (gravity vector shifting > 0.2g, meaning hand orientation changed)
        # - Sharp pierce/scoop impact compared to its own average movement
        # - Impact must be at least somewhat significant in absolute terms
        if tilt_magnitude > 0.2 and peak_jerk > (mean_jerk * 1.5) and peak_jerk > 0.5:
            valid_segments.append((start, end))
            
    logger.debug(
        "Filtered %d general segments down to %d confirmed eating cycles (phases intact)",
        len(segments), len(valid_segments)
    )
    return valid_segments


def infer_hand_from_signal(df: pd.DataFrame) -> str:
    """Infer the holding hand based on gyroscope signal physics.
    
    During the scooping motion, the wrist pronates/supinates. Because the 
    movements are mirrored for right and left hands, the dominant peaks in the 
    gyroscope (e.g. gyro_y) will have opposite signs.
    
    Args:
        df: DataFrame containing at least 'gyro_y'.
        
    Returns:
        "Right" or "Left" based on the physical direction of the signal.
    """
    if "gyro_y" not in df.columns:
        return "Right"  # Fallback
        
    y_sig = df["gyro_y"].values.astype(np.float64)
    # Basic smoothing using moving average
    window = max(3, int(len(y_sig) * 0.01))
    smoothed = np.convolve(y_sig, np.ones(window)/window, mode='valid')
    
    if len(smoothed) == 0:
        return "Right"
        
    # Calculate 95th and 5th percentiles to find dominant extremes reliably
    p95 = np.percentile(smoothed, 95)
    p05 = np.percentile(smoothed, 5)
    
    # If the positive peak is stronger (absolute distance from 0), it is Right
    if abs(p95) > abs(p05):
        return "Right"
    else:
        return "Left"


def bandpass_filter(
    signal: np.ndarray,
    fs: float = cfg.FS,
    low: float = cfg.HIGHPASS_HZ,
    high: float = cfg.LOWPASS_HZ,
    order: int = cfg.BUTTER_ORDER,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter (zero-phase, forward-backward).

    Args:
        signal: 1-D array of samples.
        fs: Sampling frequency in Hz.
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal of the same length.
    """
    nyq = 0.5 * fs
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float64)


def compute_magnitude(acc_df: pd.DataFrame) -> pd.Series:
    """Compute the Euclidean magnitude of 3-axis accelerometer data.

    Args:
        acc_df: DataFrame with columns ``acc_x``, ``acc_y``, ``acc_z``.

    Returns:
        Series of magnitude values (in *g*).
    """
    return np.sqrt(
        acc_df["acc_x"] ** 2 + acc_df["acc_y"] ** 2 + acc_df["acc_z"] ** 2
    )


def detect_activity(
    magnitude: np.ndarray,
    fs: float = cfg.FS,
    threshold: float = cfg.ACTIVITY_THRESHOLD,
    gap_fill_sec: float = cfg.GAP_FILL_SEC,
    min_segment_sec: float = cfg.MIN_SEGMENT_SEC,
) -> List[Tuple[int, int]]:
    """Detect contiguous eating-activity segments via magnitude thresholding.

    Procedure:
        1. Mark samples where ``|magnitude − 1 g| > threshold``.
        2. Fill short gaps (< *gap_fill_sec*) to merge nearby bursts.
        3. Drop segments shorter than *min_segment_sec*.

    Args:
        magnitude: 1-D array of accelerometer magnitude values.
        fs: Sampling frequency in Hz.
        threshold: Deviation from 1 g (rest) in *g* units.
        gap_fill_sec: Max gap duration (seconds) to fill.
        min_segment_sec: Minimum segment duration (seconds).

    Returns:
        List of ``(start_idx, end_idx)`` tuples (inclusive on both ends).
    """
    active = np.abs(magnitude - 1.0) > threshold

    # Fill short gaps
    gap_samples = int(gap_fill_sec * fs)
    filled = active.copy()
    i = 0
    while i < len(filled):
        if not filled[i]:
            # count gap length
            gap_start = i
            while i < len(filled) and not filled[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= gap_samples and gap_start > 0 and i < len(filled):
                filled[gap_start:i] = True
        else:
            i += 1

    # Extract contiguous segments
    min_samples = int(min_segment_sec * fs)
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < len(filled):
        if filled[i]:
            start = i
            while i < len(filled) and filled[i]:
                i += 1
            end = i - 1
            if (end - start + 1) >= min_samples:
                segments.append((start, end))
        else:
            i += 1

    logger.debug(
        "Detected %d activity segments (threshold=%.2f g, min=%.1f s)",
        len(segments), threshold, min_segment_sec,
    )
    return segments


def segment_signal(
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
) -> List[pd.DataFrame]:
    """Slice the IMU DataFrame according to detected activity segments.

    Args:
        df: Full IMU DataFrame (7 columns).
        segments: List of ``(start_idx, end_idx)`` tuples.

    Returns:
        List of DataFrame slices (with reset index).
    """
    sliced = []
    
    for start_idx, end_idx in segments:
        # Revert to standard slicing from start to end
        sliced.append(df.iloc[start_idx:end_idx + 1].reset_index(drop=True))
            
    return sliced
