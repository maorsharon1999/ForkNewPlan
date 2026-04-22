"""
Feature extraction: time-domain, frequency-domain, jerk, cross-axis,
magnitude, wavelet, and spectral 2nd-order features per segment.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

import config as cfg
from preprocessing import bandpass_filter

logger = logging.getLogger("fork_pipeline.feature_extraction")

# ── Axes used throughout ───────────────────────────────────────────────────
_ACC_AXES = ["acc_x", "acc_y", "acc_z"]
_GYRO_AXES = ["gyro_x", "gyro_y", "gyro_z"]
_ALL_AXES = _ACC_AXES + _GYRO_AXES


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Time-domain features (per axis)
# ═══════════════════════════════════════════════════════════════════════════

def _time_features(vals: np.ndarray, prefix: str) -> Dict[str, float]:
    """Mean, std, RMS, skewness, kurtosis, peak-to-peak for one axis."""
    f: Dict[str, float] = {}
    f[f"{prefix}_mean"] = float(np.mean(vals))
    f[f"{prefix}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    f[f"{prefix}_rms"] = float(np.sqrt(np.mean(vals ** 2)))
    f[f"{prefix}_skew"] = float(skew(vals, bias=False)) if len(vals) > 2 else 0.0
    f[f"{prefix}_kurt"] = float(kurtosis(vals, bias=False)) if len(vals) > 3 else 0.0
    f[f"{prefix}_ptp"] = float(np.ptp(vals))
    return f


def extract_time_features(segment: pd.DataFrame) -> Dict[str, float]:
    """Compute time-domain features for each IMU axis.

    Args:
        segment: DataFrame with IMU columns.

    Returns:
        Dict mapping ``"{axis}_{stat}"`` → value.
    """
    features: Dict[str, float] = {}
    for axis in _ALL_AXES:
        features.update(_time_features(segment[axis].values.astype(np.float64), axis))
    return features


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Jerk features (derivative of acceleration / gyroscope)
# ═══════════════════════════════════════════════════════════════════════════

def extract_jerk_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Compute jerk (first derivative) statistics for each axis.

    Jerk = d(signal)/dt.  For discrete data: jerk[i] = (x[i+1] - x[i]) * fs.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with jerk mean, std, RMS, max for each axis.
    """
    features: Dict[str, float] = {}
    for axis in _ALL_AXES:
        vals = segment[axis].values.astype(np.float64)
        jerk = np.diff(vals) * fs
        if len(jerk) == 0:
            for s in ("mean", "std", "rms", "max"):
                features[f"{axis}_jerk_{s}"] = 0.0
            continue
        features[f"{axis}_jerk_mean"] = float(np.mean(np.abs(jerk)))
        features[f"{axis}_jerk_std"] = float(np.std(jerk, ddof=1)) if len(jerk) > 1 else 0.0
        features[f"{axis}_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
        features[f"{axis}_jerk_max"] = float(np.max(np.abs(jerk)))
    return features


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Cross-axis correlation features
# ═══════════════════════════════════════════════════════════════════════════

_CROSS_PAIRS = [
    ("acc_x", "acc_y"), ("acc_x", "acc_z"), ("acc_y", "acc_z"),
    ("gyro_x", "gyro_y"), ("gyro_x", "gyro_z"), ("gyro_y", "gyro_z"),
    ("acc_x", "gyro_x"), ("acc_y", "gyro_y"), ("acc_z", "gyro_z"),
]


def extract_cross_axis_features(segment: pd.DataFrame) -> Dict[str, float]:
    """Compute Pearson correlation between axis pairs.

    Args:
        segment: IMU DataFrame.

    Returns:
        Dict with ``"corr_{a}_{b}"`` → correlation coefficient.
    """
    features: Dict[str, float] = {}
    for a, b in _CROSS_PAIRS:
        va = segment[a].values.astype(np.float64)
        vb = segment[b].values.astype(np.float64)
        if len(va) < 3 or np.std(va) == 0 or np.std(vb) == 0:
            features[f"corr_{a}_{b}"] = 0.0
        else:
            features[f"corr_{a}_{b}"] = float(np.corrcoef(va, vb)[0, 1])
    return features


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Magnitude features (acc and gyro)
# ═══════════════════════════════════════════════════════════════════════════

def extract_magnitude_features(segment: pd.DataFrame) -> Dict[str, float]:
    """Compute features on the Euclidean magnitude of acc and gyro.

    Args:
        segment: IMU DataFrame.

    Returns:
        Dict with magnitude statistics for acc and gyro.
    """
    features: Dict[str, float] = {}
    for group_name, axes in [("acc_mag", _ACC_AXES), ("gyro_mag", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        features.update(_time_features(mag, group_name))
    return features


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Frequency-domain features (per axis)
# ═══════════════════════════════════════════════════════════════════════════

def extract_freq_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Dominant freq, median freq, spectral energy, ET-band power ratio.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with frequency features per axis.
    """
    features: Dict[str, float] = {}
    for axis in _ALL_AXES:
        vals = segment[axis].values.astype(np.float64)
        filtered = bandpass_filter(vals, fs)
        n = len(filtered)
        freqs = rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(rfft(filtered)) ** 2

        total = float(np.sum(spectrum))
        if total == 0:
            for s in ("dom_freq", "med_freq", "spectral_energy", "power_4_12hz"):
                features[f"{axis}_{s}"] = 0.0
            continue

        features[f"{axis}_dom_freq"] = float(freqs[np.argmax(spectrum)])

        cum = np.cumsum(spectrum)
        med_idx = np.searchsorted(cum, total / 2.0)
        features[f"{axis}_med_freq"] = float(freqs[min(med_idx, len(freqs) - 1)])

        features[f"{axis}_spectral_energy"] = total

        band = (freqs >= cfg.ET_FREQ_LOW) & (freqs <= cfg.ET_FREQ_HIGH)
        features[f"{axis}_power_4_12hz"] = float(np.sum(spectrum[band])) / total

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Spectral 2nd-order features (entropy, flatness, centroid, rolloff)
# ═══════════════════════════════════════════════════════════════════════════

def extract_spectral_shape_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Spectral entropy, flatness, centroid, and rolloff for each axis.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with spectral shape descriptors per axis.
    """
    features: Dict[str, float] = {}
    for axis in _ALL_AXES:
        vals = segment[axis].values.astype(np.float64)
        filtered = bandpass_filter(vals, fs)
        n = len(filtered)
        freqs = rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(rfft(filtered)) ** 2

        total = float(np.sum(spectrum))
        if total == 0:
            for s in ("spec_entropy", "spec_flatness", "spec_centroid", "spec_rolloff"):
                features[f"{axis}_{s}"] = 0.0
            continue

        # Normalised PSD (probability distribution)
        psd_norm = spectrum / total
        psd_norm = np.clip(psd_norm, 1e-12, None)

        # Spectral entropy
        features[f"{axis}_spec_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm)))

        # Spectral flatness  (geometric mean / arithmetic mean)
        log_mean = np.mean(np.log(spectrum + 1e-12))
        arith_mean = np.mean(spectrum)
        features[f"{axis}_spec_flatness"] = float(np.exp(log_mean) / arith_mean) if arith_mean > 0 else 0.0

        # Spectral centroid
        features[f"{axis}_spec_centroid"] = float(np.sum(freqs * psd_norm))

        # Spectral rolloff (frequency below which 85% of energy lies)
        cum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cum, 0.85 * total)
        features[f"{axis}_spec_rolloff"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Wavelet features (CWT in ET band)
# ═══════════════════════════════════════════════════════════════════════════

def extract_wavelet_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """CWT energy in the ET tremor band (4-12 Hz) using Morlet wavelet.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with wavelet energy statistics per axis.
    """
    features: Dict[str, float] = {}
    # Convert desired frequencies to CWT scales for Morlet
    # scale = w * fs / (2 * pi * freq), with w=5 (default width)
    w = 5.0
    target_freqs = np.array(cfg.WAVELET_SCALES_HZ, dtype=np.float64)
    scales = w * fs / (2.0 * np.pi * target_freqs)

    for axis in _ALL_AXES:
        vals = segment[axis].values.astype(np.float64)
        n = len(vals)
        if n < 10:
            features[f"{axis}_cwt_energy_mean"] = 0.0
            features[f"{axis}_cwt_energy_std"] = 0.0
            features[f"{axis}_cwt_energy_max"] = 0.0
            features[f"{axis}_cwt_energy_ratio"] = 0.0
            continue

        # Manual CWT: convolve signal with scaled Morlet wavelets
        et_energy = np.zeros(len(scales))
        for si, s in enumerate(scales):
            # Generate complex Morlet wavelet at this scale
            M = min(n, int(10 * s))
            t = np.arange(M) - (M - 1) / 2
            t_scaled = t / s
            wavelet = (np.pi ** -0.25) * np.exp(1j * w * t_scaled) * np.exp(-0.5 * t_scaled ** 2)
            wavelet /= np.sqrt(s)  # normalise
            conv = np.convolve(vals, wavelet, mode="same")
            et_energy[si] = np.sum(np.abs(conv) ** 2)

        # Total signal energy for ratio
        total_energy = float(np.sum(vals ** 2))
        et_total = float(np.sum(et_energy))

        features[f"{axis}_cwt_energy_mean"] = float(np.mean(et_energy))
        features[f"{axis}_cwt_energy_std"] = float(np.std(et_energy))
        features[f"{axis}_cwt_energy_max"] = float(np.max(et_energy))
        features[f"{axis}_cwt_energy_ratio"] = (
            et_total / total_energy if total_energy > 0 else 0.0
        )

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Weighted spectral features (from Cup pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def _sliding_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Moving-average smoothing for FFT magnitude."""
    win = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, win, mode="same")


def extract_weighted_spectral_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Weighted spectral features on acc/gyro magnitude (from Cup).

    Extracts weighted mean/median/max frequency, frequency std & skewness,
    and max/mean/median FFT amplitude — all computed on a smoothed spectrum.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with weighted spectral features for acc_mag and gyro_mag.
    """
    features: Dict[str, float] = {}

    for group_name, axes in [("acc", _ACC_AXES), ("gyro", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        filtered = bandpass_filter(mag, fs)
        n = len(filtered)

        fft_result = rfft(filtered)
        fft_freq = rfftfreq(n, d=1.0 / fs)
        fft_mag = np.abs(fft_result) / max(n, 1)

        if len(fft_freq) < 3 or np.sum(fft_mag) == 0:
            for s in ("wt_mean_freq", "wt_median_freq", "wt_max_freq",
                       "freq_std", "freq_skew",
                       "fft_amp_max", "fft_amp_mean", "fft_amp_median"):
                features[f"{group_name}_{s}"] = 0.0
            continue

        # Smooth spectrum (0.25 Hz resolution window)
        freq_res = fft_freq[1] - fft_freq[0] if len(fft_freq) > 1 else 1.0
        if freq_res > 0:
            win_len = max(3, int(np.ceil(0.25 / freq_res)))
            if win_len % 2 == 0:
                win_len += 1
            smooth_fft = _sliding_average(fft_mag, win_len)
        else:
            smooth_fft = fft_mag

        total_mag = float(np.sum(fft_mag))

        # Weighted mean frequency
        wt_mean = float(np.sum(fft_freq * fft_mag) / total_mag)
        features[f"{group_name}_wt_mean_freq"] = wt_mean

        # Weighted median frequency
        cum = np.cumsum(fft_mag)
        half = total_mag / 2.0
        med_idx = np.searchsorted(cum, half)
        features[f"{group_name}_wt_median_freq"] = float(
            fft_freq[min(med_idx, len(fft_freq) - 1)]
        )

        # Weighted max frequency (from smoothed)
        features[f"{group_name}_wt_max_freq"] = float(
            fft_freq[np.argmax(smooth_fft)]
        )

        # Frequency std
        freq_std = float(np.sqrt(
            np.sum(fft_mag * (fft_freq - wt_mean) ** 2) / total_mag
        ))
        features[f"{group_name}_freq_std"] = freq_std

        # Frequency skewness
        if freq_std > 1e-12:
            features[f"{group_name}_freq_skew"] = float(
                np.sum(fft_mag * (fft_freq - wt_mean) ** 3)
                / total_mag / (freq_std ** 3)
            )
        else:
            features[f"{group_name}_freq_skew"] = 0.0

        # FFT amplitudes
        features[f"{group_name}_fft_amp_max"] = float(np.max(smooth_fft))
        features[f"{group_name}_fft_amp_mean"] = float(np.mean(fft_mag))
        features[f"{group_name}_fft_amp_median"] = float(np.median(fft_mag))

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Peak-to-peak timing features (from Cup pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def extract_peak_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Peak-to-peak interval statistics and signal duration.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with peak-to-peak features for acc_mag and gyro_mag,
        plus signal duration.
    """
    from scipy.signal import find_peaks

    features: Dict[str, float] = {}
    n = len(segment)
    features["signal_duration_sec"] = float(n / fs)

    for group_name, axes in [("acc", _ACC_AXES), ("gyro", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        filtered = bandpass_filter(mag, fs)

        peaks, _ = find_peaks(filtered)

        if len(peaks) < 2:
            for s in ("p2p_min", "p2p_mean", "p2p_median", "peak_freq"):
                features[f"{group_name}_{s}"] = 0.0
            continue

        p2p = np.diff(peaks) / fs  # in seconds
        features[f"{group_name}_p2p_min"] = float(np.min(p2p))
        features[f"{group_name}_p2p_mean"] = float(np.mean(p2p))
        features[f"{group_name}_p2p_median"] = float(np.median(p2p))
        features[f"{group_name}_peak_freq"] = float(len(peaks) / (n / fs))

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 10.  Tremor-specific features
# ═══════════════════════════════════════════════════════════════════════════

def extract_tremor_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Tremor-specific features: stability index, power ratio, HNR.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with tremor-specific features per acc_mag and gyro_mag.
    """
    features: Dict[str, float] = {}

    for group_name, axes in [("acc", _ACC_AXES), ("gyro", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        filtered = bandpass_filter(mag, fs)
        n = len(filtered)

        fft_result = rfft(filtered)
        fft_freq = rfftfreq(n, d=1.0 / fs)
        fft_mag = np.abs(fft_result) ** 2  # power spectrum

        total_power = float(np.sum(fft_mag))
        et_mask = (fft_freq >= cfg.ET_FREQ_LOW) & (fft_freq <= cfg.ET_FREQ_HIGH)
        et_power = float(np.sum(fft_mag[et_mask]))

        # Tremor Power Ratio
        features[f"{group_name}_tremor_power_ratio"] = (
            et_power / total_power if total_power > 0 else 0.0
        )

        # Tremor Stability Index (TSI)
        # Split into 1-sec windows, find peak freq in ET-band per window
        win_samples = int(fs)
        peak_freqs = []
        for start in range(0, n - win_samples + 1, win_samples):
            chunk = filtered[start : start + win_samples]
            c_fft = np.abs(rfft(chunk)) ** 2
            c_freq = rfftfreq(len(chunk), d=1.0 / fs)
            c_mask = (c_freq >= cfg.ET_FREQ_LOW) & (c_freq <= cfg.ET_FREQ_HIGH)
            if np.any(c_mask) and np.sum(c_fft[c_mask]) > 0:
                peak_freqs.append(float(c_freq[c_mask][np.argmax(c_fft[c_mask])]))
        if len(peak_freqs) >= 2:
            features[f"{group_name}_tsi"] = 1.0 / (float(np.std(peak_freqs)) + 1e-6)
            features[f"{group_name}_tsi_mean_freq"] = float(np.mean(peak_freqs))
        else:
            features[f"{group_name}_tsi"] = 0.0
            features[f"{group_name}_tsi_mean_freq"] = 0.0

        # Harmonic-to-Noise Ratio (HNR)
        if np.any(et_mask) and total_power > 0:
            noise_power = total_power - et_power
            features[f"{group_name}_hnr"] = (
                10 * np.log10(et_power / max(noise_power, 1e-12))
            )
        else:
            features[f"{group_name}_hnr"] = 0.0

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 11.  Multi-resolution CWT analysis
# ═══════════════════════════════════════════════════════════════════════════

def extract_multiresolution_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """CWT energy variance across windows (tremor temporal stability).

    Splits signal into 2-sec windows and computes CWT energy per window,
    then takes the variance — stable tremor → low variance.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with multi-resolution features.
    """
    features: Dict[str, float] = {}
    win_samples = int(2 * fs)  # 2-second windows

    for group_name, axes in [("acc", _ACC_AXES), ("gyro", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        filtered = bandpass_filter(mag, fs)
        n = len(filtered)

        if n < win_samples:
            for s in ("cwt_energy_var", "cwt_energy_cv", "cwt_energy_trend"):
                features[f"{group_name}_{s}"] = 0.0
            continue

        window_energies = []
        for start in range(0, n - win_samples + 1, win_samples // 2):
            chunk = filtered[start : start + win_samples]
            c_fft = np.abs(rfft(chunk)) ** 2
            c_freq = rfftfreq(len(chunk), d=1.0 / fs)
            c_mask = (c_freq >= cfg.ET_FREQ_LOW) & (c_freq <= cfg.ET_FREQ_HIGH)
            window_energies.append(float(np.sum(c_fft[c_mask])))

        if len(window_energies) >= 2:
            we = np.array(window_energies)
            features[f"{group_name}_cwt_energy_var"] = float(np.var(we))
            features[f"{group_name}_cwt_energy_cv"] = float(np.std(we) / (np.mean(we) + 1e-12))
            # Trend: slope of energy over time
            x = np.arange(len(we))
            if len(x) >= 2:
                slope = float(np.polyfit(x, we, 1)[0])
                features[f"{group_name}_cwt_energy_trend"] = slope
            else:
                features[f"{group_name}_cwt_energy_trend"] = 0.0
        else:
            for s in ("cwt_energy_var", "cwt_energy_cv", "cwt_energy_trend"):
                features[f"{group_name}_{s}"] = 0.0

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 12.  Dual bandpass features (narrow tremor band 3-15 Hz)
# ═══════════════════════════════════════════════════════════════════════════

def extract_dual_bandpass_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Features from a narrow tremor-only bandpass (3-15 Hz).

    Complements the standard 0.5-20 Hz features by isolating tremor-band
    energy and computing statistics on the tremor-filtered signal alone.

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with narrow-band features.
    """
    from scipy.signal import butter, sosfiltfilt

    features: Dict[str, float] = {}
    nyq = 0.5 * fs
    low, high = 3.0 / nyq, 15.0 / nyq
    if high >= 1.0:
        high = 0.99
    sos = butter(4, [low, high], btype="band", output="sos")

    for group_name, axes in [("acc", _ACC_AXES), ("gyro", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        try:
            narrow = sosfiltfilt(sos, mag).astype(np.float64)
        except ValueError:
            for s in ("narrow_rms", "narrow_std", "narrow_max", "narrow_energy_ratio"):
                features[f"{group_name}_{s}"] = 0.0
            continue

        wide = bandpass_filter(mag, fs)

        features[f"{group_name}_narrow_rms"] = float(np.sqrt(np.mean(narrow ** 2)))
        features[f"{group_name}_narrow_std"] = float(np.std(narrow))
        features[f"{group_name}_narrow_max"] = float(np.max(np.abs(narrow)))
        wide_energy = float(np.sum(wide ** 2))
        narrow_energy = float(np.sum(narrow ** 2))
        features[f"{group_name}_narrow_energy_ratio"] = (
            narrow_energy / wide_energy if wide_energy > 0 else 0.0
        )

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 13.  Temporal features (autocorrelation, sample entropy)
# ═══════════════════════════════════════════════════════════════════════════

def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute Sample Entropy (SampEn) of a time series."""
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_factor * np.std(x)
    if r == 0:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([x[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(diffs <= r)
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return 0.0
    return -np.log(A / B) if A > 0 else 0.0


def extract_temporal_features(
    segment: pd.DataFrame,
    fs: float = cfg.FS,
) -> Dict[str, float]:
    """Autocorrelation and sample entropy features.

    - Autocorrelation at ET-band lag (~0.1-0.25 sec)
    - Sample entropy (regularity measure)

    Args:
        segment: IMU DataFrame.
        fs: Sampling frequency.

    Returns:
        Dict with temporal features.
    """
    features: Dict[str, float] = {}

    # ET tremor period: 1/4 Hz to 1/12 Hz → lag of 8 to 25 samples at 100 Hz
    et_lag_min = int(fs / cfg.ET_FREQ_HIGH)  # ~8
    et_lag_max = int(fs / cfg.ET_FREQ_LOW)   # ~25

    for group_name, axes in [("acc", _ACC_AXES), ("gyro", _GYRO_AXES)]:
        mag = np.sqrt(sum(segment[ax].values.astype(np.float64) ** 2 for ax in axes))
        filtered = bandpass_filter(mag, fs)
        n = len(filtered)

        # Normalized autocorrelation at ET-band lags
        if n > et_lag_max + 1:
            sig = filtered - np.mean(filtered)
            var = np.var(sig)
            if var > 1e-12:
                acf_values = []
                for lag in range(et_lag_min, et_lag_max + 1):
                    acf = float(np.mean(sig[:n - lag] * sig[lag:])) / var
                    acf_values.append(acf)
                features[f"{group_name}_acf_max"] = float(np.max(acf_values))
                features[f"{group_name}_acf_mean"] = float(np.mean(acf_values))
                best_lag = et_lag_min + int(np.argmax(acf_values))
                features[f"{group_name}_acf_peak_freq"] = float(fs / best_lag)
            else:
                features[f"{group_name}_acf_max"] = 0.0
                features[f"{group_name}_acf_mean"] = 0.0
                features[f"{group_name}_acf_peak_freq"] = 0.0
        else:
            features[f"{group_name}_acf_max"] = 0.0
            features[f"{group_name}_acf_mean"] = 0.0
            features[f"{group_name}_acf_peak_freq"] = 0.0

        # Sample entropy (on downsampled signal for speed)
        downsample = max(1, n // 200)
        sig_ds = filtered[::downsample]
        features[f"{group_name}_sample_entropy"] = float(_sample_entropy(sig_ds))

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 14.  Aggregate extractor
# ═══════════════════════════════════════════════════════════════════════════

def extract_segment_features(segment: pd.DataFrame) -> Dict[str, float]:
    """Extract ALL feature groups from a single segment.

    Args:
        segment: IMU DataFrame for one activity segment.

    Returns:
        Combined dict of all features.
    """
    features: Dict[str, float] = {}
    features.update(extract_time_features(segment))
    features.update(extract_jerk_features(segment))
    features.update(extract_cross_axis_features(segment))
    features.update(extract_magnitude_features(segment))
    features.update(extract_freq_features(segment))
    features.update(extract_spectral_shape_features(segment))
    features.update(extract_wavelet_features(segment))
    features.update(extract_weighted_spectral_features(segment))
    features.update(extract_peak_features(segment))
    features.update(extract_tremor_features(segment))
    features.update(extract_multiresolution_features(segment))
    features.update(extract_dual_bandpass_features(segment))
    features.update(extract_temporal_features(segment))
    return features


def extract_all_features(
    segments: List[pd.DataFrame],
    patient_id: str,
    hand: str,
    group: str,
    local_score: float,
    global_score: float,
    age: float = 65.0,
    gender: float = 0.0,
) -> Optional[pd.DataFrame]:
    """Extract features from all segments.

    If ``config.PER_SEGMENT`` is True, returns one row **per segment**
    (for per-segment prediction).  Otherwise, features are averaged across
    segments into a single row.

    Args:
        segments: List of segment DataFrames.
        patient_id: E.g. ``"006"``.
        hand: ``"Right"`` or ``"Left"``.
        group: ``"ET"`` or ``"Control"``.
        local_score: Fork feeding score.
        global_score: Subtotal B Extended score.
        age: Patient age in years.
        gender: 0 for Female, 1 for Male.

    Returns:
        DataFrame with feature rows plus metadata, or *None*.
    """
    if not segments:
        logger.warning(
            "Patient %s (%s, %s): no valid segments — skipping",
            patient_id, group, hand,
        )
        return None

    all_dicts: List[Dict[str, float]] = []
    for seg in segments:
        all_dicts.append(extract_segment_features(seg))

    meta = {
        "patient_id": patient_id,
        "group": group,
        "hand": hand,
        "local_score": local_score,
        "global_score": global_score,
        "age": age,
        "gender": gender,
        "is_et": 1 if group == "ET" else 0,
    }

    if cfg.PER_SEGMENT:
        # One row per segment
        rows = []
        for d in all_dicts:
            rows.append({**d, **meta})
        return pd.DataFrame(rows)
    else:
        # Average across segments → single row
        keys = all_dicts[0].keys()
        averaged = {k: float(np.mean([d[k] for d in all_dicts])) for k in keys}
        return pd.DataFrame([{**averaged, **meta}])
