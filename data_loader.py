"""
Data loading: scan participant folders for Fork CSVs, read IMU data, parse CRF.
"""

import logging
import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional

import pandas as pd

import config as cfg
from utils import normalize_hand_label

logger = logging.getLogger("fork_pipeline.data_loader")

# ---------------------------------------------------------------------------
# Internal cache for CRF data (parsed once, reused)
# ---------------------------------------------------------------------------
_crf_cache: Optional[Dict[str, Dict[str, Any]]] = None


# ── public API ─────────────────────────────────────────────────────────────


def scan_participants(data_root: str = cfg.DATA_ROOT) -> List[Dict[str, str]]:
    """Recursively scan *data_root* for Fork CSV files, keeping only the largest appending file per device.

    Args:
        data_root: Root directory containing participant folders.

    Returns:
        List of dicts, each with keys ``patient_id`` (e.g. ``"006"``),
        ``group`` (``"ET"`` or ``"Control"``), and ``filepath`` (absolute path to the CSV).
    """
    records: List[Dict[str, str]] = []

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        clean = _strip_unicode(folder)
        parsed = _parse_folder_name(clean)
        if not parsed:
            continue

        for group, patient_id in parsed:
            # Group files by device prefix to deduplicate appended tests
            patient_files = {}

            # Walk recursively to find all Fork CSVs
            for root, _dirs, files in os.walk(folder_path):
                for fname in files:
                    if not fname.lower().endswith(".csv"):
                        continue
                    lower = fname.lower()
                    if not lower.startswith("fork"):
                        continue

                    # Extract prefix (device identifier)
                    prefix = "fork"
                    if lower.startswith("fork1"): prefix = "fork1"
                    elif lower.startswith("fork2"): prefix = "fork2"
                    elif lower.startswith("fork_"): prefix = "fork_"
                    
                    if prefix == "fork_" and not cfg.INCLUDE_AMBIGUOUS_FORK:
                        continue

                    filepath = os.path.join(root, fname)
                    size = os.path.getsize(filepath)

                    # Keep the largest file per device (since it contains all appended data)
                    if prefix not in patient_files or size > patient_files[prefix]["size"]:
                        patient_files[prefix] = {
                            "patient_id": patient_id,
                            "group": group,
                            "filepath": filepath,
                            "size": size
                        }
            
            for dev_data in patient_files.values():
                records.append({
                    "patient_id": dev_data["patient_id"],
                    "group": dev_data["group"],
                    "filepath": dev_data["filepath"],
                })

    logger.info("Scanned %d unique Fork devices across participants", len(records))
    return records


def load_imu(filepath: str) -> pd.DataFrame:
    """Read a single Fork IMU CSV and return the 7 relevant columns.

    The raw CSV has 10 columns:
        0: unix timestamp (ms)
        1: counter
        2: datetime string  (skipped)
        ...

    Args:
        filepath: Absolute path to the CSV file.

    Returns:
        DataFrame with columns
        ``[timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]``.

    Raises:
        ValueError: If the file has fewer than 10 columns.
    """
    df = pd.read_csv(filepath, header=None)
    if df.shape[1] < 10:
        raise ValueError(
            f"{filepath} has only {df.shape[1]} columns, expected ≥ 10"
        )
    # Select: col0 (Unix timestamp in ms), cols 4-9 (IMU)
    df = df.iloc[:, [0, 4, 5, 6, 7, 8, 9]].copy()
    df.columns = [
        "timestamp", "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
    ]
    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df




def load_crf_scores(
    patient_id: str,
    hand: str,
    group: str = "ET",
) -> Optional[Dict[str, float]]:
    """Return clinical scores for a patient from the CRF Excel.

    For the **local score** the function averages the scooping and stabbing
    fork columns for the relevant hand.  For **Bilateral** tremor all four
    fork columns are averaged.

    Control patients receive ``local_score = 0`` and ``global_score = 0``.

    Args:
        patient_id: Three-digit string, e.g. ``"006"``.
        hand: ``"Right"`` or ``"Left"`` (physical hand holding the fork).
        group: ``"ET"`` or ``"Control"``.

    Returns:
        Dict with ``local_score``, ``global_score``, ``age``, and ``gender``, or *None* if the
        patient cannot be found or scores are missing.
    """
    if group == "Control":
        # Missing data imputation logic: Controls might not have CRF entries, so we need to fetch them
        # if possible. Assuming controls are also in the CRF. If they have age/gender, we want it.
        # But `tremor_hand` logic breaks for controls. So let's just try to get demographics if possible.
        crf = _get_crf_data()
        record = crf.get(patient_id)
        if record is not None:
            return {
                "local_score": 0.0, "global_score": 0.0,
                "age": record.get("age", 65.0),
                "gender": record.get("gender", 0.0)
            }
        return {"local_score": 0.0, "global_score": 0.0, "age": 65.0, "gender": 0.0}

    crf = _get_crf_data()
    record = crf.get(patient_id)
    if record is None:
        logger.warning("Patient %s not found in CRF — skipping", patient_id)
        return None

    tremor_hand = record["tremor_hand"]

    # Local score: average scooping + stabbing for the relevant hand(s)
    if tremor_hand == "Right":
        vals = [record["rt_fork_scoop"], record["rt_fork_stab"]]
    elif tremor_hand == "Left":
        vals = [record["lf_fork_scoop"], record["lf_fork_stab"]]
    elif tremor_hand == "Bilateral":
        vals = [
            record["rt_fork_scoop"], record["lf_fork_scoop"],
            record["rt_fork_stab"], record["lf_fork_stab"],
        ]
    else:
        logger.warning(
            "Patient %s: unknown tremor hand '%s' — skipping",
            patient_id, tremor_hand,
        )
        return None

    valid = [v for v in vals if v is not None]
    if not valid:
        logger.warning(
            "Patient %s: all fork score columns are None — skipping",
            patient_id,
        )
        return None
    local_score = mean(valid)

    global_score = record["subtotal_b_ext"]
    if global_score is None:
        logger.warning(
            "Patient %s: Subtotal B Extended is None — skipping", patient_id,
        )
        return None

    return {
        "local_score": float(local_score), 
        "global_score": float(global_score),
        "age": record.get("age", 65.0),
        "gender": record.get("gender", 0.0)
    }


# ── internal helpers ───────────────────────────────────────────────────────


_UNICODE_MARKS = re.compile(r"[\u200f\u200e\ufeff\u200b\u200c\u200d]")


def _strip_unicode(text: str) -> str:
    return _UNICODE_MARKS.sub("", text)


def _parse_folder_name(name: str) -> list:
    """Return list of ``(group, patient_id)`` tuples found in *name*.

    Handles combined folders like ``"Control-003 and ET-020"`` by returning
    all matches, not just the first one.

    Returns:
        List of (group, pid) tuples, e.g. ``[("Control", "003"), ("ET", "020")]``.
        Empty list if no match.
    """
    matches = re.findall(r"(Control|ET)-(\d{3})", name)
    return matches


def _hand_from_filename(filename: str) -> Optional[str]:
    """Determine which hand based on the filename prefix.

    ``Fork1`` → Right, ``Fork2`` → Left.
    Files starting with just ``Fork`` (no digit) use the default hand from
    config if ``INCLUDE_AMBIGUOUS_FORK`` is True, otherwise are skipped.
    """
    base = os.path.splitext(filename)[0]
    lower = base.lower()
    if not lower.startswith("fork"):
        return None
    # Fork1_... or Fork2_...
    for prefix, hand in cfg.FORK_HAND_MAP.items():
        if lower.startswith(prefix.lower()):
            return hand
    # Bare "Fork_..." — use default hand or skip
    if lower.startswith("fork_") or lower == "fork":
        if cfg.INCLUDE_AMBIGUOUS_FORK:
            logger.info(
                "Including %s with default hand=%s",
                filename, cfg.FORK_DEFAULT_HAND,
            )
            return cfg.FORK_DEFAULT_HAND
        logger.debug("Skipping %s — cannot determine hand", filename)
        return None
    return None



def _safe_float(val: Any) -> Optional[float]:
    """Convert a CRF cell to float; return *None* on failure or NaN."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _get_crf_data() -> Dict[str, Dict[str, Any]]:
    """Parse CRF Excel (once) and return a lookup keyed by 3-digit patient ID."""
    global _crf_cache
    if _crf_cache is not None:
        return _crf_cache

    _crf_cache = {}
    path = cfg.CRF_PATH
    if not os.path.exists(path):
        logger.error("CRF file not found: %s", path)
        return _crf_cache

    try:
        df = pd.read_excel(
            path,
            sheet_name=cfg.CRF_SHEET_ET,
            header=None,
        )
    except ValueError:
        logger.error(
            "Sheet '%s' not found in CRF file", cfg.CRF_SHEET_ET,
        )
        return _crf_cache

    for i in range(2, len(df)):  # skip header rows
        raw_id = df.iloc[i, cfg.CRF_COL_SUBJECT]
        if pd.isna(raw_id):
            continue
        raw_id = str(raw_id).strip()
        num_match = re.match(r"(\d+)", raw_id)
        if not num_match:
            continue
        patient_id = num_match.group(1).zfill(3)

        tremor_raw = df.iloc[i, cfg.CRF_COL_TREMOR_HAND]
        tremor_hand = normalize_hand_label(
            str(tremor_raw) if not pd.isna(tremor_raw) else "",
        )

        _crf_cache[patient_id] = {
            "tremor_hand": tremor_hand,
            "rt_fork_scoop": _safe_float(df.iloc[i, cfg.CRF_COL_RT_FORK_SCOOP]),
            "lf_fork_scoop": _safe_float(df.iloc[i, cfg.CRF_COL_LF_FORK_SCOOP]),
            "rt_fork_stab": _safe_float(df.iloc[i, cfg.CRF_COL_RT_FORK_STAB]),
            "lf_fork_stab": _safe_float(df.iloc[i, cfg.CRF_COL_LF_FORK_STAB]),
            "subtotal_b_ext": _safe_float(df.iloc[i, cfg.CRF_COL_SUBTOTAL_B_EXT]),
            "age": _safe_float(df.iloc[i, cfg.CRF_COL_AGE]),
        }
        
        # Gender mapping: 'Male', 'זכר' -> 1 | 'Female', 'נקבה' -> 0
        raw_gender = df.iloc[i, cfg.CRF_COL_GENDER]
        gender_val = 0.0
        if not pd.isna(raw_gender):
            lower_g = str(raw_gender).strip().lower()
            if "male" in lower_g and "female" not in lower_g:
                gender_val = 1.0
            elif "זכר" in lower_g:
                gender_val = 1.0
        _crf_cache[patient_id]["gender"] = gender_val

    logger.info("Loaded CRF scores for %d ET patients", len(_crf_cache))
    return _crf_cache
