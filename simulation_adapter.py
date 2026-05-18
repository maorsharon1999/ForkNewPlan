"""Adapter: ForkNewPlan per-cycle records → Forkevg-simulator annotated CSVs.

Read-only with respect to the analytical pipeline. Re-loads raw IMU files via
data_loader.load_imu and writes one annotated CSV per source recording, with
two extra columns (Motion, Hand) the dynamic_simulation viewer expects.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

import config as cfg
from data_loader import load_imu

logger = logging.getLogger("fork_pipeline.simulation_adapter")

_MOVEMENT_TO_MOTION = {"scoop": "Scoop", "stab": "Stab"}  # else → "Noise"


def _resolve_hand(hand_from_file: Optional[str], cycle_df: pd.DataFrame, hand_clf) -> str:
    if hand_from_file in ("Right", "Left"):
        return hand_from_file
    try:
        pred = hand_clf.predict(cycle_df)
    except Exception:                       # noqa: BLE001 — ML fallback may raise if unfitted
        return "Unknown"
    return pred if pred in ("Right", "Left") else "Unknown"


def write_annotated_csvs(
    all_cycle_records: List[Dict],
    movement_types: List[str],
    hand_clf,
    out_dir: str = None,
) -> List[str]:
    """Emit one CSV per source recording for the dynamic simulator.

    Returns the list of paths written.
    """
    out_dir = out_dir or cfg.ANNOTATED_CSV_DIR
    os.makedirs(out_dir, exist_ok=True)

    by_file: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(all_cycle_records):
        by_file[r["filepath"]].append(i)

    written: List[str] = []
    for filepath, idxs in by_file.items():
        first = all_cycle_records[idxs[0]]
        group = first["group"]
        patient_id = first["patient_id"]

        try:
            df_raw = load_imu(filepath)
        except Exception as exc:            # noqa: BLE001
            logger.warning("Sim adapter: cannot load %s — %s", filepath, exc)
            continue

        df_out = df_raw.copy()
        df_out["Motion"] = "Noise"
        df_out["Hand"] = "Unknown"

        # Recover (start, end) per cycle by zipping records-of-a-test with test_segs.
        # Records within a test_key were appended in the same order as test_segs
        # (main.py:194-209), so no schema change is needed.
        by_test: Dict[str, List[int]] = defaultdict(list)
        for i in idxs:
            by_test[all_cycle_records[i]["test_key"]].append(i)

        for test_key, test_idxs in by_test.items():
            test_segs = all_cycle_records[test_idxs[0]]["test_segs"]
            if len(test_idxs) != len(test_segs):
                logger.warning(
                    "Sim adapter: %s has %d records vs %d test_segs — skipping test",
                    test_key, len(test_idxs), len(test_segs),
                )
                continue
            for rec_i, (s, e) in zip(test_idxs, test_segs):
                rec = all_cycle_records[rec_i]
                mt = movement_types[rec_i]
                motion = _MOVEMENT_TO_MOTION.get(mt, "Noise")
                hand = _resolve_hand(rec["hand_from_file"], rec["cycle_df"], hand_clf)
                df_out.loc[s:e, "Motion"] = motion
                df_out.loc[s:e, "Hand"] = hand

        group_dir = os.path.join(out_dir, group)
        os.makedirs(group_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(group_dir, f"{patient_id}_{basename}.csv")
        try:
            df_out.to_csv(out_path, index=False)
            written.append(out_path)
        except PermissionError:
            logger.warning("Sim adapter: %s open elsewhere — close and re-run", out_path)

    logger.info("Sim adapter: wrote %d annotated CSV(s) to %s", len(written), out_dir)
    return written
