"""
Main orchestrator for the Fork ET Detection Pipeline.

Usage::

    python main.py
"""

import logging
import os
import sys
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import config as cfg
from utils import setup_logging, ensure_output_dirs
from data_loader import scan_participants, load_imu, load_crf_scores, _hand_from_filename
from preprocessing import (
    bandpass_filter,
    compute_magnitude,
    compute_gyro_magnitude,
    reject_outliers,
    reject_spikes,
    smooth_signal,
    detect_activity,
    classify_cycle_quality,
    segment_signal,
    infer_hand_from_signal,
)
from handedness import HandednessClassifier
from movement_classifier import MovementClassifier
from feature_extraction import extract_all_features
from ml_pipeline import (
    run_regression,
    run_classification,
    run_regress_then_classify,
    run_shap_analysis,
    run_calibrated_classification,
    run_bucketed_regression,
    _build_cv,
)
import visualization as viz
from report_generator import generate_report

logger: logging.Logger

_IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _group_into_tests(
    eating_segs: List, raw_df: pd.DataFrame
) -> List[List]:
    """Group eating segments into separate behavioural tests by >10 s gaps."""
    if not eating_segs:
        return []
    tests = []
    current_test = [eating_segs[0]]
    for j in range(1, len(eating_segs)):
        prev_seg = eating_segs[j - 1]
        curr_seg = eating_segs[j]
        time_gap_ms = (
            raw_df["timestamp"].iloc[curr_seg[0]]
            - raw_df["timestamp"].iloc[prev_seg[1]]
        )
        if time_gap_ms > 10_000:
            tests.append(current_test)
            current_test = [curr_seg]
        else:
            current_test.append(curr_seg)
    tests.append(current_test)
    return tests


def main() -> None:
    global logger
    logger = setup_logging()
    ensure_output_dirs()

    # ── STAGE 1 ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Scanning participants…")
    logger.info("=" * 60)
    records = scan_participants()
    if not records:
        logger.error("No Fork CSV files found — aborting.")
        sys.exit(1)

    n_patients = len({r["patient_id"] for r in records})
    n_et = len({r["patient_id"] for r in records if r["group"] == "ET"})
    n_ctrl = len({r["patient_id"] for r in records if r["group"] == "Control"})
    logger.info(
        "Found %d files from %d patients (ET=%d, Control=%d)",
        len(records), n_patients, n_et, n_ctrl,
    )

    # Print raw CRF values for logging
    from data_loader import _get_crf_data
    crf_data_cache = _get_crf_data()
    if crf_data_cache:
        df_raw_crf = pd.DataFrame.from_dict(crf_data_cache, orient="index")
        df_raw_crf.index.name = "patient_id"
        logger.info("Raw Extracted CRF Data:\n" + df_raw_crf.reset_index().to_string())

    out_dir_signals = os.path.join(cfg.OUTPUT_DIR, "figures", "patient_signals")
    os.makedirs(out_dir_signals, exist_ok=True)

    # ── STAGE 2-4a: Preprocessing + segmentation (data collection pass) ──
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "STAGE 2-4a: Robust preprocessing, enhanced segmentation, "
        "cycle collection…"
    )
    logger.info("=" * 60)

    # Each entry: {patient_id, group, filepath, hand_from_file,
    #              test_key, cycle_df, test_segs, test_df_full, test_df_raw}
    all_cycle_records: List[Dict] = []
    # For visualization: store first valid test per patient
    visual_sample = None
    file_counts: Dict[str, int] = {}

    for rec in records:
        patient_id = rec["patient_id"]
        group = rec["group"]
        filepath = rec["filepath"]

        try:
            df_raw = load_imu(filepath)
        except (ValueError, pd.errors.ParserError) as exc:
            logger.warning(
                "Patient %s: cannot read %s — %s", patient_id, filepath, exc
            )
            continue

        if len(df_raw) == 0:
            continue

        # Stage A: Robust preprocessing
        df_clean = reject_outliers(df_raw.copy())
        df_clean = reject_spikes(df_clean)
        df_smooth = smooth_signal(df_clean.copy())

        # Smoothed acc magnitude for ||a(t)| - 1g| in Cup criterion
        acc_smooth_mag = compute_magnitude(df_smooth).values

        # Narrow BPF (2-15 Hz) for segmentation
        df_narrow = df_clean.copy()
        for col in _IMU_COLS:
            df_narrow[col] = bandpass_filter(
                df_narrow[col].values,
                low=cfg.NARROW_BPF_LOW,
                high=cfg.NARROW_BPF_HIGH,
            )

        # Wide BPF (0.5-20 Hz) for feature extraction
        df_wide = df_clean.copy()
        for col in _IMU_COLS:
            df_wide[col] = bandpass_filter(df_wide[col].values)

        # Stage B: Enhanced segmentation (Cup combined criterion)
        segments = detect_activity(df_narrow, acc_smooth_mag)
        if not segments:
            continue

        quality_labels = classify_cycle_quality(segments, df_raw)
        eating_segs = [
            (s, e)
            for (s, e), lbl in zip(segments, quality_labels)
            if lbl == "cycle"
        ]
        if not eating_segs:
            continue

        # Get hand from filename
        fname = os.path.basename(filepath)
        hand_from_file: Optional[str] = _hand_from_filename(fname)

        # Group into behavioural tests
        tests = _group_into_tests(eating_segs, df_raw)

        for test_idx, test_segs in enumerate(tests):
            test_key = f"{patient_id}__{fname}__{test_idx}"

            for seg_start, seg_end in test_segs:
                cycle_df = df_wide.iloc[seg_start : seg_end + 1].reset_index(drop=True)
                all_cycle_records.append(
                    {
                        "patient_id": patient_id,
                        "group": group,
                        "filepath": filepath,
                        "hand_from_file": hand_from_file,
                        "test_key": test_key,
                        "test_segs": test_segs,
                        "cycle_df": cycle_df,
                    }
                )

            # Visualization for first valid test per patient
            count = file_counts.get(patient_id, 0) + 1
            file_counts[patient_id] = count

            start_pad = max(0, test_segs[0][0] - int(cfg.FS * 2))
            end_pad = min(len(df_raw) - 1, test_segs[-1][1] + int(cfg.FS * 2))
            df_test_raw = df_raw.iloc[start_pad : end_pad + 1].copy().reset_index(drop=True)

            if visual_sample is None:
                mag_full = compute_magnitude(df_raw)
                shifted_segs = [(s - start_pad, e - start_pad) for s, e in test_segs]
                visual_sample = {
                    "magnitude": mag_full[start_pad : end_pad + 1].values,
                    "activity_segments": shifted_segs,
                    "patient_id": patient_id,
                }

            viz.plot_and_save_patient_signal(
                df=df_test_raw,
                patient_run_id=f"{patient_id}_test{count}",
                group=group,
                local_score=0.0,  # placeholder; CRF loaded in pass 2
                out_dir=out_dir_signals,
            )

    logger.info(
        "Data collection complete: %d cycle records from %d patients",
        len(all_cycle_records),
        len({r["patient_id"] for r in all_cycle_records}),
    )

    if not all_cycle_records:
        logger.error("No valid cycles found — aborting.")
        sys.exit(1)

    # ── STAGE 4b: Train handedness classifier ────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4b: Training handedness classifier…")
    logger.info("=" * 60)

    hand_clf = HandednessClassifier()
    handedness_accuracy: Optional[float] = None

    labeled = [
        (r["cycle_df"], r["hand_from_file"], r["patient_id"])
        for r in all_cycle_records
        if r["hand_from_file"] is not None
    ]
    if len(labeled) >= 10:
        hand_clf.fit([x[0] for x in labeled], [x[1] for x in labeled])
        handedness_accuracy, _ = hand_clf.evaluate_loso(
            [x[0] for x in labeled],
            [x[1] for x in labeled],
            [x[2] for x in labeled],
        )
    else:
        logger.warning(
            "Only %d labeled cycles for handedness training "
            "(need ≥10) — using heuristic fallback",
            len(labeled),
        )

    # ── STAGE 4c: Movement-type clustering (GMM k=4) ─────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4c: Movement-type clustering (GMM k=%d)…", cfg.GMM_N_COMPONENTS)
    logger.info("=" * 60)

    move_clf = MovementClassifier()
    all_cycle_dfs = [r["cycle_df"] for r in all_cycle_records]
    movement_types: List[str] = ["unknown"] * len(all_cycle_records)

    min_fit_cycles = cfg.GMM_N_COMPONENTS * 5
    if len(all_cycle_dfs) >= min_fit_cycles:
        move_clf.fit(all_cycle_dfs)
        cluster_assignments = move_clf.predict_all_clusters(all_cycle_dfs)
        movement_types = [
            move_clf.predict_label(all_cycle_dfs[i])
            for i in range(len(all_cycle_dfs))
        ]

        inspection_pdf = os.path.join(cfg.OUTPUT_DIR, "cluster_inspection.pdf")
        move_clf.generate_inspection_pdf(
            all_cycle_dfs,
            cluster_assignments=cluster_assignments,
            output_path=inspection_pdf,
        )
        logger.info(
            "Cluster label map: %s. "
            "If empty, inspect %s and populate config.GMM_CLUSTER_LABEL_MAP.",
            move_clf.cluster_label_map,
            inspection_pdf,
        )
    else:
        logger.warning(
            "Only %d cycles — need ≥%d for GMM clustering. "
            "All cycles labelled 'unknown'.",
            len(all_cycle_dfs), min_fit_cycles,
        )

    # ── STAGE 5: Feature extraction (second pass) ────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "STAGE 5: Feature extraction (PER_SEGMENT=%s)…", cfg.PER_SEGMENT
    )
    logger.info("=" * 60)

    # Group cycle records by test_key (preserves test structure)
    records_by_test: Dict[str, List] = defaultdict(list)
    for i, r in enumerate(all_cycle_records):
        records_by_test[r["test_key"]].append((i, r))

    feature_rows: List[pd.DataFrame] = []
    skipped_patients: set = set()
    total_segments = 0

    for test_key, idx_rec_pairs in records_by_test.items():
        first_rec = idx_rec_pairs[0][1]
        patient_id = first_rec["patient_id"]
        group = first_rec["group"]

        # Majority-vote hand for this test
        hands = []
        for cidx, crec in idx_rec_pairs:
            hf = crec["hand_from_file"]
            hands.append(hf if hf is not None else hand_clf.predict(crec["cycle_df"]))
        hand = Counter(hands).most_common(1)[0][0]

        scores = load_crf_scores(patient_id, hand, group)
        if not scores:
            skipped_patients.add(patient_id)
            continue

        cycle_dfs = [r["cycle_df"] for _, r in idx_rec_pairs]
        test_mtypes = [movement_types[cidx] for cidx, _ in idx_rec_pairs]

        row_df = extract_all_features(
            segments=cycle_dfs,
            patient_id=patient_id,
            hand=hand,
            group=group,
            local_score=scores["local_score"],
            global_score=scores["global_score"],
            age=scores.get("age", 65.0),
            gender=scores.get("gender", 0.0),
        )

        if row_df is not None:
            row_df = row_df.copy()
            n_rows = len(row_df)

            # Assign movement_type per row (one per cycle when PER_SEGMENT=True)
            if cfg.PER_SEGMENT:
                mt_padded = (test_mtypes + ["unknown"] * n_rows)[:n_rows]
                row_df["movement_type"] = mt_padded
            else:
                row_df["movement_type"] = Counter(test_mtypes).most_common(1)[0][0]

            # 4-cell CRF scores for bucketed regression
            for k in ["rt_scoop", "lf_scoop", "rt_stab", "lf_stab"]:
                row_df[k] = scores.get(k, 0.0)

            feature_rows.append(row_df)
            total_segments += n_rows

    logger.info(
        "Feature extraction: %d valid tests → %d segments "
        "(%d patients skipped due to missing CRF)",
        len(feature_rows), total_segments, len(skipped_patients),
    )

    if not feature_rows:
        logger.error("Feature extraction produced no rows — aborting.")
        sys.exit(1)

    features_df = pd.concat(feature_rows, ignore_index=True)
    logger.info(
        "Feature matrix: %d rows × %d columns",
        features_df.shape[0], features_df.shape[1],
    )

    # ── STAGE 6: Regression (existing, unchanged) ─────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "STAGE 6: ML Regression — LOSO=%s, TUNE=%s, STACK=%s, AUG=%s",
        cfg.USE_LOSO, cfg.TUNE_HYPERPARAMS, cfg.USE_STACKING, cfg.USE_AUGMENTATION,
    )
    logger.info("=" * 60)

    et_df = features_df[features_df["group"] == "ET"].copy()
    if len(et_df) >= 5:
        logger.info("── Local score regression (fork feeding) ──")
        reg_local = run_regression(et_df, target_col="local_score")
        logger.info("")
        logger.info("── Global score regression (Subtotal B Ext) ──")
        reg_global = run_regression(et_df, target_col="global_score")
    else:
        logger.warning("Only %d ET samples — skipping regression", len(et_df))
        reg_local = {}
        reg_global = {}

    # ── STAGE 6b: Bucketed regression ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6b: Bucketed regression (Hand × Movement Type)…")
    logger.info("=" * 60)
    bucketed_results = run_bucketed_regression(features_df)

    # Log bucket-level cycle counts for sparsity audit
    if "movement_type" in features_df.columns and "hand" in features_df.columns:
        et_subset = features_df[features_df["group"] == "ET"]
        bucket_counts = (
            et_subset.groupby(["hand", "movement_type"])["patient_id"]
            .agg(["count", "nunique"])
            .rename(columns={"count": "cycles", "nunique": "patients"})
        )
        logger.info("Bucket sparsity report:\n%s", bucket_counts.to_string())

    # ── STAGE 7: Classification ───────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 7: Classification (ET vs Control)…")
    logger.info("=" * 60)
    cls_results = run_classification(features_df)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 7b: Regress-then-classify…")
    logger.info("=" * 60)
    rtc_results = run_regress_then_classify(features_df, score_col="global_score")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 7c: Calibrated classifiers…")
    logger.info("=" * 60)
    cal_results = run_calibrated_classification(features_df)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 7d: SHAP analysis…")
    logger.info("=" * 60)
    et_df = features_df[features_df["group"] == "ET"].copy()
    shap_reg = run_shap_analysis(et_df, target="local_score", task="regression")
    shap_cls = run_shap_analysis(features_df, task="classification")

    # ── STAGE 8: Visualisations ───────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 8: Generating visualisations…")
    logger.info("=" * 60)

    meta_cols = [
        "patient_id", "group", "hand", "local_score", "global_score", "is_et",
        "movement_type", "rt_scoop", "lf_scoop", "rt_stab", "lf_stab",
    ]

    viz.plot_pca(features_df, label_col="is_et", filename="pca_et_vs_control.png")
    viz.plot_boxplot(features_df, group_col="group", filename="boxplot_features.png")

    if "movement_type" in features_df.columns:
        viz.plot_cluster_pca(features_df, cluster_col="movement_type",
                             filename="pca_movement_clusters.png")

    if visual_sample:
        viz.plot_signal_with_segments(
            visual_sample["magnitude"],
            visual_sample["activity_segments"],
            visual_sample["patient_id"],
            filename=f"activity_{visual_sample['patient_id']}.png",
        )

    if len(et_df) >= 5:
        feat_cols = [c for c in et_df.columns if c not in meta_cols]
        X = et_df[feat_cols].copy()
        y = et_df["local_score"].copy()
        groups = et_df["patient_id"].copy()
        valid = y.notna()
        X, y, groups = (
            X[valid].reset_index(drop=True),
            y[valid].reset_index(drop=True),
            groups[valid].reset_index(drop=True),
        )
        if len(X) >= 5:
            from sklearn.preprocessing import StandardScaler

            pipe_reg = Pipeline([
                ("scaler", StandardScaler()),
                ("rfe", RFE(
                    estimator=RandomForestRegressor(
                        n_estimators=50, max_depth=5,
                        random_state=cfg.RANDOM_STATE, n_jobs=-1,
                    ),
                    n_features_to_select=cfg.TOP_K_FEATURES,
                    step=0.1,
                )),
                ("model", RandomForestRegressor(
                    n_estimators=100, max_depth=10,
                    random_state=cfg.RANDOM_STATE, n_jobs=-1,
                )),
            ])
            cv = _build_cv(groups, task="regression")
            try:
                y_pred = cross_val_predict(pipe_reg, X, y, cv=cv, groups=groups)
            except ValueError:
                y_pred = cross_val_predict(
                    pipe_reg, X, y,
                    cv=KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE),
                )
            viz.plot_scatter(
                y.values, y_pred,
                title="Local Score - RandomForest (LOSO CV)",
                filename="scatter_local_score.png",
            )
            viz.plot_bland_altman(
                y.values, y_pred,
                title="Bland-Altman - Local Score (RF)",
                filename="bland_altman_local.png",
            )

    if cls_results and len(features_df) >= 5:
        feat_cols = [c for c in features_df.columns if c not in meta_cols]
        X = features_df[feat_cols].copy()
        y = features_df["is_et"].astype(int).copy()
        groups = features_df["patient_id"].copy()
        from sklearn.preprocessing import StandardScaler

        pipe_cls = Pipeline([
            ("scaler", StandardScaler()),
            ("rfe", RFE(
                estimator=RandomForestClassifier(
                    n_estimators=50, max_depth=5, class_weight="balanced",
                    random_state=cfg.RANDOM_STATE, n_jobs=-1,
                ),
                n_features_to_select=cfg.TOP_K_FEATURES,
                step=0.1,
            )),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=cfg.RANDOM_STATE,
            )),
        ])
        cv = _build_cv(groups, task="classification")
        try:
            y_pred = cross_val_predict(pipe_cls, X, y, cv=cv, groups=groups)
            viz.plot_confusion_matrix(y.values, y_pred, filename="confusion_matrix.png")
        except ValueError:
            pass
        try:
            y_score = cross_val_predict(
                pipe_cls, X, y, cv=cv, groups=groups, method="predict_proba"
            )[:, 1]
            viz.plot_roc(y.values, y_score, filename="roc_et_vs_control.png")
        except ValueError as exc:
            logger.warning("ROC plot skipped: %s", exc)

    # ── STAGE 9: Clinical report ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 9: Generating clinical report…")
    logger.info("=" * 60)

    rfe_stab = cls_results.pop("__rfe_stability__", {})
    report_metrics = {
        "n_patients": n_patients,
        "n_et": n_et,
        "n_control": n_ctrl,
        "n_segments": total_segments,
        "n_features": cfg.TOP_K_FEATURES,
        "handedness_accuracy": handedness_accuracy,
        "reg_local": reg_local,
        "reg_global": reg_global,
        "bucketed_regression": bucketed_results,
        "cls_results": cls_results,
        "cal_results": cal_results,
        "rtc_results": rtc_results,
        "shap_reg": shap_reg,
        "shap_cls": shap_cls,
        "rfe_stability": rfe_stab,
    }
    report_path = generate_report(
        metrics_dict=report_metrics,
        figures_dir=cfg.OUTPUT_DIR,
        output_dir=cfg.OUTPUT_DIR,
    )
    logger.info("Report saved to %s", report_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE.  Results saved to %s", cfg.OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
