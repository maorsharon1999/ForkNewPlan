"""
Main orchestrator for the Fork ET Detection Pipeline.

Usage::

    python main.py
"""

import logging
import os
import sys
import warnings
from typing import Dict, List

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
from data_loader import scan_participants, load_imu, load_crf_scores
from preprocessing import (
    bandpass_filter,
    compute_magnitude,
    detect_activity,
    segment_signal,
    infer_hand_from_signal,
    filter_eating_cycles
)
from feature_extraction import extract_all_features
from ml_pipeline import (
    run_regression, run_classification, run_regress_then_classify,
    run_shap_analysis, run_calibrated_classification,
    _build_cv,
)
import visualization as viz
from report_generator import generate_report

logger: logging.Logger


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
    logger.info("Found %d files from %d patients (ET=%d, Control=%d)",
                len(records), n_patients, n_et, n_ctrl)

    # ── STAGE 2-4: Process pipeline ───────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2-4: Processing, Filtering, & Feature Extraction…")
    logger.info("=" * 60)

    # Print raw extracted values from CRF for logging
    from data_loader import _get_crf_data
    crf_data_cache = _get_crf_data()
    if crf_data_cache:
        df_raw_crf = pd.DataFrame.from_dict(crf_data_cache, orient="index")
        df_raw_crf.index.name = "patient_id"
        logger.info("Raw Extracted CRF Data:\n" + df_raw_crf.reset_index().to_string())

    out_dir_signals = os.path.join(cfg.OUTPUT_DIR, "figures", "patient_signals")
    os.makedirs(out_dir_signals, exist_ok=True)

    feature_rows: List[pd.DataFrame] = []
    file_counts = {}
    skipped_patients = set()
    total_segments = 0
    visual_sample = None

    for i, rec in enumerate(records):
        patient_id = rec["patient_id"]
        group = rec["group"]
        filepath = rec["filepath"]

        try:
            df_full = load_imu(filepath)
        except (ValueError, pd.errors.ParserError) as exc:
            logger.warning("Patient %s: cannot read %s — %s", patient_id, filepath, exc)
            continue
            
        if len(df_full) == 0:
            continue

        raw_full = df_full.copy()
        mag = compute_magnitude(df_full)

        for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            df_full[col] = bandpass_filter(df_full[col].values)

        segments = detect_activity(mag.values)
        if not segments:
            continue
            
        eating_segments = filter_eating_cycles(segments, raw_full, cfg.FS)
        if not eating_segments:
            continue

        # Group into separate tests by >10s gaps between actual eating segments
        tests = []
        current_test = [eating_segments[0]]
        
        for j in range(1, len(eating_segments)):
            prev_seg = eating_segments[j-1]
            curr_seg = eating_segments[j]
            
            # Time gap between the end of previous eating cycle and start of current one
            time_gap_ms = raw_full["timestamp"].iloc[curr_seg[0]] - raw_full["timestamp"].iloc[prev_seg[1]]
            
            if time_gap_ms > 10000:
                tests.append(current_test)
                current_test = [curr_seg]
            else:
                current_test.append(curr_seg)
        tests.append(current_test)
        
        # Process each behavioral test
        from collections import Counter
        for test_idx, test_segs in enumerate(tests):
            seg_dfs = segment_signal(df_full, test_segs)
            
            # 2. Detect used hand for THIS specific behavioral test
            # Use majority vote across all eating movements in this test
            hands_detected = [infer_hand_from_signal(s) for s in seg_dfs]
            hand = Counter(hands_detected).most_common(1)[0][0]
            
            scores = load_crf_scores(patient_id, hand, group)
            if not scores:
                skipped_patients.add(patient_id)
                continue

            total_segments += len(seg_dfs)
            
            # 5. Extract features
            row = extract_all_features(
                segments=seg_dfs,
                patient_id=patient_id,
                hand=hand,
                group=group,
                local_score=scores["local_score"],
                global_score=scores["global_score"]
            )
            
            if row is not None:
                feature_rows.append(row)
                
                # Store first valid sample for Stage 7 visualization
                if visual_sample is None:
                    start_pad = max(0, test_segs[0][0] - int(cfg.FS * 2))
                    end_pad = min(len(mag)-1, test_segs[-1][1] + int(cfg.FS * 2))
                    shifted_segs = [(s[0] - start_pad, s[1] - start_pad) for s in test_segs]
                    
                    visual_sample = {
                        "magnitude": mag[start_pad:end_pad+1],
                        "activity_segments": shifted_segs,
                        "patient_id": patient_id
                    }
                
                # 6. Visualization plotting for this test
                start_pad = max(0, test_segs[0][0] - int(cfg.FS * 2))
                end_pad = min(len(raw_full)-1, test_segs[-1][1] + int(cfg.FS * 2))
                df_test_raw = raw_full.iloc[start_pad:end_pad+1].copy().reset_index(drop=True)
                
                count = file_counts.get(patient_id, 0) + 1
                file_counts[patient_id] = count
                viz.plot_and_save_patient_signal(
                    df=df_test_raw,
                    patient_run_id=f"{patient_id}_test{count}",
                    group=group,
                    local_score=scores["local_score"],
                    out_dir=out_dir_signals
                )

    logger.info("Processed features: %d valid tests -> %d segments (%d patients skipped due to missing CRF)",
                len(feature_rows), total_segments, len(skipped_patients))

    if not feature_rows:
        logger.error("Feature extraction produced no rows — aborting.")
        sys.exit(1)

    features_df = pd.concat(feature_rows, ignore_index=True)
    logger.info("Feature matrix: %d rows × %d columns (PER_SEGMENT=%s)",
                features_df.shape[0], features_df.shape[1], cfg.PER_SEGMENT)

    # ── STAGE 5: Regression ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 5: ML Pipeline (Regression) — LOSO=%s, TUNE=%s, STACK=%s, AUG=%s",
                cfg.USE_LOSO, cfg.TUNE_HYPERPARAMS, cfg.USE_STACKING, cfg.USE_AUGMENTATION)
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

    # ── STAGE 6: Classification ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6: ML Pipeline (Classification — ET vs Control)…")
    logger.info("=" * 60)
    cls_results = run_classification(features_df)

    # ── STAGE 6b: Regress-then-classify ──────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6b: Regress-then-classify (ET vs Control via score)…")
    logger.info("=" * 60)
    rtc_results = run_regress_then_classify(features_df, score_col="global_score")

    # ── STAGE 6c: Calibrated classification ──────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6c: Calibrated classifiers (Platt scaling)…")
    logger.info("=" * 60)
    cal_results = run_calibrated_classification(features_df)

    # ── STAGE 6d: SHAP interpretability ──────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6d: SHAP analysis…")
    logger.info("=" * 60)
    et_df = features_df[features_df["group"] == "ET"].copy()
    shap_reg = run_shap_analysis(et_df, target="local_score", task="regression")
    shap_cls = run_shap_analysis(features_df, task="classification")

    # ── STAGE 7: Visualisations ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 7: Generating visualisations…")
    logger.info("=" * 60)

    meta_cols = ["patient_id", "group", "hand", "local_score",
                 "global_score", "is_et"]

    # PCA
    viz.plot_pca(features_df, label_col="is_et", filename="pca_et_vs_control.png")

    # Boxplots
    viz.plot_boxplot(features_df, group_col="group", filename="boxplot_features.png")

    # Activity detection sample
    if visual_sample:
        viz.plot_signal_with_segments(
            visual_sample["magnitude"], visual_sample["activity_segments"],
            visual_sample["patient_id"], filename=f"activity_{visual_sample['patient_id']}.png")

    # Scatter + Bland-Altman for regression (Pipeline: no leakage)
    if len(et_df) >= 5:
        feat_cols = [c for c in et_df.columns if c not in meta_cols]
        X = et_df[feat_cols].copy()
        y = et_df["local_score"].copy()
        groups = et_df["patient_id"].copy()
        valid = y.notna()
        X, y, groups = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True), groups[valid].reset_index(drop=True)
        if len(X) >= 5:
            from sklearn.preprocessing import StandardScaler
            pipe_reg = Pipeline([
                ("scaler", StandardScaler()),
                ("rfe", RFE(
                    estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                                     random_state=cfg.RANDOM_STATE, n_jobs=-1),
                    n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
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
                y_pred = cross_val_predict(pipe_reg, X, y,
                                           cv=KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE))
            viz.plot_scatter(y.values, y_pred,
                             title="Local Score - RandomForest (LOSO CV)",
                             filename="scatter_local_score.png")
            viz.plot_bland_altman(y.values, y_pred,
                                  title="Bland-Altman - Local Score (RF)",
                                  filename="bland_altman_local.png")

    # Confusion matrix + ROC for classification (Pipeline: no leakage)
    if cls_results and len(features_df) >= 5:
        feat_cols = [c for c in features_df.columns if c not in meta_cols]
        X = features_df[feat_cols].copy()
        y = features_df["is_et"].astype(int).copy()
        groups = features_df["patient_id"].copy()
        from sklearn.preprocessing import StandardScaler
        pipe_cls = Pipeline([
            ("scaler", StandardScaler()),
            ("rfe", RFE(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced",
                                                  random_state=cfg.RANDOM_STATE, n_jobs=-1),
                n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
            )),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.RANDOM_STATE)),
        ])
        cv = _build_cv(groups, task="classification")

        # Confusion matrix
        try:
            y_pred = cross_val_predict(pipe_cls, X, y, cv=cv, groups=groups)
            viz.plot_confusion_matrix(y.values, y_pred, filename="confusion_matrix.png")
        except ValueError:
            pass

        # ROC
        try:
            y_score = cross_val_predict(
                pipe_cls, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
            viz.plot_roc(y.values, y_score, filename="roc_et_vs_control.png")
        except ValueError as exc:
            logger.warning("ROC plot skipped: %s", exc)

    # ── STAGE 8: Clinical report ─────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 8: Generating clinical report…")
    logger.info("=" * 60)

    # FIXED: cls_results.pop() — хрупкая мутация
    rfe_stab = cls_results.pop("__rfe_stability__", {})
    report_metrics = {
        "n_patients": n_patients,
        "n_et": n_et,
        "n_control": n_ctrl,
        "n_segments": total_segments,
        "n_features": cfg.TOP_K_FEATURES,
        "reg_local": reg_local,
        "reg_global": reg_global,
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

    # ── DONE ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE.  Results saved to %s", cfg.OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
