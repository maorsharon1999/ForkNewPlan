"""
Clinical PDF report generator for the Fork ET Detection Pipeline.

Generates an automated PDF report after each pipeline run, containing:
- Title with run timestamp
- Summary statistics (patients, segments, features after RFE)
- Regression results tables (Local Score & Global Score)
- Classification results table
- Top SHAP features (embedded bar chart)
- All generated figures (PCA, scatter, Bland-Altman, confusion matrix, ROC)
- Limitations and future work paragraph

All content is driven by the ``metrics_dict`` passed in at runtime —
no numbers are hardcoded.

Dependencies:
    fpdf2 (``pip install fpdf2``)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fpdf import FPDF

import config as cfg

logger = logging.getLogger("fork_pipeline.report_generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MARGIN = 15
_PAGE_W = 210 - 2 * _MARGIN  # A4 usable width in mm

# Figures that should be embedded (order matters for readability)
_FIGURE_FILES = [
    ("PCA - ET vs Control", "pca_et_vs_control.png"),
    ("Scatter - Local Score", "scatter_local_score.png"),
    ("Bland-Altman - Local Score", "bland_altman_local.png"),
    ("Confusion Matrix", "confusion_matrix.png"),
    ("ROC Curve", "roc_et_vs_control.png"),
    ("SHAP Bar - Regression", "shap_bar_regression.png"),
    ("SHAP Bar - Classification", "shap_bar_classification.png"),
    ("SHAP Beeswarm - Regression", "shap_beeswarm_regression.png"),
    ("SHAP Beeswarm - Classification", "shap_beeswarm_classification.png"),
    ("PCA - Movement Clusters", "pca_movement_clusters.png"),
    ("Cluster Inspection (from PDF, page 3 PCA)", "cluster_inspection_pca.png"),
]


# ---------------------------------------------------------------------------
# Helper: draw a table on the PDF
# ---------------------------------------------------------------------------
def _draw_table(
    pdf: FPDF,
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[float]] = None,
) -> None:
    """Render a bordered table with shaded header row."""
    if col_widths is None:
        col_widths = [_PAGE_W / len(headers)] * len(headers)

    # Header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 50, 80)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()

    # Body
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    fill = False
    for row in rows:
        if fill:
            pdf.set_fill_color(235, 235, 245)
        else:
            pdf.set_fill_color(255, 255, 255)
        for w, val in zip(col_widths, row):
            pdf.cell(w, 6, val, border=1, align="C", fill=True)
        pdf.ln()
        fill = not fill


def _section_title(pdf: FPDF, title: str) -> None:
    """Draw a styled section heading with a coloured underline."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 10, title)
    pdf.ln(5)
    pdf.set_draw_color(80, 80, 160)
    pdf.set_line_width(0.6)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + _PAGE_W, pdf.get_y())
    pdf.ln(6)


def _safe_fmt(value: Any, decimals: int = 3) -> str:
    """Format a numeric value safely, returning '-' for None/NaN."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "-"


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def generate_report(
    metrics_dict: Dict[str, Any],
    figures_dir: str,
    output_dir: str,
) -> str:
    """Generate a clinical PDF report and return the file path.

    Args:
        metrics_dict: Dictionary containing all pipeline results.
            Expected keys:
                ``n_patients``       – int
                ``n_et``             – int
                ``n_control``        – int
                ``n_segments``       – int
                ``n_features``       – int  (after RFE)
                ``reg_local``        – Dict[model_name, metrics_dict]
                ``reg_global``       – Dict[model_name, metrics_dict]
                ``cls_results``      – Dict[model_name, metrics_dict]
                ``cal_results``      – Dict[model_name, metrics_dict]
                ``rtc_results``      – Dict["fixed_threshold"/"youden_optimal", metrics]
                ``shap_reg``         – Optional[Dict[feature, importance]]
                ``shap_cls``         – Optional[Dict[feature, importance]]
        figures_dir: Absolute path to the folder containing .png figures.
        output_dir:  Absolute path to the folder where the PDF will be saved.

    Returns:
        Absolute path to the generated PDF file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = f"clinical_report_{timestamp}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(_MARGIN)
    pdf.set_right_margin(_MARGIN)

    # ── PAGE 1 — Title ────────────────────────────────────────────────────
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 14, "Fork ET Detection Pipeline", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Clinical Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    run_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    pdf.cell(0, 7, f"Generated: {run_time}", ln=True, align="C")
    pdf.ln(10)

    # ── Summary ───────────────────────────────────────────────────────────
    _section_title(pdf, "1. Summary")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    summary_lines = [
        f"Patients analysed: {metrics_dict.get('n_patients', '?')} "
        f"(ET={metrics_dict.get('n_et', '?')}, "
        f"Control={metrics_dict.get('n_control', '?')})",
        f"Total segments: {metrics_dict.get('n_segments', '?')}",
        f"Features after RFE selection: {metrics_dict.get('n_features', cfg.TOP_K_FEATURES)}",
        f"Cross-validation: Leave-One-Subject-Out (LOSO)"
        if cfg.USE_LOSO else f"Cross-validation: {cfg.CV_FOLDS}-fold",
        f"Feature selection: {cfg.FEATURE_SELECTION_METHOD.upper()} "
        f"(top {cfg.TOP_K_FEATURES})",
    ]
    for line in summary_lines:
        pdf.cell(0, 7, line, ln=True)
    pdf.ln(6)

    # ── Handedness Accuracy ───────────────────────────────────────────────
    handedness_acc = metrics_dict.get("handedness_accuracy")
    if handedness_acc is not None:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(
            0, 7,
            f"Handedness classifier LOSO accuracy: {handedness_acc:.3f} "
            f"(target >= {0.90:.2f})",
            ln=True,
        )
        pdf.ln(4)

    # ── Regression Results ────────────────────────────────────────────────
    _section_title(pdf, "2. Regression Results")

    reg_headers = ["Model", "MAE", "RMSE", "R²", "Pearson r"]
    reg_widths = [42, 28, 28, 28, 28]

    # Local score
    reg_local: Dict = metrics_dict.get("reg_local", {})
    if reg_local:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Local Score (Fork Feeding)", ln=True)
        pdf.ln(2)
        rows = []
        for model, m in reg_local.items():
            rows.append([
                model,
                _safe_fmt(m.get("MAE")),
                _safe_fmt(m.get("RMSE")),
                _safe_fmt(m.get("R2")),
                _safe_fmt(m.get("Pearson_r")),
            ])
        _draw_table(pdf, reg_headers, rows, reg_widths)
        pdf.ln(6)

    # Global score
    reg_global: Dict = metrics_dict.get("reg_global", {})
    if reg_global:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Global Score (Subtotal B Extended)", ln=True)
        pdf.ln(2)
        rows = []
        for model, m in reg_global.items():
            rows.append([
                model,
                _safe_fmt(m.get("MAE")),
                _safe_fmt(m.get("RMSE")),
                _safe_fmt(m.get("R2")),
                _safe_fmt(m.get("Pearson_r")),
            ])
        _draw_table(pdf, reg_headers, rows, reg_widths)
        pdf.ln(6)

    # ── Bucketed Regression (4 buckets) ──────────────────────────────────
    bucketed: Dict = metrics_dict.get("bucketed_regression", {})
    if bucketed:
        _section_title(pdf, "2b. Bucketed Regression (Hand × Movement Type)")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 6,
            "Per-bucket LOSO CV regression. RFE run independently per bucket. "
            "Target = matching CRF cell (rt_scoop / lf_scoop / rt_stab / lf_stab).",
            ln=True,
        )
        pdf.ln(3)

        bucket_order = [
            "Right_scoop", "Left_scoop", "Right_stab", "Left_stab"
        ]
        for bkey in bucket_order:
            bres = bucketed.get(bkey)
            if bres is None:
                continue

            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"Bucket: {bkey}", ln=True)
            pdf.ln(1)

            # Selected features (if logged)
            feat_list = bucketed.get(f"{bkey}_features", [])
            if feat_list:
                pdf.set_font("Helvetica", "I", 8)
                feat_str = "Features: " + ", ".join(feat_list[:15])
                pdf.multi_cell(0, 5, feat_str)
                pdf.ln(1)

            if isinstance(bres, dict) and "status" in bres:
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(0, 6, f"  Status: {bres['status']}", ln=True)
                pdf.ln(3)
                continue

            if isinstance(bres, dict) and bres:
                rows = []
                for model, m in bres.items():
                    if not isinstance(m, dict):
                        continue
                    rows.append([
                        model,
                        _safe_fmt(m.get("MAE")),
                        _safe_fmt(m.get("RMSE")),
                        _safe_fmt(m.get("R2")),
                        _safe_fmt(m.get("Pearson_r")),
                    ])
                if rows:
                    _draw_table(pdf, reg_headers, rows, reg_widths)
            pdf.ln(4)

    # ── Classification Results ────────────────────────────────────────────
    _section_title(pdf, "3. Classification Results")

    cls_headers = ["Model", "Accuracy", "Sensitivity", "Specificity", "AUC", "PPV", "NPV"]
    cls_widths = [36, 24, 26, 26, 22, 22, 22]

    cls_results: Dict = metrics_dict.get("cls_results", {})
    if cls_results:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "ET vs Control - Direct Classification", ln=True)
        pdf.ln(2)
        rows = []
        for model, m in cls_results.items():
            rows.append([
                model,
                _safe_fmt(m.get("Accuracy")),
                _safe_fmt(m.get("Sensitivity")),
                _safe_fmt(m.get("Specificity")),
                _safe_fmt(m.get("AUC")),
                _safe_fmt(m.get("PPV")),
                _safe_fmt(m.get("NPV")),
            ])
        _draw_table(pdf, cls_headers, rows, cls_widths)
        pdf.ln(6)

    cal_results: Dict = metrics_dict.get("cal_results", {})
    if cal_results:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Calibrated Classifiers (Platt Scaling)", ln=True)
        pdf.ln(2)
        rows = []
        for model, m in cal_results.items():
            rows.append([
                model,
                _safe_fmt(m.get("Accuracy")),
                _safe_fmt(m.get("Sensitivity")),
                _safe_fmt(m.get("Specificity")),
                _safe_fmt(m.get("AUC")),
                _safe_fmt(m.get("PPV")),
                _safe_fmt(m.get("NPV")),
            ])
        _draw_table(pdf, cls_headers, rows, cls_widths)
        pdf.ln(6)

    # Regress-then-classify
    rtc_results: Dict = metrics_dict.get("rtc_results", {})
    if rtc_results:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Regress-then-Classify (Score-based)", ln=True)
        pdf.ln(2)
        rtc_headers = ["Threshold", "Accuracy", "Sensitivity", "Specificity", "AUC"]
        rtc_widths = [36, 30, 30, 30, 30]
        rows = []
        for label, key in [("Fixed (0.5)", "fixed_threshold"), ("Youden Optimal", "youden_optimal")]:
            m = rtc_results.get(key, {})
            if m:
                rows.append([
                    label,
                    _safe_fmt(m.get("Accuracy")),
                    _safe_fmt(m.get("Sensitivity")),
                    _safe_fmt(m.get("Specificity")),
                    _safe_fmt(m.get("AUC")),
                ])
        if rows:
            _draw_table(pdf, rtc_headers, rows, rtc_widths)
        pdf.ln(6)

    # ── Patient-Level Results ─────────────────────────────────────────────
    patient_rows = []
    
    def _extract_patient_metrics(res_dict):
        for k, m in res_dict.items():
            if k.endswith("_patient"):
                model_name = k.replace("_patient", "")
                patient_rows.append([
                    model_name,
                    _safe_fmt(m.get("Sensitivity")),
                    _safe_fmt(m.get("Specificity")),
                    _safe_fmt(m.get("AUC")),
                    _safe_fmt(m.get("Threshold")),
                    str(m.get("N_patients", ""))
                ])

    _extract_patient_metrics(metrics_dict.get("cls_results", {}))
    _extract_patient_metrics(metrics_dict.get("cal_results", {}))
    _extract_patient_metrics(metrics_dict.get("rtc_results", {}))

    if patient_rows:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Patient-Level Results (Mean Ensembling, Specificity>=80%)", ln=True)
        pdf.ln(2)
        p_headers = ["Model", "Sensitivity", "Specificity", "AUC", "Threshold", "N"]
        p_widths = [46, 25, 25, 20, 24, 10]
        _draw_table(pdf, p_headers, patient_rows, p_widths)
        pdf.ln(6)

    # ── Top Features (SHAP) ──────────────────────────────────────────────
    _section_title(pdf, "4. Top Features (SHAP Importance)")

    shap_reg: Optional[Dict] = metrics_dict.get("shap_reg")
    shap_cls: Optional[Dict] = metrics_dict.get("shap_cls")

    if shap_reg:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Regression - Top 15 Features", ln=True)
        pdf.ln(2)
        shap_headers = ["Rank", "Feature", "Mean |SHAP|"]
        shap_widths = [16, 100, 40]
        rows = []
        for i, (feat, val) in enumerate(list(shap_reg.items())[:15], 1):
            rows.append([str(i), feat, _safe_fmt(val, 4)])
        _draw_table(pdf, shap_headers, rows, shap_widths)
        pdf.ln(4)

    if shap_cls:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Classification - Top 15 Features", ln=True)
        pdf.ln(2)
        shap_headers = ["Rank", "Feature", "Mean |SHAP|"]
        shap_widths = [16, 100, 40]
        rows = []
        for i, (feat, val) in enumerate(list(shap_cls.items())[:15], 1):
            rows.append([str(i), feat, _safe_fmt(val, 4)])
        _draw_table(pdf, shap_headers, rows, shap_widths)
        pdf.ln(4)

    # ── RFE Feature Stability ──────────────────────────────────────────────
    rfe_stability: Dict = metrics_dict.get("rfe_stability", {})
    if rfe_stability:
        _section_title(pdf, "4b. RFE Feature Stability")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7,
            "Percentage of LOSO folds in which each feature was selected by RFE.",
            ln=True)
        pdf.ln(2)
        stab_headers = ["Rank", "Feature", "Folds Selected (%)"]
        stab_widths = [16, 120, 44]
        rows = []
        for i, (feat, pct) in enumerate(list(rfe_stability.items())[:15], 1):
            rows.append([str(i), feat, f"{pct*100:.0f}%"])
        _draw_table(pdf, stab_headers, rows, stab_widths)
        pdf.ln(6)

    # ── Figures ───────────────────────────────────────────────────────────
    _section_title(pdf, "5. Figures")

    for caption, fname in _FIGURE_FILES:
        fpath = os.path.join(figures_dir, fname)
        if not os.path.isfile(fpath):
            logger.debug("Figure not found, skipping: %s", fpath)
            continue

        # Check if enough space on current page (need ~120mm for image + caption)
        if pdf.get_y() > 160:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, caption, ln=True)
        try:
            pdf.image(fpath, x=_MARGIN, w=_PAGE_W)
        except Exception as exc:
            logger.warning("Could not embed figure %s: %s", fname, exc)
        pdf.ln(6)

    # ── Limitations ──────────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "6. Limitations & Future Work")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)

    limitations_text = (
        "This analysis is based on a modest sample size of "
        f"{metrics_dict.get('n_patients', '~24')} participants. "
        "While Leave-One-Subject-Out cross-validation provides an unbiased "
        "estimate of generalisation performance, the limited number of "
        "Control subjects may affect Specificity estimates. "
        "The feature set is derived from a single IMU sensor placed on "
        "the dominant hand during a fork-feeding task; multi-sensor or "
        "multi-task protocols could improve robustness. "
        "Future work should include: (1) prospective validation on an "
        "independent cohort, (2) longitudinal tracking of tremor progression, "
        "(3) integration of additional sensor modalities (e.g. EMG), and "
        "(4) exploration of deep-learning architectures for end-to-end "
        "feature learning from raw signals."
    )
    pdf.multi_cell(0, 5.5, limitations_text)
    pdf.ln(10)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "Report generated automatically by Fork ET Detection Pipeline", align="C")

    # ── Save ─────────────────────────────────────────────────────────────
    pdf.output(pdf_path)
    logger.info("Clinical report saved to %s", pdf_path)
    return pdf_path
