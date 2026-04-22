"""
Visualization helpers — all plots save to ``config.OUTPUT_DIR``.

**Must be imported AFTER ``matplotlib.use('Agg')``** (enforced below).
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — MUST come before pyplot

import logging
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger("fork_pipeline.visualization")


def _savefig(fig: plt.Figure, filename: str) -> None:
    """Save figure to ``config.OUTPUT_DIR`` and close it."""
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ── Public API ─────────────────────────────────────────────────────────────


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Scatter plot of true vs predicted scores.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        title: Plot title.
        filename: Output filename (e.g. ``"scatter_local.png"``).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
    lims = [
        min(np.min(y_true), np.min(y_pred)) - 0.5,
        max(np.max(y_true), np.max(y_pred)) + 0.5,
    ]
    ax.plot(lims, lims, "--", color="grey", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(title)
    _savefig(fig, filename)


def plot_boxplot(
    features_df: pd.DataFrame,
    group_col: str,
    filename: str,
    max_features: int = 8,
) -> None:
    """Side-by-side box plots for top features coloured by group.

    Args:
        features_df: DataFrame with features + a group column.
        group_col: Column name used for grouping (e.g. ``"group"``).
        filename: Output filename.
        max_features: Maximum number of feature subplots.
    """
    meta_cols = {"patient_id", "group", "hand", "local_score",
                 "global_score", "is_et"}
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    # Pick features with highest variance
    variances = features_df[feat_cols].var().sort_values(ascending=False)
    top = variances.head(max_features).index.tolist()

    n = len(top)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    groups = features_df[group_col].unique()
    for ax, col in zip(axes, top):
        data = [features_df.loc[features_df[group_col] == g, col].dropna()
                for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        # FIXED: plot_boxplot() — динамическая палитра для групп (IndexError при >2)
        colours = plt.cm.tab10.colors
        for patch, colour in zip(bp["boxes"], colours[:len(groups)]):
            patch.set_facecolor(colour)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature distributions by group", fontsize=12)
    fig.tight_layout()
    _savefig(fig, filename)


def plot_pca(
    features_df: pd.DataFrame,
    label_col: str,
    filename: str,
) -> None:
    """2-D PCA projection coloured by label.

    Args:
        features_df: DataFrame with features + label column.
        label_col: Column to colour by (e.g. ``"is_et"``).
        filename: Output filename.
    """
    meta_cols = {"patient_id", "group", "hand", "local_score", "global_score", "is_et"}
    feat_cols = [c for c in features_df.columns if c not in meta_cols]

    from sklearn.impute import SimpleImputer
    
    # Keep only numeric feature columns; movement_type and similar labels may be strings.
    X_df = features_df[feat_cols].select_dtypes(include=[np.number])
    if X_df.shape[1] == 0:
        logger.warning("PCA plot skipped: no numeric feature columns available.")
        return

    X = X_df.values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X_scaled)

    labels = features_df[label_col].values
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        comp[:, 0], comp[:, 1], c=labels, cmap="coolwarm",
        alpha=0.8, edgecolors="k", linewidths=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — ET vs Control")
    fig.colorbar(scatter, ax=ax, label=label_col)
    _savefig(fig, filename)


def plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    filename: str,
) -> None:
    """ROC curve with AUC annotation.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities for the positive class.
        filename: Output filename.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — ET vs Control")
    ax.legend(loc="lower right")
    _savefig(fig, filename)


def plot_signal_with_segments(
    magnitude: np.ndarray,
    segments: List[Tuple[int, int]],
    patient_id: str,
    filename: str,
    fs: float = cfg.FS,
) -> None:
    """Plot accelerometer magnitude highlighting detected activity segments.

    Args:
        magnitude: 1-D array of magnitude values.
        segments: List of ``(start_idx, end_idx)`` tuples.
        patient_id: Used in the title.
        filename: Output filename.
        fs: Sampling frequency.
    """
    t = np.arange(len(magnitude)) / fs
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, magnitude, linewidth=0.5, color="steelblue")
    for s, e in segments:
        ax.axvspan(s / fs, e / fs, alpha=0.25, color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (g)")
    ax.set_title(f"Activity detection — {patient_id}")
    _savefig(fig, filename)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str,
    labels: Optional[List[str]] = None,
) -> None:
    """Confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        filename: Output filename.
        labels: Class names for display.
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    if labels is None:
        labels = ["Control", "ET"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    _savefig(fig, filename)


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Bland-Altman plot (agreement between prediction and ground truth).

    Args:
        y_true: True values.
        y_pred: Predicted values.
        title: Plot title.
        filename: Output filename.
    """
    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_true - y_pred
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(mean_diff, color="blue", linestyle="-", linewidth=1,
               label=f"Mean diff = {mean_diff:.3f}")
    ax.axhline(mean_diff + 1.96 * std_diff, color="red", linestyle="--",
               linewidth=1, label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:.3f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color="red", linestyle="--",
               linewidth=1, label=f"−1.96 SD = {mean_diff - 1.96 * std_diff:.3f}")
    ax.set_xlabel("Mean of True & Predicted")
    ax.set_ylabel("True − Predicted")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, filename)


def plot_cluster_pca(
    features_df: pd.DataFrame,
    cluster_col: str = "movement_type",
    filename: str = "pca_movement_clusters.png",
) -> None:
    """2-D PCA scatter coloured by movement-type cluster label.

    Args:
        features_df: DataFrame with feature columns + a cluster label column.
        cluster_col: Column containing cluster labels (e.g. "movement_type").
        filename: Output filename.
    """
    meta_cols = {"patient_id", "group", "hand", "local_score",
                 "global_score", "is_et", "movement_type",
                 "rt_scoop", "lf_scoop", "rt_stab", "lf_stab"}
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    if cluster_col not in features_df.columns or not feat_cols:
        return

    from sklearn.impute import SimpleImputer

    X = features_df[feat_cols].values
    imputer = SimpleImputer(strategy="mean")
    X_s = StandardScaler().fit_transform(imputer.fit_transform(X))
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X_s)

    labels = features_df[cluster_col].values
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            comp[mask, 0], comp[mask, 1],
            label=str(lbl), alpha=0.7, s=30,
            color=colors[i % len(colors)],
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — movement-type clusters")
    ax.legend(title=cluster_col, fontsize=9)
    fig.tight_layout()
    _savefig(fig, filename)


def plot_and_save_patient_signal(
    df: pd.DataFrame, 
    patient_run_id: str, 
    group: str, 
    local_score: float, 
    out_dir: str
) -> None:
    """Plot accelerometer and gyroscope signals for a single patient segment.
    
    Args:
        df: DataFrame with acc_x, acc_y, acc_z and gyro_x, gyro_y, gyro_z.
        patient_run_id: Patient ID including run number.
        group: Group ('ET' or 'Control').
        local_score: Clinical score for the task.
        out_dir: Directory to save the figure in.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Time axis in seconds (based on sampling frequency)
    t = np.arange(len(df)) / cfg.FS
    
    # 1. Accelerometer
    ax_acc = axes[0]
    ax_acc.plot(t, df["acc_x"], color="tab:blue", label="X", linewidth=1.5)
    ax_acc.plot(t, df["acc_y"], color="tab:orange", label="Y", linewidth=1.5)
    ax_acc.plot(t, df["acc_z"], color="tab:green", label="Z", linewidth=1.5)
    ax_acc.set_title("Accelerometer", fontsize=14)
    ax_acc.set_ylabel("Acceleration [m/s²]", fontsize=12)
    ax_acc.legend(loc="upper right", fontsize=10)
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Gyroscope
    ax_gyr = axes[1]
    ax_gyr.plot(t, df["gyro_x"], color="tab:blue", label="X", linewidth=1.5)
    ax_gyr.plot(t, df["gyro_y"], color="tab:orange", label="Y", linewidth=1.5)
    ax_gyr.plot(t, df["gyro_z"], color="tab:green", label="Z", linewidth=1.5)
    ax_gyr.set_title("Gyroscope", fontsize=14)
    ax_gyr.set_ylabel("Angular Velocity [rad/s]", fontsize=12)
    ax_gyr.set_xlabel("Time [sec]", fontsize=12)
    ax_gyr.legend(loc="upper right", fontsize=10)
    ax_gyr.grid(True, linestyle="--", alpha=0.6)
    
    if not isinstance(local_score, (int, float)) or pd.isna(local_score):
        score_str = "NaN"
    else:
        score_str = f"{local_score:.1f}"
        
    fig.suptitle(
        f"Patient: {patient_run_id} | Group: {group} | Local Fork Score: {score_str}", 
        fontsize=16, fontweight='bold'
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    filename = f"{group}_{patient_run_id}_score_{score_str}.png"
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved patient signal plot to %s", filepath)

