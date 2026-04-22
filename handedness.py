"""
Per-cycle hand classification using chirality-sensitive gyroscope/accelerometer features.

Labels come from filenames (Fork1_* → Right, Fork2_* → Left).
HandednessClassifier trains on labeled cycles and predicts for ambiguous (Fork_*) files.
Aggregation: majority vote across cycles in a behavioral test (unchanged architecture).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger("fork_pipeline.handedness")


class HandednessClassifier:
    """Per-cycle logistic regression hand classifier (6 chirality-sensitive features).

    Feature set:
        - p95(gyro_y) + p05(gyro_y)  : signed asymmetry of dominant rotation axis
        - skew(gyro_y), skew(gyro_z) : pronation/supination sign
        - corr(acc_x, gyro_y)        : cross-axis chirality (mirrored L vs R)
        - corr(acc_z, gyro_x)        : cross-axis chirality
        - weighted_mean(gyro_y)       : mean gyro_y weighted by |a| (emphasises scoop moment)
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=cfg.RANDOM_STATE
        )
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.fitted: bool = False

    # ── Feature extraction ─────────────────────────────────────────────────

    def _extract_features(self, cycle_df: pd.DataFrame) -> dict:
        gy = cycle_df["gyro_y"].values.astype(np.float64)
        gz = cycle_df["gyro_z"].values.astype(np.float64)
        gx = cycle_df["gyro_x"].values.astype(np.float64)
        ax = cycle_df["acc_x"].values.astype(np.float64)
        ay = cycle_df["acc_y"].values.astype(np.float64)
        az = cycle_df["acc_z"].values.astype(np.float64)
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2) + 1e-10

        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        return {
            "gyro_y_asym": float(np.percentile(gy, 95) + np.percentile(gy, 5)),
            "skew_gyro_y": float(skew(gy)) if len(gy) > 2 else 0.0,
            "skew_gyro_z": float(skew(gz)) if len(gz) > 2 else 0.0,
            "corr_ax_gy": _safe_corr(ax, gy),
            "corr_az_gx": _safe_corr(az, gx),
            "weighted_mean_gy": float(np.average(gy, weights=acc_mag)),
        }

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        cycles: List[pd.DataFrame],
        labels: List[str],
    ) -> None:
        """Train on labeled cycles.

        Args:
            cycles: List of per-cycle IMU DataFrames.
            labels: Per-cycle hand labels ("Right" or "Left").
        """
        X = pd.DataFrame([self._extract_features(c) for c in cycles]).fillna(0.0)
        self.feature_names = list(X.columns)
        y = np.array([1 if lbl == "Right" else 0 for lbl in labels])
        X_s = self.scaler.fit_transform(X.values)
        self.model.fit(X_s, y)
        self.fitted = True
        logger.info(
            "HandednessClassifier trained on %d cycles (%d Right, %d Left)",
            len(cycles), int(sum(y)), int(len(y) - sum(y)),
        )

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, cycle_df: pd.DataFrame) -> str:
        """Predict hand for a single cycle.

        Falls back to heuristic if classifier is not fitted.
        """
        if not self.fitted:
            return _heuristic_hand(cycle_df)
        feats = self._extract_features(cycle_df)
        X = pd.DataFrame([feats])[self.feature_names].fillna(0.0)
        X_s = self.scaler.transform(X.values)
        return "Right" if self.model.predict(X_s)[0] == 1 else "Left"

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate_loso(
        self,
        cycles: List[pd.DataFrame],
        labels: List[str],
        patient_ids: List[str],
    ) -> Tuple[float, np.ndarray]:
        """Leave-one-subject-out CV evaluation.

        Returns:
            (accuracy, per-cycle predictions array)
        """
        X = pd.DataFrame([self._extract_features(c) for c in cycles]).fillna(0.0)
        y = np.array([1 if lbl == "Right" else 0 for lbl in labels])
        groups = np.array(patient_ids)

        loso = LeaveOneGroupOut()
        y_pred = np.zeros_like(y)

        for train_idx, test_idx in loso.split(X, y, groups):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X.iloc[train_idx].values)
            X_te = sc.transform(X.iloc[test_idx].values)
            m = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=cfg.RANDOM_STATE
            )
            m.fit(X_tr, y[train_idx])
            y_pred[test_idx] = m.predict(X_te)

        acc = float(np.mean(y_pred == y))

        # Confusion matrix for logging
        cm = confusion_matrix(y, y_pred)
        logger.info(
            "Handedness LOSO accuracy: %.3f (%d cycles). Confusion:\n%s",
            acc, len(cycles), str(cm),
        )
        if acc < cfg.HANDEDNESS_MIN_ACCURACY:
            logger.warning(
                "Handedness accuracy %.3f is below target %.3f — "
                "check feature quality or retrain with more data",
                acc, cfg.HANDEDNESS_MIN_ACCURACY,
            )
        return acc, y_pred


# ── Fallback heuristic ─────────────────────────────────────────────────────


def _heuristic_hand(cycle_df: pd.DataFrame) -> str:
    """Fallback: gyro_y asymmetry heuristic (original signal-based method)."""
    if "gyro_y" not in cycle_df.columns:
        return "Right"
    gy = cycle_df["gyro_y"].values.astype(np.float64)
    p95 = np.percentile(gy, 95)
    p05 = np.percentile(gy, 5)
    return "Right" if abs(p95) >= abs(p05) else "Left"
