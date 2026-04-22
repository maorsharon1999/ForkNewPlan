"""
Per-cycle movement-type clustering using GMM (k=4).

Clusters are assigned post-hoc from config.GMM_CLUSTER_LABEL_MAP.
Expected mapping (to be confirmed after inspecting the PDF):
    Two clusters → "scoop" / "stab"  (fed to bucketed regression)
    One-two clusters → "fragment" / "other"  (excluded from regression)

Usage:
    clf = MovementClassifier()
    clf.fit(all_cycle_dfs)
    clf.generate_inspection_pdf(all_cycle_dfs, output_path="cluster_inspection.pdf")
    # → inspect PDF, set GMM_CLUSTER_LABEL_MAP in config.py, re-run
    labels = [clf.predict_label(c) for c in all_cycle_dfs]
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger("fork_pipeline.movement_classifier")


class MovementClassifier:
    """GMM-based movement-type classifier (k=4 clusters, post-hoc labeling).

    Feature families (15 total):
        - Jerk  : peak, ratio, variance, N peaks
        - Tilt  : acc_y/z range, tilt path length
        - Gyro  : std(gy)/std(gx), std(gy)/std(gz)
        - Time  : cycle duration
        - Spectrum : dominant freq, HF ratio, spectral centroid
        - Shape : kurtosis of |a|, kurtosis of |ω|
    """

    def __init__(self, n_components: int = cfg.GMM_N_COMPONENTS) -> None:
        self.n_components = n_components
        self.gmm: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.cluster_label_map: Dict[int, str] = dict(cfg.GMM_CLUSTER_LABEL_MAP)

    # ── Feature extraction ─────────────────────────────────────────────────

    def _extract_features(
        self, cycle_df: pd.DataFrame, fs: float = cfg.FS
    ) -> dict:
        ax = cycle_df["acc_x"].values.astype(np.float64)
        ay = cycle_df["acc_y"].values.astype(np.float64)
        az = cycle_df["acc_z"].values.astype(np.float64)
        gx = cycle_df["gyro_x"].values.astype(np.float64)
        gy = cycle_df["gyro_y"].values.astype(np.float64)
        gz = cycle_df["gyro_z"].values.astype(np.float64)

        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # Jerk
        jerk = np.abs(np.gradient(acc_mag, 1.0 / fs))
        mean_jerk = float(jerk.mean()) if jerk.mean() > 1e-10 else 1e-10
        peaks_idx, _ = find_peaks(jerk, height=np.percentile(jerk, 75))

        # Gyro dominance
        gy_std = max(float(gy.std()), 1e-10)
        gx_std = max(float(gx.std()), 1e-10)
        gz_std = max(float(gz.std()), 1e-10)

        # Spectrum on acc magnitude
        n = len(acc_mag)
        fft_vals = np.abs(np.fft.rfft(acc_mag))
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        total_power = float(np.sum(fft_vals**2)) + 1e-10
        dom_freq = float(freqs[np.argmax(fft_vals[1:]) + 1]) if n > 2 else 0.0
        hf_mask = freqs > 10.0
        hf_ratio = (
            float(np.sum(fft_vals[hf_mask] ** 2) / total_power)
            if np.any(hf_mask)
            else 0.0
        )
        centroid = float(np.sum(freqs * fft_vals**2) / total_power)

        return {
            "peak_jerk": float(jerk.max()),
            "jerk_ratio": float(jerk.max() / mean_jerk),
            "jerk_var": float(jerk.var()),
            "n_jerk_peaks": len(peaks_idx),
            "acc_y_range": float(ay.max() - ay.min()),
            "acc_z_range": float(az.max() - az.min()),
            "tilt_path": float(
                np.sum(np.abs(np.gradient(ay)) + np.abs(np.gradient(az)))
            ),
            "gyro_dom_yx": gy_std / gx_std,
            "gyro_dom_yz": gy_std / gz_std,
            "duration": float(n / fs),
            "dominant_freq": dom_freq,
            "hf_ratio": hf_ratio,
            "spectral_centroid": centroid,
            "kurtosis_acc": float(kurtosis(acc_mag)),
            "kurtosis_gyro": float(kurtosis(gyro_mag)),
        }

    def _build_feature_matrix(
        self, cycles: List[pd.DataFrame], fs: float = cfg.FS
    ) -> pd.DataFrame:
        rows = [self._extract_features(c, fs) for c in cycles]
        X = pd.DataFrame(rows).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return X

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self, cycles: List[pd.DataFrame], fs: float = cfg.FS
    ) -> None:
        """Fit GMM on all cycles.

        Args:
            cycles: All per-cycle IMU DataFrames across the full cohort.
            fs: Sampling frequency.
        """
        X = self._build_feature_matrix(cycles, fs)
        self.feature_names = list(X.columns)
        X_s = self.scaler.fit_transform(X.values)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            n_init=5,
            random_state=cfg.RANDOM_STATE,
        )
        self.gmm.fit(X_s)
        preds = self.gmm.predict(X_s)
        uniq, counts = np.unique(preds, return_counts=True)
        logger.info(
            "MovementClassifier fitted on %d cycles. Cluster counts: %s",
            len(cycles),
            dict(zip(uniq.tolist(), counts.tolist())),
        )
        if not self.cluster_label_map:
            logger.warning(
                "GMM_CLUSTER_LABEL_MAP is empty — cycle labels will be "
                "'cluster_N'. Inspect the PDF and populate config.GMM_CLUSTER_LABEL_MAP."
            )

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_cluster(
        self, cycle_df: pd.DataFrame, fs: float = cfg.FS
    ) -> int:
        """Return raw cluster index for one cycle."""
        if self.gmm is None or self.feature_names is None:
            return -1
        feats = self._extract_features(cycle_df, fs)
        X = pd.DataFrame([feats])[self.feature_names].fillna(0.0)
        X_s = self.scaler.transform(X.values)
        return int(self.gmm.predict(X_s)[0])

    def predict_label(
        self, cycle_df: pd.DataFrame, fs: float = cfg.FS
    ) -> str:
        """Return movement-type label for one cycle."""
        cluster = self.predict_cluster(cycle_df, fs)
        return self.cluster_label_map.get(cluster, f"cluster_{cluster}")

    def predict_all_clusters(
        self, cycles: List[pd.DataFrame], fs: float = cfg.FS
    ) -> List[int]:
        """Batch prediction of cluster indices."""
        if self.gmm is None or self.feature_names is None:
            return [-1] * len(cycles)
        X = self._build_feature_matrix(cycles, fs)
        X = X[self.feature_names]
        X_s = self.scaler.transform(X.values)
        return self.gmm.predict(X_s).tolist()

    # ── Inspection PDF ────────────────────────────────────────────────────

    def generate_inspection_pdf(
        self,
        cycles: List[pd.DataFrame],
        cluster_assignments: Optional[List[int]] = None,
        fs: float = cfg.FS,
        output_path: str = "cluster_inspection.pdf",
    ) -> None:
        """Generate a cluster inspection PDF.

        Pages:
            1. Feature distributions per cluster (histograms)
            2. Mean acc-magnitude profile per cluster
            3. PCA scatter coloured by cluster

        After inspection, assign cluster→label in config.GMM_CLUSTER_LABEL_MAP.
        """
        if self.gmm is None or self.feature_names is None:
            logger.warning("GMM not fitted — cannot generate inspection PDF")
            return

        if cluster_assignments is None:
            cluster_assignments = self.predict_all_clusters(cycles, fs)

        X = self._build_feature_matrix(cycles, fs)[self.feature_names]
        clusters = np.array(cluster_assignments)
        colors = plt.cm.tab10.colors

        from matplotlib.backends.backend_pdf import PdfPages

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with PdfPages(output_path) as pdf:
            # ── Page 1: Feature distributions ─────────────────────────────
            n_feats = len(self.feature_names)
            ncols = 5
            nrows = int(np.ceil(n_feats / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows))
            axes = axes.ravel()
            for idx, col in enumerate(self.feature_names):
                ax = axes[idx]
                for ci in range(self.n_components):
                    vals = X.loc[clusters == ci, col].values
                    if len(vals) > 0:
                        lbl = self.cluster_label_map.get(ci, f"C{ci}")
                        ax.hist(vals, bins=15, alpha=0.5, label=str(lbl), density=True,
                                color=colors[ci % len(colors)])
                ax.set_title(col, fontsize=7)
                ax.legend(fontsize=5)
                ax.tick_params(labelsize=6)
            for ax in axes[n_feats:]:
                ax.set_visible(False)
            fig.suptitle("Feature distributions per cluster", fontsize=13)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── Page 2: Mean acc-magnitude profile per cluster ─────────────
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.ravel()
            for ci in range(min(self.n_components, 4)):
                ax = axes[ci]
                mask = clusters == ci
                cluster_cycles = [cycles[i] for i in range(len(cycles)) if mask[i]]
                if not cluster_cycles:
                    ax.set_title(f"Cluster {ci} — empty")
                    continue
                target_len = int(np.median([len(c) for c in cluster_cycles]))
                target_len = max(target_len, 10)
                profiles = []
                for cyc in cluster_cycles[:30]:
                    mag = np.sqrt(
                        cyc["acc_x"] ** 2 + cyc["acc_y"] ** 2 + cyc["acc_z"] ** 2
                    ).values.astype(np.float64)
                    if len(mag) < 5:
                        continue
                    x_old = np.linspace(0, 1, len(mag))
                    x_new = np.linspace(0, 1, target_len)
                    resampled = np.interp(x_new, x_old, mag)
                    profiles.append(resampled)
                if profiles:
                    mean_p = np.mean(profiles, axis=0)
                    std_p = np.std(profiles, axis=0)
                    t = np.linspace(0, target_len / fs, target_len)
                    ax.plot(t, mean_p, color=colors[ci % len(colors)], linewidth=2)
                    ax.fill_between(t, mean_p - std_p, mean_p + std_p,
                                    alpha=0.2, color=colors[ci % len(colors)])
                lbl = self.cluster_label_map.get(ci, f"C{ci}")
                ax.set_title(
                    f"Cluster {ci} ({lbl}) — n={int(sum(mask))}", fontsize=10
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("|acc| (g)")
            fig.suptitle("Mean acc-magnitude profile per cluster", fontsize=13)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── Page 3: PCA scatter ────────────────────────────────────────
            X_s = self.scaler.transform(X.values)
            pca = PCA(n_components=2)
            comp = pca.fit_transform(X_s)
            fig, ax = plt.subplots(figsize=(9, 7))
            for ci in range(self.n_components):
                mask = clusters == ci
                lbl = self.cluster_label_map.get(ci, f"C{ci}")
                ax.scatter(
                    comp[mask, 0], comp[mask, 1],
                    label=f"C{ci} ({lbl})", alpha=0.7, s=30,
                    color=colors[ci % len(colors)],
                )
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.set_title("PCA — movement clusters")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        logger.info("Cluster inspection PDF saved to %s", output_path)
