"""
ML pipeline: LOSO CV, hyperparameter tuning, stacking ensemble,
per-segment aggregation, augmentation, additional models & metrics,
Youden's J threshold optimization, regress-then-classify,
SHAP interpretability, calibrated classifiers.
"""

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Suppress mathematical warnings from numpy/sklearn
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

import config as cfg

logger = logging.getLogger("fork_pipeline.ml_pipeline")

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    logger.warning("xgboost not installed — XGBoost models will be skipped")
    _HAS_XGB = False


# ── helpers ────────────────────────────────────────────────────────────────

META_COLS = [
    "patient_id", "group", "hand", "local_score",
    "global_score", "is_et",
]


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


# ── Data augmentation (from Cup) ──────────────────────────────────────────

def augment_and_balance(
    X: pd.DataFrame,
    y: pd.Series,
    noise_scale: float = cfg.NOISE_SCALE,
    mixup_alpha: float = cfg.MIXUP_ALPHA,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Augment minority classes with noise injection + mixup (from Cup).

    Ensures each discrete target bin has the same number of samples.
    """
    # Fill NaNs and ensure numeric
    y = y.fillna(y.mean()).astype(float)
    X = X.fillna(0.0).astype(float)

    # Bin targets for balancing
    y_binned = pd.cut(y, bins=5, labels=False)
    counts = Counter(y_binned)
    
    if len(counts) == 0:
        return X, y
        
    max_count = max(counts.values())

    feature_stds = X.std(axis=0).values.copy()
    feature_stds[np.isnan(feature_stds)] = 0.0
    
    X_new, y_new = [X.copy()], [y.copy()]

    for cls, count in counts.items():
        if pd.isna(cls):
            continue
        mask = y_binned == cls
        X_cls = X[mask].values
        y_cls = y[mask].values

        n_to_add = max_count - count
        if n_to_add <= 0:
            continue

        aug_X, aug_y = [], []
        for _ in range(n_to_add):
            i = np.random.randint(0, len(X_cls))
            if np.random.rand() < 0.5:
                # Noise injection
                noise = np.random.normal(0, noise_scale * feature_stds, size=X_cls[i].shape)
                aug_X.append(X_cls[i] + noise)
                aug_y.append(y_cls[i])
            else:
                # Mixup
                j = np.random.randint(0, len(X_cls))
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                aug_X.append(lam * X_cls[i] + (1 - lam) * X_cls[j])
                aug_y.append(lam * y_cls[i] + (1 - lam) * y_cls[j])

        X_new.append(pd.DataFrame(aug_X, columns=X.columns))
        y_new.append(pd.Series(aug_y))

    return pd.concat(X_new, ignore_index=True), pd.concat(y_new, ignore_index=True)


# ── Feature selection ──────────────────────────────────────────────────────

def select_features_rf(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = cfg.RF_IMPORTANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, List[str]]:
    """Select features via RandomForest importance (from Brush)."""
    rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    mask = importances > threshold
    selected = list(X.columns[mask])
    if len(selected) < 3:
        # Fallback: top 15
        top_idx = np.argsort(importances)[-cfg.TOP_K_FEATURES:]
        selected = list(X.columns[top_idx])
    logger.info("RF importance: selected %d/%d features", len(selected), X.shape[1])
    return X[selected], selected


def select_features_kbest(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = cfg.TOP_K_FEATURES,
    task: str = "regression",
) -> Tuple[pd.DataFrame, List[str]]:
    """Select top-k features using univariate scoring."""
    k = min(k, X.shape[1])
    score_func = f_regression if task == "regression" else f_classif
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
    mask = selector.get_support()
    selected = list(X.columns[mask])
    logger.info("SelectKBest: selected %d features", len(selected))
    return X[selected], selected


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = cfg.TOP_K_FEATURES,
    task: str = "regression",
) -> Tuple[pd.DataFrame, List[str]]:
    """Select features using configured method.
    Options: 'rf_importance', 'select_k_best', 'rfe'.
    RFE limits the feature set to k to combat the curse of dimensionality.
    """
    if cfg.FEATURE_SELECTION_METHOD == "rf_importance":
        return select_features_rf(X, y)
    elif cfg.FEATURE_SELECTION_METHOD == "rfe":
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        logger.info("Starting RFE (target %d features)...", k)
        if task == "regression":
            estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=cfg.RANDOM_STATE, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=cfg.RANDOM_STATE, n_jobs=-1)
            
        # step=0.1 means dropping 10% of features at each step (faster than step=0.1)
        selector = RFE(estimator, n_features_to_select=k, step=0.1)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = list(X.columns[mask])
        logger.info("RFE selected exactly %d features: %s", len(selected), selected)
        return X[selected], selected
        
    return select_features_kbest(X, y, k, task)


# ── Hyperparameter grids ──────────────────────────────────────────────────

_RF_REG_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3],
}
_RF_CLF_GRID = _RF_REG_GRID.copy()

_XGB_REG_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.05, 0.1],
}
_XGB_CLF_GRID = _XGB_REG_GRID.copy()


def _tune_model(model, param_grid, X, y, cv, scoring, groups=None):
    """Tune a model via GridSearchCV and return the best estimator."""
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
    gs.fit(X, y, groups=groups)
    logger.info("  Best params for %s: %s (score=%.3f)",
                type(model).__name__, gs.best_params_, gs.best_score_)
    return gs.best_estimator_


# ── LOSO / GroupKFold builder ─────────────────────────────────────────────

def _build_cv(groups: pd.Series, task: str = "regression"):
    n_groups = groups.nunique()
    if cfg.USE_LOSO and n_groups >= 3:
        if n_groups <= 15:
            return LeaveOneGroupOut()
        else:
            return GroupKFold(n_splits=min(cfg.CV_FOLDS, n_groups))
    return KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)


# ── Per-segment aggregation ──────────────────────────────────────────────

def _aggregate_segment_predictions(df, y_pred, target_col):
    key = df["patient_id"].astype(str) + "_" + df["hand"].astype(str)
    tmp = pd.DataFrame({"key": key.values, "y_true": df[target_col].values, "y_pred": y_pred})
    agg = tmp.groupby("key").agg(y_true=("y_true", "first"), y_pred=("y_pred", "median"))
    return agg["y_true"].values, agg["y_pred"].values


# ── Additional metrics ───────────────────────────────────────────────────

def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute comprehensive regression metrics."""
    metrics: Dict[str, float] = {}
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["R2"] = r2_score(y_true, y_pred)

    # Pearson correlation
    if len(y_true) >= 3:
        r, p = sp_stats.pearsonr(y_true, y_pred)
        metrics["Pearson_r"] = r
        metrics["Pearson_p"] = p
        rho, rho_p = sp_stats.spearmanr(y_true, y_pred)
        metrics["Spearman_rho"] = rho
        metrics["Spearman_p"] = rho_p
    else:
        metrics["Pearson_r"] = 0.0
        metrics["Pearson_p"] = 1.0
        metrics["Spearman_rho"] = 0.0
        metrics["Spearman_p"] = 1.0

    return metrics


def compute_classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    metrics: Dict[str, float] = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["F1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["PPV"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["NPV"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["AUC"] = roc_auc_score(y_true, y_score)

    return metrics


# ── Regression-to-ROC (from Brush) ───────────────────────────────────────

def regression_to_roc(y_true, y_pred, threshold="median") -> Dict[str, Any]:
    """Binarise regression targets and compute ROC curve (from Brush)."""
    if threshold == "median":
        cutoff = np.median(y_true)
    elif threshold == "mean":
        cutoff = np.mean(y_true)
    else:
        cutoff = float(threshold)

    y_binary = (y_true > cutoff).astype(int)
    if len(np.unique(y_binary)) < 2:
        return {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": 0.5}

    mm = MinMaxScaler()
    scores = mm.fit_transform(y_pred.reshape(-1, 1)).ravel()
    fpr, tpr, _ = roc_curve(y_binary, scores)
    roc_auc = roc_auc_score(y_binary, scores)
    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc}


# ── Public runners ────────────────────────────────────────────────────────

def run_regression(features_df, target_col="local_score") -> Dict[str, Dict[str, float]]:
    """Run regression with LOSO CV, stacking, augmentation."""
    # FIXED: "Regression only on ET" guarantee
    df = features_df[features_df["group"] == "ET"].copy()
    assert len(df) > 0, "run_regression() requires ET-only data"
    
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import RFE

    feat_cols = _feature_cols(df)
    X = df[feat_cols].copy()
    y = df[target_col].copy()
    groups = df["patient_id"].copy()

    valid = y.notna()
    X, y, groups = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True), groups[valid].reset_index(drop=True)
    valid_df = df[valid].reset_index(drop=True)

    if len(X) < 5:
        logger.warning("Only %d samples for regression — skipping", len(X))
        return {}

    cv = _build_cv(groups, task="regression")

    # Log which features RFE picks (informational only — no leakage, not used for CV)
    _rfe_info = RFE(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=cfg.RANDOM_STATE, n_jobs=-1),
        n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
    )
    _rfe_info.fit(StandardScaler().fit_transform(X), y)
    _selected_names = [X.columns[i] for i in range(len(X.columns)) if _rfe_info.support_[i]]
    logger.info("RFE selected %d features (informational): %s", len(_selected_names), _selected_names)

    # Build Pipeline-wrapped models: scaler → RFE → estimator
    def _make_pipeline(estimator):
        return Pipeline([
            ("var_thresh", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("rfe", RFE(
                estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                                 random_state=cfg.RANDOM_STATE, n_jobs=-1),
                n_features_to_select=cfg.TOP_K_FEATURES, step=1,
            )),
            ("model", estimator),
        ])

    base_models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=cfg.RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=cfg.RANDOM_STATE),
    }
    if _HAS_XGB:
        base_models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=cfg.RANDOM_STATE, verbosity=0)

    # Evaluate — manual CV fold to apply augmentation strictly on train
    results: Dict[str, Dict[str, float]] = {}
    for name, estimator in base_models.items():
        pipe = _make_pipeline(estimator)
        y_pred = np.zeros(len(y))
        
        cv_splitter = cv.split(X, y, groups) if hasattr(cv, "split") else KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE).split(X, y)
        for train_idx, test_idx in cv_splitter:
            X_train_raw = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            X_test_raw = X.iloc[test_idx].copy()
            
            # FIXED: Data leakage — аугментация до CV
            if cfg.USE_AUGMENTATION:
                X_train_raw, y_train = augment_and_balance(X_train_raw, y_train)
                
            from sklearn.base import clone
            fold_pipe = clone(pipe)
            fold_pipe.fit(X_train_raw, y_train)
            y_pred[test_idx] = fold_pipe.predict(X_test_raw)

        if cfg.PER_SEGMENT and "patient_id" in valid_df.columns:
            # FIXED: PER_SEGMENT агрегация — убрать fallback без агрегации, Only original rows
            y_true_agg, y_pred_agg = _aggregate_segment_predictions(valid_df, y_pred, target_col)
        else:
            y_true_agg, y_pred_agg = y.values, y_pred

        metrics = compute_regression_metrics(y_true_agg, y_pred_agg)
        results[name] = metrics
        logger.info("  %-20s  MAE=%.3f  R²=%.3f  Pearson_r=%.3f",
                     name, metrics["MAE"], metrics["R2"], metrics.get("Pearson_r", 0))

    # Stacking — also manual fold for train augmentation
    if cfg.USE_STACKING and len(base_models) >= 2:
        logger.info("  Building stacking ensemble…")
        estimators_list = [(n, _make_pipeline(m)) for n, m in base_models.items()]
        stack = StackingRegressor(estimators=estimators_list, final_estimator=RidgeCV(),
                                  cv=min(cfg.CV_FOLDS, groups.nunique()))
        
        y_pred_stack = np.zeros(len(y))
        cv_splitter = cv.split(X, y, groups) if hasattr(cv, "split") else KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE).split(X, y)
        for train_idx, test_idx in cv_splitter:
            X_train_raw = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            X_test_raw = X.iloc[test_idx].copy()
            
            # FIXED: Data leakage — аугментация до CV
            if cfg.USE_AUGMENTATION:
                X_train_raw, y_train = augment_and_balance(X_train_raw, y_train)
                
            from sklearn.base import clone
            fold_stack = clone(stack)
            fold_stack.fit(X_train_raw, y_train)
            y_pred_stack[test_idx] = fold_stack.predict(X_test_raw)

        if cfg.PER_SEGMENT and "patient_id" in valid_df.columns:
            # FIXED: PER_SEGMENT агрегация — убрать fallback без агрегации
            y_true_agg, y_pred_agg = _aggregate_segment_predictions(valid_df, y_pred_stack, target_col)
        else:
            y_true_agg, y_pred_agg = y.values, y_pred_stack
        
        stack_metrics = compute_regression_metrics(y_true_agg, y_pred_agg)
        results["StackingEnsemble"] = stack_metrics
        logger.info("  %-20s  MAE=%.3f  R²=%.3f  Pearson_r=%.3f",
                    "StackingEnsemble", stack_metrics["MAE"], stack_metrics["R2"], stack_metrics.get("Pearson_r", 0))

    return results


# ── Patient-level evaluation ─────────────────────────────────────────────

def compute_patient_metrics(
    y_true, y_score, patient_ids, target_specificity=cfg.TARGET_SPECIFICITY
) -> Dict[str, Any]:
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "y_true": y_true,
        "y_score": y_score
    })
    patient_df = df.groupby("patient_id").agg(
        y_true=("y_true", "first"),
        y_score=("y_score", lambda x: np.percentile(x, 75))
    )
    y_p = patient_df["y_true"].values
    score_p = patient_df["y_score"].values

    try:
        auc_val = roc_auc_score(y_p, score_p)
    except ValueError:
        auc_val = 0.5

    fpr, tpr, thresholds = roc_curve(y_p, score_p)
    specificities = 1 - fpr
    valid_idx = np.where(specificities >= target_specificity)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(tpr[valid_idx])]
    else:
        best_idx = np.argmax(specificities)
    opt_thresh = thresholds[best_idx]

    y_pred_p = (score_p >= opt_thresh).astype(int)
    cm = confusion_matrix(y_p, y_pred_p)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "Accuracy": accuracy_score(y_p, y_pred_p),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "AUC": auc_val,
        "Threshold": float(opt_thresh),
        "N_patients": len(patient_df),
    }


def run_classification(features_df) -> Dict[str, Dict[str, float]]:
    """Run classification (ET vs Control) with LOSO, SMOTE, stacking.

    Scaling and feature selection (RFE) are applied **within** each CV fold
    to prevent data leakage.  SMOTE is applied after scaling+RFE on the
    training fold only.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import RFE

    feat_cols = _feature_cols(features_df)
    X = features_df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    y = features_df["is_et"].astype(int).copy()
    groups = features_df["patient_id"].copy()

    if len(X) < 5:
        logger.warning("Only %d samples for classification — skipping", len(X))
        return {}

    cv = _build_cv(groups, task="classification")

    # Log which features RFE picks (informational only)
    _rfe_info = RFE(
        estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=cfg.RANDOM_STATE, n_jobs=-1),
        n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
    )
    _rfe_info.fit(StandardScaler().fit_transform(X), y)
    _selected_names = [X.columns[i] for i in range(len(X.columns)) if _rfe_info.support_[i]]
    logger.info("RFE selected %d features (informational): %s", len(_selected_names), _selected_names)

    # Models including SVC with class_weight (from Cup)
    n_et = sum(y == 1)
    n_ctrl = sum(y == 0)
    svc_weight = {0: 1, 1: n_ctrl / max(n_et, 1)} if n_et > 0 else None

    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.RANDOM_STATE),
        "SVC": SVC(kernel="rbf", probability=True, class_weight=svc_weight, random_state=cfg.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=cfg.RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=cfg.RANDOM_STATE),
    }
    if _HAS_XGB:
        # scale_pos_weight is for the positive class (ET).
        # If n_et > n_ctrl, we scale it down.
        spw = n_ctrl / max(n_et, 1) if n_et > 0 else 1.0
        models["XGBoost"] = XGBClassifier(n_estimators=100, scale_pos_weight=spw,
                                           random_state=cfg.RANDOM_STATE,
                                           verbosity=0, use_label_encoder=False, eval_metric="logloss")

    results: Dict[str, Dict[str, float]] = {}
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=cfg.RANDOM_STATE)

    from collections import defaultdict
    feature_selection_counts: Dict[str, int] = defaultdict(int)
    n_folds_tracked = 0

    # Manual CV loop: scaler + RFE + SMOTE per fold
    for name, model in models.items():
        y_pred = np.zeros(len(y))
        y_score = np.zeros(len(y))
        has_proba = hasattr(model, "predict_proba")

        cv_splitter = cv.split(X, y, groups) if hasattr(cv, "split") else StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE).split(X, y)
        for train_idx, test_idx in cv_splitter:
            X_train_raw = X.iloc[train_idx].values
            X_test_raw = X.iloc[test_idx].values
            y_train = y.iloc[train_idx]

            # 1. VarianceThreshold then Scale on train, transform test
            vt = VarianceThreshold()
            X_train_vt = vt.fit_transform(X_train_raw)
            X_test_vt = vt.transform(X_test_raw)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_vt)
            X_test_s = scaler.transform(X_test_vt)

            # 2. RFE on train, apply mask to test
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced",
                                                  random_state=cfg.RANDOM_STATE, n_jobs=-1),
                n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
            )
            rfe.fit(X_train_s, y_train)
            X_train_sel = rfe.transform(X_train_s)
            X_test_sel = rfe.transform(X_test_s)

            # 3. SMOTE on selected+scaled train data
            try:
                X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
            except ValueError:
                X_train_res, y_train_res = X_train_sel, y_train

            # 4. Fit and predict
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_res, y_train_res)
            y_pred[test_idx] = fold_model.predict(X_test_sel)
            if has_proba:
                y_score[test_idx] = fold_model.predict_proba(X_test_sel)[:, 1]

        if not has_proba:
            y_score = None

        metrics = compute_classification_metrics(y, y_pred, y_score)
        results[name] = metrics
        parts = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        logger.info("  %-20s  %s", name, parts)
        
        if has_proba:
            patient_metrics = compute_patient_metrics(y.values, y_score, groups.values)
            results[f"{name}_patient"] = patient_metrics
            parts_p = "  ".join(f"{k}={v:.3f}" for k, v in patient_metrics.items())
            logger.info("  %-20s  %s", f"{name}_patient", parts_p)

    # Stacking blending
    if cfg.USE_STACKING and len(models) >= 2:
        logger.info("  Building stacking ensemble…")

        stack = StackingClassifier(
            estimators=list(models.items()),
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.RANDOM_STATE),
            cv=min(cfg.CV_FOLDS, groups.nunique()),
        )
        
        y_pred_stack = np.zeros(len(y))
        y_score_stack = np.zeros(len(y))
        cv_splitter = cv.split(X, y, groups) if hasattr(cv, "split") else StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE).split(X, y)
        for train_idx, test_idx in cv_splitter:
            X_train_raw = X.iloc[train_idx].values
            X_test_raw = X.iloc[test_idx].values
            y_train = y.iloc[train_idx]
            
            vt = VarianceThreshold()
            X_train_vt = vt.fit_transform(X_train_raw)
            X_test_vt = vt.transform(X_test_raw)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_vt)
            X_test_s = scaler.transform(X_test_vt)
            
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced",
                                                  random_state=cfg.RANDOM_STATE, n_jobs=-1),
                n_features_to_select=cfg.TOP_K_FEATURES, step=1,
            )
            rfe.fit(X_train_s, y_train)
            X_train_sel = rfe.transform(X_train_s)
            X_test_sel = rfe.transform(X_test_s)
            
            # FIXED: Data leakage — SMOTE внутри CV для stacking
            try:
                X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
            except ValueError:
                X_train_res, y_train_res = X_train_sel, y_train
                
            from sklearn.base import clone
            fold_stack = clone(stack)
            fold_stack.fit(X_train_res, y_train_res)
            y_pred_stack[test_idx] = fold_stack.predict(X_test_sel)
            y_score_stack[test_idx] = fold_stack.predict_proba(X_test_sel)[:, 1]

        stack_metrics = compute_classification_metrics(y, y_pred_stack, y_score_stack)
        results["StackingEnsemble"] = stack_metrics
        parts = "  ".join(f"{k}={v:.3f}" for k, v in stack_metrics.items())
        logger.info("  %-20s  %s", "StackingEnsemble", parts)
        
        if y_score_stack is not None:
            patient_metrics = compute_patient_metrics(y.values, y_score_stack, groups.values)
            results["StackingEnsemble_patient"] = patient_metrics
            parts_p = "  ".join(f"{k}={v:.3f}" for k, v in patient_metrics.items())
            logger.info("  %-20s  %s", "Stacking_patient", parts_p)

    # ── Youden's J threshold optimization ────────────────────────────────
    logger.info("")
    logger.info("  ── Youden's J threshold optimization ──")
    for name, metrics in list(results.items()):
        if "AUC" not in metrics:
            continue
        model = models.get(name)
        if model is None:
            continue

        # Use Pipeline for Youden's J too (no leakage)
        pipe_j = Pipeline([
            ("var_thresh", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("rfe", RFE(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced",
                                                  random_state=cfg.RANDOM_STATE, n_jobs=-1),
                n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
            )),
            ("model", model),
        ])
        try:
            y_score_j = cross_val_predict(pipe_j, X, y, cv=cv, groups=groups,
                                           method="predict_proba")[:, 1]
        except (AttributeError, ValueError):
            continue

        opt_metrics = optimize_threshold_youden(y.values, y_score_j)
        results[f"{name}_Youden"] = opt_metrics
        parts = "  ".join(f"{k}={v:.3f}" for k, v in opt_metrics.items())
        logger.info("  %-20s  %s", f"{name}_Youden", parts)

    if n_folds_tracked > 0:
        rfe_stability = {
            feat: round(count / n_folds_tracked, 3)
            for feat, count in sorted(
                feature_selection_counts.items(),
                key=lambda x: -x[1]
            )
        }
        logger.info("── RFE Feature Stability (%d folds) ──", n_folds_tracked)
        for feat, pct in list(rfe_stability.items())[:15]:
            logger.info("  %5.0f%%  %s", pct * 100, feat)
        results["__rfe_stability__"] = rfe_stability

    return results


# ── Youden's J threshold optimization ────────────────────────────────────

def optimize_threshold_youden(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Find the optimal probability threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1
    The optimal threshold maximizes J, balancing sensitivity and specificity.

    Returns:
        Dict with metrics at the optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Youden's J = tpr - fpr (equivalent to sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred_opt = (y_score >= best_threshold).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred_opt, y_score)
    metrics["Threshold"] = float(best_threshold)
    metrics["Youden_J"] = float(j_scores[best_idx])

    logger.info("    Optimal threshold=%.3f  Youden_J=%.3f", best_threshold, j_scores[best_idx])
    return metrics


# ── Regress-then-classify ────────────────────────────────────────────────

def run_regress_then_classify(
    features_df: pd.DataFrame,
    score_col: str = "local_score",
    score_threshold: float = 0.5,
) -> Dict[str, float]:
    """Classify ET vs Control by first predicting tremor score via regression.

    Scaling and RFE are wrapped in a Pipeline to prevent data leakage.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import RFE

    feat_cols = _feature_cols(features_df)
    X = features_df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    groups = features_df["patient_id"].copy()

    # Build regression target: Controls get 0, ET get their actual score
    y_reg = features_df[score_col].copy()
    is_control = features_df["group"] == "Control"
    y_reg[is_control] = 0.0
    y_reg = y_reg.fillna(0.0)

    y_true_binary = features_df["is_et"].astype(int).copy()

    if len(X) < 5:
        logger.warning("Only %d samples for regress-then-classify — skipping", len(X))
        return {}

    cv = _build_cv(groups, task="regression")

    # Use the best regression model wrapped in Pipeline
    if _HAS_XGB:
        base_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=cfg.RANDOM_STATE, verbosity=0)
    else:
        base_model = Ridge(alpha=1.0)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rfe", RFE(
            estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                             random_state=cfg.RANDOM_STATE, n_jobs=-1),
            n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
        )),
        ("model", base_model),
    ])

    try:
        y_pred_score = cross_val_predict(pipe, X, y_reg, cv=cv, groups=groups)
    except ValueError:
        y_pred_score = cross_val_predict(pipe, X, y_reg,
                                          cv=KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE))

    # Classify based on predicted score
    y_pred_binary = (y_pred_score > score_threshold).astype(int)

    # Standard metrics at fixed threshold
    metrics = compute_classification_metrics(y_true_binary.values, y_pred_binary, y_pred_score)
    logger.info("  Regress-then-classify (threshold=%.2f):", score_threshold)
    parts = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    logger.info("    Fixed threshold:    %s", parts)

    # Also optimize with Youden's J
    opt_metrics = optimize_threshold_youden(y_true_binary.values, y_pred_score)
    logger.info("    Youden optimal:")
    parts = "  ".join(f"{k}={v:.3f}" for k, v in opt_metrics.items())
    logger.info("                        %s", parts)

    # Patient-level
    patient_metrics = compute_patient_metrics(y_true_binary.values, y_pred_score, groups.values)
    logger.info("    Patient-level (Spec>=80):")
    parts_p = "  ".join(f"{k}={v:.3f}" for k, v in patient_metrics.items())
    logger.info("                        %s", parts_p)

    return {"fixed_threshold": metrics, "youden_optimal": opt_metrics, "rtc_patient": patient_metrics}


# ── SHAP interpretability ────────────────────────────────────────────────

def run_shap_analysis(
    features_df: pd.DataFrame,
    target: str = "local_score",
    task: str = "regression",
    output_dir: str = None,
) -> Optional[Dict[str, float]]:
    """Run SHAP analysis on the best model and save plots.

    Args:
        features_df: Full feature DataFrame.
        target: Target column.
        task: "regression" or "classification".
        output_dir: Where to save SHAP plots.

    Returns:
        Dict mapping feature name → mean |SHAP value|, or None if shap unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis. "
                       "Install with: pip install shap")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = cfg.OUTPUT_DIR

    feat_cols = _feature_cols(features_df)
    X = features_df[feat_cols].copy()

    if task == "regression":
        y = features_df[target].copy()
        valid = y.notna()
        X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    else:
        y = features_df["is_et"].astype(int).copy()

    if len(X) < 5:
        return None

    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_sel, selected = select_features(X_s, y, task=task)

    # Train model on full data for SHAP
    if _HAS_XGB:
        if task == "regression":
            model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=cfg.RANDOM_STATE, verbosity=0)
        else:
            model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=cfg.RANDOM_STATE, verbosity=0,
                                   use_label_encoder=False, eval_metric="logloss")
    else:
        if task == "regression":
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=cfg.RANDOM_STATE)
        else:
            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=cfg.RANDOM_STATE)

    model.fit(X_sel, y)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sel)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sel, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    fname = f"shap_bar_{task}.png"
    import os
    fig_path = os.path.join(output_dir, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP bar plot: %s", fig_path)

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sel, show=False, max_display=20)
    plt.tight_layout()
    fname = f"shap_beeswarm_{task}.png"
    fig_path = os.path.join(output_dir, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP beeswarm plot: %s", fig_path)

    # Return feature importances
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(X_sel.columns, mean_abs_shap))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("  Top 10 SHAP features (%s):", task)
    for i, (feat, val) in enumerate(importance.items()):
        if i >= 10:
            break
        logger.info("    %2d. %-30s  %.4f", i + 1, feat, val)

    return importance


# ── Calibrated classifiers ───────────────────────────────────────────────

def run_calibrated_classification(
    features_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Run classification with CalibratedClassifierCV (Platt scaling).

    Scaling and RFE are applied **within** each CV fold to prevent
    data leakage.  SMOTE is applied after scaling+RFE on training fold.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import RFE

    feat_cols = _feature_cols(features_df)
    X = features_df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    y = features_df["is_et"].astype(int).copy()
    groups = features_df["patient_id"].copy()

    if len(X) < 5:
        return {}

    cv = _build_cv(groups, task="classification")

    # Models to calibrate
    # Use balanced weights since CalibratedClassifierCV will fit on resampled training data anyway,
    # but it helps the base estimator.
    n_et = sum(y == 1)
    n_ctrl = sum(y == 0)
    bal_weight = {0: 1.0, 1: n_ctrl / max(n_et, 1)} if n_et > 0 else None

    base_models = {
        "SVC_calibrated": SVC(kernel="rbf", probability=True, class_weight=bal_weight, random_state=cfg.RANDOM_STATE),
        "RF_calibrated": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=cfg.RANDOM_STATE),
    }

    results: Dict[str, Dict[str, float]] = {}
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=cfg.RANDOM_STATE)

    for name, base in base_models.items():
        y_pred = np.zeros(len(y))
        y_score = np.zeros(len(y))

        cv_splitter = cv.split(X, y, groups) if hasattr(cv, "split") else StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE).split(X, y)
        success = True

        for train_idx, test_idx in cv_splitter:
            X_train_raw = X.iloc[train_idx].values
            X_test_raw = X.iloc[test_idx].values
            y_train = y.iloc[train_idx]

            # 1. VarianceThreshold then Scale on train, transform test
            vt = VarianceThreshold()
            X_train_vt = vt.fit_transform(X_train_raw)
            X_test_vt = vt.transform(X_test_raw)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_vt)
            X_test_s = scaler.transform(X_test_vt)

            # 2. RFE on train, apply mask to test
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced",
                                                  random_state=cfg.RANDOM_STATE, n_jobs=-1),
                n_features_to_select=cfg.TOP_K_FEATURES, step=0.1,
            )
            rfe.fit(X_train_s, y_train)
            X_train_sel = rfe.transform(X_train_s)
            X_test_sel = rfe.transform(X_test_s)

            # 3. SMOTE on selected+scaled train data
            try:
                X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
            except ValueError:
                X_train_res, y_train_res = X_train_sel, y_train

            # 4. Fit calibrated model and predict
            from sklearn.base import clone
            calibrated = CalibratedClassifierCV(clone(base), cv=3, method="sigmoid")
            try:
                calibrated.fit(X_train_res, y_train_res)
                y_pred[test_idx] = calibrated.predict(X_test_sel)
                y_score[test_idx] = calibrated.predict_proba(X_test_sel)[:, 1]
            except (ValueError, AttributeError):
                success = False
                break

        if not success:
            continue

        metrics = compute_classification_metrics(y.values, y_pred, y_score)
        results[name] = metrics

        # Also apply Youden's J
        opt = optimize_threshold_youden(y.values, y_score)
        results[f"{name}_Youden"] = opt

        parts = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        logger.info("  %-25s  %s", name, parts)
        parts_y = "  ".join(f"{k}={v:.3f}" for k, v in opt.items())
        logger.info("  %-25s  %s", f"{name}_Youden", parts_y)

        # Patient-level
        patient_metrics = compute_patient_metrics(y.values, y_score, groups.values)
        results[f"{name}_patient"] = patient_metrics
        parts_p = "  ".join(f"{k}={v:.3f}" for k, v in patient_metrics.items())
        logger.info("  %-25s  %s", f"{name}_patient", parts_p)

    return results

