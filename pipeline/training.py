"""
Two-stage TB count model — mirrors the production `TwoStageHurdleModel`
(AI-ACF-ML-Prediction, api/services/two_stage_model.py) so this open-source
notebook is a faithful distillation of what actually runs in staging:

  Stage 1 — XGBClassifier gate: P(any cases next month)
  Isotonic calibration — corrects gate probabilities distorted by
      `scale_pos_weight` class-imbalance reweighting (fit on the Stage-1
      validation fold, never on training data, to avoid overfitting the
      calibration itself)
  Stage 2 — XGBRegressor Poisson: E[cases | cases > 0] (fit only on
      non-zero rows — a standard hurdle-model simplification; see
      docs/methodology.md for the zero-truncation caveat this implies)
  Combine — expected_cases = calibrated_gate_prob * count_pred
      (continuous, no hard threshold cutoff — matches production; a hard
      threshold discards the probability information right at the gate
      boundary, hurting ranking quality)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import ARTIFACTS_DIR, ensure_dirs, logger

# Hyperparameters mirror production's GATE_MODEL_CONFIG / COUNT_MODEL_CONFIG
# (AI-ACF-ML-Prediction, api/config.py) so offline notebook results are
# representative of what the deployed model actually does.
GATE_MODEL_CONFIG: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 200,
    "max_depth": 4,
    "min_child_weight": 50,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "gamma": 1.0,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "random_state": 42,
    "tree_method": "hist",
}

COUNT_MODEL_CONFIG: Dict[str, Any] = {
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
    "n_estimators": 150,
    "max_depth": 4,
    "min_child_weight": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "random_state": 42,
    "tree_method": "hist",
}


def predict_two_stage(
    stage1_model,
    stage2_model,
    X: pd.DataFrame,
    calibrator: Optional[IsotonicRegression] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score with the gate + count models and combine multiplicatively:
    expected_cases = P(active) x E[count | active].

    `calibrator` is the isotonic regressor fit during training on the
    Stage-1 validation fold; pass it (from the saved bundle) so gate
    probabilities used for ranking/tiering are calibrated, not raw.
    """
    gate_proba_raw = stage1_model.predict_proba(X)[:, 1]
    gate_proba = calibrator.transform(gate_proba_raw) if calibrator is not None else gate_proba_raw
    count_pred = np.maximum(stage2_model.predict(X), 0)
    expected_cases = gate_proba * count_pred
    return expected_cases, gate_proba


def train_two_stage_models(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit Stage 1 (gate) + isotonic calibrator + Stage 2 (count) + a
    single-stage Poisson baseline, matching production's train() flow.

    Returns dict with models, calibrator, metrics, and feature_cols.
    """
    X = train_df[feature_cols].fillna(0)
    y_count = train_df["target_cases"].astype(float).values
    y_bin = train_df["has_any_cases"].astype(int).values

    # Class weight for rare positives (same imbalance handling as production)
    pos = max(y_bin.sum(), 1)
    neg = max(len(y_bin) - pos, 1)
    scale_pos = neg / pos

    # ---- Stage 1: gate ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_bin, test_size=0.2, random_state=random_state, stratify=y_bin if y_bin.sum() > 1 else None
    )
    gate_config = dict(GATE_MODEL_CONFIG, scale_pos_weight=scale_pos, random_state=random_state, n_jobs=-1)
    stage1 = xgb.XGBClassifier(**gate_config)
    stage1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    gate_proba_val_raw = stage1.predict_proba(X_val)[:, 1]
    gate_auc = roc_auc_score(y_val, gate_proba_val_raw) if y_val.sum() > 0 else float("nan")
    gate_ap = average_precision_score(y_val, gate_proba_val_raw) if y_val.sum() > 0 else float("nan")
    logger.info("Stage 1 gate val AUC: %.4f | AP: %.4f", gate_auc, gate_ap)

    # ---- Isotonic calibration of gate probabilities ----
    # Fit on the held-out validation fold only (never on training rows the
    # gate model already saw) so the calibration itself doesn't overfit.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(gate_proba_val_raw, y_val)
    gate_proba_val_calibrated = calibrator.transform(gate_proba_val_raw)
    calibrated_ap = average_precision_score(y_val, gate_proba_val_calibrated) if y_val.sum() > 0 else float("nan")
    logger.info("Gate calibrated AP: %.4f (raw AP: %.4f)", calibrated_ap, gate_ap)

    # ---- Stage 2: Poisson count, non-zero rows only ----
    mask = y_count > 0
    X_nz, y_nz = X.loc[mask], y_count[mask]
    if len(X_nz) < 10:
        logger.warning("Very few non-zero samples (%d); Stage 2 will be weak", len(X_nz))

    count_config = dict(COUNT_MODEL_CONFIG, random_state=random_state, n_jobs=-1)
    stage2 = xgb.XGBRegressor(**count_config)
    if len(X_nz) >= 5:
        X2_tr, X2_val, y2_tr, y2_val = train_test_split(X_nz, y_nz, test_size=0.2, random_state=random_state)
        stage2.fit(X2_tr, y2_tr, eval_set=[(X2_val, y2_val)], verbose=False)
        pred_nz = np.maximum(stage2.predict(X2_val), 0)
        rmse_nz = float(np.sqrt(mean_squared_error(y2_val, pred_nz)))
    else:
        stage2.fit(X_nz if len(X_nz) else X, y_nz if len(X_nz) else y_count)
        rmse_nz = float("nan")
    logger.info("Stage 2 non-zero val RMSE: %s", rmse_nz)

    # ---- Single-stage Poisson baseline (for comparison only) ----
    single = xgb.XGBRegressor(
        objective="count:poisson",
        max_depth=5,
        n_estimators=120,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    single.fit(X, y_count)

    return {
        "stage1": stage1,
        "stage2": stage2,
        "calibrator": calibrator,
        "single_poisson": single,
        "feature_cols": feature_cols,
        "metrics": {
            "gate_auc": gate_auc,
            "gate_ap": gate_ap,
            "calibrated_ap": calibrated_ap,
            "stage2_rmse_nz": rmse_nz,
        },
    }


def save_models(bundle: Dict[str, Any], out_dir: Optional[Path] = None) -> Path:
    """Persist models + calibrator + feature list under artifacts/."""
    ensure_dirs()
    out_dir = Path(out_dir) if out_dir else ARTIFACTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["stage1"], out_dir / "stage1_gate.joblib")
    joblib.dump(bundle["stage2"], out_dir / "stage2_poisson.joblib")
    joblib.dump(bundle["calibrator"], out_dir / "calibrator.joblib")
    joblib.dump(bundle["single_poisson"], out_dir / "single_poisson.joblib")
    joblib.dump(
        {"feature_cols": bundle["feature_cols"], "metrics": bundle["metrics"]},
        out_dir / "metadata.joblib",
    )
    logger.info("Models saved to %s", out_dir)
    return out_dir


def load_models(model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a previously saved model bundle."""
    model_dir = Path(model_dir) if model_dir else ARTIFACTS_DIR / "models"
    meta = joblib.load(model_dir / "metadata.joblib")
    return {
        "stage1": joblib.load(model_dir / "stage1_gate.joblib"),
        "stage2": joblib.load(model_dir / "stage2_poisson.joblib"),
        "calibrator": joblib.load(model_dir / "calibrator.joblib"),
        "single_poisson": joblib.load(model_dir / "single_poisson.joblib"),
        **meta,
    }
