"""Classification, regression, operational (@K), and walk-forward metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

from pipeline.training import predict_two_stage, train_two_stage_models
from utils import logger


def evaluate_classification(y_binary, gate_proba, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (gate_proba >= threshold).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_binary, gate_proba)) if len(np.unique(y_binary)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_binary, y_pred)),
        "precision": float(precision_score(y_binary, y_pred, zero_division=0)),
        "recall": float(recall_score(y_binary, y_pred, zero_division=0)),
        "f1": float(f1_score(y_binary, y_pred, zero_division=0)),
        "threshold": threshold,
    }
    logger.info("Gate AUC=%.3f P=%.3f R=%.3f", metrics["auc_roc"], metrics["precision"], metrics["recall"])
    return metrics


def evaluate_regression(y_count, y_pred) -> Dict[str, float]:
    y_count = np.asarray(y_count, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    corr = spearmanr(y_count, y_pred).correlation if len(y_count) > 2 else float("nan")
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_count, y_pred))),
        "mae": float(mean_absolute_error(y_count, y_pred)),
        "r2": float(r2_score(y_count, y_pred)) if len(y_count) > 1 else float("nan"),
        "spearman": float(corr) if corr == corr else float("nan"),
    }
    logger.info("Count RMSE=%.3f MAE=%.3f Spearman=%.3f", metrics["rmse"], metrics["mae"], metrics["spearman"])
    return metrics


def evaluate_operational(
    y_count,
    y_pred,
    gate_proba,
    k_values: Sequence[int] = (10, 20, 50),
) -> Dict[str, float]:
    """Precision@K / Efficiency@K / Lift@K for field prioritization."""
    ranked = pd.DataFrame(
        {
            "y": np.asarray(y_count, dtype=float),
            "pred": np.asarray(y_pred, dtype=float),
            "gate": np.asarray(gate_proba, dtype=float),
        }
    )
    ranked["score"] = ranked["pred"] + 0.01 * ranked["gate"]
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    total_cases = ranked["y"].sum()
    results: Dict[str, float] = {}
    for k in k_values:
        if k > n:
            continue
        top = ranked.head(k)
        precision = (top["y"] > 0).mean()
        cases_k = top["y"].sum()
        efficiency = cases_k / total_cases if total_cases > 0 else 0.0
        expected_random = (total_cases / n) * k if n else 0.0
        lift = cases_k / expected_random if expected_random > 0 else 0.0
        results[f"precision_at_{k}"] = float(precision)
        results[f"efficiency_at_{k}"] = float(efficiency)
        results[f"cases_at_{k}"] = float(cases_k)
        results[f"lift_at_{k}"] = float(lift)
        logger.info("K=%d Precision@K=%.2f Efficiency@K=%.2f Lift=%.1fx", k, precision, efficiency, lift)
    return results


def walk_forward_auc(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_train_months: int = 3,
) -> pd.DataFrame:
    """
    Retrain each month on all prior months; score the next month.
    Returns a small table of per-fold AUC / RMSE.
    """
    months = sorted(df["month"].unique())
    rows = []
    for i in range(min_train_months, len(months)):
        train_m = months[:i]
        test_m = months[i]
        train = df[df["month"].isin(train_m)]
        test = df[df["month"] == test_m]
        if test.empty or train["has_any_cases"].sum() < 2:
            continue
        bundle = train_two_stage_models(train, feature_cols)
        X_te = test[feature_cols].fillna(0)
        pred, gate = predict_two_stage(bundle["stage1"], bundle["stage2"], X_te, bundle["calibrator"])
        auc = (
            float(roc_auc_score(test["has_any_cases"], gate))
            if test["has_any_cases"].nunique() > 1
            else float("nan")
        )
        rmse = float(np.sqrt(mean_squared_error(test["target_cases"], pred)))
        rows.append({"test_month": test_m, "n_train": len(train), "n_test": len(test), "auc": auc, "rmse": rmse})
        logger.info("Walk-forward %s AUC=%.3f RMSE=%.3f", pd.Timestamp(test_m).date(), auc, rmse)
    return pd.DataFrame(rows)
