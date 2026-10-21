"""Temporal train/test split and feature column selection."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd

from utils import logger

# Columns that must never be used as model inputs
EXCLUDE_COLS = {
    "h3_index",
    "month",
    "target_month",
    "year",
    "month_num",
    "target_cases",
    "future_cases",
    "has_any_cases",
    "is_cluster",
    "is_emerging",
    "case_change",
    "current_cases",
    "target",
    "risk_tier",
    "predicted_cases",
    "gate_probability",
    "lat",
    "lng",
}


def get_feature_columns(df: pd.DataFrame, extra_exclude: Sequence[str] = ()) -> List[str]:
    """Numeric feature columns excluding labels / ids / geography."""
    exclude = EXCLUDE_COLS.union(extra_exclude)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            cols.append(c)
    return cols


def prepare_train_test(df: pd.DataFrame, test_months: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_months` calendar months as the test set."""
    df = df.sort_values("month")
    unique_months = sorted(df["month"].unique())
    if len(unique_months) <= test_months:
        raise ValueError(f"Need more than {test_months} months; got {len(unique_months)}")
    split_month = unique_months[-test_months]
    train = df[df["month"] < split_month].copy()
    test = df[df["month"] >= split_month].copy()
    logger.info(
        "Split: train %s→%s (%d) | test %s→%s (%d)",
        train["month"].min().date(),
        train["month"].max().date(),
        len(train),
        test["month"].min().date(),
        test["month"].max().date(),
        len(test),
    )
    return train, test


def validate_no_future_leakage(df: pd.DataFrame) -> None:
    """Hard check: feature month must be strictly before target_month."""
    if "target_month" not in df.columns:
        return
    bad = ~(df["month"] < df["target_month"])
    if bad.any():
        raise ValueError(f"Temporal leakage: {bad.sum()} rows have month >= target_month")
    logger.info("Leakage check passed (month < target_month for all rows)")
