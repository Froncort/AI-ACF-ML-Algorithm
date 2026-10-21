"""
Target creation: features at month M predict case count at month M+1.

Primary target: target_cases (regression)
Binary flags: has_any_cases (gate), is_cluster, is_emerging
"""
from __future__ import annotations

import pandas as pd

from utils import logger


def create_future_target(features_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift so each row's features come from month M and targets from M+1.
    Last month per hexagon is dropped (no future label).
    """
    patients = patients_df.copy()
    # Align patient months to month-start
    date_col = "diagnosis_date" if "diagnosis_date" in patients.columns else "notification_date"
    patients["month"] = pd.to_datetime(patients[date_col]).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

    future_counts = (
        patients.groupby(["h3_index", "month"]).size().reset_index(name="future_case_count")
    )

    rows = []
    for h3_idx, hex_feats in features_df.groupby("h3_index"):
        hex_feats = hex_feats.sort_values("month")
        for i in range(len(hex_feats) - 1):
            cur = hex_feats.iloc[i]
            next_month = cur["month"] + pd.DateOffset(months=1)
            # Prefer exact next calendar month match from patient counts
            match = future_counts[
                (future_counts["h3_index"] == h3_idx) & (future_counts["month"] == next_month)
            ]
            n_future = int(match["future_case_count"].iloc[0]) if len(match) else 0
            current_cases = int(cur.get("total_cases", 0) or 0)

            row = cur.to_dict()
            row["target_month"] = next_month
            row["current_cases"] = current_cases
            row["target_cases"] = n_future
            row["future_cases"] = n_future
            row["has_any_cases"] = 1 if n_future > 0 else 0
            row["is_cluster"] = 1 if n_future >= 3 else 0
            row["is_emerging"] = 1 if (current_cases <= 1 and n_future >= 2) else 0
            row["case_change"] = n_future - current_cases
            rows.append(row)

    out = pd.DataFrame(rows)
    logger.info(
        "Targets: %d rows | positive hex-months: %d (%.1f%%)",
        len(out),
        out["has_any_cases"].sum(),
        100 * out["has_any_cases"].mean() if len(out) else 0,
    )
    return out
