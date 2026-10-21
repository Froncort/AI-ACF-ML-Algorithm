"""
Feature engineering on an H3 hexagon × month grid.

Distilled from the production two-stage TB hotspot pipeline:
clinical counts, temporal lags, optional contact tracing, simple spatial
neighbor risk, and interaction terms. No database access.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils import logger


def _patient_month(patients: pd.DataFrame) -> pd.DataFrame:
    """Attach a month-start timestamp from diagnosis_date (or fallbacks)."""
    df = patients.copy()
    for col in ("diagnosis_date", "notification_date", "created_at"):
        if col in df.columns and df[col].notna().any():
            df["month"] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
            break
    else:
        raise ValueError("patients need a date column: diagnosis_date / notification_date / created_at")
    return df


def create_monthly_grid(patients: pd.DataFrame, hexagons: pd.DataFrame) -> pd.DataFrame:
    """Cartesian product of AOI hexagons × months covered by patient dates."""
    p = _patient_month(patients)
    months = pd.date_range(p["month"].min(), p["month"].max(), freq="MS")
    hex_ids = hexagons["h3_index"].unique()
    grid = pd.MultiIndex.from_product([hex_ids, months], names=["h3_index", "month"]).to_frame(index=False)
    grid["year"] = grid["month"].dt.year
    grid["month_num"] = grid["month"].dt.month
    logger.info("Grid: %d hex-months (%d hex × %d months)", len(grid), len(hex_ids), len(months))
    return grid


def calculate_clinical_features(patients: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Aggregate case counts and comorbidity proxies per hex-month."""
    p = _patient_month(patients)
    id_col = "patient_id" if "patient_id" in p.columns else p.columns[0]

    agg = (
        p.groupby(["h3_index", "month"])
        .agg(
            total_cases=(id_col, "count"),
            avg_age=("age", "mean") if "age" in p.columns else (id_col, "count"),
        )
        .reset_index()
    )
    if "age" not in p.columns:
        agg = agg.rename(columns={"avg_age": "total_cases_dup"}).drop(columns=["total_cases_dup"], errors="ignore")
        agg["avg_age"] = np.nan

    if "site_of_disease" in p.columns:
        pulm = (
            p[p["site_of_disease"] == "Pulmonary"]
            .groupby(["h3_index", "month"])
            .size()
            .reset_index(name="pulmonary_cases")
        )
        agg = agg.merge(pulm, on=["h3_index", "month"], how="left")
    else:
        agg["pulmonary_cases"] = 0

    if "is_recidivism" in p.columns:
        rel = p.groupby(["h3_index", "month"])["is_recidivism"].mean().reset_index(name="relapse_rate")
        agg = agg.merge(rel, on=["h3_index", "month"], how="left")
    else:
        agg["relapse_rate"] = 0.0

    if "hiv_status" in p.columns:
        hiv = (
            p[p["hiv_status"] == "Positive"]
            .groupby(["h3_index", "month"])
            .size()
            .reset_index(name="hiv_positive_count")
        )
        agg = agg.merge(hiv, on=["h3_index", "month"], how="left")
    else:
        agg["hiv_positive_count"] = 0

    if "diabetes_status" in p.columns:
        dm = (
            p[p["diabetes_status"] == "Yes"]
            .groupby(["h3_index", "month"])
            .size()
            .reset_index(name="diabetes_count")
        )
        agg = agg.merge(dm, on=["h3_index", "month"], how="left")
    else:
        agg["diabetes_count"] = 0

    if "infectivity_score" in p.columns:
        load = p.groupby(["h3_index", "month"])["infectivity_score"].sum().reset_index(name="weighted_infectivity_load")
        agg = agg.merge(load, on=["h3_index", "month"], how="left")
    else:
        agg["weighted_infectivity_load"] = agg["total_cases"]

    out = grid.merge(agg, on=["h3_index", "month"], how="left")
    for c in ["total_cases", "pulmonary_cases", "hiv_positive_count", "diabetes_count", "weighted_infectivity_load"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    out["relapse_rate"] = out.get("relapse_rate", 0).fillna(0)
    out["pulmonary_percentage"] = np.where(out["total_cases"] > 0, out["pulmonary_cases"] / out["total_cases"], 0)
    return out


def calculate_temporal_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling sum/mean, MoM trend, persistence, days since last case."""
    g = grid.sort_values(["h3_index", "month"]).copy()
    for lag in (1, 2, 3):
        g[f"cases_t_minus_{lag}"] = g.groupby("h3_index")["total_cases"].shift(lag).fillna(0)

    g["cases_sum_3mo"] = (
        g.groupby("h3_index")["total_cases"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    g["cases_3month_rolling_avg"] = (
        g.groupby("h3_index")["total_cases"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    g["cases_sum_6mo"] = (
        g.groupby("h3_index")["total_cases"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    g["mom_trend"] = g.groupby("h3_index")["total_cases"].diff().fillna(0)

    def _persistence(s: pd.Series) -> pd.Series:
        vals = []
        for i in range(len(s)):
            window = s.iloc[max(0, i - 6) : i]
            vals.append(int((window > 0).sum()))
        return pd.Series(vals, index=s.index)

    g["persistence_score"] = g.groupby("h3_index")["total_cases"].transform(_persistence)

    def _days_since(group: pd.DataFrame) -> pd.Series:
        out, last = [], None
        for _, row in group.iterrows():
            if row["total_cases"] > 0:
                last = row["month"]
                out.append(0)
            elif last is not None:
                out.append((row["month"] - last).days)
            else:
                out.append(999)
        return pd.Series(out, index=group.index)

    g["days_since_last_case"] = g.groupby("h3_index", group_keys=False).apply(_days_since)
    return g


def calculate_contact_features(contacts: Optional[pd.DataFrame], grid: pd.DataFrame) -> pd.DataFrame:
    """Merge optional contact-tracing counts; zeros if file absent."""
    g = grid.copy()
    if contacts is None or contacts.empty:
        g["n_contacts"] = 0
        return g
    c = contacts.copy()
    c["month"] = pd.to_datetime(c["month"]).dt.to_period("M").dt.to_timestamp()
    agg = c.groupby(["h3_index", "month"])["n_contacts"].sum().reset_index()
    g = g.merge(agg, on=["h3_index", "month"], how="left")
    g["n_contacts"] = g["n_contacts"].fillna(0)
    return g


def calculate_spatial_features(grid: pd.DataFrame, hexagons: pd.DataFrame) -> pd.DataFrame:
    """
    Neighbor risk via H3 ring-1 when possible; else lat/lng k-NN proxy.
    Also attach optional slum_density from hexagon metadata.
    """
    g = grid.copy()
    if "slum_density" in hexagons.columns:
        g = g.merge(hexagons[["h3_index", "slum_density"]], on="h3_index", how="left")
        g["slum_density"] = g["slum_density"].fillna(0)
    else:
        g["slum_density"] = 0.0

    # Month-level case lookup for neighbors
    case_lookup = g.set_index(["month", "h3_index"])["total_cases"].to_dict()
    neighbor_map = {}

    try:
        import h3

        for h in g["h3_index"].unique():
            if hasattr(h3, "grid_disk"):
                nbrs = list(h3.grid_disk(h, 1))
            else:
                nbrs = list(h3.k_ring(h, 1))
            neighbor_map[h] = [n for n in nbrs if n != h]
    except Exception:
        # Fallback: approximate neighbors by proximity in hexagon table
        if {"lat", "lng"}.issubset(hexagons.columns):
            coords = hexagons.set_index("h3_index")[["lat", "lng"]]
            for h, row in coords.iterrows():
                d = ((coords["lat"] - row["lat"]) ** 2 + (coords["lng"] - row["lng"]) ** 2) ** 0.5
                neighbor_map[h] = list(d.nsmallest(7).index.drop(h, errors="ignore"))
        else:
            neighbor_map = {h: [] for h in g["h3_index"].unique()}

    def _neighbor_risk(row):
        nbrs = neighbor_map.get(row["h3_index"], [])
        return float(sum(case_lookup.get((row["month"], n), 0) for n in nbrs))

    g["neighbor_risk_ring1"] = g.apply(_neighbor_risk, axis=1)
    g["hotspot_encirclement"] = (g["neighbor_risk_ring1"] > 0).astype(float)
    return g


def calculate_interaction_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Simple interactions used by the two-stage models."""
    g = grid.copy()
    g["cases_x_slum"] = g["total_cases"] * g.get("slum_density", 0)
    g["neighbor_x_slum"] = g.get("neighbor_risk_ring1", 0) * g.get("slum_density", 0)
    g["lag1_x_persistence"] = g.get("cases_t_minus_1", 0) * g.get("persistence_score", 0)
    g["transmission_vulnerability_score"] = (
        0.4 * g.get("slum_density", 0)
        + 0.3 * np.clip(g.get("neighbor_risk_ring1", 0) / 10.0, 0, 1)
        + 0.3 * np.clip(g.get("weighted_infectivity_load", 0) / 5.0, 0, 1)
    )
    return g


def build_feature_table(
    patients: pd.DataFrame,
    hexagons: pd.DataFrame,
    contacts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Full feature pipeline: grid → clinical → temporal → contacts → spatial → interactions."""
    grid = create_monthly_grid(patients, hexagons)
    grid = calculate_clinical_features(patients, grid)
    grid = calculate_temporal_features(grid)
    grid = calculate_contact_features(contacts, grid)
    grid = calculate_spatial_features(grid, hexagons)
    grid = calculate_interaction_features(grid)
    logger.info("Feature table ready: %d rows, %d cols", len(grid), grid.shape[1])
    return grid
