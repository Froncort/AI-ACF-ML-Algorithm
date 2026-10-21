"""Operational top-K recommendations, risk tiers, and dashboard JSON export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.training import predict_two_stage
from utils import ARTIFACTS_DIR, ensure_dirs, logger


def assign_risk_tier(predicted_cases, gate_proba) -> List[str]:
    """Map count + gate probability to High / Medium / Low."""
    tiers = []
    for cases, proba in zip(predicted_cases, gate_proba):
        if cases >= 3 or (cases >= 1 and proba >= 0.7):
            tiers.append("High")
        elif cases >= 1 and proba >= 0.4:
            tiers.append("Medium")
        else:
            tiers.append("Low")
    return tiers


def _rationale(row: pd.Series) -> str:
    parts = []
    if row.get("predicted_cases", 0) >= 3:
        parts.append(f"High predicted count ({row['predicted_cases']:.1f})")
    elif row.get("predicted_cases", 0) >= 1:
        parts.append(f"Expected {row['predicted_cases']:.1f} cases")
    if row.get("gate_probability", 0) >= 0.5:
        parts.append(f"Case probability {row['gate_probability']*100:.0f}%")
    if row.get("total_cases", 0) > 0:
        parts.append(f"Recent history ({int(row['total_cases'])} last month)")
    if row.get("neighbor_risk_ring1", 0) > 2:
        parts.append("Elevated neighbor activity")
    if row.get("total_cases", 0) == 0 and row.get("predicted_cases", 0) > 0:
        parts.append("Emerging / cold-start risk")
    return "; ".join(parts) if parts else "Elevated baseline risk"


def generate_recommendations(
    feature_df: pd.DataFrame,
    stage1_model,
    stage2_model,
    feature_cols: List[str],
    calibrator=None,
    top_k: int = 50,
) -> pd.DataFrame:
    """Score the latest month and return top-K hexes with tiers + rationale."""
    latest = feature_df["month"].max()
    cur = feature_df[feature_df["month"] == latest].copy()
    X = cur[feature_cols].fillna(0)
    pred, gate = predict_two_stage(stage1_model, stage2_model, X, calibrator)
    cur["predicted_cases"] = pred
    cur["gate_probability"] = gate
    cur["risk_tier"] = assign_risk_tier(pred, gate)
    cur["rank_score"] = cur["predicted_cases"] + 0.01 * cur["gate_probability"]
    cur = cur.sort_values("rank_score", ascending=False).reset_index(drop=True)
    cur["rank"] = np.arange(1, len(cur) + 1)
    cur["rationale"] = cur.apply(_rationale, axis=1)
    out = cur.head(top_k)
    logger.info("Recommendations for %s: top %d (High=%d)", latest.date(), len(out), (out["risk_tier"] == "High").sum())
    return out


def export_dashboard_json(recommendations: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """
    Write a compact JSON list suitable for a map / dashboard prototype.

    Includes lat/lng centroids (derived from H3 when missing) so a UI can
    draw the same hexagons the main product site visualizes.
    """
    from pipeline.viz import attach_hex_centroids

    ensure_dirs()
    path = Path(path) if path else ARTIFACTS_DIR / "dashboard_recommendations.json"
    recs = attach_hex_centroids(recommendations)
    records = []
    for _, row in recs.iterrows():
        rec = {
            "rank": int(row["rank"]),
            "h3_index": str(row["h3_index"]),
            "predicted_cases": float(row["predicted_cases"]),
            "gate_probability": float(row["gate_probability"]),
            "risk_tier": row["risk_tier"],
            "rationale": row["rationale"],
        }
        # Centroids let a map UI place polygons without recomputing H3
        if pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
            rec["lat"] = float(row["lat"])
            rec["lng"] = float(row["lng"])
        records.append(rec)
    path.write_text(json.dumps({"recommendations": records}, indent=2))
    logger.info("Dashboard JSON -> %s", path)
    return path
