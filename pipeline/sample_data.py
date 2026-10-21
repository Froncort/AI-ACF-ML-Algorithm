"""
Synthetic sample data + documented input schema.

Real patient / facility data is NOT shipped. Users point DATA_DIR at their
own CSVs that match the contract below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils import SAMPLE_DATA_DIR, ensure_dirs, logger, save_data

# Required columns for bring-your-own data (Nikshay / ACF-style extracts)
DATA_CONTRACT: Dict[str, Dict[str, str]] = {
    "patients": {
        "patient_id": "Unique case id (string or int)",
        "h3_index": "H3 cell id at your chosen resolution (e.g. res 10)",
        "diagnosis_date": "ISO date when TB was diagnosed",
        "age": "Age in years (optional but useful)",
        "site_of_disease": "Pulmonary | Extra-pulmonary | Other (optional)",
        "hiv_status": "Positive | Negative | Unknown (optional)",
        "diabetes_status": "Yes | No | Unknown (optional)",
        "is_recidivism": "1 if retreatment / relapse, else 0 (optional)",
    },
    "hexagons": {
        "h3_index": "H3 cell id covering the AOI / region (list of hexes to score — not the whole globe)",
        "lat": "Cell centroid latitude (optional, for maps)",
        "lng": "Cell centroid longitude (optional, for maps)",
        "slum_density": "0-1 density / vulnerability proxy (optional)",
    },
    "contacts": {
        "h3_index": "H3 of index / contact location (optional file)",
        "month": "Month (YYYY-MM-01) of contact-tracing activity (optional)",
        "n_contacts": "Count of contacts registered (optional)",
    },
}


def _random_h3_cells(n: int, seed: int = 42) -> list:
    """
    Build a compact cluster of H3 cells around a fixed origin.
    Uses h3 if installed; otherwise invents opaque string ids.
    """
    rng = np.random.default_rng(seed)
    try:
        import h3

        # Arbitrary inland origin — not a real program AOI
        origin = h3.latlng_to_cell(18.52, 73.85, 10) if hasattr(h3, "latlng_to_cell") else h3.geo_to_h3(18.52, 73.85, 10)
        cells = {origin}
        frontier = [origin]
        while len(cells) < n and frontier:
            cur = frontier.pop(0)
            if hasattr(h3, "grid_disk"):
                nbrs = h3.grid_disk(cur, 1)
            else:
                nbrs = h3.k_ring(cur, 1)
            for nb in nbrs:
                if nb not in cells:
                    cells.add(nb)
                    frontier.append(nb)
                if len(cells) >= n:
                    break
        return sorted(cells)[:n]
    except Exception:
        # Fallback: fake ids so notebooks still run without h3
        return [f"SYNTH_HEX_{i:04d}" for i in range(n)]


def generate_synthetic_dataset(
    n_hex: int = 80,
    n_months: int = 8,
    start: str = "2024-01-01",
    seed: int = 42,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create small synthetic patients / hexagons / contacts tables.

    Spatial clustering + temporal persistence so two-stage models have signal.
    """
    ensure_dirs()
    out_dir = Path(out_dir) if out_dir else SAMPLE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    h3_ids = _random_h3_cells(n_hex, seed=seed)
    months = pd.date_range(start=start, periods=n_months, freq="MS")

    # Hexagon metadata with a few high-risk cells
    risk = rng.uniform(0.05, 0.95, size=n_hex)
    risk[rng.choice(n_hex, size=max(5, n_hex // 10), replace=False)] = rng.uniform(0.7, 1.0, size=max(5, n_hex // 10))
    hexagons = pd.DataFrame(
        {
            "h3_index": h3_ids,
            "lat": 18.5 + rng.normal(0, 0.02, n_hex),
            "lng": 73.8 + rng.normal(0, 0.02, n_hex),
            "slum_density": risk,
        }
    )

    # Generate cases: Poisson intensity driven by slum_density + month seasonality
    patients_rows = []
    pid = 0
    for m_i, month in enumerate(months):
        season = 1.0 + 0.3 * np.sin(2 * np.pi * m_i / 12)
        for h_i, h3_idx in enumerate(h3_ids):
            lam = 0.15 + 1.8 * risk[h_i] * season
            # Mild temporal autocorrelation from previous intensity
            n_cases = int(rng.poisson(lam))
            for _ in range(n_cases):
                site = rng.choice(["Pulmonary", "Extra-pulmonary", "Other"], p=[0.7, 0.2, 0.1])
                patients_rows.append(
                    {
                        "patient_id": f"P{pid:05d}",
                        "h3_index": h3_idx,
                        "diagnosis_date": month + pd.Timedelta(days=int(rng.integers(0, 27))),
                        "age": int(rng.integers(5, 80)),
                        "site_of_disease": site,
                        "hiv_status": rng.choice(["Positive", "Negative", "Unknown"], p=[0.05, 0.85, 0.1]),
                        "diabetes_status": rng.choice(["Yes", "No", "Unknown"], p=[0.1, 0.8, 0.1]),
                        "is_recidivism": int(rng.random() < 0.08),
                        "infectivity_score": 1.0 if site == "Pulmonary" else 0.2,
                    }
                )
                pid += 1

    patients = pd.DataFrame(patients_rows)
    if patients.empty:
        # Guarantee at least a few rows so notebooks don't break
        patients = pd.DataFrame(
            [
                {
                    "patient_id": "P00000",
                    "h3_index": h3_ids[0],
                    "diagnosis_date": months[0],
                    "age": 40,
                    "site_of_disease": "Pulmonary",
                    "hiv_status": "Negative",
                    "diabetes_status": "No",
                    "is_recidivism": 0,
                    "infectivity_score": 1.0,
                }
            ]
        )

    # Optional contact-tracing aggregates (sparse)
    contact_rows = []
    for month in months:
        for h3_idx in rng.choice(h3_ids, size=max(3, n_hex // 5), replace=False):
            contact_rows.append(
                {
                    "h3_index": h3_idx,
                    "month": month,
                    "n_contacts": int(rng.integers(1, 12)),
                }
            )
    contacts = pd.DataFrame(contact_rows)

    save_data(patients, out_dir / "patients.csv")
    save_data(hexagons, out_dir / "hexagons.csv")
    save_data(contacts, out_dir / "contacts.csv")
    logger.info(
        "Synthetic data: %d patients, %d hexes, %d months -> %s",
        len(patients),
        len(hexagons),
        n_months,
        out_dir,
    )
    return patients, hexagons, contacts
