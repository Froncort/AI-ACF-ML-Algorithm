"""
Ingestion for real Nikshay-style TB program exports (Notification Register /
Contact Tracing Register, as downloaded from the Nikshay portal).

WHY THIS MODULE EXISTS
-----------------------
Government TB registers (any Indian state/district) share three quirks that
break a naive `pd.read_csv`:

1. **Report metadata rows before the header.** Nikshay exports start with
   ~5 rows like "@Report Name, Notification Register" and
   "Data Filter Criteria, State = Maharashtra, ...". The real header
   (`Spectrum_Current_State`, `Episode_ID`, ...) is usually row 5 (0-indexed).
2. **Health-system hierarchy, not lat/lon.** Cases are tagged by
   State -> District -> TB Unit (TBU) -> Health Facility (HF/PHI), plus a
   free-text `Address` field. There is **no coordinate column** to build an
   H3 index directly.
3. **Locale-specific dates** such as `1/29/25` (M/D/YY) instead of ISO.

This module handles all three generically, so it works on *any* district's
export with the same Nikshay report structure -- not just one program's data.

GEOCODING STRATEGY (facility-centroid, not per-patient) — mirrors V1
---------------------------------------------------------------------
Production AI-ACF ingestion (V1) uses the **Google Maps Geocoding API** to
turn facility / address text into lat/lng, then H3. Geocoding every patient's
free-text `Address` is slow and unreliable; instead we geocode **once per
unique health facility** (`Spectrum_Current_HF`) and reuse that centroid for
every patient at that facility:

- Far fewer network calls (dozens of facilities vs thousands of patients)
- Reproducible: save the centroid table once, reuse for every future export
- Editable: you can manually correct/fill any row in the centroid CSV

Providers in ``geocode_facility_table``:
  - ``provider="google"`` (default) — same as V1; needs ``GOOGLE_MAPS_API_KEY``
  - ``provider="nominatim"`` — free OSM fallback via geopy

If you have real GPS coordinates per patient (e.g. from a field app), skip
geocoding entirely and merge them directly -- see `assign_h3_from_latlon`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import logger

PathLike = "str | Path"

# ---------------------------------------------------------------------------
# 1. Column mapping: Nikshay Notification Register -> our internal schema
# ---------------------------------------------------------------------------
# Left = exact column name as it appears in the Nikshay CSV export.
# Right = internal name used by the rest of the pipeline (pipeline/features.py).
# Extend this dict if your export has extra columns you want to keep.
NOTIFICATION_REGISTER_COLUMN_MAP: Dict[str, str] = {
    "Episode_ID": "patient_id",
    "Patient_Name": "patient_name",
    "Address": "address",
    "Spectrum_Diagnosis_Date": "diagnosis_date",
    "Treatment_Initiation_Date": "treatment_initiation_date",
    "Age": "age",
    "Gender": "gender",
    "ACF_PCF": "case_finding_mode",
    "Type_of_Case": "type_of_case",  # "New" / "Recurrent" / "Relapse" ...
    "Site_of_TBDisease": "site_of_disease",
    "Status_of_HIV": "hiv_status",
    "Status_of_Diabetes": "diabetes_status",
    "Bacteriologically_Confirmed": "bacteriologically_confirmed",
    "Keypopulation": "key_population",
    "Treatment_Outcome": "treatment_outcome",
    # Facility / geography hierarchy used for facility-centroid geocoding
    "Spectrum_Current_State": "state",
    "Spectrum_Current_District": "district",
    "Spectrum_Current_TBU": "tb_unit",
    "Spectrum_Current_HF": "facility_name",
}

# Substrings in `Type_of_Case` that indicate prior treatment (used for is_recidivism).
# Nikshay wording varies by year/state (e.g. "Retreatment: Recurrent",
# "Retreatment: Others", "Relapse") -- matching "retreatment"/"relapse"/
# "recurrent" as substrings is more robust than an exact-value list.
RECIDIVISM_KEYWORDS = ("retreatment", "relapse", "recurrent")


def _detect_header_row(path: "PathLike", probe_rows: int = 15, min_columns: int = 10) -> int:
    """
    Find the real header row in a Nikshay export by scanning the first
    `probe_rows` lines and returning the first one with more than
    `min_columns` non-empty cells (the metadata rows above it have 1-3).
    """
    import csv

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= probe_rows:
                break
            non_empty = sum(1 for c in row if c.strip())
            if non_empty >= min_columns:
                logger.info("Detected header row at index %d (%d populated columns)", i, non_empty)
                return i
    raise ValueError(f"Could not find a header row in the first {probe_rows} rows of {path}")


def load_raw_register(path: "PathLike", header_row: Optional[int] = None) -> pd.DataFrame:
    """
    Load a raw Nikshay-style register CSV, automatically skipping the
    report-metadata rows above the real header.

    Set `header_row` explicitly if auto-detection picks the wrong row
    (e.g. an unusual export with unusually few populated metadata cells).
    """
    path = Path(path)
    if header_row is None:
        header_row = _detect_header_row(path)
    df = pd.read_csv(path, skiprows=header_row, low_memory=False)
    df.columns = df.columns.str.strip()
    logger.info("Loaded raw register: %d rows x %d columns from %s", len(df), df.shape[1], path.name)
    return df


def _parse_flexible_date(series: pd.Series) -> pd.Series:
    """Nikshay dates are usually M/D/YY (e.g. '1/29/25'); fall back to generic parsing."""
    parsed = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    still_missing = parsed.isna() & series.notna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(series[still_missing], errors="coerce")
    return parsed


def standardize_patients(
    raw_df: pd.DataFrame,
    column_map: Dict[str, str] = NOTIFICATION_REGISTER_COLUMN_MAP,
) -> pd.DataFrame:
    """
    Rename known Nikshay columns to the internal schema, parse dates, and
    derive the binary flags the feature/target pipeline expects
    (`is_recidivism`, `infectivity_score`).

    Unknown / unmapped columns are dropped -- extend `column_map` if you
    need to keep something extra.
    """
    present = {k: v for k, v in column_map.items() if k in raw_df.columns}
    missing = [k for k in column_map if k not in raw_df.columns]
    if missing:
        logger.warning("Columns not found in this export (skipped): %s", missing)

    df = raw_df[list(present.keys())].rename(columns=present).copy()

    if "diagnosis_date" in df.columns:
        df["diagnosis_date"] = _parse_flexible_date(df["diagnosis_date"])
    if "treatment_initiation_date" in df.columns:
        df["treatment_initiation_date"] = _parse_flexible_date(df["treatment_initiation_date"])

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "type_of_case" in df.columns:
        case_type_lower = df["type_of_case"].astype(str).str.strip().str.lower()
        df["is_recidivism"] = case_type_lower.str.contains("|".join(RECIDIVISM_KEYWORDS), na=False).astype(int)
    else:
        df["is_recidivism"] = 0

    # Infectivity weight mirrors production heuristic: pulmonary cases are
    # the main transmission risk; extra-pulmonary and unknown sites less so.
    if "site_of_disease" in df.columns:
        site_weight = {"Pulmonary": 1.0, "Extra Pulmonary": 0.2}
        df["infectivity_score"] = df["site_of_disease"].map(site_weight).fillna(0.6)
    else:
        df["infectivity_score"] = 1.0

    before = len(df)
    if "patient_id" in df.columns:
        df = df[df["patient_id"].notna()]
    if "diagnosis_date" in df.columns:
        df = df[df["diagnosis_date"].notna()]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows missing patient_id / diagnosis_date", dropped)

    logger.info(
        "Standardized patients: %d rows | date range %s -> %s",
        len(df),
        df["diagnosis_date"].min().date() if len(df) else "n/a",
        df["diagnosis_date"].max().date() if len(df) else "n/a",
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Facility-centroid geocoding (one lookup per facility, not per patient)
# ---------------------------------------------------------------------------

def build_facility_table(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique facilities (+ district/TBU context) that need a
    lat/lon centroid. Save this, fill in coordinates once, and reuse it
    for every future monthly export from the same district.
    """
    cols = [c for c in ["facility_name", "tb_unit", "district", "state"] if c in patients.columns]
    if "facility_name" not in cols:
        raise ValueError("standardize_patients() output has no 'facility_name' column to geocode")
    facilities = patients[cols].drop_duplicates().reset_index(drop=True)
    facilities["lat"] = np.nan
    facilities["lng"] = np.nan
    logger.info("Found %d unique facilities needing a lat/lon centroid", len(facilities))
    return facilities


def geocode_facility_table(
    facilities: pd.DataFrame,
    query_template: str = "{facility_name}, {tb_unit}, {district}, {state}, India",
    provider: str = "google",
    rate_limit_seconds: Optional[float] = None,
    user_agent: str = "tb-hotspot-notebooks",
    google_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Forward-geocode unique facilities → lat/lng (V1-style facility centroids).

    Providers (mirrors production ingestion):
      - ``google`` (default): Google Maps Geocoding API — set ``GOOGLE_MAPS_API_KEY``
        or pass ``google_api_key``. Requires ``pip install googlemaps``.
      - ``nominatim``: OpenStreetMap Nominatim via geopy (free, ~1 req/s).
        Requires ``pip install geopy``.

    Safe to re-run: only fills rows where lat/lng are still missing.
    Failed rows stay NaN — correct them manually and re-run.
    """
    import os

    facilities = facilities.copy()
    to_geocode = facilities[facilities["lat"].isna()].index
    if len(to_geocode) == 0:
        logger.info("All facilities already have lat/lng — nothing to geocode")
        return facilities

    provider = (provider or "google").strip().lower()
    if provider == "google":
        return _geocode_facilities_google(
            facilities, to_geocode, query_template,
            api_key=google_api_key or os.environ.get("GOOGLE_MAPS_API_KEY"),
            rate_limit_seconds=rate_limit_seconds if rate_limit_seconds is not None else 0.05,
        )
    if provider == "nominatim":
        return _geocode_facilities_nominatim(
            facilities, to_geocode, query_template, user_agent,
            rate_limit_seconds=rate_limit_seconds if rate_limit_seconds is not None else 1.1,
        )
    raise ValueError(f"Unknown geocode provider={provider!r}; use 'google' or 'nominatim'")


def _geocode_facilities_google(
    facilities: pd.DataFrame,
    to_geocode,
    query_template: str,
    api_key: Optional[str],
    rate_limit_seconds: float,
) -> pd.DataFrame:
    """Google Maps Geocoding API — same provider as V1 AI-ACF ingestion."""
    if not api_key:
        logger.warning(
            "GOOGLE_MAPS_API_KEY not set. Skipping Google geocoding — "
            "export the key, or pass google_api_key=, or use provider='nominatim', "
            "or fill facilities['lat']/['lng'] manually."
        )
        return facilities
    try:
        import googlemaps
    except ImportError:
        logger.warning(
            "googlemaps is not installed (pip install googlemaps). "
            "Skipping automatic geocoding."
        )
        return facilities

    client = googlemaps.Client(key=api_key)
    logger.info("Geocoding %d facilities via Google Maps...", len(to_geocode))
    for idx in to_geocode:
        row = facilities.loc[idx]
        query = query_template.format(
            **{k: row.get(k, "") for k in ("facility_name", "tb_unit", "district", "state")}
        )
        try:
            results = client.geocode(query)
            if results:
                loc = results[0]["geometry"]["location"]
                facilities.loc[idx, "lat"] = loc["lat"]
                facilities.loc[idx, "lng"] = loc["lng"]
            else:
                logger.warning("No Google geocode match for: %s", query)
        except Exception as exc:  # network / quota / invalid key
            logger.warning("Google geocoding failed for %s: %s", query, exc)
        time.sleep(rate_limit_seconds)

    found = facilities["lat"].notna().sum()
    logger.info("Geocoded %d/%d facilities (Google)", found, len(facilities))
    return facilities


def _geocode_facilities_nominatim(
    facilities: pd.DataFrame,
    to_geocode,
    query_template: str,
    user_agent: str,
    rate_limit_seconds: float,
) -> pd.DataFrame:
    """Free OSM Nominatim fallback when no Google Maps key is available."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderServiceError
    except ImportError:
        logger.warning(
            "geopy is not installed (pip install geopy). Skipping Nominatim geocoding — "
            "fill facilities['lat']/['lng'] manually, then continue."
        )
        return facilities

    geolocator = Nominatim(user_agent=user_agent)
    logger.info(
        "Geocoding %d facilities via Nominatim (~%.1fs each)...",
        len(to_geocode), rate_limit_seconds,
    )
    for idx in to_geocode:
        row = facilities.loc[idx]
        query = query_template.format(
            **{k: row.get(k, "") for k in ("facility_name", "tb_unit", "district", "state")}
        )
        try:
            location = geolocator.geocode(query, timeout=10)
            if location:
                facilities.loc[idx, "lat"] = location.latitude
                facilities.loc[idx, "lng"] = location.longitude
            else:
                logger.warning("No geocode match for: %s", query)
        except GeocoderServiceError as exc:
            logger.warning("Geocoding failed for %s: %s", query, exc)
        time.sleep(rate_limit_seconds)  # respect Nominatim usage policy

    found = facilities["lat"].notna().sum()
    logger.info("Geocoded %d/%d facilities (Nominatim)", found, len(facilities))
    return facilities


def assign_h3_from_facility_table(
    patients: pd.DataFrame,
    facilities: pd.DataFrame,
    resolution: int = 10,
) -> pd.DataFrame:
    """
    Merge facility lat/lng onto patients and convert to an H3 index at the
    given resolution. Patients whose facility has no lat/lng (not yet
    geocoded, or geocoding failed) are dropped -- log the count so you know
    how much coverage you have.
    """
    import h3

    merge_cols = [c for c in ["facility_name", "tb_unit", "district", "state"] if c in patients.columns]
    merged = patients.merge(facilities[merge_cols + ["lat", "lng"]], on=merge_cols, how="left")

    has_coords = merged["lat"].notna() & merged["lng"].notna()
    logger.info("%d/%d patients matched to a geocoded facility", has_coords.sum(), len(merged))

    def to_h3(row):
        if pd.isna(row["lat"]) or pd.isna(row["lng"]):
            return None
        if hasattr(h3, "latlng_to_cell"):
            return h3.latlng_to_cell(row["lat"], row["lng"], resolution)
        return h3.geo_to_h3(row["lat"], row["lng"], resolution)  # older h3-py API

    merged["h3_index"] = merged.apply(to_h3, axis=1)
    dropped = merged["h3_index"].isna().sum()
    if dropped:
        logger.warning(
            "Dropping %d patients with no H3 index (facility not geocoded). "
            "Fill in facilities['lat']/['lng'] for better coverage and re-run.",
            dropped,
        )
    return merged[merged["h3_index"].notna()].reset_index(drop=True)


def assign_h3_from_latlon(df: pd.DataFrame, lat_col: str = "lat", lng_col: str = "lng", resolution: int = 10) -> pd.DataFrame:
    """
    If you already have per-patient GPS coordinates (skip facility
    geocoding entirely), convert them straight to an H3 index.
    """
    import h3

    out = df.copy()

    def to_h3(row):
        if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
            return None
        if hasattr(h3, "latlng_to_cell"):
            return h3.latlng_to_cell(row[lat_col], row[lng_col], resolution)
        return h3.geo_to_h3(row[lat_col], row[lng_col], resolution)

    out["h3_index"] = out.apply(to_h3, axis=1)
    return out


def build_hexagon_table(patients_with_h3: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the `hexagons` table (one row per H3 cell) required by
    `pipeline.features.build_feature_table` from geocoded patients.
    Adds a simple case-density proxy since a real slum/vulnerability layer
    is program-specific and not something this generic loader can infer.
    """
    hexagons = (
        patients_with_h3.groupby("h3_index")
        .agg(lat=("lat", "mean"), lng=("lng", "mean"), n_cases_total=("h3_index", "count"))
        .reset_index()
    )
    max_cases = max(hexagons["n_cases_total"].max(), 1)
    hexagons["slum_density"] = (hexagons["n_cases_total"] / max_cases).clip(0, 1)
    return hexagons.drop(columns=["n_cases_total"])
