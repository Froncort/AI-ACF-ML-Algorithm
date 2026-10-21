"""
Shared helpers for the TB hotspot notebook pipeline.

No database / cloud clients — load and save local CSV/Parquet only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

# Project root = folder that contains this file
ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = ROOT / "sample_data"
ARTIFACTS_DIR = ROOT / "artifacts"


def get_logger(name: str = "tb_hotspot") -> logging.Logger:
    """Simple console logger used across notebooks and pipeline modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def ensure_dirs() -> None:
    """Create local output folders if missing."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: PathLike) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_data(df: pd.DataFrame, path: PathLike, index: bool = False) -> Path:
    """Save a DataFrame as CSV or Parquet based on the file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)
    logger.info("Saved %s rows -> %s", len(df), path)
    return path
