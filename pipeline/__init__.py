"""Distilled TB hotspot ML pipeline (dataset-agnostic, local files only)."""

from pipeline.sample_data import generate_synthetic_dataset, DATA_CONTRACT
from pipeline.ingest import (
    load_raw_register,
    standardize_patients,
    build_facility_table,
    geocode_facility_table,
    assign_h3_from_facility_table,
    assign_h3_from_latlon,
    build_hexagon_table,
    NOTIFICATION_REGISTER_COLUMN_MAP,
)
from pipeline.features import build_feature_table
from pipeline.targets import create_future_target
from pipeline.dataset import prepare_train_test, get_feature_columns
from pipeline.training import train_two_stage_models, predict_two_stage
from pipeline.evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_operational,
    walk_forward_auc,
)
from pipeline.inference import (
    assign_risk_tier,
    generate_recommendations,
    export_dashboard_json,
)

__all__ = [
    "DATA_CONTRACT",
    "generate_synthetic_dataset",
    "load_raw_register",
    "standardize_patients",
    "build_facility_table",
    "geocode_facility_table",
    "assign_h3_from_facility_table",
    "assign_h3_from_latlon",
    "build_hexagon_table",
    "NOTIFICATION_REGISTER_COLUMN_MAP",
    "build_feature_table",
    "create_future_target",
    "prepare_train_test",
    "get_feature_columns",
    "train_two_stage_models",
    "predict_two_stage",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_operational",
    "walk_forward_auc",
    "assign_risk_tier",
    "generate_recommendations",
    "export_dashboard_json",
]
