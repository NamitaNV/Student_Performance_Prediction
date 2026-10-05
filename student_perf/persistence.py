"""Persistence helpers for the trained model and its metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from student_perf.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
)
from student_perf.data import add_features


def save_model(pipeline, path: str | Path = MODEL_PATH) -> Path:
    """Serialise the fitted pipeline to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination


def load_model(path: str | Path = MODEL_PATH):
    """Load a fitted pipeline from disk."""
    return joblib.load(Path(path))


def build_metadata(
    model_name: str,
    pipeline,
    metrics: dict[str, float],
    frame: pd.DataFrame,
    seed: int,
    test_size: float,
    best_params: dict | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """Assemble the JSON side-car describing the saved model."""
    target_values = frame[TARGET].astype(float)
    model = pipeline.named_steps["model"] if hasattr(pipeline, "named_steps") else pipeline

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_class": type(model).__name__,
        "best_params": best_params or {},
        "target": TARGET,
        "raw_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "feature_columns": list(
            feature_columns
            if feature_columns is not None
            else NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
        ),
        "seed": seed,
        "test_size": test_size,
        "n_samples": int(len(frame)),
        "test_metrics": {key: round(value, 4) for key, value in metrics.items()},
        "target_stats": {
            "mean": round(float(target_values.mean()), 3),
            "std": round(float(target_values.std()), 3),
            "min": round(float(target_values.min()), 3),
            "max": round(float(target_values.max()), 3),
        },
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def save_metadata(metadata: dict, path: str | Path = METADATA_PATH) -> Path:
    """Write the metadata side-car as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def load_metadata(path: str | Path = METADATA_PATH) -> dict:
    """Read the metadata side-car."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_one(
    features: dict, pipeline, feature_columns: list[str] | None = None
) -> float:
    """Predict the final score for a single student given raw feature values.

    The same feature engineering used at training time is applied here, so the
    caller only supplies the raw columns listed in the metadata side-car.
    """
    frame = add_features(pd.DataFrame([features]))
    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column != TARGET]
    return float(pipeline.predict(frame[feature_columns])[0])
