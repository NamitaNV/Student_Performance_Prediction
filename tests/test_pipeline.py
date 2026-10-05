"""Smoke tests covering the full pipeline end to end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from student_perf.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RAW_REQUIRED_COLUMNS,
    TARGET,
    score_band,
)
from student_perf.data import add_features, load_dataset, split_data, validate_columns
from student_perf.generate_dataset import generate
from student_perf.models import baseline_predictions, regression_metrics
from student_perf.persistence import build_metadata, predict_one, save_metadata, save_model
from student_perf.preprocess import build_pipeline, build_preprocessor

N_STUDENTS = 300


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate(n_students=N_STUDENTS, seed=7)


@pytest.fixture(scope="module")
def engineered(raw: pd.DataFrame) -> pd.DataFrame:
    return add_features(raw)


def test_generate_dataset_shape_and_schema(raw):
    assert len(raw) == N_STUDENTS
    assert set(raw.columns) == set(RAW_REQUIRED_COLUMNS) | {ID_COLUMN, TARGET}
    assert not raw.isna().any().any()
    validate_columns(raw)


def test_target_is_bounded_and_continuous(raw):
    assert raw[TARGET].between(0, 100).all()
    assert raw[TARGET].nunique() > N_STUDENTS // 2


def test_features_are_informative(raw):
    """Strongest driver must actually move the target."""
    correlation = raw["study_hours_per_week"].corr(raw[TARGET])
    assert correlation > 0.25


def test_add_features_creates_engineered_columns(engineered):
    for column in ENGINEERED_FEATURES:
        assert column in engineered.columns
        assert engineered[column].notna().all()
    assert (engineered["study_hours_per_absence"] > 0).all()


def test_split_data_is_disjoint_and_ordered(engineered):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    X_train, X_test, y_train, y_test = split_data(engineered, features=features)
    assert len(X_train) + len(X_test) == len(engineered)
    assert list(X_train.columns) == features
    assert len(set(X_train.index) & set(X_test.index)) == 0


def test_split_data_is_reproducible(engineered):
    first = split_data(engineered, seed=3)[0]
    second = split_data(engineered, seed=3)[0]
    pd.testing.assert_frame_equal(first, second)


def test_preprocessor_output_is_finite_dense(engineered):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(engineered)
    assert transformed.shape[0] == len(engineered)
    assert transformed.ndim == 2
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_categories(engineered):
    preprocessor = build_preprocessor().fit(engineered)
    sample = engineered.iloc[[0]].copy()
    sample["gender"] = "non_binary"
    assert np.isfinite(preprocessor.transform(sample)).all()


def test_preprocessor_handles_missing_values(engineered):
    preprocessor = build_preprocessor().fit(engineered)
    sample = engineered.iloc[[0, 1]].copy()
    sample.loc[0, "study_hours_per_week"] = np.nan
    sample.loc[1, "parent_education"] = np.nan
    assert np.isfinite(preprocessor.transform(sample)).all()


def test_pipeline_learns_better_than_the_mean_baseline(engineered):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    X_train, X_test, y_train, y_test = split_data(engineered, features=features)

    pipeline = build_pipeline(LinearRegression()).fit(X_train, y_train)
    metrics = regression_metrics(y_test, pipeline.predict(X_test))
    baseline = regression_metrics(y_test, baseline_predictions(y_train, y_test))

    assert metrics["r2"] > 0.5
    assert metrics["mae"] < baseline["mae"]
    assert 0 < metrics["mae"] < 50
    assert metrics["rmse"] >= metrics["mae"]


def test_random_forest_is_competitive_with_linear_regression(engineered):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    X_train, X_test, y_train, y_test = split_data(engineered, features=features)

    forest = build_pipeline(
        RandomForestRegressor(n_estimators=25, max_depth=8, random_state=1, n_jobs=-1)
    ).fit(X_train, y_train)
    linear = build_pipeline(LinearRegression()).fit(X_train, y_train)
    baseline = regression_metrics(y_test, baseline_predictions(y_train, y_test))

    forest_mae = regression_metrics(y_test, forest.predict(X_test))["mae"]
    linear_mae = regression_metrics(y_test, linear.predict(X_test))["mae"]
    assert forest_mae < baseline["mae"]
    assert forest_mae <= linear_mae + 1.5


def test_metrics_are_symmetric_in_arguments():
    values = np.array([70.0, 80.0, 90.0])
    assert regression_metrics(values, values)["mae"] == pytest.approx(0.0)
    assert regression_metrics(values, values + 5)["rmse"] == pytest.approx(5.0)


def test_baseline_predictions_match_training_mean(engineered):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    _, _, y_train, y_test = split_data(engineered, features=features)
    predictions = baseline_predictions(y_train, y_test)
    assert len(predictions) == len(y_test)
    assert np.allclose(predictions, float(y_train.mean()))


def test_validate_columns_reports_missing(raw):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(raw.drop(columns=["attendance_rate", TARGET]))


def test_load_dataset_round_trips_via_csv(tmp_path: Path, raw):
    csv_path = tmp_path / "students.csv"
    raw.to_csv(csv_path, index=False)
    reloaded = load_dataset(path=csv_path)
    pd.testing.assert_frame_equal(reloaded, raw)


def test_load_dataset_generates_when_missing(tmp_path: Path):
    csv_path = tmp_path / "nested" / "students.csv"
    frame = load_dataset(path=csv_path, n_students=50, seed=1)
    assert csv_path.exists()
    assert len(frame) == 50


def test_save_model_and_metadata_round_trip(tmp_path: Path, engineered, raw):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    pipeline = build_pipeline(LinearRegression()).fit(engineered[features], engineered[TARGET])

    model_path = save_model(pipeline, tmp_path / "model.joblib")
    metadata = build_metadata(
        model_name="LinearRegression",
        pipeline=pipeline,
        metrics=regression_metrics(engineered[TARGET], pipeline.predict(engineered[features])),
        frame=raw,
        seed=7,
        test_size=0.2,
        feature_columns=features,
    )
    metadata_path = save_metadata(metadata, tmp_path / "model_metadata.json")

    assert model_path.exists() and metadata_path.exists()
    assert json.loads(metadata_path.read_text())["feature_columns"] == features


def test_predict_one_reproduces_pipeline_output(engineered):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    pipeline = build_pipeline(LinearRegression()).fit(engineered[features], engineered[TARGET])

    record = engineered.iloc[0].to_dict()
    direct = float(pipeline.predict(engineered.iloc[[0]][features])[0])
    assert predict_one(record, pipeline, features) == pytest.approx(direct, abs=1e-9)


def test_predict_one_ignores_student_id_and_target(engineered):
    features = [c for c in engineered.columns if c not in (TARGET, ID_COLUMN)]
    pipeline = build_pipeline(LinearRegression()).fit(engineered[features], engineered[TARGET])
    record = {column: engineered.iloc[0][column] for column in RAW_REQUIRED_COLUMNS}
    assert 0 <= predict_one(record, pipeline, features) <= 100


def test_feature_lists_are_disjoint():
    numeric = set(NUMERIC_FEATURES)
    categorical = set(CATEGORICAL_FEATURES)
    assert not numeric & categorical
    assert not set(ENGINEERED_FEATURES) & (numeric | categorical)


def test_score_band_thresholds():
    assert score_band(95) == "Excellent"
    assert score_band(85) == "Very good"
    assert score_band(75) == "Good"
    assert score_band(65) == "Needs improvement"
    assert score_band(40) == "At risk"
