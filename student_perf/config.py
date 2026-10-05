"""Central configuration: paths, column names and defaults."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

DEFAULT_DATA_PATH = DATA_DIR / "students.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
REPORT_PATH = ARTIFACTS_DIR / "report.md"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET = "final_score"
ID_COLUMN = "student_id"

BINARY_FEATURES = ["internet_access", "tutoring"]

NUMERIC_FEATURES = [
    "grade",
    "age",
    "study_hours_per_week",
    "attendance_rate",
    "absences",
    "previous_score",
    "assignments_completed",
    "assignment_score",
    "sleep_hours",
    "extracurricular_hours",
    "class_size",
    "teacher_quality",
    *BINARY_FEATURES,
]

CATEGORICAL_FEATURES = ["gender", "school_type", "parent_education"]

ENGINEERED_FEATURES = [
    "study_attendance_interaction",
    "study_hours_per_absence",
    "prior_performance",
]

RAW_REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES

SCORE_BANDS = (
    (90.0, "Excellent"),
    (80.0, "Very good"),
    (70.0, "Good"),
    (60.0, "Needs improvement"),
    (0.0, "At risk"),
)


def score_band(score: float) -> str:
    """Map a predicted score (0-100) onto a human readable band."""
    for threshold, label in SCORE_BANDS:
        if score >= threshold:
            return label
    return SCORE_BANDS[-1][1]
