"""Dataset loading, validation and feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from student_perf.config import (
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    RAW_REQUIRED_COLUMNS,
    SEED,
    TARGET,
)
from student_perf.generate_dataset import generate


def validate_columns(frame: pd.DataFrame) -> None:
    """Raise a clear error when the dataset is missing required columns."""
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in frame.columns]
    if TARGET not in frame.columns:
        missing.append(TARGET)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def load_dataset(
    path: str | Path | None = None,
    n_students: int = 2000,
    seed: int = SEED,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load a CSV dataset, generating the synthetic one when it is missing."""
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    if regenerate or not csv_path.exists():
        frame = generate(n_students=n_students, seed=seed)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
    else:
        frame = pd.read_csv(csv_path)

    validate_columns(frame)
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features that capture interactions between raw columns."""
    data = frame.copy()

    study_hours = data["study_hours_per_week"].astype(float)
    attendance = data["attendance_rate"].astype(float)
    absences = data["absences"].astype(float)

    data["study_attendance_interaction"] = study_hours * attendance
    data["study_hours_per_absence"] = study_hours / (1.0 + absences)
    data["prior_performance"] = data[["previous_score", "assignment_score"]].mean(axis=1)

    return data


def split_data(
    frame: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    seed: int = SEED,
    features: list[str] | None = None,
):
    """Split an engineered frame into train/test design and target matrices."""
    if features is None:
        features = [
            column
            for column in frame.columns
            if column not in (target, ID_COLUMN)
        ]

    design = frame[features]
    outcome = frame[target]

    return train_test_split(
        design, outcome, test_size=test_size, random_state=seed, shuffle=True
    )
