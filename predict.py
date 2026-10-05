"""Predict a student's final score from the saved model.

Usage:
    python predict.py --study-hours 12 --attendance 0.9 --previous-score 74
    python predict.py --json student.json
    python predict.py --batch students.csv

Run ``python train.py`` first: this script reads ``artifacts/model.joblib``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from student_perf.config import MODEL_PATH, score_band
from student_perf.persistence import load_metadata, load_model, predict_one

# Median-ish defaults for every raw feature, so a partial CLI is still valid.
DEFAULTS: dict[str, object] = {
    "gender": "female",
    "age": 16,
    "grade": 11,
    "school_type": "public",
    "parent_education": "some_college",
    "internet_access": 1,
    "extracurricular_hours": 2.5,
    "tutoring": 0,
    "study_hours_per_week": 9.5,
    "attendance_rate": 0.86,
    "absences": 5,
    "previous_score": 72,
    "assignments_completed": 15,
    "assignment_score": 72,
    "sleep_hours": 7.0,
    "class_size": 24,
    "teacher_quality": 3,
}

CLI_TO_COLUMN = {
    "study-hours": "study_hours_per_week",
    "attendance": "attendance_rate",
    "previous-score": "previous_score",
    "assignment-score": "assignment_score",
    "assignments": "assignments_completed",
    "absences": "absences",
    "sleep": "sleep_hours",
    "extracurricular": "extracurricular_hours",
    "class-size": "class_size",
    "teacher-quality": "teacher_quality",
    "grade": "grade",
    "age": "age",
    "tutoring": "tutoring",
    "internet": "internet_access",
    "gender": "gender",
    "school-type": "school_type",
    "parent-education": "parent_education",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a student's final score.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--json", type=Path, help="JSON file with one student's features.")
    parser.add_argument("--batch", type=Path, help="CSV file with one student per row.")
    for flag, column in CLI_TO_COLUMN.items():
        parser.add_argument(
            f"--{flag}",
            dest=column,
            default=None,
            help=f"value for {column}",
        )
    return parser.parse_args(argv)


def build_features(args: argparse.Namespace) -> dict:
    """Merge defaults with anything supplied on the command line."""
    features = dict(DEFAULTS)
    for column in CLI_TO_COLUMN.values():
        value = getattr(args, column, None)
        if value is not None:
            features[column] = value
    return features


def coerce_types(features: dict) -> dict:
    """Cast numeric-looking values so dtypes match training."""
    coerced = dict(features)
    for column, value in list(coerced.items()):
        if isinstance(value, str):
            try:
                coerced[column] = float(value)
            except ValueError:
                pass
        if column in ("grade", "teacher_quality", "assignments_completed", "absences", "class_size"):
            coerced[column] = int(round(float(coerced[column])))
    return coerced


def report(score: float, features: dict, metadata: dict) -> None:
    stats = metadata.get("target_stats", {})
    mean = stats.get("mean")
    spread = stats.get("std")

    print()
    print("=" * 52)
    print(f"Predicted final score: {score:.1f} / 100   ({score_band(score)})")
    if mean is not None:
        delta = score - mean
        percentile = ""
        if spread:
            z = delta / spread
            percentile = f"  ({z:+.2f} sd from the cohort mean)"
        print(f"  vs cohort mean {mean:.1f}: {delta:+.1f} points{percentile}")
    print("=" * 52)
    print("Profile used for the prediction:")
    for key in (
        "study_hours_per_week",
        "attendance_rate",
        "previous_score",
        "absences",
        "sleep_hours",
        "assignments_completed",
        "tutoring",
        "parent_education",
    ):
        if key in features:
            print(f"  {key:<24} {features[key]}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not Path(args.model).exists():
        print(f"Model not found at {args.model}. Run 'python train.py' first.")
        return 1

    pipeline = load_model(args.model)
    metadata = load_metadata(Path(args.model).parent / "model_metadata.json")
    feature_columns = metadata.get("feature_columns")

    if args.batch is not None:
        frame = pd.read_csv(args.batch)
        records = [
            {**DEFAULTS, **coerce_types(record)}
            for record in frame.to_dict(orient="records")
        ]
        predictions = [
            predict_one(record, pipeline, feature_columns) for record in records
        ]
        result = frame.copy()
        result["predicted_score"] = [round(value, 2) for value in predictions]
        result["band"] = [score_band(value) for value in predictions]
        print(result.to_string(index=False))
        return 0

    if args.json is not None:
        payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        payload = build_features(args)

    features = coerce_types({**DEFAULTS, **payload})
    score = predict_one(features, pipeline, feature_columns)
    report(score, features, metadata)

    print(f"\n  model: {metadata.get('model_name')} "
          f"({'R2 ' + format(metadata['test_metrics']['r2'], '.3f') if 'test_metrics' in metadata else 'n/a'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
