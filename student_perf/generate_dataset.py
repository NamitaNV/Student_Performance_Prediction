"""Generate a synthetic student-performance dataset.

The generator produces a realistic secondary-school dataset (study habits,
attendance, prior achievement, school context) where the final score is a
noisy, non-linear function of those features. It exists so the project is
self-contained and reproducible offline; swap in your own CSV with the same
columns to run the pipeline on real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from student_perf.config import DEFAULT_DATA_PATH, SEED, TARGET

GENDERS = ["female", "male"]
SCHOOL_TYPES = ["public", "private"]
PARENT_EDUCATION = ["high_school", "some_college", "bachelor", "postgraduate"]

PARENT_EDUCATION_BONUS = {
    "high_school": 0.0,
    "some_college": 1.0,
    "bachelor": 2.0,
    "postgraduate": 3.0,
}

# Coefficients of the simulated grading process, in "points per unit".
COEFFICIENTS = {
    "study_hour": 1.35,
    "attendance": 40.0,
    "assignment_completed": 0.30,
    "previous_score": 0.45,
    "assignment_score": 0.25,
    "sleep": 1.20,
    "extracurricular": -1.10,
    "tutoring": 3.20,
    "teacher_quality": 1.10,
    "internet": 1.80,
    "absences_per_10": -3.00,
    "class_size": -0.12,
    "grade": 1.60,
    "interaction": 0.06,
    "private_school": 1.20,
    "intercept": 48.0,
    "noise_sd": 3.5,
}


def generate(n_students: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Return a synthetic student dataset with a ``final_score`` column."""
    if n_students <= 0:
        raise ValueError("n_students must be a positive integer")

    rng = np.random.default_rng(seed)
    c = COEFFICIENTS

    grade = rng.choice([9, 10, 11, 12], size=n_students, p=[0.26, 0.26, 0.24, 0.24])
    age = grade + 5 + rng.integers(0, 2, size=n_students)
    gender = rng.choice(GENDERS, size=n_students, p=[0.49, 0.51])
    school_type = rng.choice(SCHOOL_TYPES, size=n_students, p=[0.78, 0.22])
    parent_education = rng.choice(
        PARENT_EDUCATION, size=n_students, p=[0.24, 0.30, 0.28, 0.18]
    )

    internet_probability = np.where(parent_education == "postgraduate", 0.88, 0.70)
    internet_access = (rng.random(n_students) < internet_probability).astype(int)

    study_hours = np.round(np.clip(rng.normal(9.5, 3.6, n_students), 0, 30), 1)
    attendance_rate = np.round(np.clip(rng.normal(0.86, 0.09, n_students), 0.45, 1.0), 3)
    absences = np.clip(
        np.round((1.0 - attendance_rate) * 36 + rng.normal(0, 1.5, n_students)), 0, 40
    ).astype(int)

    previous_score = np.clip(
        np.round(
            72
            + rng.normal(0, 11, n_students)
            + 0.6 * study_hours
            - 8 * (1.0 - attendance_rate)
        ),
        20,
        100,
    )
    assignments_completed = np.clip(
        np.round(rng.normal(15, 4, n_students)), 0, 20
    ).astype(int)
    assignment_score = np.clip(
        np.round(previous_score + rng.normal(2, 7, n_students)), 20, 100
    )

    sleep_hours = np.round(np.clip(rng.normal(7.0, 1.1, n_students), 4.0, 10.5), 1)
    extracurricular_hours = np.round(
        np.clip(rng.normal(2.5, 1.8, n_students), 0, 14), 1
    )
    class_size = np.clip(np.round(rng.normal(24, 4, n_students)), 15, 35).astype(int)
    teacher_quality = np.clip(np.round(rng.normal(3.4, 1.0, n_students)), 1, 5).astype(int)

    tutoring_probability = np.clip(0.45 - (previous_score - 72) / 100.0, 0.05, 0.60)
    tutoring = (rng.random(n_students) < tutoring_probability).astype(int)

    parent_bonus = np.array(
        [PARENT_EDUCATION_BONUS[level] for level in parent_education], dtype=float
    )
    private_school = np.where(school_type == "private", c["private_school"], 0.0)

    final_score = (
        c["intercept"]
        + c["grade"] * (grade - 9)
        + c["study_hour"] * study_hours
        + c["attendance"] * (attendance_rate - 0.85)
        + c["assignment_completed"] * assignments_completed
        + c["previous_score"] * (previous_score - 72.0)
        + c["assignment_score"] * (assignment_score - 72.0)
        + c["sleep"] * (sleep_hours - 7.0)
        + c["extracurricular"] * (extracurricular_hours - 2.5)
        + c["tutoring"] * tutoring
        + c["teacher_quality"] * (teacher_quality - 3)
        + c["internet"] * internet_access
        + c["absences_per_10"] * (absences / 10.0)
        + c["class_size"] * (class_size - 24)
        + c["interaction"] * study_hours * attendance_rate
        + parent_bonus
        + private_school
        + rng.normal(0.0, c["noise_sd"], n_students)
    )
    final_score = np.round(np.clip(final_score, 0, 100), 1)

    return pd.DataFrame(
        {
            "student_id": [f"S{index:05d}" for index in range(1, n_students + 1)],
            "gender": gender,
            "age": age,
            "grade": grade,
            "school_type": school_type,
            "parent_education": parent_education,
            "internet_access": internet_access,
            "extracurricular_hours": extracurricular_hours,
            "tutoring": tutoring,
            "study_hours_per_week": study_hours,
            "attendance_rate": attendance_rate,
            "absences": absences,
            "previous_score": previous_score,
            "assignments_completed": assignments_completed,
            "assignment_score": assignment_score,
            "sleep_hours": sleep_hours,
            "class_size": class_size,
            "teacher_quality": teacher_quality,
            TARGET: final_score,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset.")
    parser.add_argument("--n-students", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)

    frame = generate(n_students=args.n_students, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Wrote {len(frame)} rows x {frame.shape[1]} columns to {args.output}")
    print(frame[TARGET].describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
