"""Streamlit dashboard for the student performance prediction pipeline.

Usage:
    python train.py            # once, so artifacts/model.joblib exists
    streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from predict import DEFAULTS, coerce_types
from student_perf.config import (
    CATEGORICAL_FEATURES,
    COMPARISON_PATH,
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RAW_REQUIRED_COLUMNS,
    REPORT_PATH,
    SCORE_BANDS,
    score_band,
)
from student_perf.persistence import load_metadata, load_model, predict_one

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide",
)

# Widget definitions: the slider bounds mirror the generator's own clips, so every
# value the UI can produce is a value the model was actually trained on.
SLIDERS: dict[str, tuple[str, float, float, float]] = {
    "study_hours_per_week": ("Study hours per week", 0.0, 30.0, 0.5),
    "attendance_rate": ("Attendance rate", 0.45, 1.0, 0.005),
    "absences": ("Absences this term", 0, 40, 1),
    "sleep_hours": ("Sleep hours per night", 4.0, 10.5, 0.25),
    "extracurricular_hours": ("Extracurricular hours", 0.0, 14.0, 0.5),
    "previous_score": ("Previous score", 20, 100, 1),
    "assignments_completed": ("Assignments completed", 0, 20, 1),
    "assignment_score": ("Assignment score", 20, 100, 1),
    "class_size": ("Class size", 15, 35, 1),
    "teacher_quality": ("Teacher quality (1-5)", 1, 5, 1),
}

CHOICES: dict[str, list] = {
    "parent_education": ["high_school", "some_college", "bachelor", "postgraduate"],
    "school_type": ["public", "private"],
    "gender": ["female", "male"],
}

FLAGS = ["internet_access", "tutoring"]

GRADES = [9, 10, 11, 12]

FIGURE_CAPTIONS = {
    "01_feature_distributions.png": "Distributions behind the scaling choices",
    "02_target_distribution.png": "Target spread, mean and median",
    "03_correlation_heatmap.png": "Correlation with the target",
    "04_actual_vs_predicted.png": "45-degree line, bias and spread of the best model",
    "05_residuals.png": "Heteroscedasticity and systematic error",
    "06_model_comparison.png": "MAE and R-squared per model",
    "07_feature_importance.png": "What drives the prediction",
    "08_error_distribution.png": "Absolute-error histogram",
    "09_score_by_category.png": "Mean score per category level",
}


# --------------------------------------------------------------------------- #
# cached loaders
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load the fitted pipeline once per server process."""
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return load_metadata(METADATA_PATH)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_comparison() -> pd.DataFrame | None:
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def get_report() -> str:
    if not REPORT_PATH.exists():
        return ""
    return REPORT_PATH.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def get_cohort_profile() -> dict | None:
    """Median profile of the training cohort, used as a reference point."""
    if not DEFAULT_DATA_PATH.exists():
        return None
    frame = pd.read_csv(DEFAULT_DATA_PATH)
    profile: dict = {}
    for column in NUMERIC_FEATURES:
        profile[column] = round(float(frame[column].median()), 4)
    for column in CATEGORICAL_FEATURES:
        profile[column] = frame[column].mode().iloc[0]
    return profile


# --------------------------------------------------------------------------- #
# prediction helpers
# --------------------------------------------------------------------------- #


def score(profile: dict, pipeline, feature_columns: list[str]) -> float:
    """Score one raw feature dict through the saved pipeline."""
    return predict_one(coerce_types(profile), pipeline, feature_columns)


def score_many(profiles: list[dict], pipeline, feature_columns: list[str]) -> list[float]:
    return [score(profile, pipeline, feature_columns) for profile in profiles]


def model_weights(pipeline) -> pd.DataFrame:
    """Feature weights of the fitted model, named after the transformed columns."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = [
        name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
    ]
    if hasattr(model, "coef_"):
        values = np.ravel(np.asarray(model.coef_))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    else:
        return pd.DataFrame(columns=["feature", "weight"])
    return (
        pd.DataFrame({"feature": names, "weight": values})
        .sort_values("weight", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def feature_label(key: str) -> str:
    """Human readable label for a raw feature column."""
    if key in SLIDERS:
        return SLIDERS[key][0]
    return key.replace("_", " ").capitalize()


def render_field(key: str) -> object:
    """Draw the control for one raw feature and return its value."""
    if key in SLIDERS:
        label, low, high, step = SLIDERS[key]
        return st.slider(
            label, float(low), float(high), float(DEFAULTS[key]), float(step)
        )
    if key in CHOICES:
        options = CHOICES[key]
        return st.selectbox(
            feature_label(key), options, options.index(DEFAULTS[key])
        )
    return int(st.checkbox(feature_label(key), value=bool(DEFAULTS[key])))


def clear_artifacts_cache() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Artifacts")
    if MODEL_PATH.exists():
        st.success(f"Model: `{MODEL_PATH.name}`")
    else:
        st.error("No trained model found.")
        st.code("python train.py", language="bash")
    st.caption(
        "Predictions come from `artifacts/model.joblib`. Re-run `python train.py` "
        "after retraining, then reload below."
    )
    if st.button("Reload artifacts"):
        clear_artifacts_cache()
        st.rerun()

pipeline = get_pipeline()
metadata = get_metadata()
metrics = get_metrics()
feature_columns = metadata.get("feature_columns")

if pipeline is None or not feature_columns:
    st.title("Student Performance Predictor")
    st.warning(
        "The model has not been trained yet. Run `python train.py` in the project "
        "root, then reload this page."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #

st.title("Student Performance Predictor")
st.caption(
    f"{metadata.get('model_name', 'model')} · trained on "
    f"{metadata.get('n_samples', 0):,} students · "
    f"test R² {metadata.get('test_metrics', {}).get('r2', float('nan')):.3f} · "
    f"test MAE {metadata.get('test_metrics', {}).get('mae', float('nan')):.2f}"
)

predict_tab, whatif_tab, batch_tab, model_tab, figures_tab = st.tabs(
    ["Predict", "What-if", "Batch", "Model", "Figures"]
)

# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #

with predict_tab:
    habits_col, prior_col, context_col = st.columns(3)

    profile: dict = dict(DEFAULTS)

    with habits_col:
        st.subheader("Study habits & attendance")
        for key in (
            "study_hours_per_week",
            "attendance_rate",
            "absences",
            "sleep_hours",
            "extracurricular_hours",
        ):
            profile[key] = render_field(key)

    with prior_col:
        st.subheader("Prior achievement")
        profile["grade"] = st.selectbox(
            "Grade", GRADES, GRADES.index(int(DEFAULTS["grade"]))
        )
        birthday_passed = st.checkbox("Birthday has passed this year", value=False)
        profile["age"] = profile["grade"] + 5 + int(birthday_passed)
        st.caption(f"Age: {profile['age']}")
        for key in (
            "previous_score",
            "assignments_completed",
            "assignment_score",
        ):
            profile[key] = render_field(key)

    with context_col:
        st.subheader("Context & support")
        for key in ("class_size", "teacher_quality"):
            profile[key] = render_field(key)
        for key in FLAGS:
            profile[key] = render_field(key)
        for key in CHOICES:
            profile[key] = render_field(key)

    st.divider()

    predicted = score(profile, pipeline, feature_columns)
    band = score_band(predicted)
    stats = metadata.get("target_stats", {})
    mean = stats.get("mean")
    spread = stats.get("std")

    score_col, band_col, mean_col, cohort_col = st.columns(4)
    score_col.metric("Predicted final score", f"{predicted:.1f}")
    band_col.metric("Band", band)
    if mean is not None:
        mean_col.metric("vs cohort mean", f"{predicted - mean:+.1f}")
        if spread:
            st.caption(
                f"Cohort mean {mean:.1f}, sd {spread:.1f} → "
                f"{(predicted - mean) / spread:+.2f} sd"
            )
    cohort = get_cohort_profile()
    if cohort is not None:
        cohort_score = score(cohort, pipeline, feature_columns)
        cohort_col.metric("vs median student", f"{predicted - cohort_score:+.1f}")
        st.caption(f"Median profile scores {cohort_score:.1f}")

    position_col, summary_col = st.columns([3, 2])
    with position_col:
        st.progress(min(max(predicted, 0.0), 100.0) / 100.0)
        st.caption(
            "  |  ".join(
                f"{label} ≥ {threshold:.0f}"
                for threshold, label in reversed(SCORE_BANDS)
            )
        )
    with summary_col:
        st.dataframe(
            pd.DataFrame(
                {"feature": list(profile), "value": list(profile.values())}
            ),
            hide_index=True,
            height=280,
        )

# --------------------------------------------------------------------------- #
# what-if
# --------------------------------------------------------------------------- #

with whatif_tab:
    st.subheader("Sensitivity")
    st.caption(
        "One feature is swept across its full range while everything else stays at "
        "the profile set on the Predict tab."
    )

    selected = st.multiselect(
        "Features to sweep",
        options=list(SLIDERS) + FLAGS + list(CHOICES),
        default=["study_hours_per_week", "attendance_rate", "previous_score"],
        format_func=feature_label,
    )

    if not selected:
        st.info("Pick at least one feature to sweep.")
    else:
        for start in range(0, len(selected), 2):
            for key in selected[start : start + 2]:
                if key in SLIDERS:
                    _, low, high, _ = SLIDERS[key]
                    values = list(np.linspace(float(low), float(high), 40))
                elif key in CHOICES:
                    values = list(CHOICES[key])
                else:
                    values = [0, 1]

                variants = [{**profile, key: value} for value in values]
                predictions = score_many(variants, pipeline, feature_columns)
                series = pd.DataFrame(
                    {"predicted_score": predictions},
                    index=[f"{value:g}" for value in values],
                )
                st.markdown(f"**{feature_label(key)}**")
                st.line_chart(series, height=200)
                st.caption(
                    f"Range {min(predictions):.1f} → {max(predictions):.1f} points "
                    f"({max(predictions) - min(predictions):.1f} spread). "
                    f"Profile currently scores {predicted:.1f}."
                )

# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #

with batch_tab:
    st.subheader("Batch prediction")
    st.caption(
        "Upload a CSV with one student per row. Missing feature columns fall back to "
        "the median profile, so a partial file still works."
    )

    uploaded = st.file_uploader("Students CSV", type="csv")
    if uploaded is None:
        st.info("No file uploaded yet.")
    else:
        frame = pd.read_csv(uploaded)
        absent = [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]
        if absent:
            st.warning(
                "Missing columns, filled with the median profile: "
                + ", ".join(absent)
            )
        records = [
            coerce_types({**DEFAULTS, **record})
            for record in frame.to_dict(orient="records")
        ]
        predictions = score_many(records, pipeline, feature_columns)

        result = frame.copy()
        result["predicted_score"] = [round(value, 2) for value in predictions]
        result["band"] = [score_band(value) for value in predictions]

        head, low, mid, high = st.columns(4)
        head.metric("Students scored", f"{len(result)}")
        low.metric("Mean score", f"{np.mean(predictions):.1f}")
        mid.metric("Lowest", f"{np.min(predictions):.1f}")
        high.metric("Highest", f"{np.max(predictions):.1f}")

        st.dataframe(result, hide_index=True, height=360)
        st.bar_chart(result["band"].value_counts())

        st.download_button(
            "Download predictions",
            result.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

with model_tab:
    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        cols = st.columns(4)
        cols[0].metric("MAE", f"{test_metrics.get('mae', float('nan')):.2f}")
        cols[1].metric("RMSE", f"{test_metrics.get('rmse', float('nan')):.2f}")
        cols[2].metric("R²", f"{test_metrics.get('r2', float('nan')):.3f}")
        cols[3].metric("Max error", f"{test_metrics.get('max_error', float('nan')):.1f}")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Model comparison")
        comparison = get_comparison()
        if comparison is not None:
            tidy = comparison[["model", "test_mae", "test_rmse", "test_r2"]].round(3)
            st.dataframe(tidy, hide_index=True)
            mae_col, r2_col = st.columns(2)
            mae_col.bar_chart(comparison.set_index("model")[["test_mae"]], height=220)
            r2_col.bar_chart(comparison.set_index("model")[["test_r2"]], height=220)
        else:
            st.info("`artifacts/model_comparison.csv` not found.")

    with right:
        st.subheader("Feature weights")
        weights = model_weights(pipeline)
        if not weights.empty:
            st.dataframe(
                weights.rename(columns={"weight": "coef / importance"}).head(15),
                hide_index=True,
            )
            st.caption(
                "Linear models contribute coefficients on the standardised scale; "
                "tree models contribute impurity-based importances."
            )
        else:
            st.info("No interpretable weights for this model class.")

    st.subheader("Target")
    if stats:
        stat_cols = st.columns(4)
        for column, name in zip(stat_cols, ("mean", "std", "min", "max")):
            column.metric(
                f"Target {name}", f"{stats.get(name, float('nan')):.1f}"
            )
    if "baseline" in metrics:
        baseline = metrics["baseline"]
        st.caption(
            f"Mean-only baseline: MAE {baseline['mae']:.2f}, "
            f"R² {baseline['r2']:.3f} — the bar every model has to beat."
        )

    with st.expander("Training metadata"):
        st.json(
            {
                key: metadata[key]
                for key in (
                    "model_name",
                    "model_class",
                    "best_params",
                    "seed",
                    "test_size",
                    "n_samples",
                    "created_at",
                    "sklearn_version",
                    "numpy_version",
                    "pandas_version",
                )
                if key in metadata
            }
        )

    with st.expander("Full training report"):
        report = get_report()
        if report:
            st.markdown(report)
        else:
            st.info("`artifacts/report.md` not found.")

# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

with figures_tab:
    st.subheader("Saved figures")
    available = (
        sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    )
    if not available:
        st.info("No figures found. Run `python train.py` to generate them.")
    else:
        for start in range(0, len(available), 2):
            columns = st.columns(2)
            for column, path in zip(columns, available[start : start + 2]):
                with column:
                    st.image(str(path))
                    st.caption(FIGURE_CAPTIONS.get(path.name, path.name))
