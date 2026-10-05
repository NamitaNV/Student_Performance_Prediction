"""Matplotlib visualisations for the analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from student_perf.config import CATEGORICAL_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, TARGET

PALETTE = {
    "LinearRegression": "#4C72B0",
    "Ridge": "#DD8452",
    "RandomForest": "#55A868",
    "GradientBoosting": "#C44E52",
}
ACCENT = "#4C72B0"


def _prepare_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finish(fig, path: Path) -> Path:
    output = _prepare_output(path)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_feature_distributions(frame: pd.DataFrame, output: Path | None = None) -> Path:
    """Histogram every raw feature so the modelling choices can be justified."""
    output = output or FIGURES_DIR / "01_feature_distributions.png"
    columns = [column for column in NUMERIC_FEATURES if column in frame.columns]

    rows = int(np.ceil(len(columns) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(18, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns):
        ax.hist(frame[column], bins=25, color=ACCENT, edgecolor="white")
        ax.set_title(f"{column}\nmean={frame[column].mean():.2f}", fontsize=10)
        ax.grid(alpha=0.25, axis="y")

    for ax in axes[len(columns) :]:
        ax.set_visible(False)

    fig.suptitle("Feature distributions", fontsize=14, y=1.005)
    fig.tight_layout()
    return _finish(fig, output)


def plot_target_distribution(frame: pd.DataFrame, output: Path | None = None) -> Path:
    """Distribution of the target variable with summary statistics."""
    output = output or FIGURES_DIR / "02_target_distribution.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame[TARGET], bins=30, color=ACCENT, edgecolor="white")
    ax.axvline(
        frame[TARGET].mean(), color="#C44E52", linestyle="--", label=f"mean = {frame[TARGET].mean():.2f}"
    )
    ax.axvline(
        frame[TARGET].median(), color="#DD8452", linestyle=":", label=f"median = {frame[TARGET].median():.2f}"
    )
    ax.set_xlabel(TARGET)
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of the final score")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    return _finish(fig, output)


def plot_correlation_heatmap(
    frame: pd.DataFrame, output: Path | None = None
) -> Path:
    """Correlation matrix over the numeric features and the target."""
    output = output or FIGURES_DIR / "03_correlation_heatmap.png"

    columns = [column for column in NUMERIC_FEATURES if column in frame.columns]
    columns.append(TARGET)
    matrix = frame[columns].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=9)
    ax.set_yticklabels(columns, fontsize=9)

    for row in range(len(columns)):
        for col in range(len(columns)):
            value = matrix.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.6 else "black",
            )

    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature correlation with the final score")
    fig.tight_layout()
    return _finish(fig, output)


def plot_actual_vs_predicted(
    y_true, y_pred, model_name: str, output: Path | None = None
) -> Path:
    """Predicted versus observed scores with the ideal 45-degree line."""
    output = output or FIGURES_DIR / "04_actual_vs_predicted.png"

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.55, color=ACCENT, edgecolors="none")

    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([low, high], [low, high], linestyle="--", color="#C44E52", label="perfect prediction")

    ax.set_xlabel("Actual final score")
    ax.set_ylabel("Predicted final score")
    ax.set_title(f"{model_name}: actual vs predicted")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return _finish(fig, output)


def plot_residuals(y_true, y_pred, model_name: str, output: Path | None = None) -> Path:
    """Residuals against the fitted values to check for bias or heteroscedasticity."""
    output = output or FIGURES_DIR / "05_residuals.png"

    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.asarray(y_true), residuals, s=18, alpha=0.55, color="#55A868", edgecolors="none")
    ax.axhline(0.0, linestyle="--", color="#C44E52")
    ax.set_xlabel("Actual final score")
    ax.set_ylabel("Residual (predicted - actual)")
    ax.set_title(f"{model_name}: residual plot")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return _finish(fig, output)


def plot_model_comparison(table: pd.DataFrame, output: Path | None = None) -> Path:
    """Bar chart comparing test MAE and R-squared across models."""
    output = output or FIGURES_DIR / "06_model_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [PALETTE.get(name, ACCENT) for name in table["model"]]

    axes[0].bar(table["model"], table["test_mae"], color=colors)
    axes[0].set_ylabel("MAE (lower is better)")
    axes[0].set_title("Mean Absolute Error on the test set")

    axes[1].bar(table["model"], table["test_r2"], color=colors)
    axes[1].set_ylabel("R-squared (higher is better)")
    axes[1].set_title("R-squared on the test set")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    return _finish(fig, output)


def plot_feature_importance(
    pipeline, output: Path | None = None, top_n: int = 15
) -> Path | None:
    """Importance of each transformed feature for a tree/linear pipeline."""
    output = output or FIGURES_DIR / "07_feature_importance.png"

    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return None

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        label = "Gini importance"
    else:
        importance = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        label = "Absolute coefficient"

    order = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 2))
    ax.barh([names[index] for index in order][::-1], importance[order][::-1], color=ACCENT)
    ax.set_xlabel(label)
    ax.set_title("Most influential features")
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    return _finish(fig, output)


def plot_error_distribution(
    y_true, y_pred, model_name: str, output: Path | None = None
) -> Path:
    """Histogram of absolute errors with the mean error highlighted."""
    output = output or FIGURES_DIR / "08_error_distribution.png"

    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, color="#DD8452", edgecolor="white")
    ax.axvline(
        errors.mean(), linestyle="--", color="#C44E52", label=f"mean = {errors.mean():.2f}"
    )
    ax.set_xlabel("Absolute error (points)")
    ax.set_ylabel("Number of students")
    ax.set_title(f"{model_name}: prediction error")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    return _finish(fig, output)


def plot_target_by_category(
    frame: pd.DataFrame, output: Path | None = None
) -> Path:
    """Average final score for each level of the categorical features."""
    output = output or FIGURES_DIR / "09_score_by_category.png"

    columns = [column for column in CATEGORICAL_FEATURES if column in frame.columns]
    fig, axes = plt.subplots(1, max(len(columns), 1), figsize=(5 * max(len(columns), 1), 4.5))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns):
        grouped = frame.groupby(column)[TARGET].mean().sort_values()
        ax.barh(grouped.index.astype(str), grouped.values, color=ACCENT)
        ax.set_xlabel("Mean final score")
        ax.set_title(f"{TARGET} by {column}")
        ax.grid(alpha=0.25, axis="x")

    for ax in axes[len(columns) :]:
        ax.set_visible(False)

    fig.tight_layout()
    return _finish(fig, output)
