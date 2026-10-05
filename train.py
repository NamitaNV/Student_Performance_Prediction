"""End-to-end training script for the student performance predictor.

Usage:
    python train.py                       # full pipeline on the synthetic dataset
    python train.py --tune                # add grid-search hyperparameter tuning
    python train.py --data data/mine.csv  # train on your own dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from student_perf.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    DEFAULT_DATA_PATH,
    ENGINEERED_FEATURES,
    ID_COLUMN,
    RAW_REQUIRED_COLUMNS,
    SEED,
    TARGET,
    TEST_SIZE,
)
from student_perf.data import add_features, load_dataset, split_data
from student_perf.models import baseline_predictions, regression_metrics, train_and_evaluate
from student_perf.persistence import build_metadata, save_metadata, save_model
from student_perf.plots import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_error_distribution,
    plot_feature_distributions,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_target_by_category,
    plot_target_distribution,
)

FIGURE_NAMES = {
    "distributions": "01_feature_distributions.png",
    "target": "02_target_distribution.png",
    "correlation": "03_correlation_heatmap.png",
    "by_category": "09_score_by_category.png",
    "actual_vs_predicted": "04_actual_vs_predicted.png",
    "residuals": "05_residuals.png",
    "comparison": "06_model_comparison.png",
    "importance": "07_feature_importance.png",
    "errors": "08_error_distribution.png",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate models that predict a student's final score."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input CSV path.")
    parser.add_argument(
        "--regenerate", action="store_true", help="Regenerate the synthetic dataset first."
    )
    parser.add_argument(
        "--n-students", type=int, default=2000, help="Rows to generate when synthesizing."
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--tune", action="store_true", help="Run grid search for the tunable models."
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION")
    print("=" * 70)

    # ------------------------------------------------------------------ data
    print("\n[1/6] Loading data")
    raw = load_dataset(
        path=args.data,
        n_students=args.n_students,
        seed=args.seed,
        regenerate=args.regenerate,
    )
    print(f"  rows: {len(raw)}   columns: {raw.shape[1]}")
    print(f"  target '{TARGET}': mean={raw[TARGET].mean():.2f}  sd={raw[TARGET].std():.2f}")

    frame = add_features(raw)
    feature_columns = [
        column for column in frame.columns if column not in (TARGET, ID_COLUMN)
    ]
    X_train, X_test, y_train, y_test = split_data(
        frame, features=feature_columns, test_size=args.test_size, seed=args.seed
    )
    print(f"  engineered features: {', '.join(ENGINEERED_FEATURES)}")
    print(f"  train/test split: {X_train.shape[0]} / {X_test.shape[0]}")

    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Engineered frame is missing columns: {missing}")

    # ---------------------------------------------------- exploratory plots
    print("\n[2/6] Generating exploratory figures")
    exploratory = {
        "distributions": plot_feature_distributions(
            raw, figures_dir / FIGURE_NAMES["distributions"]
        ),
        "target": plot_target_distribution(raw, figures_dir / FIGURE_NAMES["target"]),
        "correlation": plot_correlation_heatmap(
            raw, figures_dir / FIGURE_NAMES["correlation"]
        ),
        "by_category": plot_target_by_category(raw, figures_dir / FIGURE_NAMES["by_category"]),
    }
    for name, path in exploratory.items():
        print(f"  {name:<18} -> {path}")

    # -------------------------------------------------------------- training
    print(f"\n[3/6] Training and comparing models ({args.cv_folds}-fold CV)")
    trained, table = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        tune=args.tune,
    )

    baseline = regression_metrics(y_test, baseline_predictions(y_train, y_test))
    output_dir.joinpath("model_comparison.csv").write_text(
        table.to_csv(index=False), encoding="utf-8"
    )
    print("\n  model comparison (held-out test split):")
    print(
        table[
            ["model", "test_mae", "test_rmse", "test_r2", "cv_mae", "cv_rmse", "cv_r2", "fit_seconds"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(f"\n  baseline (predict the training mean): MAE {baseline['mae']:.3f} | R2 {baseline['r2']:.3f}")

    # ------------------------------------------------------ best model + eval
    best = trained[0]
    predictions = best.pipeline.predict(X_test)
    print(f"\n[4/6] Best model: {best.name}")
    if best.best_params:
        print(f"  tuned parameters: {best.best_params}")
    print(f"  MAE  {best.metrics['mae']:.3f} points ({best.metrics['mae'] / float(y_train.mean()) * 100:.1f}% of the mean score)")
    print(f"  RMSE {best.metrics['rmse']:.3f}")
    print(f"  R2   {best.metrics['r2']:.3f}")
    print(f"  largest single error {best.metrics['max_error']:.2f} points")

    residuals = pd.DataFrame(
        {
            "actual": np.asarray(y_test, dtype=float),
            "predicted": np.asarray(predictions, dtype=float),
        }
    )
    residuals["absolute_error"] = (residuals["predicted"] - residuals["actual"]).abs()
    print("\n  sample predictions:")
    print(residuals.head(5).round(2).to_string(index=False))

    diagnostic = {
        "actual_vs_predicted": plot_actual_vs_predicted(
            y_test, predictions, best.name, figures_dir / FIGURE_NAMES["actual_vs_predicted"]
        ),
        "residuals": plot_residuals(
            y_test, predictions, best.name, figures_dir / FIGURE_NAMES["residuals"]
        ),
        "comparison": plot_model_comparison(
            table, figures_dir / FIGURE_NAMES["comparison"]
        ),
        "errors": plot_error_distribution(
            y_test, predictions, best.name, figures_dir / FIGURE_NAMES["errors"]
        ),
    }
    importance_path = plot_feature_importance(
        best.pipeline, figures_dir / FIGURE_NAMES["importance"]
    )
    if importance_path is not None:
        diagnostic["importance"] = importance_path
    for name, path in diagnostic.items():
        print(f"  {name:<18} -> {path}")

    # ---------------------------------------------------------- final model
    print("\n[5/6] Refitting the best model on the full dataset")
    final_pipeline = clone(best.pipeline)
    final_pipeline.fit(frame[feature_columns], frame[TARGET])

    model_path = save_model(final_pipeline, output_dir / "model.joblib")
    metadata = build_metadata(
        model_name=best.name,
        pipeline=final_pipeline,
        metrics=best.metrics,
        frame=raw,
        seed=args.seed,
        test_size=args.test_size,
        best_params=best.best_params,
        feature_columns=feature_columns,
    )
    metadata_path = save_metadata(metadata, output_dir / "model_metadata.json")
    print(f"  model    -> {model_path}")
    print(f"  metadata -> {metadata_path}")

    # -------------------------------------------------------------- report
    print("\n[6/6] Writing metrics and report")
    metrics_payload = {
        "target": TARGET,
        "n_samples": int(len(raw)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "tuned": bool(args.tune),
        "baseline": {key: round(value, 4) for key, value in baseline.items()},
        "best_model": best.name,
        "best_model_metrics": {key: round(value, 4) for key, value in best.metrics.items()},
        "best_model_cv_metrics": {key: round(value, 4) for key, value in best.cv_metrics.items()},
        "best_params": best.best_params,
        "models": json_safe_records(table),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    report_path = write_report(
        output_dir / "report.md", metadata, metrics_payload, table
    )
    print(f"  metrics  -> {metrics_path}")
    print(f"  report   -> {report_path}")

    print("\nDone. Score a new student with:")
    print("  python predict.py --study-hours 12 --attendance 0.9 --previous-score 74")
    return 0


def json_safe_records(table: pd.DataFrame) -> list[dict]:
    """Convert a comparison table into JSON-serialisable records."""
    records = []
    for record in table.to_dict(orient="records"):
        safe: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                safe[key] = None if pd.isna(value) else float(value)
            else:
                safe[key] = None if value is None else str(value)
        records.append(safe)
    return records


def _markdown_table(table: pd.DataFrame) -> list[str]:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in record) + " |"
        for record in table.to_dict(orient="records")
    ]
    return [header, divider, *rows]


def write_report(path: Path, metadata: dict, metrics: dict, table: pd.DataFrame) -> Path:
    """Render a concise Markdown summary of the run."""
    best = metrics["best_model_metrics"]
    cv = metrics["best_model_cv_metrics"]
    baseline = metrics["baseline"]

    lines = [
        "# Student Performance Prediction - Run Report",
        "",
        f"- Generated: {metadata['created_at']}",
        f"- Best model: **{metrics['best_model']}** ({metadata['model_class']})",
        f"- Samples: {metrics['n_samples']} ({metrics['n_train']} train / {metrics['n_test']} test)",
        f"- scikit-learn: {metadata['sklearn_version']}",
        "",
        "## Metrics on the held-out test set",
        "",
        "| Metric | Best model | Mean-only baseline |",
        "| --- | --- | --- |",
        f"| MAE | {best['mae']:.3f} | {baseline['mae']:.3f} |",
        f"| RMSE | {best['rmse']:.3f} | {baseline['rmse']:.3f} |",
        f"| R-squared | {best['r2']:.3f} | {baseline['r2']:.3f} |",
        f"| Max error | {best['max_error']:.3f} | - |",
        "",
        "## Cross-validation on the training split",
        "",
        f"- MAE: {cv['mae']:.3f}",
        f"- RMSE: {cv['rmse']:.3f}",
        f"- R-squared: {cv['r2']:.3f}",
        "",
        "## All candidate models",
        "",
        *_markdown_table(table.round(3)),
        "",
        "## Target distribution",
        "",
        f"- mean: {metadata['target_stats']['mean']}",
        f"- sd: {metadata['target_stats']['std']}",
        f"- range: {metadata['target_stats']['min']} - {metadata['target_stats']['max']}",
        "",
    ]
    if metrics.get("best_params"):
        lines += [
            "## Tuned parameters",
            "",
            "```json",
            json.dumps(metrics["best_params"], indent=2),
            "```",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
