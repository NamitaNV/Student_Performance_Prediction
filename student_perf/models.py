"""Model definitions, evaluation metrics and hyperparameter search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

from student_perf.config import SEED
from student_perf.preprocess import build_pipeline


@dataclass
class TrainedModel:
    """A fitted pipeline plus its evaluation results."""

    name: str
    pipeline: BaseEstimator
    metrics: dict[str, float] = field(default_factory=dict)
    cv_metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    best_params: dict[str, str] | None = None


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE and R-squared."""
    residual = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "max_error": float(np.abs(residual).max()),
    }


def build_candidates(seed: int = SEED) -> dict[str, BaseEstimator]:
    """Return the candidate regressors to compare."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
    }


PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3, 5],
    },
    "GradientBoosting": {
        "n_estimators": [200, 400],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3, 4],
    },
    "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
}

SCORINGS = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def cross_validate_pipeline(
    pipeline: BaseEstimator,
    X,
    y,
    cv_folds: int = 5,
    seed: int = SEED,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Return mean cross-validated MAE, RMSE and R-squared over k folds."""
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        pipeline, X, y, cv=splitter, scoring=SCORINGS, n_jobs=n_jobs
    )
    return {
        "mae": float(-scores["test_mae"].mean()),
        "rmse": float(-scores["test_rmse"].mean()),
        "r2": float(scores["test_r2"].mean()),
    }


def tune_pipeline(
    pipeline: BaseEstimator,
    param_grid: dict,
    X_train,
    y_train,
    cv_folds: int = 5,
    seed: int = SEED,
    n_jobs: int = -1,
) -> tuple[BaseEstimator, dict[str, object]]:
    """Grid-search the pipeline, optimising for the lowest cross-validated MAE."""
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, dict(search.best_params_)


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    cv_folds: int = 5,
    seed: int = SEED,
    n_jobs: int = -1,
    tune: bool = False,
    verbose: bool = True,
) -> tuple[list[TrainedModel], pd.DataFrame]:
    """Fit every candidate on the training split and score it on the test split.

    Returns the trained models ordered by test MAE plus a comparison table.
    """
    trained: list[TrainedModel] = []

    for name, estimator in build_candidates(seed).items():
        pipeline = build_pipeline(clone(estimator))
        best_params = None

        if tune and name in PARAM_GRIDS:
            grid = {
                f"model__{key}": value
                for key, value in PARAM_GRIDS[name].items()
            }
            if verbose:
                print(f"  tuning {name} over {len(PARAM_GRIDS[name])} parameter sets...")
            pipeline, best_params = tune_pipeline(
                pipeline, grid, X_train, y_train, cv_folds, seed, n_jobs
            )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        predictions = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        cv_metrics = cross_validate_pipeline(
            pipeline, X_train, y_train, cv_folds, seed, n_jobs
        )

        trained.append(
            TrainedModel(
                name=name,
                pipeline=pipeline,
                metrics=metrics,
                cv_metrics=cv_metrics,
                fit_seconds=fit_seconds,
                best_params=best_params,
            )
        )
        if verbose:
            print(
                f"  {name:<17} MAE {metrics['mae']:6.3f} | "
                f"RMSE {metrics['rmse']:6.3f} | R2 {metrics['r2']:6.3f} | "
                f"{fit_seconds:5.2f}s"
            )

    trained.sort(key=lambda model: model.metrics["mae"])
    table = pd.DataFrame(
        [
            {
                "model": model.name,
                "test_mae": model.metrics["mae"],
                "test_rmse": model.metrics["rmse"],
                "test_r2": model.metrics["r2"],
                "test_max_error": model.metrics["max_error"],
                "cv_mae": model.cv_metrics["mae"],
                "cv_rmse": model.cv_metrics["rmse"],
                "cv_r2": model.cv_metrics["r2"],
                "fit_seconds": model.fit_seconds,
            }
            for model in trained
        ]
    )
    return trained, table


def baseline_predictions(y_train, y_test) -> np.ndarray:
    """Predict the training mean for every test row: the model to beat."""
    return np.full(len(y_test), float(np.mean(y_train)))
