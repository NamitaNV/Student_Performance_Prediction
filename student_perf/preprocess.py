"""Preprocessing pipeline: imputation, scaling and one-hot encoding."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_perf.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_numeric_pipeline() -> Pipeline:
    """Median imputation followed by standardisation for numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Mode imputation followed by one-hot encoding for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Combine the numeric and categorical transformers into one transformer."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), numeric_features),
            ("categorical", build_categorical_pipeline(), categorical_features),
        ],
        remainder="drop",
    )


def build_pipeline(model, numeric_features=None, categorical_features=None) -> Pipeline:
    """Full sklearn pipeline: preprocess then regress."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", model),
        ]
    )
