"""Feature engineering: scalers, encoders, and pipeline construction."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.ingestion import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)

TRANSFORMER_ARTIFACT_NAME = "preprocessor.pkl"


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for feature preprocessing.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Unfitted ColumnTransformer.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    logger.debug(
        "Built preprocessor: %d numerical, %d categorical features",
        len(numerical_features),
        len(categorical_features),
    )
    return preprocessor


def fit_preprocessor(
    df: pd.DataFrame,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Fit the preprocessor on training data.

    Args:
        df: Training DataFrame (may include target column).
        numerical_features: Numerical feature names.
        categorical_features: Categorical feature names.

    Returns:
        Fitted ColumnTransformer.
    """
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    feature_cols = (numerical_features or NUMERICAL_FEATURES) + (categorical_features or CATEGORICAL_FEATURES)
    X = df[feature_cols]

    preprocessor.fit(X)
    logger.info("Preprocessor fitted on %d samples, %d features", len(df), len(feature_cols))
    return preprocessor


def transform_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply fitted preprocessor to a DataFrame.

    Args:
        df: Input DataFrame.
        preprocessor: Fitted ColumnTransformer.
        numerical_features: Numerical feature names.
        categorical_features: Categorical feature names.

    Returns:
        Tuple (X_transformed, y) where y is None if target not present.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    feature_cols = numerical_features + categorical_features
    X = df[feature_cols]
    X_transformed = preprocessor.transform(X)

    y = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None

    logger.debug(
        "Transformed %d samples into %d features",
        len(df),
        X_transformed.shape[1],
    )
    return X_transformed, y


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract feature names after transformation.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        List of feature name strings.
    """
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "numerical":
            feature_names.extend(cols)
        elif name == "categorical":
            encoder: OneHotEncoder = transformer.named_steps["encoder"]
            for col_idx, col in enumerate(cols):
                for cat in encoder.categories_[col_idx]:
                    feature_names.append(f"{col}_{cat}")
    return feature_names


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """Serialize fitted preprocessor to disk.

    Args:
        preprocessor: Fitted ColumnTransformer.
        path: File path for pickle output.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved to %s", path)


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a fitted preprocessor from disk.

    Args:
        path: Path to pickle file.

    Returns:
        Fitted ColumnTransformer.
    """
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded from %s", path)
    return preprocessor


def log_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
) -> None:
    """Log feature importances to MLflow as metrics and artifact.

    Args:
        feature_names: Names of features.
        importances: Array of importance scores.
        top_n: Number of top features to log as individual metrics.
    """
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # Log top-N as individual metrics
    for _, row in importance_df.head(top_n).iterrows():
        safe_name = row["feature"].replace(" ", "_").replace("(", "").replace(")", "")
        mlflow.log_metric(f"feat_importance_{safe_name}", round(float(row["importance"]), 6))

    # Save full importance table as CSV artifact
    tmp_path = "/tmp/feature_importances.csv"
    importance_df.to_csv(tmp_path, index=False)
    mlflow.log_artifact(tmp_path, artifact_path="feature_analysis")
    logger.info("Logged feature importances for %d features", len(feature_names))


def run_feature_engineering(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """End-to-end feature engineering: fit on train, transform all splits.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        output_dir: Directory to save processed splits.

    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor).
    """
    preprocessor = fit_preprocessor(train_df)

    X_train, y_train = transform_features(train_df, preprocessor)
    X_val, y_val = transform_features(val_df, preprocessor)
    X_test, y_test = transform_features(test_df, preprocessor)

    # Save preprocessor artifact
    preprocessor_path = str(Path(output_dir) / TRANSFORMER_ARTIFACT_NAME)
    save_preprocessor(preprocessor, preprocessor_path)

    # Save processed splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "y_val.npy", y_val)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_test.npy", y_test)

    logger.info(
        "Feature engineering complete: X_train=%s, X_val=%s, X_test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
