"""Model training module with MLflow autolog and artifact logging."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.engineering import (
    TRANSFORMER_ARTIFACT_NAME,
    get_feature_names,
    log_feature_importance,
    run_feature_engineering,
)
from src.data.ingestion import (
    TARGET_COLUMN,
    ingest_data,
    load_latest_raw_data,
    split_dataset,
)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction")
MODEL_REGISTRY_NAME = os.getenv("MODEL_NAME", "ChurnModel")

SUPPORTED_MODELS: Dict[str, Any] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def get_default_params(model_type: str) -> Dict[str, Any]:
    """Return sensible default hyperparameters for a given model type.

    Args:
        model_type: One of 'random_forest', 'gradient_boosting', 'logistic_regression'.

    Returns:
        Dictionary of hyperparameters.
    """
    defaults: Dict[str, Dict[str, Any]] = {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "random_state": 42,
        },
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
        },
    }
    return defaults.get(model_type, {})


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dictionary of metric name → value.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "random_forest",
    params: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Tuple[Any, str, Dict[str, float]]:
    """Train a classifier with MLflow experiment tracking.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        model_type: Classifier type key.
        params: Hyperparameters; defaults used if None.
        run_name: MLflow run name.
        tags: Additional MLflow tags.

    Returns:
        Tuple of (trained_model, mlflow_run_id, validation_metrics).
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from: {list(SUPPORTED_MODELS)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    if params is None:
        params = get_default_params(model_type)

    default_tags = {
        "model_type": model_type,
        "stage": "training",
    }
    if tags:
        default_tags.update(tags)

    with mlflow.start_run(run_name=run_name or f"{model_type}_run", tags=default_tags) as run:
        logger.info("Starting MLflow run: %s", run.info.run_id)

        # Train
        model_class = SUPPORTED_MODELS[model_type]
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)

        # Log val metrics with prefix
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)

        logger.info(
            "Validation metrics: AUC=%.4f, F1=%.4f, Accuracy=%.4f",
            val_metrics["auc"],
            val_metrics["f1"],
            val_metrics["accuracy"],
        )

        # Log feature importances if available
        if hasattr(model, "feature_importances_"):
            from src.features.engineering import get_feature_names
            # We do best-effort feature name retrieval
            try:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                log_feature_importance(feature_names, model.feature_importances_)
            except Exception as exc:
                logger.warning("Could not log feature importances: %s", exc)

        run_id = run.info.run_id

    logger.info("Training complete. Run ID: %s", run_id)
    return model, run_id, val_metrics


def register_model(
    run_id: str,
    model_name: str = MODEL_REGISTRY_NAME,
    auc_threshold: float = 0.75,
    val_metrics: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """Register model to MLflow Model Registry and promote to Staging if AUC > threshold.

    Args:
        run_id: MLflow run ID that contains the logged model.
        model_name: Registry model name.
        auc_threshold: Minimum AUC to auto-promote to Staging.
        val_metrics: Metrics dict to check AUC against.

    Returns:
        Model version string if registered, else None.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = mv.version
        logger.info("Registered model '%s' version %s", model_name, version)

        # Auto-promote to Staging
        if val_metrics and val_metrics.get("auc", 0) > auc_threshold:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            logger.info(
                "Promoted model '%s' v%s to Staging (AUC=%.4f > %.2f)",
                model_name,
                version,
                val_metrics["auc"],
                auc_threshold,
            )
        else:
            auc = val_metrics.get("auc", 0) if val_metrics else 0
            logger.info(
                "Model '%s' v%s NOT promoted (AUC=%.4f <= %.2f)",
                model_name,
                version,
                auc,
                auc_threshold,
            )

        return version
    except Exception as exc:
        logger.error("Failed to register model: %s", exc)
        return None


def run_training_pipeline(
    data_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    model_type: str = "random_forest",
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Orchestrate end-to-end training: load → features → train → register.

    Args:
        data_dir: Directory with raw data partitions.
        processed_dir: Directory for processed feature arrays.
        model_type: Classifier type.
        params: Optional hyperparameters.
        tags: Optional MLflow tags.

    Returns:
        Dictionary with run_id, version, and metrics.
    """
    logger.info("Starting training pipeline")

    # Load data
    raw_df = load_latest_raw_data(data_dir)

    # Split
    train_df, val_df, test_df = split_dataset(raw_df)

    # Feature engineering
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = run_feature_engineering(
        train_df, val_df, test_df, output_dir=processed_dir
    )

    # Train
    model, run_id, val_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_type=model_type,
        params=params,
        tags=tags,
    )

    # Register
    version = register_model(run_id, val_metrics=val_metrics)

    result = {
        "run_id": run_id,
        "version": version,
        "val_metrics": val_metrics,
        "model_type": model_type,
    }
    logger.info("Training pipeline complete: %s", result)
    return result
