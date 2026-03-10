"""Evaluation metrics with MLflow artifact logging."""

import logging
import os
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class.
        prefix: Optional prefix string (e.g., 'test_') for metric names.

    Returns:
        Dictionary of metric_name → float value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    return metrics


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    artifact_path: str = "evaluation",
    labels: Optional[list] = None,
) -> None:
    """Save and log a confusion matrix plot as MLflow artifact.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        artifact_path: MLflow artifact sub-directory.
        labels: Optional label names for axis.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels or ["No Churn", "Churn"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix")

        tmp_path = "/tmp/confusion_matrix.png"
        fig.savefig(tmp_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
        logger.info("Confusion matrix logged to MLflow")
    except ImportError:
        logger.warning("matplotlib not available; skipping confusion matrix plot")


def log_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    artifact_path: str = "evaluation",
) -> None:
    """Save and log ROC curve plot as MLflow artifact.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        artifact_path: MLflow artifact sub-directory.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
        ax.set_title("ROC Curve")

        tmp_path = "/tmp/roc_curve.png"
        fig.savefig(tmp_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
        logger.info("ROC curve logged to MLflow")
    except ImportError:
        logger.warning("matplotlib not available; skipping ROC curve plot")


def evaluate_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test",
    log_plots: bool = True,
) -> Dict[str, float]:
    """Evaluate a fitted model and log all metrics + plots to the active MLflow run.

    Args:
        model: Fitted sklearn-compatible classifier.
        X: Feature matrix.
        y: True labels.
        split_name: Name of the split (used as metric prefix).
        log_plots: Whether to log confusion matrix and ROC curve.

    Returns:
        Dictionary of computed metrics.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = compute_full_metrics(y, y_pred, y_prob, prefix=f"{split_name}_")
    mlflow.log_metrics(metrics)

    logger.info(
        "%s metrics: AUC=%.4f, F1=%.4f, Accuracy=%.4f",
        split_name,
        metrics.get(f"{split_name}_auc", 0),
        metrics.get(f"{split_name}_f1", 0),
        metrics.get(f"{split_name}_accuracy", 0),
    )

    if log_plots:
        log_confusion_matrix(y, y_pred)
        log_roc_curve(y, y_prob)

    return metrics
