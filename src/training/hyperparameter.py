"""Hyperparameter optimization using Optuna."""

import logging
import os
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction")


def _rf_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for RandomForest hyperparameters."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)


def _gb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for GradientBoosting hyperparameters."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "random_state": 42,
    }
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)


OBJECTIVES = {
    "random_forest": _rf_objective,
    "gradient_boosting": _gb_objective,
}


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "random_forest",
    n_trials: int = 30,
    timeout: Optional[int] = 600,
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search and log results to MLflow.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        model_type: 'random_forest' or 'gradient_boosting'.
        n_trials: Number of Optuna trials.
        timeout: Max seconds for search (None = unlimited).
        study_name: Optional Optuna study name.

    Returns:
        Best hyperparameters dictionary.
    """
    if model_type not in OBJECTIVES:
        raise ValueError(f"Unsupported model_type for HPO: {model_type}")

    objective_fn = OBJECTIVES[model_type]
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name or f"{model_type}_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_auc = study.best_value

    logger.info(
        "HPO complete: best AUC=%.4f with params=%s",
        best_auc,
        best_params,
    )

    # Log HPO results to MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{model_type}_hpo", tags={"stage": "hpo"}):
            mlflow.log_params(best_params)
            mlflow.log_metric("hpo_best_val_auc", best_auc)
            mlflow.log_metric("hpo_n_trials", n_trials)
    except Exception as exc:
        logger.warning("Could not log HPO results to MLflow: %s", exc)

    return best_params
