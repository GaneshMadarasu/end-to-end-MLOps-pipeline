"""Model promotion: challenger vs champion comparison and registry transitions."""

import logging
import os
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient

from src.evaluation.metrics import compute_full_metrics

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = os.getenv("MODEL_NAME", "ChurnModel")
PROMOTION_AUC_DELTA = float(os.getenv("PROMOTION_AUC_DELTA", "0.01"))  # 1% improvement


def load_production_model(model_name: str = MODEL_REGISTRY_NAME) -> Optional[object]:
    """Load the current Production model from MLflow registry.

    Args:
        model_name: Registered model name.

    Returns:
        Loaded model or None if no Production model exists.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.info("No Production model found for '%s'", model_name)
            return None

        mv = versions[0]
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(
            "Loaded Production model '%s' version %s (run_id=%s)",
            model_name,
            mv.version,
            mv.run_id,
        )
        return model
    except Exception as exc:
        logger.error("Failed to load Production model: %s", exc)
        return None


def compare_models(
    champion: object,
    challenger: object,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    """Compare champion and challenger on a holdout set.

    Args:
        champion: Currently deployed Production model.
        challenger: Newly trained candidate model.
        X_holdout: Holdout feature matrix.
        y_holdout: Holdout true labels.

    Returns:
        Tuple of (champion_metrics, challenger_metrics, should_promote).
        should_promote is True if challenger AUC > champion AUC + PROMOTION_AUC_DELTA.
    """
    champ_pred = champion.predict(X_holdout)
    champ_prob = champion.predict_proba(X_holdout)[:, 1]
    champ_metrics = compute_full_metrics(y_holdout, champ_pred, champ_prob, prefix="champion_")

    chal_pred = challenger.predict(X_holdout)
    chal_prob = challenger.predict_proba(X_holdout)[:, 1]
    chal_metrics = compute_full_metrics(y_holdout, chal_pred, chal_prob, prefix="challenger_")

    champ_auc = champ_metrics["champion_auc"]
    chal_auc = chal_metrics["challenger_auc"]
    should_promote = bool(chal_auc > champ_auc + PROMOTION_AUC_DELTA)

    logger.info(
        "Champion AUC=%.4f | Challenger AUC=%.4f | Delta=%.4f | Promote=%s",
        champ_auc,
        chal_auc,
        chal_auc - champ_auc,
        should_promote,
    )
    return champ_metrics, chal_metrics, should_promote


def promote_model(
    challenger_version: str,
    model_name: str = MODEL_REGISTRY_NAME,
) -> None:
    """Transition challenger to Production and archive old Production version.

    Args:
        challenger_version: Model version string to promote.
        model_name: Registered model name.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Archive current Production versions
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    for mv in current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived",
            archive_existing_versions=False,
        )
        logger.info("Archived Production model '%s' version %s", model_name, mv.version)

    # Promote challenger
    client.transition_model_version_stage(
        name=model_name,
        version=challenger_version,
        stage="Production",
        archive_existing_versions=False,
    )
    logger.info(
        "Promoted model '%s' version %s to Production",
        model_name,
        challenger_version,
    )


def run_promotion_logic(
    challenger_run_id: str,
    challenger_version: str,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    model_name: str = MODEL_REGISTRY_NAME,
) -> Dict[str, object]:
    """Full promotion flow: load champion → compare → optionally promote.

    Args:
        challenger_run_id: MLflow run ID of the challenger.
        challenger_version: Registry version of the challenger.
        X_holdout: Holdout feature matrix.
        y_holdout: Holdout true labels.
        model_name: Registered model name.

    Returns:
        Summary dict with promotion decision and metrics.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    champion = load_production_model(model_name)
    challenger = mlflow.sklearn.load_model(f"runs:/{challenger_run_id}/model")

    if champion is None:
        # No existing Production model → auto-promote challenger
        logger.info("No champion found — promoting challenger directly to Production")
        promote_model(challenger_version, model_name)
        chal_pred = challenger.predict(X_holdout)
        chal_prob = challenger.predict_proba(X_holdout)[:, 1]
        chal_metrics = compute_full_metrics(y_holdout, chal_pred, chal_prob, prefix="challenger_")
        decision = "promoted_no_champion"
    else:
        champ_metrics, chal_metrics, should_promote = compare_models(
            champion, challenger, X_holdout, y_holdout
        )

        # Log comparison to active run
        try:
            with mlflow.start_run(run_id=challenger_run_id):
                mlflow.log_metrics({**champ_metrics, **chal_metrics})
                mlflow.log_param("promotion_decision", "promoted" if should_promote else "rejected")
                mlflow.log_param("champion_auc", champ_metrics["champion_auc"])
                mlflow.log_param("challenger_auc", chal_metrics["challenger_auc"])
        except Exception as exc:
            logger.warning("Could not log comparison to MLflow: %s", exc)

        if should_promote:
            promote_model(challenger_version, model_name)
            decision = "promoted"
        else:
            # Move challenger to Staging (not good enough for Production)
            client.transition_model_version_stage(
                name=model_name,
                version=challenger_version,
                stage="Staging",
                archive_existing_versions=False,
            )
            decision = "rejected"

    result = {
        "decision": decision,
        "challenger_version": challenger_version,
        "challenger_metrics": chal_metrics,
    }
    logger.info("Promotion result: %s", result)
    return result
