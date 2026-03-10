"""MLflow model loader with caching and version tracking."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "ChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")


@dataclass
class LoadedModel:
    """Container for a loaded model with its metadata."""

    model: Any
    version: str
    stage: str
    name: str
    run_id: str


def load_model_from_registry(
    model_name: str = MODEL_NAME,
    stage: str = MODEL_STAGE,
) -> LoadedModel:
    """Load a model from the MLflow registry by name and stage.

    Args:
        model_name: Registered model name.
        stage: Model stage ('Production', 'Staging', etc.).

    Returns:
        LoadedModel with model object and metadata.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise RuntimeError(
                f"No '{stage}' version found for model '{model_name}' in MLflow registry"
            )

        mv = versions[0]
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)

        loaded = LoadedModel(
            model=model,
            version=mv.version,
            stage=stage,
            name=model_name,
            run_id=mv.run_id,
        )

        logger.info(
            "Loaded model '%s' version=%s stage=%s run_id=%s",
            model_name,
            mv.version,
            stage,
            mv.run_id,
        )
        return loaded

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_name}' ({stage}): {exc}") from exc


def load_model_by_version(
    model_name: str = MODEL_NAME,
    version: str = "1",
) -> LoadedModel:
    """Load a specific model version from the registry.

    Args:
        model_name: Registered model name.
        version: Version string.

    Returns:
        LoadedModel with model object and metadata.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    mv = client.get_model_version(model_name, version)
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    return LoadedModel(
        model=model,
        version=version,
        stage=mv.current_stage,
        name=model_name,
        run_id=mv.run_id,
    )
