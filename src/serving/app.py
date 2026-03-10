"""FastAPI model serving application with prediction, health, metrics, and feedback endpoints."""

import csv
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from src.serving.model_loader import LoadedModel, load_model_from_registry
from src.serving.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_STORE = os.getenv("PREDICTIONS_STORE", "data/predictions.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "ChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# In-memory metrics counters
_metrics: Dict[str, Any] = {
    "request_count": 0,
    "prediction_0_count": 0,
    "prediction_1_count": 0,
    "total_latency_ms": 0.0,
    "error_count": 0,
}

# Global model holder
_loaded_model: Optional[LoadedModel] = None


def _ensure_predictions_store() -> None:
    """Create predictions CSV with header if it does not exist."""
    path = Path(PREDICTIONS_STORE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "prediction_id", "timestamp", "prediction", "probability",
                "age", "tenure_months", "monthly_charges", "total_charges",
                "num_products", "has_tech_support", "has_online_security",
                "has_backup", "has_device_protection", "is_senior_citizen",
                "has_partner", "has_dependents", "contract_type",
                "payment_method", "internet_service", "actual_label",
            ])


def _log_prediction(
    prediction_id: str,
    request: PredictRequest,
    prediction: int,
    probability: float,
    actual_label: Optional[int] = None,
) -> None:
    """Append a prediction record to the predictions CSV store.

    Args:
        prediction_id: Unique ID for this prediction.
        request: Original prediction request.
        prediction: Binary prediction result.
        probability: Churn probability.
        actual_label: Optional ground truth label from feedback.
    """
    _ensure_predictions_store()
    row = [
        prediction_id,
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prediction,
        round(probability, 6),
        request.age,
        request.tenure_months,
        request.monthly_charges,
        request.total_charges,
        request.num_products,
        request.has_tech_support,
        request.has_online_security,
        request.has_backup,
        request.has_device_protection,
        request.is_senior_citizen,
        request.has_partner,
        request.has_dependents,
        request.contract_type,
        request.payment_method,
        request.internet_service,
        actual_label if actual_label is not None else "",
    ]
    with open(PREDICTIONS_STORE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global _loaded_model
    logger.info("Loading model '%s' (stage=%s) from MLflow...", MODEL_NAME, MODEL_STAGE)
    try:
        _loaded_model = load_model_from_registry(MODEL_NAME, MODEL_STAGE)
        logger.info(
            "Model loaded: %s v%s",
            _loaded_model.name,
            _loaded_model.version,
        )
    except Exception as exc:
        logger.error("Could not load model on startup: %s", exc)
        _loaded_model = None
    _ensure_predictions_store()
    yield
    logger.info("Shutting down model server")


app = FastAPI(
    title="Churn Prediction API",
    description="Production model serving API for customer churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service health status and loaded model metadata."""
    if _loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=_loaded_model.name,
        model_version=_loaded_model.version,
        model_stage=_loaded_model.stage,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Run churn prediction for a single customer.

    Accepts customer feature payload and returns binary churn prediction
    with associated probability.
    """
    if _loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    try:
        features_df = pd.DataFrame([request.model_dump()])
        y_pred = _loaded_model.model.predict(features_df)[0]
        y_prob = _loaded_model.model.predict_proba(features_df)[0][1]

        prediction = int(y_pred)
        probability = round(float(y_prob), 6)

        # Update metrics
        _metrics["request_count"] += 1
        _metrics["prediction_0_count" if prediction == 0 else "prediction_1_count"] += 1
        latency_ms = (time.time() - start_time) * 1000
        _metrics["total_latency_ms"] += latency_ms

        # Persist prediction
        _log_prediction(prediction_id, request, prediction, probability)

        logger.info(
            "Prediction id=%s: churn=%d prob=%.4f latency=%.1fms",
            prediction_id,
            prediction,
            probability,
            latency_ms,
        )

        return PredictResponse(
            prediction=prediction,
            probability=probability,
            model_name=_loaded_model.name,
            model_version=_loaded_model.version,
        )

    except Exception as exc:
        _metrics["error_count"] += 1
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics() -> str:
    """Return Prometheus-format metrics."""
    total = _metrics["request_count"] or 1
    avg_latency = _metrics["total_latency_ms"] / total

    lines = [
        "# HELP churn_api_requests_total Total prediction requests",
        "# TYPE churn_api_requests_total counter",
        f"churn_api_requests_total {_metrics['request_count']}",
        "",
        "# HELP churn_api_errors_total Total prediction errors",
        "# TYPE churn_api_errors_total counter",
        f"churn_api_errors_total {_metrics['error_count']}",
        "",
        "# HELP churn_prediction_distribution Prediction class distribution",
        "# TYPE churn_prediction_distribution gauge",
        f'churn_prediction_distribution{{label="0"}} {_metrics["prediction_0_count"]}',
        f'churn_prediction_distribution{{label="1"}} {_metrics["prediction_1_count"]}',
        "",
        "# HELP churn_api_avg_latency_ms Average prediction latency in milliseconds",
        "# TYPE churn_api_avg_latency_ms gauge",
        f"churn_api_avg_latency_ms {avg_latency:.2f}",
        "",
    ]

    if _loaded_model:
        lines += [
            "# HELP churn_model_info Model version metadata",
            "# TYPE churn_model_info gauge",
            f'churn_model_info{{name="{_loaded_model.name}",version="{_loaded_model.version}",stage="{_loaded_model.stage}"}} 1',
            "",
        ]

    return "\n".join(lines)


@app.post("/feedback", response_model=FeedbackResponse, tags=["prediction"])
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Accept ground truth label for a previous prediction.

    Stores actual outcome for drift monitoring and model evaluation.
    """
    store_path = Path(PREDICTIONS_STORE)
    if not store_path.exists():
        raise HTTPException(status_code=404, detail="Predictions store not found")

    try:
        df = pd.read_csv(store_path)
        mask = df["prediction_id"] == request.prediction_id

        if not mask.any():
            raise HTTPException(
                status_code=404,
                detail=f"Prediction ID '{request.prediction_id}' not found",
            )

        df.loc[mask, "actual_label"] = request.actual_label
        df.to_csv(store_path, index=False)

        logger.info(
            "Feedback recorded: id=%s actual=%d",
            request.prediction_id,
            request.actual_label,
        )
        return FeedbackResponse(
            status="ok",
            prediction_id=request.prediction_id,
            message="Feedback recorded successfully",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Feedback error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {exc}")
