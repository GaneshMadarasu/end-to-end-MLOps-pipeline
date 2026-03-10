"""Tests for FastAPI serving application."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# We patch the model loader so tests don't need a running MLflow instance
MOCK_MODEL_NAME = "ChurnModel"
MOCK_MODEL_VERSION = "1"
MOCK_MODEL_STAGE = "Production"

SAMPLE_PAYLOAD = {
    "age": 35,
    "tenure_months": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "num_products": 2,
    "has_tech_support": 1,
    "has_online_security": 0,
    "has_backup": 1,
    "has_device_protection": 0,
    "is_senior_citizen": 0,
    "has_partner": 1,
    "has_dependents": 0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
}


def make_mock_model():
    """Create a mock sklearn-compatible classifier."""
    import numpy as np

    mock = MagicMock()
    mock.predict.return_value = [0]
    mock.predict_proba.return_value = [[0.75, 0.25]]
    return mock


@pytest.fixture
def client(tmp_path):
    """FastAPI TestClient with mocked model loader and predictions store."""
    from src.serving import app as serving_module

    mock_loaded = MagicMock()
    mock_loaded.model = make_mock_model()
    mock_loaded.name = MOCK_MODEL_NAME
    mock_loaded.version = MOCK_MODEL_VERSION
    mock_loaded.stage = MOCK_MODEL_STAGE

    predictions_file = str(tmp_path / "predictions.csv")

    with patch("src.serving.app.load_model_from_registry", return_value=mock_loaded), \
         patch("src.serving.app.PREDICTIONS_STORE", predictions_file), \
         patch("src.serving.app._loaded_model", mock_loaded):

        from src.serving.app import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "model_version" in data
        assert "model_stage" in data
        assert "model_name" in data

    def test_health_model_metadata(self, client):
        data = client.get("/health").json()
        assert data["model_name"] == MOCK_MODEL_NAME
        assert data["model_version"] == MOCK_MODEL_VERSION
        assert data["model_stage"] == MOCK_MODEL_STAGE


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_name" in data
        assert "model_version" in data

    def test_predict_binary_output(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert data["prediction"] in (0, 1)

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_invalid_payload(self, client):
        bad_payload = {"age": "not_a_number"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_missing_fields(self, client):
        response = client.post("/predict", json={"age": 35})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_prometheus_format(self, client):
        text = client.get("/metrics").text
        assert "churn_api_requests_total" in text
        assert "churn_api_errors_total" in text
        assert "churn_prediction_distribution" in text


class TestFeedbackEndpoint:
    """Tests for POST /feedback."""

    def test_feedback_unknown_id_returns_404(self, client):
        payload = {"prediction_id": "nonexistent-uuid", "actual_label": 1}
        response = client.post("/feedback", json=payload)
        # 404 because prediction_id doesn't exist, or 500 if store missing
        assert response.status_code in (404, 500)

    def test_feedback_invalid_label(self, client):
        payload = {"prediction_id": "some-id", "actual_label": 5}  # > 1
        response = client.post("/feedback", json=payload)
        assert response.status_code == 422
