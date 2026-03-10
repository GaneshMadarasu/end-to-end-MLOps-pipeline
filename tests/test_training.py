"""Tests for training, evaluation, and promotion modules (MLflow mocked)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call
from sklearn.ensemble import RandomForestClassifier

from src.data.ingestion import generate_synthetic_churn_data, split_dataset
from src.features.engineering import run_feature_engineering


@pytest.fixture
def small_xy(tmp_path):
    """Small processed feature arrays for fast training tests."""
    df = generate_synthetic_churn_data(n_samples=400, random_state=7)
    train_df, val_df, test_df = split_dataset(df)
    X_tr, y_tr, X_v, y_v, X_te, y_te, _ = run_feature_engineering(
        train_df, val_df, test_df, output_dir=str(tmp_path)
    )
    return X_tr, y_tr, X_v, y_v, X_te, y_te


# ──────────────────────────────────────────────────────────────────────────────
# src/training/train.py
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_keys(self, small_xy):
        from src.training.train import compute_metrics

        X_tr, y_tr, X_v, y_v, *_ = small_xy
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_v)
        y_prob = clf.predict_proba(X_v)[:, 1]

        metrics = compute_metrics(y_v, y_pred, y_prob)
        assert set(metrics.keys()) == {"accuracy", "f1", "precision", "recall", "auc"}

    def test_metrics_in_range(self, small_xy):
        from src.training.train import compute_metrics

        X_tr, y_tr, X_v, y_v, *_ = small_xy
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_v)
        y_prob = clf.predict_proba(X_v)[:, 1]

        metrics = compute_metrics(y_v, y_pred, y_prob)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


class TestGetDefaultParams:
    def test_random_forest_params(self):
        from src.training.train import get_default_params

        params = get_default_params("random_forest")
        assert "n_estimators" in params
        assert "random_state" in params

    def test_unknown_model_returns_empty(self):
        from src.training.train import get_default_params

        assert get_default_params("nonexistent") == {}


class TestTrainModel:
    def test_train_model_runs(self, small_xy):
        from src.training.train import train_model

        X_tr, y_tr, X_v, y_v, *_ = small_xy

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_active = MagicMock()
        mock_active.__enter__ = MagicMock(return_value=mock_run)
        mock_active.__exit__ = MagicMock(return_value=False)

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.sklearn.autolog"), \
             patch("mlflow.start_run", return_value=mock_active), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"):

            model, run_id, val_metrics = train_model(
                X_tr, y_tr, X_v, y_v,
                model_type="random_forest",
                params={"n_estimators": 10, "random_state": 42, "n_jobs": 1},
            )

        assert run_id == "test-run-123"
        assert "auc" in val_metrics
        assert 0.0 <= val_metrics["auc"] <= 1.0

    def test_unsupported_model_raises(self, small_xy):
        from src.training.train import train_model

        X_tr, y_tr, X_v, y_v, *_ = small_xy
        with pytest.raises(ValueError, match="Unsupported model_type"):
            train_model(X_tr, y_tr, X_v, y_v, model_type="xgboost_nonexistent")

    def test_gradient_boosting_trains(self, small_xy):
        from src.training.train import train_model

        X_tr, y_tr, X_v, y_v, *_ = small_xy

        mock_run = MagicMock()
        mock_run.info.run_id = "gb-run-456"
        mock_active = MagicMock()
        mock_active.__enter__ = MagicMock(return_value=mock_run)
        mock_active.__exit__ = MagicMock(return_value=False)

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.sklearn.autolog"), \
             patch("mlflow.start_run", return_value=mock_active), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"):

            model, run_id, val_metrics = train_model(
                X_tr, y_tr, X_v, y_v,
                model_type="gradient_boosting",
                params={"n_estimators": 10, "random_state": 42},
            )

        assert run_id == "gb-run-456"
        assert val_metrics["auc"] > 0.4


class TestRegisterModel:
    def test_registers_and_promotes_above_threshold(self):
        from src.training.train import register_model

        mock_mv = MagicMock()
        mock_mv.version = "3"
        mock_client = MagicMock()

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.register_model", return_value=mock_mv), \
             patch("mlflow.tracking.MlflowClient", return_value=mock_client):

            version = register_model(
                run_id="run-abc",
                val_metrics={"auc": 0.85},
                auc_threshold=0.75,
            )

        assert version == "3"
        mock_client.transition_model_version_stage.assert_called_once()

    def test_does_not_promote_below_threshold(self):
        from src.training.train import register_model

        mock_mv = MagicMock()
        mock_mv.version = "4"
        mock_client = MagicMock()

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.register_model", return_value=mock_mv), \
             patch("mlflow.tracking.MlflowClient", return_value=mock_client):

            version = register_model(
                run_id="run-xyz",
                val_metrics={"auc": 0.60},
                auc_threshold=0.75,
            )

        assert version == "4"
        mock_client.transition_model_version_stage.assert_not_called()

    def test_returns_none_on_registry_failure(self):
        from src.training.train import register_model

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.register_model", side_effect=Exception("registry error")), \
             patch("mlflow.tracking.MlflowClient"):

            version = register_model(run_id="run-fail")

        assert version is None


# ──────────────────────────────────────────────────────────────────────────────
# src/evaluation/metrics.py
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeFullMetrics:
    def test_prefix_applied(self):
        from src.evaluation.metrics import compute_full_metrics

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8])

        metrics = compute_full_metrics(y_true, y_pred, y_prob, prefix="test_")
        assert all(k.startswith("test_") for k in metrics)
        assert "test_auc" in metrics

    def test_no_prefix(self):
        from src.evaluation.metrics import compute_full_metrics

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.6, 0.9])

        metrics = compute_full_metrics(y_true, y_pred, y_prob)
        assert "auc" in metrics
        assert 0 <= metrics["auc"] <= 1


class TestEvaluateModel:
    def test_logs_metrics_to_mlflow(self, small_xy):
        from src.evaluation.metrics import evaluate_model

        X_tr, y_tr, X_v, y_v, X_te, y_te, *_ = (*small_xy, None)
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X_tr, y_tr)

        with patch("mlflow.log_metrics") as mock_log, \
             patch("mlflow.log_artifact"), \
             patch("src.evaluation.metrics.log_confusion_matrix"), \
             patch("src.evaluation.metrics.log_roc_curve"):

            metrics = evaluate_model(clf, X_te, y_te, split_name="test", log_plots=False)

        mock_log.assert_called_once()
        assert "test_auc" in metrics


class TestLogConfusionMatrix:
    def test_does_not_crash(self):
        from src.evaluation.metrics import log_confusion_matrix

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        with patch("mlflow.log_artifact"):
            # Should not raise even if matplotlib unavailable
            try:
                log_confusion_matrix(y_true, y_pred)
            except Exception as exc:
                pytest.fail(f"log_confusion_matrix raised: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# src/evaluation/promotion.py
# ──────────────────────────────────────────────────────────────────────────────

class TestCompareModels:
    def test_detects_improvement(self, small_xy):
        from src.evaluation.promotion import compare_models

        X_tr, y_tr, X_v, y_v, X_te, y_te, *_ = (*small_xy, None)

        # Train a weaker champion
        champion = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=1)
        champion.fit(X_tr, y_tr)

        # Train a stronger challenger
        challenger = RandomForestClassifier(n_estimators=100, random_state=42)
        challenger.fit(X_tr, y_tr)

        champ_metrics, chal_metrics, should_promote = compare_models(
            champion, challenger, X_te, y_te
        )

        assert "champion_auc" in champ_metrics
        assert "challenger_auc" in chal_metrics
        assert isinstance(should_promote, bool)

    def test_no_promotion_if_worse(self, small_xy):
        from src.evaluation.promotion import compare_models

        X_tr, y_tr, X_v, y_v, X_te, y_te, *_ = (*small_xy, None)

        # Same model for both → delta = 0, should not promote
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_tr, y_tr)

        _, _, should_promote = compare_models(clf, clf, X_te, y_te)
        # Identical models → delta is 0 → should not promote (needs >1% improvement)
        assert not should_promote


class TestLoadProductionModel:
    def test_returns_none_when_no_production(self):
        from src.evaluation.promotion import load_production_model

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        with patch("mlflow.set_tracking_uri"), \
             patch("src.evaluation.promotion.MlflowClient", return_value=mock_client):

            result = load_production_model("ChurnModel")

        assert result is None

    def test_loads_production_model(self):
        from src.evaluation.promotion import load_production_model

        mock_mv = MagicMock()
        mock_mv.version = "2"
        mock_mv.run_id = "run-prod"

        mock_model = MagicMock()
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_mv]

        with patch("mlflow.set_tracking_uri"), \
             patch("src.evaluation.promotion.MlflowClient", return_value=mock_client), \
             patch("mlflow.sklearn.load_model", return_value=mock_model):

            result = load_production_model("ChurnModel")

        assert result is mock_model


# ──────────────────────────────────────────────────────────────────────────────
# src/monitoring/alerting.py
# ──────────────────────────────────────────────────────────────────────────────

class TestTriggerRetrainingDag:
    def test_returns_true_on_success(self):
        from src.monitoring.alerting import trigger_retraining_dag

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"dag_run_id": "run-triggered"}

        with patch("requests.post", return_value=mock_resp):
            result = trigger_retraining_dag("drift_detected")

        assert result is True

    def test_returns_false_on_failure(self):
        from src.monitoring.alerting import trigger_retraining_dag

        import requests as req_module
        with patch("requests.post", side_effect=req_module.exceptions.ConnectionError("refused")):
            result = trigger_retraining_dag("drift_detected")

        assert result is False


class TestSendSlackNotification:
    def test_skips_when_no_webhook(self):
        from src.monitoring.alerting import send_slack_notification

        result = send_slack_notification("test", webhook_url="")
        assert result is False

    def test_sends_when_webhook_set(self):
        from src.monitoring.alerting import send_slack_notification

        mock_resp = MagicMock()
        with patch("requests.post", return_value=mock_resp):
            result = send_slack_notification("test msg", webhook_url="http://fake.webhook")

        assert result is True


# ──────────────────────────────────────────────────────────────────────────────
# src/serving/model_loader.py
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadModelFromRegistry:
    def test_raises_when_no_stage_version(self):
        from src.serving.model_loader import load_model_from_registry

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.tracking.MlflowClient", return_value=mock_client):

            with pytest.raises(RuntimeError, match="No 'Production' version found"):
                load_model_from_registry("ChurnModel", "Production")

    def test_loads_correctly(self):
        from src.serving.model_loader import load_model_from_registry, LoadedModel

        mock_mv = MagicMock()
        mock_mv.version = "5"
        mock_mv.run_id = "run-999"
        mock_model = MagicMock()
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_mv]

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.tracking.MlflowClient", return_value=mock_client), \
             patch("mlflow.sklearn.load_model", return_value=mock_model):

            loaded = load_model_from_registry("ChurnModel", "Production")

        assert isinstance(loaded, LoadedModel)
        assert loaded.version == "5"
        assert loaded.model is mock_model
        assert loaded.stage == "Production"


# ──────────────────────────────────────────────────────────────────────────────
# src/training/hyperparameter.py
# ──────────────────────────────────────────────────────────────────────────────

class TestOptimizeHyperparameters:
    def test_returns_best_params(self, small_xy):
        from src.training.hyperparameter import optimize_hyperparameters

        X_tr, y_tr, X_v, y_v, *_ = small_xy

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run", return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metric"):

            best = optimize_hyperparameters(
                X_tr, y_tr, X_v, y_v,
                model_type="random_forest",
                n_trials=3,
                timeout=30,
            )

        assert isinstance(best, dict)
        assert "n_estimators" in best

    def test_unsupported_model_raises(self, small_xy):
        from src.training.hyperparameter import optimize_hyperparameters

        X_tr, y_tr, X_v, y_v, *_ = small_xy
        with pytest.raises(ValueError):
            optimize_hyperparameters(X_tr, y_tr, X_v, y_v, model_type="unsupported")
