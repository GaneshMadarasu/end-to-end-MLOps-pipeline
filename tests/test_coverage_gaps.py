"""Targeted tests to cover remaining uncovered lines."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────────────────────────────────────
# src/data/ingestion.py – save_raw_data, load_or_generate, load_latest, ingest
# ──────────────────────────────────────────────────────────────────────────────

class TestIngestionIO:
    def test_save_raw_data_creates_file(self, tmp_path):
        from src.data.ingestion import save_raw_data, generate_synthetic_churn_data

        df = generate_synthetic_churn_data(n_samples=50)
        path = save_raw_data(df, base_dir=str(tmp_path), partition_date="2024-03-01")
        assert Path(path).exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 50

    def test_save_raw_data_defaults_to_today(self, tmp_path):
        from src.data.ingestion import save_raw_data, generate_synthetic_churn_data
        from datetime import date

        df = generate_synthetic_churn_data(n_samples=20)
        path = save_raw_data(df, base_dir=str(tmp_path))
        assert date.today().isoformat() in path

    def test_load_or_generate_uses_file_if_exists(self, tmp_path):
        from src.data.ingestion import load_or_generate_data, generate_synthetic_churn_data

        df = generate_synthetic_churn_data(n_samples=30)
        csv_path = str(tmp_path / "data.csv")
        df.to_csv(csv_path, index=False)

        result = load_or_generate_data(data_path=csv_path)
        assert len(result) == 30

    def test_load_or_generate_generates_if_no_file(self, tmp_path):
        from src.data.ingestion import load_or_generate_data

        result = load_or_generate_data(
            data_path=str(tmp_path / "nonexistent.csv"),
            n_samples=100,
        )
        assert len(result) == 100

    def test_load_or_generate_generates_when_no_path(self):
        from src.data.ingestion import load_or_generate_data

        result = load_or_generate_data(data_path=None, n_samples=80)
        assert len(result) == 80

    def test_load_latest_raw_data(self, tmp_path):
        from src.data.ingestion import save_raw_data, load_latest_raw_data, generate_synthetic_churn_data

        df = generate_synthetic_churn_data(n_samples=40)
        save_raw_data(df, base_dir=str(tmp_path), partition_date="2024-06-01")

        loaded = load_latest_raw_data(base_dir=str(tmp_path))
        assert len(loaded) == 40

    def test_load_latest_raises_if_empty(self, tmp_path):
        from src.data.ingestion import load_latest_raw_data

        with pytest.raises(FileNotFoundError):
            load_latest_raw_data(base_dir=str(tmp_path))

    def test_load_latest_picks_most_recent(self, tmp_path):
        from src.data.ingestion import save_raw_data, load_latest_raw_data, generate_synthetic_churn_data

        df_old = generate_synthetic_churn_data(n_samples=30, random_state=1)
        df_new = generate_synthetic_churn_data(n_samples=50, random_state=2)
        save_raw_data(df_old, base_dir=str(tmp_path), partition_date="2024-01-01")
        save_raw_data(df_new, base_dir=str(tmp_path), partition_date="2024-12-31")

        loaded = load_latest_raw_data(base_dir=str(tmp_path))
        assert len(loaded) == 50

    def test_ingest_data_end_to_end(self, tmp_path):
        from src.data.ingestion import ingest_data

        path = ingest_data(
            data_path=None,
            base_dir=str(tmp_path),
            n_samples=60,
            partition_date="2024-05-01",
        )
        assert Path(path).exists()
        df = pd.read_csv(path)
        assert len(df) == 60


# ──────────────────────────────────────────────────────────────────────────────
# src/evaluation/promotion.py – promote_model, run_promotion_logic
# ──────────────────────────────────────────────────────────────────────────────

class TestPromoteModel:
    def test_promotes_and_archives(self):
        from src.evaluation.promotion import promote_model

        mock_mv_current = MagicMock()
        mock_mv_current.version = "2"
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_mv_current]

        with patch("mlflow.set_tracking_uri"), \
             patch("src.evaluation.promotion.MlflowClient", return_value=mock_client):
            promote_model(challenger_version="3")

        # Should have been called twice: archive old + promote new
        assert mock_client.transition_model_version_stage.call_count == 2

    def test_promotes_when_no_existing_production(self):
        from src.evaluation.promotion import promote_model

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        with patch("mlflow.set_tracking_uri"), \
             patch("src.evaluation.promotion.MlflowClient", return_value=mock_client):
            promote_model(challenger_version="1")

        # No archive, only promotion
        mock_client.transition_model_version_stage.assert_called_once()


class TestRunPromotionLogic:
    def _make_xy(self, tmp_path):
        from src.data.ingestion import generate_synthetic_churn_data, split_dataset
        from src.features.engineering import run_feature_engineering

        df = generate_synthetic_churn_data(n_samples=300, random_state=5)
        tr, v, te = split_dataset(df)
        X_tr, y_tr, X_v, y_v, X_te, y_te, _ = run_feature_engineering(tr, v, te, str(tmp_path))
        return X_te, y_te

    def test_auto_promotes_when_no_champion(self, tmp_path):
        from src.evaluation.promotion import run_promotion_logic
        from sklearn.ensemble import RandomForestClassifier

        X_te, y_te = self._make_xy(tmp_path)

        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        # Fit on the test data itself just to get a loadable model
        clf.fit(X_te, y_te)

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []  # no champion

        with patch("mlflow.set_tracking_uri"), \
             patch("src.evaluation.promotion.MlflowClient", return_value=mock_client), \
             patch("src.evaluation.promotion.load_production_model", return_value=None), \
             patch("mlflow.sklearn.load_model", return_value=clf), \
             patch("src.evaluation.promotion.promote_model") as mock_promote:

            result = run_promotion_logic(
                challenger_run_id="run-new",
                challenger_version="1",
                X_holdout=X_te,
                y_holdout=y_te,
            )

        assert result["decision"] == "promoted_no_champion"
        mock_promote.assert_called_once_with("1", "ChurnModel")

    def test_rejects_weaker_challenger(self, tmp_path):
        from src.evaluation.promotion import run_promotion_logic
        from sklearn.ensemble import RandomForestClassifier

        X_te, y_te = self._make_xy(tmp_path)
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X_te, y_te)

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        with patch("mlflow.set_tracking_uri"), \
             patch("src.evaluation.promotion.MlflowClient", return_value=mock_client), \
             patch("src.evaluation.promotion.load_production_model", return_value=clf), \
             patch("mlflow.sklearn.load_model", return_value=clf), \
             patch("mlflow.start_run") as mock_run, \
             patch("src.evaluation.promotion.PROMOTION_AUC_DELTA", 1.0):  # impossible threshold

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            result = run_promotion_logic(
                challenger_run_id="run-weak",
                challenger_version="2",
                X_holdout=X_te,
                y_holdout=y_te,
            )

        assert result["decision"] == "rejected"


# ──────────────────────────────────────────────────────────────────────────────
# src/monitoring/alerting.py – alert_on_drift
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertOnDrift:
    def test_alert_triggers_retrain(self):
        from src.monitoring.alerting import alert_on_drift

        with patch("src.monitoring.alerting.send_slack_notification") as mock_slack, \
             patch("src.monitoring.alerting.trigger_retraining_dag") as mock_trigger:

            mock_trigger.return_value = True
            alert_on_drift(report_dir="/tmp/reports", drift_score=0.45, auto_retrain=True)

        mock_slack.assert_called_once()
        mock_trigger.assert_called_once()

    def test_alert_no_auto_retrain(self):
        from src.monitoring.alerting import alert_on_drift

        with patch("src.monitoring.alerting.send_slack_notification") as mock_slack, \
             patch("src.monitoring.alerting.trigger_retraining_dag") as mock_trigger:

            alert_on_drift(report_dir="/tmp/reports", auto_retrain=False)

        mock_slack.assert_called_once()
        mock_trigger.assert_not_called()

    def test_alert_logs_error_when_trigger_fails(self):
        from src.monitoring.alerting import alert_on_drift

        with patch("src.monitoring.alerting.send_slack_notification"), \
             patch("src.monitoring.alerting.trigger_retraining_dag", return_value=False):

            # Should not raise even when trigger fails
            alert_on_drift(report_dir="/tmp/reports", auto_retrain=True)


# ──────────────────────────────────────────────────────────────────────────────
# src/monitoring/drift_detector.py – load_reference_data
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadReferenceData:
    def test_loads_existing_file(self, tmp_path):
        from src.data.ingestion import generate_synthetic_churn_data
        from src.monitoring.drift_detector import load_reference_data

        df = generate_synthetic_churn_data(n_samples=100)
        ref_path = str(tmp_path / "reference.csv")
        df.to_csv(ref_path, index=False)

        loaded = load_reference_data(reference_path=ref_path)
        assert len(loaded) == 100

    def test_raises_on_missing_file(self, tmp_path):
        from src.monitoring.drift_detector import load_reference_data

        with pytest.raises(FileNotFoundError):
            load_reference_data(reference_path=str(tmp_path / "nonexistent.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# src/evaluation/metrics.py – log_roc_curve
# ──────────────────────────────────────────────────────────────────────────────

class TestLogRocCurve:
    def test_does_not_crash(self):
        from src.evaluation.metrics import log_roc_curve

        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.6, 0.8])

        with patch("mlflow.log_artifact"):
            try:
                log_roc_curve(y_true, y_prob)
            except Exception as exc:
                pytest.fail(f"log_roc_curve raised: {exc}")
