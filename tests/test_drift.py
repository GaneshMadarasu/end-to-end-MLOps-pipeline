"""Tests for drift detection module."""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.data.ingestion import generate_synthetic_churn_data
from src.monitoring.drift_detector import (
    load_production_data,
    generate_drift_reports,
    run_drift_detection,
    FEATURE_COLUMNS,
)


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Reference dataset for drift tests."""
    return generate_synthetic_churn_data(n_samples=500, random_state=42)


@pytest.fixture
def current_df_no_drift(reference_df) -> pd.DataFrame:
    """Production data from same distribution (no drift)."""
    return generate_synthetic_churn_data(n_samples=200, random_state=99)


@pytest.fixture
def current_df_with_drift() -> pd.DataFrame:
    """Production data with artificial drift (all elderly, high charges)."""
    df = generate_synthetic_churn_data(n_samples=200, random_state=42)
    df["age"] = 75  # All customers now 75 years old
    df["monthly_charges"] = 200.0  # Much higher charges
    df["contract_type"] = "Two year"  # All same contract
    return df


@pytest.fixture
def predictions_csv(tmp_path, current_df_no_drift) -> str:
    """Write a predictions CSV to tmp_path and return its path."""
    from datetime import datetime, timezone

    df = current_df_no_drift.copy()
    df["prediction_id"] = [f"pred-{i}" for i in range(len(df))]
    df["timestamp"] = datetime.now(timezone.utc).isoformat()
    df["prediction"] = np.random.randint(0, 2, len(df))
    df["probability"] = np.random.uniform(0, 1, len(df))
    df["actual_label"] = ""

    path = tmp_path / "predictions.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadProductionData:
    """Tests for load_production_data()."""

    def test_loads_csv(self, predictions_csv):
        df = load_production_data(predictions_store=predictions_csv, lookback_hours=9999)
        assert len(df) > 0

    def test_returns_empty_if_no_file(self, tmp_path):
        df = load_production_data(predictions_store=str(tmp_path / "missing.csv"))
        assert df.empty

    def test_time_filter_excludes_old(self, tmp_path):
        """Records with very old timestamps should be filtered out."""
        df = pd.DataFrame({
            "timestamp": ["2020-01-01T00:00:00+00:00"] * 5,
            "prediction": [0] * 5,
        })
        path = str(tmp_path / "old_predictions.csv")
        df.to_csv(path, index=False)
        result = load_production_data(predictions_store=path, lookback_hours=1)
        assert result.empty


class TestGenerateDriftReports:
    """Tests for generate_drift_reports()."""

    def test_reports_directory_created(self, reference_df, current_df_no_drift, tmp_path):
        with patch("src.monitoring.drift_detector.REPORTS_DIR", str(tmp_path)):
            report_dir, _ = generate_drift_reports(
                reference_df, current_df_no_drift,
                report_date="2024-01-15",
                reports_dir=str(tmp_path),
            )
        assert Path(report_dir).exists()

    def test_returns_bool_drift_flag(self, reference_df, current_df_no_drift, tmp_path):
        _, drift_detected = generate_drift_reports(
            reference_df, current_df_no_drift,
            report_date="2024-01-15",
            reports_dir=str(tmp_path),
        )
        assert isinstance(drift_detected, bool)

    def test_no_crash_on_mismatched_columns(self, reference_df, tmp_path):
        """Partial columns should not crash the report generation."""
        current = reference_df[["age", "monthly_charges", "contract_type", "churn"]].copy()
        try:
            generate_drift_reports(
                reference_df, current,
                report_date="2024-01-15",
                reports_dir=str(tmp_path),
            )
        except Exception as exc:
            pytest.fail(f"generate_drift_reports raised unexpectedly: {exc}")


class TestRunDriftDetection:
    """Integration tests for run_drift_detection()."""

    def test_returns_false_when_no_reference(self, tmp_path):
        with patch("src.monitoring.drift_detector.REFERENCE_DATA_PATH", str(tmp_path / "missing.csv")):
            detected, report_path = run_drift_detection()
        assert not detected
        assert report_path == ""

    def test_returns_false_when_no_production_data(self, tmp_path, reference_df):
        ref_path = str(tmp_path / "reference.csv")
        reference_df.to_csv(ref_path, index=False)

        with patch("src.monitoring.drift_detector.REFERENCE_DATA_PATH", ref_path), \
             patch("src.monitoring.drift_detector.PREDICTIONS_STORE", str(tmp_path / "missing.csv")):
            detected, report_path = run_drift_detection()
        assert not detected
