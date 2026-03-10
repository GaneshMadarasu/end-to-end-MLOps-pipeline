"""Drift detection using Evidently AI: data drift, target drift, and data quality reports."""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PREDICTIONS_STORE = os.getenv("PREDICTIONS_STORE", "data/predictions.csv")
REPORTS_DIR = os.getenv("REPORTS_DIR", "monitoring/evidently_reports")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/processed/reference.csv")

FEATURE_COLUMNS = [
    "age", "tenure_months", "monthly_charges", "total_charges", "num_products",
    "has_tech_support", "has_online_security", "has_backup", "has_device_protection",
    "is_senior_citizen", "has_partner", "has_dependents",
    "contract_type", "payment_method", "internet_service",
]


def load_reference_data(reference_path: str = REFERENCE_DATA_PATH) -> pd.DataFrame:
    """Load the training reference dataset for drift comparison.

    Args:
        reference_path: Path to reference CSV file.

    Returns:
        Reference DataFrame.

    Raises:
        FileNotFoundError: If reference data is not found.
    """
    path = Path(reference_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {reference_path}. "
            "Run training pipeline first to generate reference data."
        )
    df = pd.read_csv(path)
    logger.info("Loaded reference data: %d rows from %s", len(df), reference_path)
    return df


def load_production_data(
    predictions_store: str = PREDICTIONS_STORE,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """Load recent production predictions for drift analysis.

    Args:
        predictions_store: Path to predictions CSV file.
        lookback_hours: How many hours of data to include.

    Returns:
        Production DataFrame from last N hours.
    """
    path = Path(predictions_store)
    if not path.exists():
        logger.warning("Predictions store not found: %s", predictions_store)
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        logger.warning("Predictions store is empty")
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.utcnow() - timedelta(hours=lookback_hours)
        df = df[df["timestamp"] >= cutoff]

    logger.info("Loaded %d production records (last %dh)", len(df), lookback_hours)
    return df


def generate_drift_reports(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_date: Optional[str] = None,
    reports_dir: str = REPORTS_DIR,
) -> Tuple[str, bool]:
    """Generate Evidently drift reports and save as HTML.

    Args:
        reference_df: Reference (training) dataset.
        current_df: Current production dataset.
        report_date: Date string YYYY-MM-DD for output directory.
        reports_dir: Base directory for reports.

    Returns:
        Tuple of (report_directory_path, drift_detected_flag).
    """
    if report_date is None:
        report_date = date.today().isoformat()

    output_dir = Path(reports_dir) / report_date
    output_dir.mkdir(parents=True, exist_ok=True)

    drift_detected = False

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import ColumnDriftMetric

        # Ensure common columns
        common_cols = [c for c in FEATURE_COLUMNS if c in reference_df.columns and c in current_df.columns]
        ref_subset = reference_df[common_cols].copy()
        cur_subset = current_df[common_cols].copy()

        # Data Drift Report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=ref_subset, current_data=cur_subset)
        drift_report_path = str(output_dir / "data_drift.html")
        drift_report.save_html(drift_report_path)

        # Parse drift score
        drift_result = drift_report.as_dict()
        metrics_list = drift_result.get("metrics", [])
        for metric in metrics_list:
            result = metric.get("result", {})
            drift_share = result.get("share_of_drifted_columns", 0)
            if drift_share > DRIFT_THRESHOLD:
                drift_detected = True
                logger.warning(
                    "DRIFT DETECTED: share_of_drifted_columns=%.2f > threshold=%.2f",
                    drift_share,
                    DRIFT_THRESHOLD,
                )
            break

        # Data Quality Report
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(reference_data=ref_subset, current_data=cur_subset)
        quality_report.save_html(str(output_dir / "data_quality.html"))

        # Target Drift Report (if prediction column exists)
        if "prediction" in current_df.columns and "churn" in reference_df.columns:
            ref_with_target = reference_df[common_cols + ["churn"]].rename(columns={"churn": "prediction"})
            cur_with_target = current_df[common_cols + ["prediction"]]

            target_drift_report = Report(metrics=[ColumnDriftMetric(column_name="prediction")])
            target_drift_report.run(reference_data=ref_with_target, current_data=cur_with_target)
            target_drift_report.save_html(str(output_dir / "target_drift.html"))

        logger.info("Drift reports saved to %s (drift_detected=%s)", output_dir, drift_detected)

    except ImportError as exc:
        logger.error("Evidently not installed: %s", exc)
        # Write a simple placeholder
        placeholder = output_dir / "drift_not_available.txt"
        placeholder.write_text("Evidently not installed. Install with: pip install evidently")
    except Exception as exc:
        logger.error("Error generating drift reports: %s", exc, exc_info=True)

    return str(output_dir), drift_detected


def run_drift_detection(
    lookback_hours: int = 24,
    report_date: Optional[str] = None,
) -> Tuple[bool, str]:
    """Full drift detection pipeline: load data → generate reports → return result.

    Args:
        lookback_hours: Hours of production data to analyze.
        report_date: Optional override for report date.

    Returns:
        Tuple of (drift_detected, report_directory_path).
    """
    logger.info("Starting drift detection (lookback=%dh)", lookback_hours)

    try:
        reference_df = load_reference_data()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return False, ""

    current_df = load_production_data(lookback_hours=lookback_hours)
    if current_df.empty:
        logger.warning("No production data available for drift analysis")
        return False, ""

    if len(current_df) < 50:
        logger.warning(
            "Too few production samples (%d) for reliable drift detection (need >= 50)",
            len(current_df),
        )

    report_dir, drift_detected = generate_drift_reports(
        reference_df, current_df, report_date=report_date
    )
    return drift_detected, report_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detected, report_path = run_drift_detection()
    if detected:
        logger.warning("Drift detected! Reports at: %s", report_path)
    else:
        logger.info("No significant drift detected. Reports at: %s", report_path)
