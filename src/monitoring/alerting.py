"""Alerting module: triggers Airflow retraining DAG and sends Slack/email notifications."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
RETRAINING_DAG_ID = "retraining_pipeline"


def trigger_retraining_dag(
    trigger_reason: str = "drift_detected",
    conf: Optional[Dict] = None,
) -> bool:
    """Trigger the retraining DAG via Airflow REST API.

    Args:
        trigger_reason: Reason string logged with the DAG run.
        conf: Optional configuration dict passed to the DAG run.

    Returns:
        True if the DAG was triggered successfully, False otherwise.
    """
    url = f"{AIRFLOW_API_URL}/dags/{RETRAINING_DAG_ID}/dagRuns"
    payload = {
        "conf": conf or {"trigger": trigger_reason},
        "logical_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    try:
        response = requests.post(
            url,
            json=payload,
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        run_id = response.json().get("dag_run_id", "unknown")
        logger.info(
            "Retraining DAG triggered successfully: dag_run_id=%s reason=%s",
            run_id,
            trigger_reason,
        )
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to trigger retraining DAG: %s", exc)
        return False


def send_slack_notification(
    message: str,
    webhook_url: str = SLACK_WEBHOOK_URL,
) -> bool:
    """Send a notification to Slack via incoming webhook.

    Args:
        message: Message text to send.
        webhook_url: Slack incoming webhook URL.

    Returns:
        True if sent successfully, False otherwise.
    """
    if not webhook_url:
        logger.debug("SLACK_WEBHOOK_URL not configured; skipping Slack notification")
        return False

    payload = {
        "text": message,
        "username": "MLOps Bot",
        "icon_emoji": ":robot_face:",
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Slack notification sent")
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to send Slack notification: %s", exc)
        return False


def alert_on_drift(
    report_dir: str,
    drift_score: Optional[float] = None,
    auto_retrain: bool = True,
) -> None:
    """Send drift alerts and optionally trigger retraining.

    Args:
        report_dir: Path to drift reports directory.
        drift_score: Optional numeric drift score for the message.
        auto_retrain: Whether to automatically trigger retraining DAG.
    """
    score_str = f" (score={drift_score:.3f})" if drift_score is not None else ""
    message = (
        f":warning: *MLOps Drift Alert*{score_str}\n"
        f"Data drift detected in production predictions.\n"
        f"Reports: `{report_dir}`\n"
        f"{'Retraining pipeline triggered automatically.' if auto_retrain else 'Manual retraining required.'}"
    )

    send_slack_notification(message)

    if auto_retrain:
        triggered = trigger_retraining_dag(
            trigger_reason="drift_detected",
            conf={"report_dir": report_dir, "drift_score": drift_score},
        )
        if not triggered:
            logger.error(
                "CRITICAL: Drift detected but retraining DAG could not be triggered. "
                "Manual intervention required."
            )
