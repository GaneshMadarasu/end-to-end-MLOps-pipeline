"""Airflow DAG: retraining pipeline triggered by drift detection or schedule."""

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")
logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "/opt/airflow/data/processed")
PREDICTIONS_STORE = os.getenv("PREDICTIONS_STORE", "/opt/airflow/data/predictions.csv")
MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def check_trigger(**context) -> str:
    """Determine trigger reason (drift_detected or scheduled)."""
    conf = context["dag_run"].conf or {}
    trigger = conf.get("trigger", "scheduled")
    logger.info("Retraining triggered by: %s", trigger)
    context["ti"].xcom_push(key="trigger", value=trigger)
    return trigger


def load_combined_data(**context) -> str:
    """Merge historical training data with recent production data."""
    import pandas as pd
    from pathlib import Path
    from src.data.ingestion import load_latest_raw_data

    # Load historical data
    try:
        historical_df = load_latest_raw_data(DATA_DIR)
    except FileNotFoundError:
        logger.warning("No historical data found; generating synthetic data")
        from src.data.ingestion import generate_synthetic_churn_data
        historical_df = generate_synthetic_churn_data()

    # Load recent production predictions with feedback
    prod_path = Path(PREDICTIONS_STORE)
    if prod_path.exists():
        prod_df = pd.read_csv(prod_path)
        # Only include rows with ground truth labels
        labeled = prod_df[prod_df["actual_label"].notna()].copy()
        if not labeled.empty:
            feature_cols = [
                "age", "tenure_months", "monthly_charges", "total_charges", "num_products",
                "has_tech_support", "has_online_security", "has_backup", "has_device_protection",
                "is_senior_citizen", "has_partner", "has_dependents",
                "contract_type", "payment_method", "internet_service",
            ]
            labeled = labeled[[c for c in feature_cols if c in labeled.columns] + ["actual_label"]]
            labeled = labeled.rename(columns={"actual_label": "churn"})
            historical_df = pd.concat([historical_df, labeled], ignore_index=True)
            logger.info(
                "Combined dataset: %d historical + %d production = %d total",
                len(historical_df) - len(labeled),
                len(labeled),
                len(historical_df),
            )

    tmp_path = f"/tmp/combined_data_{context['ds_nodash']}.csv"
    historical_df.to_csv(tmp_path, index=False)
    context["ti"].xcom_push(key="combined_data_path", value=tmp_path)
    return tmp_path


def retrain(**context) -> None:
    """Run training on combined data with drift trigger tag."""
    import pandas as pd
    import numpy as np
    from src.data.ingestion import split_dataset
    from src.features.engineering import run_feature_engineering
    from src.training.train import train_model, register_model

    trigger = context["ti"].xcom_pull(key="trigger", task_ids="check_trigger")
    data_path = context["ti"].xcom_pull(key="combined_data_path", task_ids="load_combined_data")

    df = pd.read_csv(data_path)
    train_df, val_df, test_df = split_dataset(df)
    run_feature_engineering(train_df, val_df, test_df, output_dir=PROCESSED_DIR)

    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    X_val = np.load(f"{PROCESSED_DIR}/X_val.npy")
    y_val = np.load(f"{PROCESSED_DIR}/y_val.npy")

    tags = {
        "trigger": trigger,
        "dag_id": context["dag"].dag_id,
        "run_date": context["ds"],
        "pipeline": "retraining",
    }

    model, run_id, val_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_type=MODEL_TYPE,
        run_name=f"retrain_{trigger}_{context['ds_nodash']}",
        tags=tags,
    )
    version = register_model(run_id, val_metrics=val_metrics)

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="model_version", value=version)
    context["ti"].xcom_push(key="val_auc", value=val_metrics["auc"])
    logger.info(
        "Retraining complete: run_id=%s version=%s val_auc=%.4f",
        run_id, version, val_metrics["auc"],
    )


def promote_retrained_model(**context) -> None:
    """Run champion/challenger promotion logic for the retrained model."""
    import numpy as np
    from src.evaluation.promotion import run_promotion_logic

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="retrain")
    version = context["ti"].xcom_pull(key="model_version", task_ids="retrain")

    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

    result = run_promotion_logic(
        challenger_run_id=run_id,
        challenger_version=str(version),
        X_holdout=X_test,
        y_holdout=y_test,
    )
    context["ti"].xcom_push(key="promotion_decision", value=result["decision"])
    logger.info("Promotion decision: %s", result["decision"])


def notify_completion(**context) -> None:
    """Send Slack notification on retraining completion."""
    from src.monitoring.alerting import send_slack_notification

    trigger = context["ti"].xcom_pull(key="trigger", task_ids="check_trigger")
    val_auc = context["ti"].xcom_pull(key="val_auc", task_ids="retrain")
    version = context["ti"].xcom_pull(key="model_version", task_ids="retrain")
    decision = context["ti"].xcom_pull(key="promotion_decision", task_ids="promote_model")

    message = (
        f":white_check_mark: *Retraining Complete*\n"
        f"Trigger: `{trigger}`\n"
        f"Model version: `{version}`\n"
        f"Val AUC: `{val_auc:.4f}`\n"
        f"Promotion decision: `{decision}`"
    )
    send_slack_notification(message)
    logger.info("Completion notification sent")


with DAG(
    dag_id="retraining_pipeline",
    default_args=DEFAULT_ARGS,
    description="Drift-triggered or scheduled model retraining pipeline",
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "retraining"],
) as dag:

    check_trigger_task = PythonOperator(
        task_id="check_trigger",
        python_callable=check_trigger,
    )

    load_data_task = PythonOperator(
        task_id="load_combined_data",
        python_callable=load_combined_data,
    )

    retrain_task = PythonOperator(
        task_id="retrain",
        python_callable=retrain,
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_retrained_model,
    )

    notify_task = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
    )

    check_trigger_task >> load_data_task >> retrain_task >> promote_task >> notify_task
