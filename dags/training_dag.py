"""Airflow DAG: full training pipeline (extract → transform → train → evaluate → register)."""

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
    "retry_delay": timedelta(minutes=10),
}

DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "/opt/airflow/data/processed")
MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")


def extract(**context) -> str:
    """Load latest raw data and push path to XCom."""
    from src.data.ingestion import load_latest_raw_data
    import pandas as pd

    df = load_latest_raw_data(DATA_DIR)
    tmp_path = f"/tmp/raw_data_{context['ds_nodash']}.csv"
    df.to_csv(tmp_path, index=False)
    context["ti"].xcom_push(key="raw_data_path", value=tmp_path)
    logger.info("Extracted %d rows from %s", len(df), DATA_DIR)
    return tmp_path


def transform(**context) -> None:
    """Run feature engineering and save processed splits."""
    import pandas as pd
    from src.data.ingestion import split_dataset
    from src.features.engineering import run_feature_engineering

    raw_path = context["ti"].xcom_pull(key="raw_data_path", task_ids="extract")
    df = pd.read_csv(raw_path)

    train_df, val_df, test_df = split_dataset(df)
    run_feature_engineering(train_df, val_df, test_df, output_dir=PROCESSED_DIR)
    logger.info("Feature engineering complete")


def train(**context) -> None:
    """Train the model using processed features and log to MLflow."""
    import numpy as np
    from pathlib import Path
    from src.training.train import train_model, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    import mlflow

    processed = Path(PROCESSED_DIR)
    X_train = np.load(processed / "X_train.npy")
    y_train = np.load(processed / "y_train.npy")
    X_val = np.load(processed / "X_val.npy")
    y_val = np.load(processed / "y_val.npy")

    tags = {
        "trigger": context["dag_run"].conf.get("trigger", "scheduled"),
        "dag_id": context["dag"].dag_id,
        "run_date": context["ds"],
    }

    model, run_id, val_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_type=MODEL_TYPE,
        tags=tags,
    )

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="val_auc", value=val_metrics["auc"])
    logger.info("Training complete: run_id=%s val_auc=%.4f", run_id, val_metrics["auc"])


def evaluate(**context) -> None:
    """Evaluate model on test set and log metrics."""
    import numpy as np
    from pathlib import Path
    import mlflow
    from src.training.train import MLFLOW_TRACKING_URI
    from src.evaluation.metrics import evaluate_model

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train")
    processed = Path(PROCESSED_DIR)
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test.npy")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    with mlflow.start_run(run_id=run_id):
        test_metrics = evaluate_model(model, X_test, y_test, split_name="test")

    context["ti"].xcom_push(key="test_auc", value=test_metrics.get("test_auc", 0))
    logger.info("Test evaluation complete: test_auc=%.4f", test_metrics.get("test_auc", 0))


def register_model_task(**context) -> None:
    """Register trained model to MLflow registry and promote if AUC threshold met."""
    from src.training.train import register_model

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train")
    val_auc = context["ti"].xcom_pull(key="val_auc", task_ids="train")

    version = register_model(
        run_id=run_id,
        val_metrics={"auc": val_auc},
    )

    if version:
        context["ti"].xcom_push(key="model_version", value=version)
        logger.info("Model registered as version %s", version)
    else:
        logger.warning("Model registration failed or was skipped")


with DAG(
    dag_id="training_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end model training pipeline: extract → transform → train → evaluate → register",
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=train,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
    )

    extract_task >> transform_task >> train_task >> evaluate_task >> register_task
