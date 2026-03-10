"""Airflow DAG: data ingestion and schema validation pipeline."""

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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data/raw")
N_SAMPLES = int(os.getenv("N_SAMPLES", "10000"))


def generate_and_save_data(**context) -> str:
    """Generate synthetic churn data and save partitioned by today's date."""
    from src.data.ingestion import ingest_data

    partition_date = context["ds"]
    output_path = ingest_data(
        data_path=None,
        base_dir=DATA_DIR,
        n_samples=N_SAMPLES,
        partition_date=partition_date,
    )
    logger.info("Data ingested to: %s", output_path)
    context["ti"].xcom_push(key="raw_data_path", value=output_path)
    return output_path


def validate_data(**context) -> None:
    """Run schema validation on the ingested data."""
    import pandas as pd
    from src.data.validation import validate_schema

    raw_data_path = context["ti"].xcom_pull(key="raw_data_path", task_ids="ingest_data")
    df = pd.read_csv(raw_data_path)
    result = validate_schema(df)

    if not result.passed:
        raise ValueError(f"Data validation FAILED:\n{result.summary()}")

    logger.info("Data validation PASSED: %d rows, churn_rate=%.2f%%",
                len(df), result.stats.get("churn_rate", 0) * 100)


def save_reference_data(**context) -> None:
    """Copy today's data as reference dataset for drift detection."""
    import pandas as pd
    from pathlib import Path

    raw_data_path = context["ti"].xcom_pull(key="raw_data_path", task_ids="ingest_data")
    df = pd.read_csv(raw_data_path)

    ref_dir = Path("/opt/airflow/data/processed")
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / "reference.csv"
    df.to_csv(ref_path, index=False)
    logger.info("Reference data saved to %s", ref_path)


with DAG(
    dag_id="data_ingestion",
    default_args=DEFAULT_ARGS,
    description="Generate and validate synthetic churn data",
    schedule="@daily",
    catchup=False,
    tags=["mlops", "ingestion"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=generate_and_save_data,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    reference_task = PythonOperator(
        task_id="save_reference_data",
        python_callable=save_reference_data,
    )

    ingest_task >> validate_task >> reference_task
