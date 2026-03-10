"""Data ingestion module: loads, generates, and partitions churn data."""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Feature schema
FEATURE_SCHEMA = {
    "age": "int64",
    "tenure_months": "int64",
    "monthly_charges": "float64",
    "total_charges": "float64",
    "num_products": "int64",
    "has_tech_support": "int64",
    "has_online_security": "int64",
    "has_backup": "int64",
    "has_device_protection": "int64",
    "is_senior_citizen": "int64",
    "has_partner": "int64",
    "has_dependents": "int64",
    "contract_type": "object",
    "payment_method": "object",
    "internet_service": "object",
    "churn": "int64",
}

NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "has_tech_support",
    "has_online_security",
    "has_backup",
    "has_device_protection",
    "is_senior_citizen",
    "has_partner",
    "has_dependents",
]

CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]

TARGET_COLUMN = "churn"


def generate_synthetic_churn_data(
    n_samples: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic customer churn dataset.

    Args:
        n_samples: Number of rows to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with churn dataset (features + target).
    """
    rng = np.random.RandomState(random_state)

    # Demographics
    age = rng.randint(18, 80, n_samples)
    is_senior_citizen = (age >= 65).astype(int)
    has_partner = rng.binomial(1, 0.5, n_samples)
    has_dependents = rng.binomial(1, 0.3, n_samples)

    # Service tenure and charges
    tenure_months = rng.randint(1, 72, n_samples)
    monthly_charges = rng.uniform(20, 120, n_samples).round(2)
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.9, 1.1, n_samples)).round(2)

    # Products and services
    num_products = rng.randint(1, 5, n_samples)
    has_tech_support = rng.binomial(1, 0.4, n_samples)
    has_online_security = rng.binomial(1, 0.45, n_samples)
    has_backup = rng.binomial(1, 0.35, n_samples)
    has_device_protection = rng.binomial(1, 0.35, n_samples)

    # Categorical features
    contract_types = ["Month-to-month", "One year", "Two year"]
    contract_weights = [0.55, 0.25, 0.20]
    contract_type = rng.choice(contract_types, n_samples, p=contract_weights)

    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    payment_method = rng.choice(payment_methods, n_samples)

    internet_services = ["DSL", "Fiber optic", "No"]
    internet_service = rng.choice(internet_services, n_samples, p=[0.4, 0.45, 0.15])

    # Churn probability (realistic signal)
    churn_logit = (
        -3.0
        + 0.8 * (contract_type == "Month-to-month").astype(float)
        - 0.5 * (contract_type == "Two year").astype(float)
        + 0.5 * (payment_method == "Electronic check").astype(float)
        + 0.3 * (internet_service == "Fiber optic").astype(float)
        - 0.03 * tenure_months
        + 0.01 * monthly_charges
        - 0.3 * has_tech_support
        - 0.3 * has_online_security
        + 0.2 * is_senior_citizen
        + rng.normal(0, 0.5, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (rng.uniform(0, 1, n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_products": num_products,
            "has_tech_support": has_tech_support,
            "has_online_security": has_online_security,
            "has_backup": has_backup,
            "has_device_protection": has_device_protection,
            "is_senior_citizen": is_senior_citizen,
            "has_partner": has_partner,
            "has_dependents": has_dependents,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "churn": churn,
        }
    )

    logger.info(
        "Generated synthetic dataset: %d rows, churn rate=%.2f%%",
        n_samples,
        churn.mean() * 100,
    )
    return df


def load_or_generate_data(
    data_path: Optional[str] = None,
    n_samples: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load data from disk or generate synthetically if not available.

    Args:
        data_path: Path to existing CSV file, or None to generate.
        n_samples: Number of samples if generating.
        random_state: Random seed.

    Returns:
        DataFrame with churn data.
    """
    if data_path and Path(data_path).exists():
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)
    else:
        logger.info("No data file found. Generating synthetic data (%d rows).", n_samples)
        df = generate_synthetic_churn_data(n_samples=n_samples, random_state=random_state)
    return df


def save_raw_data(df: pd.DataFrame, base_dir: str = "data/raw", partition_date: Optional[str] = None) -> str:
    """Save raw data partitioned by date.

    Args:
        df: DataFrame to save.
        base_dir: Root directory for raw data.
        partition_date: Date string YYYY-MM-DD; defaults to today.

    Returns:
        Path to saved CSV file.
    """
    if partition_date is None:
        partition_date = date.today().isoformat()

    output_dir = Path(base_dir) / partition_date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "churn_raw.csv"

    df.to_csv(output_path, index=False)
    logger.info("Raw data saved to %s (%d rows)", output_path, len(df))
    return str(output_path)


def load_latest_raw_data(base_dir: str = "data/raw") -> pd.DataFrame:
    """Load the most recently partitioned raw data.

    Args:
        base_dir: Root directory for raw data.

    Returns:
        DataFrame with latest raw data.

    Raises:
        FileNotFoundError: If no partitions exist.
    """
    raw_dir = Path(base_dir)
    partitions = sorted(raw_dir.glob("*/churn_raw.csv"), reverse=True)

    if not partitions:
        raise FileNotFoundError(f"No raw data partitions found in {base_dir}")

    latest = partitions[0]
    logger.info("Loading latest raw data from %s", latest)
    return pd.read_csv(latest)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.

    Args:
        df: Full dataset.
        test_size: Fraction for test set.
        val_size: Fraction of train set to use for validation.
        random_state: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COLUMN]
    )
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_df[TARGET_COLUMN],
    )

    logger.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def ingest_data(
    data_path: Optional[str] = None,
    base_dir: str = "data/raw",
    n_samples: int = 10000,
    random_state: int = 42,
    partition_date: Optional[str] = None,
) -> str:
    """Full ingestion pipeline: load/generate, validate, and save.

    Args:
        data_path: Optional path to source CSV.
        base_dir: Directory for raw data partitions.
        n_samples: Number of samples if generating.
        random_state: Random seed.
        partition_date: Date partition string.

    Returns:
        Path to saved raw data file.
    """
    df = load_or_generate_data(data_path, n_samples, random_state)
    output_path = save_raw_data(df, base_dir, partition_date)
    return output_path
