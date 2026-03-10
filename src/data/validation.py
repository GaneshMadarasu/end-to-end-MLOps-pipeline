"""Data validation module: schema checks, null rates, value range assertions."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.ingestion import CATEGORICAL_FEATURES, FEATURE_SCHEMA, NUMERICAL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for schema validation outcome."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        """Append an error and mark validation as failed."""
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Append a warning (does not fail validation)."""
        self.warnings.append(msg)

    def summary(self) -> str:
        """Return human-readable validation summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            lines.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


# Expected value ranges for numerical features
FEATURE_RANGES: Dict[str, Dict[str, Optional[float]]] = {
    "age": {"min": 0, "max": 120},
    "tenure_months": {"min": 0, "max": 600},
    "monthly_charges": {"min": 0, "max": 10000},
    "total_charges": {"min": 0, "max": None},
    "num_products": {"min": 1, "max": 20},
    "has_tech_support": {"min": 0, "max": 1},
    "has_online_security": {"min": 0, "max": 1},
    "has_backup": {"min": 0, "max": 1},
    "has_device_protection": {"min": 0, "max": 1},
    "is_senior_citizen": {"min": 0, "max": 1},
    "has_partner": {"min": 0, "max": 1},
    "has_dependents": {"min": 0, "max": 1},
}

# Expected categories for categorical features
EXPECTED_CATEGORIES: Dict[str, List[str]] = {
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "internet_service": ["DSL", "Fiber optic", "No"],
}

MAX_NULL_RATE = 0.05  # 5% max null rate per column


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> ValidationResult:
    """Validate DataFrame columns, dtypes, nulls, ranges, and categories.

    Args:
        df: DataFrame to validate.
        require_target: Whether to require the target column.

    Returns:
        ValidationResult with pass/fail status and details.
    """
    result = ValidationResult(passed=True)
    result.stats["num_rows"] = len(df)
    result.stats["num_cols"] = len(df.columns)

    # --- Column presence ---
    expected_cols = list(FEATURE_SCHEMA.keys())
    if not require_target:
        expected_cols = [c for c in expected_cols if c != TARGET_COLUMN]

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        result.add_error(f"Missing columns: {sorted(missing_cols)}")

    extra_cols = set(df.columns) - set(FEATURE_SCHEMA.keys())
    if extra_cols:
        result.add_warning(f"Unexpected extra columns: {sorted(extra_cols)}")

    # Only validate columns that are present
    present_cols = [c for c in expected_cols if c in df.columns]

    # --- Null rates ---
    null_rates = df[present_cols].isnull().mean()
    result.stats["null_rates"] = null_rates.to_dict()
    for col, rate in null_rates.items():
        if rate > MAX_NULL_RATE:
            result.add_error(f"Column '{col}' has null rate {rate:.1%} (max {MAX_NULL_RATE:.0%})")
        elif rate > 0:
            result.add_warning(f"Column '{col}' has null rate {rate:.1%}")

    # --- Value ranges ---
    for col, bounds in FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if bounds["min"] is not None and (series < bounds["min"]).any():
            n_violations = (series < bounds["min"]).sum()
            result.add_error(f"Column '{col}': {n_violations} values below min={bounds['min']}")
        if bounds["max"] is not None and (series > bounds["max"]).any():
            n_violations = (series > bounds["max"]).sum()
            result.add_error(f"Column '{col}': {n_violations} values above max={bounds['max']}")

    # --- Categorical values ---
    for col, expected_vals in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].dropna().unique())
        unknown_vals = actual_vals - set(expected_vals)
        if unknown_vals:
            result.add_warning(f"Column '{col}' has unknown categories: {unknown_vals}")

    # --- Row count ---
    if len(df) == 0:
        result.add_error("DataFrame is empty")
    elif len(df) < 100:
        result.add_warning(f"DataFrame has very few rows: {len(df)}")

    # --- Target distribution ---
    if require_target and TARGET_COLUMN in df.columns:
        churn_rate = df[TARGET_COLUMN].mean()
        result.stats["churn_rate"] = churn_rate
        if churn_rate < 0.01 or churn_rate > 0.99:
            result.add_warning(f"Extreme class imbalance: churn_rate={churn_rate:.2%}")

    logger.info(result.summary())
    return result


def validate_inference_payload(data: Dict[str, Any]) -> ValidationResult:
    """Validate a single inference request payload.

    Args:
        data: Dictionary with feature values.

    Returns:
        ValidationResult.
    """
    df = pd.DataFrame([data])
    return validate_schema(df, require_target=False)
