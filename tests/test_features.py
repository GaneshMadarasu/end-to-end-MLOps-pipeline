"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import generate_synthetic_churn_data, split_dataset, TARGET_COLUMN
from src.data.validation import validate_schema
from src.features.engineering import (
    build_preprocessor,
    fit_preprocessor,
    get_feature_names,
    run_feature_engineering,
    transform_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic churn dataset for testing."""
    return generate_synthetic_churn_data(n_samples=500, random_state=42)


@pytest.fixture
def split_dfs(sample_df):
    """Split sample_df into train/val/test."""
    return split_dataset(sample_df)


class TestDataIngestion:
    """Tests for src/data/ingestion.py."""

    def test_generate_shape(self, sample_df):
        assert sample_df.shape == (500, 16)

    def test_target_binary(self, sample_df):
        assert set(sample_df[TARGET_COLUMN].unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self, sample_df):
        rate = sample_df[TARGET_COLUMN].mean()
        assert 0.05 <= rate <= 0.6, f"Unexpected churn rate: {rate:.2%}"

    def test_split_sizes(self, split_dfs):
        train_df, val_df, test_df = split_dfs
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 500
        # test ~20% of 500 = 100, val ~9% of 400 = ~36-40
        assert 90 <= len(test_df) <= 110
        assert len(val_df) > 0

    def test_no_duplicates_across_splits(self, split_dfs):
        train_df, val_df, test_df = split_dfs
        # Indices should be disjoint
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


class TestDataValidation:
    """Tests for src/data/validation.py."""

    def test_valid_data_passes(self, sample_df):
        result = validate_schema(sample_df)
        assert result.passed, result.summary()

    def test_missing_column_fails(self, sample_df):
        df_missing = sample_df.drop(columns=["age"])
        result = validate_schema(df_missing)
        assert not result.passed
        assert any("age" in e for e in result.errors)

    def test_high_null_rate_fails(self, sample_df):
        df_nulls = sample_df.copy()
        df_nulls.loc[:250, "monthly_charges"] = None  # >5% nulls
        result = validate_schema(df_nulls)
        assert not result.passed

    def test_out_of_range_value_fails(self, sample_df):
        df_bad = sample_df.copy()
        df_bad.loc[0, "age"] = 200  # > max 120
        result = validate_schema(df_bad)
        assert not result.passed


class TestFeatureEngineering:
    """Tests for src/features/engineering.py."""

    def test_preprocessor_builds(self):
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_fit_and_transform(self, sample_df):
        preprocessor = fit_preprocessor(sample_df)
        X, y = transform_features(sample_df, preprocessor)
        assert X.shape[0] == len(sample_df)
        assert X.ndim == 2
        assert y is not None
        assert len(y) == len(sample_df)

    def test_no_nan_after_transform(self, sample_df):
        preprocessor = fit_preprocessor(sample_df)
        X, _ = transform_features(sample_df, preprocessor)
        assert not np.isnan(X).any()

    def test_feature_names_count(self, sample_df):
        preprocessor = fit_preprocessor(sample_df)
        names = get_feature_names(preprocessor)
        # 12 numerical + OHE expanded categoricals
        assert len(names) >= 12

    def test_run_feature_engineering(self, split_dfs, tmp_path):
        train_df, val_df, test_df = split_dfs
        result = run_feature_engineering(
            train_df, val_df, test_df, output_dir=str(tmp_path)
        )
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = result
        assert X_train.shape[0] == len(train_df)
        assert X_val.shape[0] == len(val_df)
        assert X_test.shape[0] == len(test_df)
        # All splits have same number of features
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

    def test_transform_unseen_categories(self, sample_df):
        """OneHotEncoder should handle unknown categories without error."""
        preprocessor = fit_preprocessor(sample_df)
        test_df = sample_df.copy()
        test_df["contract_type"] = "Unknown Contract"
        X, _ = transform_features(test_df, preprocessor)
        assert not np.isnan(X).any()
