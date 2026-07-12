"""Tests for src/features.py: feature calculation correctness, edge cases."""

import pandas as pd
import numpy as np
import pytest

from src.features import validate_input, check_feature_correlations, build_feature_table


class TestValidateInput:
    """validate_input() must reject None, empty, or column-missing DataFrames."""

    def test_none_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_input(None, "test")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_input(pd.DataFrame(), "test")

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"user": ["a"], "wrong": ["x"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input(df, "test")

    def test_valid_passes(self):
        df = pd.DataFrame({"user": ["a"], "date": pd.to_datetime(["2010-01-04"])})
        validate_input(df, "test")  # should not raise


class TestFeatureCorrelations:
    """check_feature_correlations() must detect highly correlated pairs."""

    def test_high_correlation_detected(self):
        np.random.seed(42)
        x = np.random.randn(100)
        y = x * 0.98 + np.random.randn(100) * 0.01  # almost perfectly correlated
        z = np.random.randn(100)  # independent
        df = pd.DataFrame({"user": ["u{}".format(i) for i in range(100)], "a": x, "b": y, "c": z})
        # Should not raise — just logs a warning
        check_feature_correlations(df, threshold=0.95)

    def test_no_high_correlation(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "user": ["u{}".format(i) for i in range(100)],
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
        })
        check_feature_correlations(df, threshold=0.95)


class TestBuildFeatureTable:
    """build_feature_table() must merge correctly and fill NAs."""

    def test_merge_adds_device_columns(self):
        logon = pd.DataFrame({
            "user": ["a", "b"],
            "login_count": [10, 20],
            "off_hour_logins": [5, 10],
            "weekend_logins": [1, 2],
            "late_night_logins": [3, 4],
            "unique_pcs_logon": [2, 3],
            "off_hour_ratio": [0.5, 0.5],
            "weekend_ratio": [0.1, 0.1],
            "avg_session_gap": [3600, 7200],
        })
        device = pd.DataFrame({
            "user": ["a"],
            "device_connections": [100],
            "unique_pcs_device": [5],
        })
        result = build_feature_table(logon, device)
        assert "device_connections" in result.columns
        assert "unique_pcs_device" in result.columns
        user_a = result[result["user"] == "a"]
        user_b = result[result["user"] == "b"]
        assert user_a["device_connections"].iloc[0] == 100
        assert user_b["device_connections"].iloc[0] == 0

    def test_derived_features_computed(self):
        logon = pd.DataFrame({
            "user": ["a"],
            "login_count": [100],
            "off_hour_logins": [50],
            "weekend_logins": [10],
            "late_night_logins": [20],
            "unique_pcs_logon": [5],
            "off_hour_ratio": [0.5],
            "weekend_ratio": [0.1],
            "avg_session_gap": [3600],
        })
        device = pd.DataFrame({
            "user": ["a"],
            "device_connections": [75],
            "unique_pcs_device": [3],
        })
        result = build_feature_table(logon, device)
        row = result.iloc[0]
        assert row["pc_diversity_score"] == round(5 / 100, 4)
        assert row["device_to_login_ratio"] == round(75 / 100, 4)

    def test_zero_login_divide_by_zero(self):
        """When login_count is 0, ratios must be 0, not NaN or Inf."""
        logon = pd.DataFrame({
            "user": ["a"],
            "login_count": [0],
            "off_hour_logins": [0],
            "weekend_logins": [0],
            "late_night_logins": [0],
            "unique_pcs_logon": [0],
            "off_hour_ratio": [0.0],
            "weekend_ratio": [0.0],
            "avg_session_gap": [0.0],
        })
        device = pd.DataFrame({
            "user": ["a"],
            "device_connections": [0],
            "unique_pcs_device": [0],
        })
        result = build_feature_table(logon, device)
        row = result.iloc[0]
        assert row["pc_diversity_score"] == 0.0
        assert row["device_to_login_ratio"] == 0.0
