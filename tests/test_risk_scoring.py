"""Tests for src/risk_scoring.py: threshold assignment, explanation generation."""

import pandas as pd
import pytest

from src.risk_scoring import assign_risk_level, generate_explanation, _load_population_means, THRESHOLD_HIGH, THRESHOLD_MEDIUM


class TestAssignRiskLevel:
    """assign_risk_level() must map numeric scores to correct categories."""

    def test_high_at_threshold(self):
        assert assign_risk_level(THRESHOLD_HIGH) == "High"

    def test_high_above_threshold(self):
        assert assign_risk_level(THRESHOLD_HIGH + 1) == "High"

    def test_medium_at_threshold(self):
        assert assign_risk_level(THRESHOLD_MEDIUM) == "Medium"

    def test_medium_mid_range(self):
        mid = (THRESHOLD_HIGH + THRESHOLD_MEDIUM) / 2
        assert assign_risk_level(mid) == "Medium"

    def test_low_below_threshold(self):
        assert assign_risk_level(THRESHOLD_MEDIUM - 10) == "Low"

    def test_low_zero(self):
        assert assign_risk_level(0) == "Low"

    def test_low_negative(self):
        assert assign_risk_level(-5) == "Low"

    def test_high_max(self):
        assert assign_risk_level(100) == "High"

    def test_edge_case_69(self):
        assert assign_risk_level(69.9) == "Medium"

    def test_edge_case_70(self):
        assert assign_risk_level(70.0) == "High"

    def test_edge_case_39(self):
        assert assign_risk_level(39.9) == "Low"

    def test_edge_case_40(self):
        assert assign_risk_level(40.0) == "Medium"


class TestGenerateExplanation:
    """generate_explanation() must flag out-of-range behaviour and mention population averages."""

    def setup_method(self):
        # Load population means so explanations reference real averages
        try:
            risk_df = pd.read_csv("outputs/risk_report.csv")
            _load_population_means(risk_df)
        except (FileNotFoundError, Exception):
            # Fallback: set minimal means for test isolation
            _load_population_means(pd.DataFrame({
                "login_count": [100], "off_hour_logins": [10], "weekend_logins": [1],
                "late_night_logins": [2], "unique_pcs_logon": [3], "off_hour_ratio": [0.1],
                "weekend_ratio": [0.01], "pc_diversity_score": [0.02],
                "device_connections": [5], "unique_pcs_device": [1],
                "device_to_login_ratio": [0.05], "avg_session_gap": [50000],
            }))

    def test_normal_user_no_flags(self):
        row = pd.Series({
            "login_count": 10, "off_hour_logins": 1, "weekend_logins": 0,
            "late_night_logins": 0, "unique_pcs_logon": 1, "off_hour_ratio": 0.1,
            "weekend_ratio": 0.0, "pc_diversity_score": 0.1,
            "device_connections": 1, "unique_pcs_device": 0,
            "device_to_login_ratio": 0.1, "avg_session_gap": 80000,
            "risk_score": 5, "is_anomaly": 0,
        })
        explanation = generate_explanation(row)
        assert explanation == "No significant anomalies detected."

    def test_high_login_count_flagged(self):
        row = pd.Series({
            "login_count": 200, "off_hour_logins": 0, "weekend_logins": 0,
            "late_night_logins": 0, "unique_pcs_logon": 1, "off_hour_ratio": 0.0,
            "weekend_ratio": 0.0, "pc_diversity_score": 0.005,
            "device_connections": 0, "unique_pcs_device": 0,
            "device_to_login_ratio": 0.0, "avg_session_gap": 80000,
            "risk_score": 50, "is_anomaly": 0,
        })
        explanation = generate_explanation(row)
        assert "Excessive logins" in explanation

    def test_many_pcs_flagged(self):
        row = pd.Series({
            "login_count": 50, "off_hour_logins": 0, "weekend_logins": 0,
            "late_night_logins": 0, "unique_pcs_logon": 20, "off_hour_ratio": 0.0,
            "weekend_ratio": 0.0, "pc_diversity_score": 0.4,
            "device_connections": 0, "unique_pcs_device": 0,
            "device_to_login_ratio": 0.0, "avg_session_gap": 80000,
            "risk_score": 80, "is_anomaly": 1,
        })
        explanation = generate_explanation(row)
        assert "distinct machines" in explanation

    def test_explanation_contains_population_average(self):
        """Explanations should reference the user average for context."""
        row = pd.Series({
            "login_count": 200, "off_hour_logins": 0, "weekend_logins": 0,
            "late_night_logins": 0, "unique_pcs_logon": 1, "off_hour_ratio": 0.0,
            "weekend_ratio": 0.0, "pc_diversity_score": 0.005,
            "device_connections": 0, "unique_pcs_device": 0,
            "device_to_login_ratio": 0.0, "avg_session_gap": 80000,
            "risk_score": 50, "is_anomaly": 0,
        })
        explanation = generate_explanation(row)
        assert "user average" in explanation
