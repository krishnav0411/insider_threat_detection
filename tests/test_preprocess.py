"""Tests for src/preprocess.py: date parsing, off-hour detection, text standardisation."""

import pandas as pd
import pytest

from src.preprocess import (
    drop_duplicates,
    parse_dates,
    standardize_text,
    add_time_features,
    drop_missing_critical,
    OFF_HOUR_START,
    OFF_HOUR_END,
)


class TestDateParsing:
    """parse_dates() should convert the 'date' column to datetime and drop NaT rows."""

    def test_valid_dates_parsed(self):
        df = pd.DataFrame({"date": ["01/04/2010 00:10:37", "01/05/2010 08:30:00", "01/06/2010 23:59:59"]})
        result = parse_dates(df, "test")
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert len(result) == 3

    def test_invalid_dates_dropped(self):
        df = pd.DataFrame({"date": ["01/04/2010 00:10:37", "not-a-date", ""]})
        result = parse_dates(df, "test")
        assert len(result) == 1
        assert pd.notna(result["date"].iloc[0])


class TestTextStandardization:
    """standardize_text() must apply .title() only to 'activity', not user/pc."""

    def test_activity_gets_title_case(self):
        df = pd.DataFrame({"activity": ["logon", "LOGOFF", "  connect  "]})
        result = standardize_text(df, ["activity"], "test")
        assert list(result["activity"]) == ["Logon", "Logoff", "Connect"]

    def test_user_column_keeps_casing(self):
        """Regression test for the B1 bug: .title() was mangling user IDs."""
        df = pd.DataFrame({"user": ["DTAA/KEE0997", "dtaa/abc1234"]})
        result = standardize_text(df, ["user"], "test")
        assert list(result["user"]) == ["DTAA/KEE0997", "dtaa/abc1234"]

    def test_pc_column_keeps_casing(self):
        df = pd.DataFrame({"pc": ["PC-1914", "pc-abcd"]})
        result = standardize_text(df, ["pc"], "test")
        assert list(result["pc"]) == ["PC-1914", "pc-abcd"]


class TestDeduplication:
    """drop_duplicates() should remove exact duplicate rows."""

    def test_duplicates_dropped(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = drop_duplicates(df, "test")
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = drop_duplicates(df, "test")
        assert len(result) == 3


class TestMissingCritical:
    """drop_missing_critical() should remove rows with null-like identifiers."""

    def test_drops_nan_strings(self):
        df = pd.DataFrame({
            "user": ["alice", "nan", "bob"],
            "pc": ["PC-1", "PC-2", ""],
            "activity": ["Logon", "Logon", "Logon"],
        })
        result = drop_missing_critical(df, ["user", "pc", "activity"], "test")
        assert len(result) == 1
        assert result["user"].iloc[0] == "alice"

    def test_drops_none_and_null_variants(self):
        df = pd.DataFrame({
            "user": ["alice", "null", "none", "na", "n/a"],
            "pc": ["PC-1"] * 5,
            "activity": ["Logon"] * 5,
        })
        result = drop_missing_critical(df, ["user", "pc", "activity"], "test")
        assert len(result) == 1


class TestTimeFeatures:
    """add_time_features() must correctly compute hour, day_of_week, is_weekend, is_off_hours."""

    def test_hour_extracted(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2010-01-04 14:30:00"])})
        result = add_time_features(df, "test")
        assert result["hour"].iloc[0] == 14

    def test_weekend_detection(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2010-01-09 12:00:00"])})  # Saturday
        result = add_time_features(df, "test")
        assert bool(result["is_weekend"].iloc[0]) is True

    def test_weekday_is_not_weekend(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2010-01-04 12:00:00"])})  # Monday
        result = add_time_features(df, "test")
        assert bool(result["is_weekend"].iloc[0]) is False

    def test_off_hours_midnight(self):
        """Hour before OFF_HOUR_END (8 AM) is off-hours."""
        df = pd.DataFrame({"date": pd.to_datetime(["2010-01-04 03:00:00"])})
        result = add_time_features(df, "test")
        assert bool(result["is_off_hours"].iloc[0]) is True

    def test_off_hours_evening(self):
        """Hour after OFF_HOUR_START (6 PM) is off-hours."""
        df = pd.DataFrame({"date": pd.to_datetime(["2010-01-04 20:00:00"])})
        result = add_time_features(df, "test")
        assert bool(result["is_off_hours"].iloc[0]) is True

    def test_business_hours_are_not_off_hours(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2010-01-04 10:00:00"])})
        result = add_time_features(df, "test")
        assert bool(result["is_off_hours"].iloc[0]) is False
