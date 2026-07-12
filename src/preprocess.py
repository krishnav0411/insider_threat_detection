# src/preprocess.py
"""
File: preprocess.py
Purpose: Stage 2 — Data Preprocessing. Cleans logon.csv and device.csv,
         validates schemas, parses dates, standardises text, and adds
         time-based features (hour, day_of_week, is_weekend, is_off_hours).
Inputs:  config.yaml — paths, preprocess settings
         data/logon.csv, data/device.csv
Outputs: Cleaned DataFrames with time features (consumed by features.py)
Dependencies: pandas, numpy, logging; src.config
"""

import logging
import os
from typing import List, Tuple

import pandas as pd

from src.config import LOGON_PATH, DEVICE_PATH, OFF_HOUR_START, OFF_HOUR_END

logger = logging.getLogger(__name__)

_NULL_LIKE: List[str] = ["", "nan", "null", "none", "na", "n/a"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame:
    """
    Load a CSV file from disk with existence validation.

    Args:
        path:  Absolute or relative path to the CSV file.
        label: Human-readable label for log messages (e.g. "logon.csv").

    Returns:
        DataFrame with raw CSV contents.

    Raises:
        FileNotFoundError: If the file does not exist at *path*.

    Example:
        >>> df = load_csv("data/logon.csv", "logon.csv")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{label}] File not found: '{path}'. "
            "Make sure your CSV files are inside the /data folder."
        )
    df = pd.read_csv(path)
    logger.info("  Loaded %s: %s rows | %s columns", label, f"{df.shape[0]:,}", df.shape[1])
    return df


def validate_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    """
    Assert that all required columns exist in a DataFrame.

    Args:
        df:       DataFrame to check.
        required: Column names that must be present.
        label:    Identifier for error messages.

    Raises:
        ValueError: If any required columns are missing.

    Example:
        >>> validate_columns(df, ["id", "date", "user"], "logon.csv")
    """
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[{label}] Missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.info("  [%s] All required columns present.", label)


def drop_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Remove fully duplicate rows from a DataFrame.

    Args:
        df:    Input DataFrame.
        label: Identifier for log messages.

    Returns:
        Deduplicated DataFrame.

    Example:
        >>> df = drop_duplicates(df, "logon.csv")
    """
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info("  [%s] Dropped %s duplicate rows.", label, f"{dropped:,}")
    else:
        logger.info("  [%s] No duplicate rows found.", label)
    return df


def parse_dates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Convert the 'date' column to datetime and drop unparseable rows.

    Args:
        df:    DataFrame with a 'date' column.
        label: Identifier for log messages.

    Returns:
        DataFrame with 'date' as datetime; invalid-date rows removed.

    Example:
        >>> df = parse_dates(df, "logon.csv")
    """
    len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isnull().sum()
    if bad_dates > 0:
        logger.warning("  [%s] Dropped %s rows with unparseable dates.", label, f"{bad_dates:,}")
        df = df.dropna(subset=["date"])
    else:
        logger.info("  [%s] All dates parsed successfully.", label)
    return df


def standardize_text(df: pd.DataFrame, columns: List[str], label: str) -> pd.DataFrame:
    """
    Strip whitespace and standardise text columns.

    Only the 'activity' column gets title-case (Logon, Logoff, Connect, Disconnect).
    User IDs and PC names keep their original casing.

    Args:
        df:      Input DataFrame.
        columns: List of text column names to standardize.
        label:   Identifier for log messages.

    Returns:
        DataFrame with stripped text columns.

    Example:
        >>> df = standardize_text(df, ["activity"], "logon.csv")
    """
    for col in columns:
        if col in df.columns:
            if col == "activity":
                df[col] = df[col].astype(str).str.strip().str.title()
            else:
                df[col] = df[col].astype(str).str.strip()
    logger.info("  [%s] Text columns standardized: %s", label, columns)
    return df


def add_time_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Extract hour, day_of_week, is_weekend, and is_off_hours from the 'date' column.

    Args:
        df:    DataFrame with a datetime 'date' column.
        label: Identifier for log messages.

    Returns:
        DataFrame with four new time-based columns.

    Example:
        >>> df = add_time_features(df, "logon.csv")
    """
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_off_hours"] = (df["hour"] >= OFF_HOUR_START) | (df["hour"] < OFF_HOUR_END)
    logger.info("  [%s] Time features added: hour, day_of_week, is_weekend, is_off_hours", label)
    return df


def drop_missing_critical(df: pd.DataFrame, critical_cols: List[str], label: str) -> pd.DataFrame:
    """
    Drop rows where identifier columns (user, pc, activity) are null or
    contain null-like string values.

    Args:
        df:            Input DataFrame.
        critical_cols: Column names to check for missing values.
        label:         Identifier for log messages.

    Returns:
        Cleaned DataFrame with index reset.

    Example:
        >>> df = drop_missing_critical(df, ["user", "pc", "activity"], "logon.csv")
    """
    before = len(df)
    df = df.dropna(subset=critical_cols)
    for col in critical_cols:
        df = df[~df[col].astype(str).str.strip().str.lower().isin(_NULL_LIKE)]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("  [%s] Dropped %s rows with missing critical values.", label, f"{dropped:,}")
    else:
        logger.info("  [%s] No missing critical values found.", label)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def preprocess_logon(path: str = LOGON_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for logon.csv.

    Loads, validates, deduplicates, cleans, parses, standardises, and
    adds time features. Returns both the full cleaned dataset and a
    Logon-events-only subset.

    Args:
        path: Path to logon.csv. Defaults to config path.

    Returns:
        Tuple of (df_all, df_logon):
            df_all   — Full cleaned DataFrame.
            df_logon — Subset where activity == "Logon".

    Example:
        >>> all_events, logon_only = preprocess_logon()
    """
    _s = "-" * 65
    logger.info("")
    logger.info(_s)
    logger.info("  Preprocessing: logon.csv")
    logger.info(_s)

    REQUIRED_COLS = ["id", "date", "user", "pc", "activity"]

    df = load_csv(path, "logon.csv")
    validate_columns(df, REQUIRED_COLS, "logon.csv")
    df = drop_duplicates(df, "logon.csv")
    df = drop_missing_critical(df, ["user", "pc", "activity"], "logon.csv")
    df = parse_dates(df, "logon.csv")
    for col in ["user", "pc"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df = standardize_text(df, ["activity"], "logon.csv")
    df = add_time_features(df, "logon.csv")

    df_logon = df[df["activity"] == "Logon"].copy()

    logger.info("")
    logger.info("  Final logon.csv stats:")
    logger.info("     Total clean rows  : %s", f"{len(df):,}")
    logger.info("     Logon events only : %s", f"{len(df_logon):,}")
    logger.info("     Unique users      : %s", f"{df['user'].nunique():,}")
    logger.info("     Unique PCs        : %s", f"{df['pc'].nunique():,}")

    return df, df_logon


def preprocess_device(path: str = DEVICE_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for device.csv.

    Loads, validates, deduplicates, cleans, parses, standardises, and
    adds time features. Returns both the full cleaned dataset and a
    Connect-events-only subset.

    Args:
        path: Path to device.csv. Defaults to config path.

    Returns:
        Tuple of (df_all, df_connect):
            df_all     — Full cleaned DataFrame.
            df_connect — Subset where activity == "Connect".

    Example:
        >>> all_events, connect_only = preprocess_device()
    """
    _s = "-" * 65
    logger.info("")
    logger.info(_s)
    logger.info("  Preprocessing: device.csv")
    logger.info(_s)

    REQUIRED_COLS = ["id", "date", "user", "pc", "activity"]

    df = load_csv(path, "device.csv")
    validate_columns(df, REQUIRED_COLS, "device.csv")
    df = drop_duplicates(df, "device.csv")
    df = drop_missing_critical(df, ["user", "pc", "activity"], "device.csv")
    df = parse_dates(df, "device.csv")
    for col in ["user", "pc"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df = standardize_text(df, ["activity"], "device.csv")
    df = add_time_features(df, "device.csv")

    df_connect = df[df["activity"] == "Connect"].copy()

    logger.info("")
    logger.info("  Final device.csv stats:")
    logger.info("     Total clean rows   : %s", f"{len(df):,}")
    logger.info("     Connect events only: %s", f"{len(df_connect):,}")
    logger.info("     Unique users       : %s", f"{df['user'].nunique():,}")
    logger.info("     Unique PCs         : %s", f"{df['pc'].nunique():,}")

    return df, df_connect


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    logon_all, logon_events = preprocess_logon()
    device_all, device_events = preprocess_device()

    print("\n\n  Sample of cleaned logon data (first 3 rows):")
    print(logon_all[["user", "pc", "activity",
                      "hour", "is_weekend", "is_off_hours"]].head(3).to_string(index=False))

    print("\n  Sample of cleaned device data (first 3 rows):")
    print(device_all[["user", "pc", "activity",
                       "hour", "is_weekend", "is_off_hours"]].head(3).to_string(index=False))
