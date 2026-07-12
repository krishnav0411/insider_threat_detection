# src/features.py
"""
File: features.py
Purpose: Stage 3 — Feature Engineering. Extracts 12 per-user behavioural
         features from preprocessed logon and device data.
Inputs:  config.yaml — feature paths and column names
         Cleaned DataFrames from preprocess.py
Outputs: outputs/user_features.csv — 1000 users x 13 columns
Dependencies: pandas, numpy, logging; src.config, src.preprocess
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import FEATURE_COLS, FEATURE_PATH
from src.preprocess import preprocess_logon, preprocess_device

logger = logging.getLogger(__name__)


OUTPUT_PATH: str = FEATURE_PATH


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────

def validate_input(df: Optional[pd.DataFrame], label: str) -> None:
    """
    Assert that a DataFrame is non-empty and contains mandatory columns.

    Args:
        df:    Input DataFrame. May be None.
        label: Identifier for error messages.

    Raises:
        ValueError: If *df* is None, empty, or missing required columns.

    Example:
        >>> validate_input(logon_events, "logon")
    """
    if df is None or len(df) == 0:
        raise ValueError(f"[{label}] Input DataFrame is empty. No events to extract features from.")
    required = ["user", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {missing}. Found: {list(df.columns)}")


def _merge_features(base: pd.DataFrame, *others: pd.DataFrame) -> pd.DataFrame:
    """Merge multiple per-user feature DataFrames on 'user' using left joins."""
    result = base
    for o in others:
        result = result.merge(o, on="user", how="left")
    return result


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — LOGON DATA
# ─────────────────────────────────────────────

def extract_logon_features(df_logon: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-user features from Logon events.

    Produces: login_count, off_hour_logins, weekend_logins, late_night_logins,
    unique_pcs_logon, off_hour_ratio, weekend_ratio, avg_session_gap.

    Args:
        df_logon: DataFrame of Logon events only (from preprocess_logon).

    Returns:
        DataFrame with one row per user and 8 logon-derived feature columns.

    Example:
        >>> logon_feats = extract_logon_features(logon_events)
    """
    logger.info("  Extracting logon features...")
    validate_input(df_logon, "logon")

    login_count = df_logon.groupby("user").size().reset_index(name="login_count")
    off_hour_logins = (
        df_logon[df_logon["is_off_hours"]]
        .groupby("user").size().reset_index(name="off_hour_logins")
    )
    weekend_logins = (
        df_logon[df_logon["is_weekend"]]
        .groupby("user").size().reset_index(name="weekend_logins")
    )
    late_night_logins = (
        df_logon[(df_logon["hour"] >= 22) | (df_logon["hour"] < 4)]
        .groupby("user").size().reset_index(name="late_night_logins")
    )
    unique_pcs_logon = (
        df_logon.groupby("user")["pc"]
        .nunique().reset_index(name="unique_pcs_logon")
    )

    # Avg session gap: mean seconds between consecutive logins per user.
    df_sorted = df_logon.sort_values(["user", "date"]).copy()
    df_sorted["prev_date"] = df_sorted.groupby("user")["date"].shift(1)
    df_sorted["gap_seconds"] = (
        df_sorted["date"] - df_sorted["prev_date"]
    ).dt.total_seconds()
    avg_session_gap = (
        df_sorted.dropna(subset=["gap_seconds"])
        .groupby("user")["gap_seconds"]
        .mean().round(1).reset_index(name="avg_session_gap")
    )

    features = _merge_features(login_count, off_hour_logins, weekend_logins,
                                late_night_logins, unique_pcs_logon, avg_session_gap)

    for col in ["off_hour_logins", "weekend_logins", "late_night_logins"]:
        features[col] = features[col].fillna(0).astype(int)
    features["avg_session_gap"] = features["avg_session_gap"].fillna(0.0)

    features["off_hour_ratio"] = np.where(
        features["login_count"] > 0,
        features["off_hour_logins"] / features["login_count"],
        0.0
    ).round(4)
    features["weekend_ratio"] = np.where(
        features["login_count"] > 0,
        features["weekend_logins"] / features["login_count"],
        0.0
    ).round(4)

    logger.info("  Logon features extracted for %s users.", f"{len(features):,}")
    return features


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — DEVICE DATA
# ─────────────────────────────────────────────

def extract_device_features(df_connect: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-user features from device Connect events.

    Produces: device_connections, unique_pcs_device.

    Args:
        df_connect: DataFrame of Connect events only (from preprocess_device).

    Returns:
        DataFrame with one row per user and 2 device-derived feature columns.

    Example:
        >>> dev_feats = extract_device_features(connect_events)
    """
    logger.info("  Extracting device features...")
    validate_input(df_connect, "device")

    device_connections = (
        df_connect.groupby("user")
        .size().reset_index(name="device_connections")
    )
    unique_pcs_device = (
        df_connect.groupby("user")["pc"]
        .nunique().reset_index(name="unique_pcs_device")
    )

    features = device_connections.merge(unique_pcs_device, on="user", how="left")
    logger.info("  Device features extracted for %s users.", f"{len(features):,}")
    return features


# ─────────────────────────────────────────────
# COMBINE ALL FEATURES
# ─────────────────────────────────────────────

def build_feature_table(
    logon_features: pd.DataFrame,
    device_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge logon and device features into a single user feature table.

    Users without device activity receive 0 for device columns. Two derived
    features (pc_diversity_score, device_to_login_ratio) are computed during
    the merge.

    Args:
        logon_features:  Output of extract_logon_features.
        device_features: Output of extract_device_features.

    Returns:
        DataFrame with columns: user + 12 feature columns.

    Example:
        >>> ft = build_feature_table(logon_feats, dev_feats)
    """
    logger.info("  Building unified feature table...")

    feature_table = logon_features.merge(device_features, on="user", how="left")
    feature_table["device_connections"] = feature_table["device_connections"].fillna(0).astype(int)
    feature_table["unique_pcs_device"] = feature_table["unique_pcs_device"].fillna(0).astype(int)

    feature_table["pc_diversity_score"] = np.where(
        feature_table["login_count"] > 0,
        (feature_table["unique_pcs_logon"] / feature_table["login_count"]).round(4),
        0.0
    )
    feature_table["device_to_login_ratio"] = np.where(
        feature_table["login_count"] > 0,
        (feature_table["device_connections"] / feature_table["login_count"]).round(4),
        0.0
    )

    col_order: List[str] = ["user"] + FEATURE_COLS
    feature_table = feature_table[[c for c in col_order if c in feature_table.columns]]

    logger.info("  Feature table built: %s users | %s features",
                f"{len(feature_table):,}", len(feature_table.columns) - 1)
    return feature_table


# ─────────────────────────────────────────────
# FEATURE CORRELATION CHECK
# ─────────────────────────────────────────────

def check_feature_correlations(feature_table: pd.DataFrame, threshold: float = 0.95) -> None:
    """
    Compute pairwise correlations between numeric features and warn if any
    pair exceeds the given threshold.

    Args:
        feature_table: Feature DataFrame (the 'user' column is excluded internally).
        threshold:     Correlation magnitude threshold. Default 0.95.

    Example:
        >>> check_feature_correlations(feature_table)
    """
    numeric = feature_table.drop(columns=["user"])
    corr_matrix = numeric.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_pairs: List[Tuple[str, str, float]] = [
        (col, row, round(upper_tri[col][row], 3))
        for col in upper_tri.columns
        for row in upper_tri.index
        if upper_tri[col][row] > threshold
    ]

    if high_pairs:
        for col, row, val in high_pairs:
            logger.warning("  High correlation: %s <-> %s = %s", col, row, val)
        logger.warning("  %s feature pair(s) above %s — consider removing redundant features.",
                       len(high_pairs), threshold)
    else:
        logger.info("  No feature pair exceeds correlation threshold of %s.", threshold)


# ─────────────────────────────────────────────
# FEATURE SUMMARY REPORT
# ─────────────────────────────────────────────

def print_feature_summary(feature_table: pd.DataFrame) -> None:
    """
    Print descriptive statistics for every feature in the table.

    Args:
        feature_table: Feature DataFrame with a 'user' column.

    Example:
        >>> print_feature_summary(feature_table)
    """
    _s = "=" * 60
    logger.info("")
    logger.info(_s)
    logger.info("  FEATURE SUMMARY STATISTICS")
    logger.info(_s)

    stats = feature_table.drop(columns=["user"]).describe().round(2)
    for col in stats.columns:
        logger.info("")
        logger.info("  %s", col)
        logger.info("     Mean   : %s", stats[col]['mean'])
        logger.info("     Std    : %s", stats[col]['std'])
        logger.info("     Min    : %s", stats[col]['min'])
        logger.info("     Median : %s", stats[col]['50%'])
        logger.info("     Max    : %s", stats[col]['max'])

    logger.info("")
    logger.info(_s)
    logger.info("  TOP 5 USERS — by login_count")
    logger.info(_s)
    top = feature_table.nlargest(5, "login_count")[
        ["user", "login_count", "off_hour_logins", "unique_pcs_logon"]
    ]
    print(top.to_string(index=False))

    logger.info("")
    logger.info(_s)
    logger.info("  TOP 5 USERS — by device_connections")
    logger.info(_s)
    top_dev = feature_table.nlargest(5, "device_connections")[
        ["user", "device_connections", "unique_pcs_device"]
    ]
    print(top_dev.to_string(index=False))


# ─────────────────────────────────────────────
# SAVE FEATURE TABLE
# ─────────────────────────────────────────────

def save_feature_table(feature_table: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    """
    Write the feature table to a CSV file, creating the output directory if needed.

    Args:
        feature_table: DataFrame to save.
        path:          Destination file path. Defaults to config path.

    Example:
        >>> save_feature_table(feature_table)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feature_table.to_csv(path, index=False)
    logger.info("  Feature table saved -> %s  (%s users x %s features)",
                path, f"{len(feature_table):,}", len(feature_table.columns) - 1)


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline.

    Returns:
        DataFrame with one row per user and 12 feature columns.

    Example:
        >>> ft = run_feature_engineering()
    """
    _s = "=" * 60
    logger.info("")
    logger.info(_s)
    logger.info("  STAGE 3 — Feature Engineering Pipeline")
    logger.info(_s)

    logger.info("")
    logger.info("  [1/4] Running preprocessing...")
    _, logon_events = preprocess_logon()
    _, device_events = preprocess_device()

    logger.info("")
    logger.info("  [2/4] Extracting features...")
    logon_features = extract_logon_features(logon_events)
    device_features = extract_device_features(device_events)

    logger.info("")
    logger.info("  [3/4] Combining features...")
    feature_table = build_feature_table(logon_features, device_features)

    logger.info("")
    logger.info("  [Intermediate] Checking feature correlations...")
    check_feature_correlations(feature_table)

    logger.info("")
    logger.info("  [4/4] Saving feature table...")
    save_feature_table(feature_table)

    return feature_table


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    ft = run_feature_engineering()
    print_feature_summary(ft)
