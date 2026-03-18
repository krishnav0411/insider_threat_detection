# src/features.py
# Stage 4: Feature Engineering
# Aggregates per-user behavioral features from preprocessed data

import pandas as pd
import numpy as np
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_logon, preprocess_device

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_PATH = "outputs/user_features.csv"


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — LOGON DATA
# ─────────────────────────────────────────────

def extract_logon_features(df_logon: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts per-user behavioral features from Logon events only.

    Features produced:
    - login_count      : total number of logins
    - off_hour_logins  : logins outside 8AM–6PM
    - weekend_logins   : logins on Saturday or Sunday
    - unique_pcs_logon : number of distinct PCs logged into
    - off_hour_ratio   : off_hour_logins / login_count
    - weekend_ratio    : weekend_logins / login_count
    """
    print("\n  Extracting logon features...")

    # ── Total login count per user
    login_count = (
        df_logon.groupby("user")
        .size()
        .reset_index(name="login_count")
    )

    # ── Off-hour login count per user
    off_hour_logins = (
        df_logon[df_logon["is_off_hours"] == True]
        .groupby("user")
        .size()
        .reset_index(name="off_hour_logins")
    )

    # ── Weekend login count per user
    weekend_logins = (
        df_logon[df_logon["is_weekend"] == True]
        .groupby("user")
        .size()
        .reset_index(name="weekend_logins")
    )

    # ── Unique PCs used per user
    unique_pcs_logon = (
        df_logon.groupby("user")["pc"]
        .nunique()
        .reset_index(name="unique_pcs_logon")
    )

    # ── Merge all logon features together
    features = login_count \
        .merge(off_hour_logins,  on="user", how="left") \
        .merge(weekend_logins,   on="user", how="left") \
        .merge(unique_pcs_logon, on="user", how="left")

    # ── Fill NaN with 0 (users who had no off-hour or weekend logins)
    features["off_hour_logins"] = features["off_hour_logins"].fillna(0).astype(int)
    features["weekend_logins"]  = features["weekend_logins"].fillna(0).astype(int)

    # ── Derived ratio features (avoid divide-by-zero)
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

    print(f"  ✅ Logon features extracted for {len(features):,} users.")
    return features


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — DEVICE DATA
# ─────────────────────────────────────────────

def extract_device_features(df_connect: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts per-user behavioral features from device Connect events only.

    Features produced:
    - device_connections : total number of device connections
    - unique_pcs_device  : number of distinct PCs used for device connections
    """
    print("\n  Extracting device features...")

    # ── Total device connections per user
    device_connections = (
        df_connect.groupby("user")
        .size()
        .reset_index(name="device_connections")
    )

    # ── Unique PCs where devices were connected
    unique_pcs_device = (
        df_connect.groupby("user")["pc"]
        .nunique()
        .reset_index(name="unique_pcs_device")
    )

    # ── Merge device features
    features = device_connections.merge(unique_pcs_device, on="user", how="left")

    print(f"  ✅ Device features extracted for {len(features):,} users.")
    return features


# ─────────────────────────────────────────────
# COMBINE ALL FEATURES
# ─────────────────────────────────────────────

def build_feature_table(
    logon_features: pd.DataFrame,
    device_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges logon and device features into one unified user feature table.

    - Users with no device activity get 0 for device columns.
    - Final table has one row per user.
    """
    print("\n  Building unified feature table...")

    feature_table = logon_features.merge(
        device_features, on="user", how="left"
    )

    # Fill NaN for users with no device connections
    feature_table["device_connections"] = (
        feature_table["device_connections"].fillna(0).astype(int)
    )
    feature_table["unique_pcs_device"] = (
        feature_table["unique_pcs_device"].fillna(0).astype(int)
    )

    # ── Reorder columns cleanly
    feature_table = feature_table[[
        "user",
        "login_count",
        "off_hour_logins",
        "weekend_logins",
        "unique_pcs_logon",
        "off_hour_ratio",
        "weekend_ratio",
        "device_connections",
        "unique_pcs_device"
    ]]

    print(f"  ✅ Feature table built: {len(feature_table):,} users | "
          f"{len(feature_table.columns) - 1} features")

    return feature_table


# ─────────────────────────────────────────────
# FEATURE SUMMARY REPORT
# ─────────────────────────────────────────────

def print_feature_summary(feature_table: pd.DataFrame):
    """
    Prints a descriptive statistics summary of the feature table.
    Helps spot outliers and validate feature distributions.
    """
    SEPARATOR = "=" * 60

    print(f"\n{SEPARATOR}")
    print("  FEATURE SUMMARY STATISTICS")
    print(SEPARATOR)

    stats = feature_table.drop(columns=["user"]).describe().round(2)

    # Print each feature with clean formatting
    for col in stats.columns:
        print(f"\n  📌 {col}")
        print(f"     Mean   : {stats[col]['mean']}")
        print(f"     Std    : {stats[col]['std']}")
        print(f"     Min    : {stats[col]['min']}")
        print(f"     Median : {stats[col]['50%']}")
        print(f"     Max    : {stats[col]['max']}")

    # Flag high-value outliers per feature
    print(f"\n{SEPARATOR}")
    print("  TOP 5 USERS — by login_count")
    print(SEPARATOR)
    top = feature_table.nlargest(5, "login_count")[
        ["user", "login_count", "off_hour_logins", "unique_pcs_logon"]
    ]
    print(top.to_string(index=False))

    print(f"\n{SEPARATOR}")
    print("  TOP 5 USERS — by device_connections")
    print(SEPARATOR)
    top_dev = feature_table.nlargest(5, "device_connections")[
        ["user", "device_connections", "unique_pcs_device"]
    ]
    print(top_dev.to_string(index=False))


# ─────────────────────────────────────────────
# SAVE FEATURE TABLE
# ─────────────────────────────────────────────

def save_feature_table(feature_table: pd.DataFrame, path: str = OUTPUT_PATH):
    """
    Saves the feature table to a CSV file in /outputs.
    Creates the directory if it does not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feature_table.to_csv(path, index=False)
    print(f"\n  💾 Feature table saved → {path}")


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """
    Master function that runs the full feature engineering pipeline.
    Called by later stages (model training, dashboard).

    Returns:
        feature_table : DataFrame with one row per user and 8 features
    """
    SEPARATOR = "=" * 60

    print(f"\n{SEPARATOR}")
    print("  STAGE 4 — Feature Engineering Pipeline")
    print(SEPARATOR)

    # Step 1 — Preprocess raw data
    print("\n  [1/4] Running preprocessing...")
    _, logon_events   = preprocess_logon()
    _, device_events  = preprocess_device()

    # Step 2 — Extract features from each source
    print("\n  [2/4] Extracting features...")
    logon_features  = extract_logon_features(logon_events)
    device_features = extract_device_features(device_events)

    # Step 3 — Combine into one feature table
    print("\n  [3/4] Combining features...")
    feature_table = build_feature_table(logon_features, device_features)

    # Step 4 — Save to disk
    print("\n  [4/4] Saving feature table...")
    save_feature_table(feature_table)

    return feature_table


# ─────────────────────────────────────────────
# SELF-TEST  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    feature_table = run_feature_engineering()

    # Print summary statistics
    print_feature_summary(feature_table)

    print("\n" + "=" * 60)
    print("  ✅ Stage 4 complete — feature table ready for modelling")
    print("=" * 60)