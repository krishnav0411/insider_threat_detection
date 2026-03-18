# src/preprocess.py
# Stage 3: Data Preprocessing
# Cleans and prepares logon.csv and device.csv for feature engineering

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Off-hours = before 8 AM or after 6 PM
OFF_HOUR_START = 18   # 6 PM
OFF_HOUR_END   = 8    # 8 AM

# Paths (relative to project root)
LOGON_PATH  = "data/logon.csv"
DEVICE_PATH = "data/device.csv"


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame:
    """
    Safely load a CSV file with basic validation.
    Returns a DataFrame or raises a clear error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{label}] File not found: '{path}'\n"
            f"→ Make sure your CSV files are inside the /data folder."
        )

    df = pd.read_csv(path)
    print(f"  ✅ Loaded {label}: {df.shape[0]:,} rows | {df.shape[1]} columns")
    return df


def validate_columns(df: pd.DataFrame, required: list, label: str):
    """
    Check that all required columns exist in the DataFrame.
    Raises a clear error if any are missing.
    """
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[{label}] Missing expected columns: {missing_cols}\n"
            f"→ Found columns: {list(df.columns)}"
        )
    print(f"  ✅ [{label}] All required columns present.")


def drop_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Remove fully duplicate rows and report how many were dropped.
    """
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  ⚠️  [{label}] Dropped {dropped:,} duplicate rows.")
    else:
        print(f"  ✅ [{label}] No duplicate rows found.")
    return df


def parse_dates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Convert the 'date' column to datetime.
    Rows with unparseable dates are dropped with a warning.
    """
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isnull().sum()

    if bad_dates > 0:
        print(f"  ⚠️  [{label}] Dropped {bad_dates:,} rows with unparseable dates.")
        df = df.dropna(subset=["date"])
    else:
        print(f"  ✅ [{label}] All dates parsed successfully.")

    return df


def standardize_text(df: pd.DataFrame, columns: list, label: str) -> pd.DataFrame:
    """
    Strip whitespace and apply title case to text columns.
    Ensures 'logon', 'Logon', '  Logon  ' all become 'Logon'.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    print(f"  ✅ [{label}] Text columns standardized: {columns}")
    return df


def add_time_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Extract time-based features from the 'date' column.

    New columns added:
    - hour        : hour of the event (0–23)
    - day_of_week : 0=Monday ... 6=Sunday
    - is_weekend  : True if Saturday or Sunday
    - is_off_hours: True if before OFF_HOUR_END or after OFF_HOUR_START
    """
    df["hour"]         = df["date"].dt.hour
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["is_weekend"]   = df["day_of_week"] >= 5   # 5=Saturday, 6=Sunday
    df["is_off_hours"] = (df["hour"] >= OFF_HOUR_START) | (df["hour"] < OFF_HOUR_END)

    print(f"  ✅ [{label}] Time features added: hour, day_of_week, is_weekend, is_off_hours")
    return df


def drop_missing_critical(df: pd.DataFrame, critical_cols: list, label: str) -> pd.DataFrame:
    """
    Drop rows where critical columns (user, pc, activity) are null or empty.
    """
    before = len(df)
    df = df.dropna(subset=critical_cols)

    # Also drop rows where any critical column is the string 'nan' or empty
    for col in critical_cols:
        df = df[df[col].astype(str).str.strip() != ""]
        df = df[df[col].astype(str).str.lower() != "nan"]

    dropped = before - len(df)
    if dropped > 0:
        print(f"  ⚠️  [{label}] Dropped {dropped:,} rows with missing critical values.")
    else:
        print(f"  ✅ [{label}] No missing critical values found.")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def preprocess_logon(path: str = LOGON_PATH) -> pd.DataFrame:
    """
    Full preprocessing pipeline for logon.csv.

    Steps:
    1. Load CSV
    2. Validate columns
    3. Drop duplicates
    4. Drop missing critical values
    5. Parse dates
    6. Standardize text
    7. Add time features
    8. Filter to Logon events only (for feature counting)

    Returns:
        df_all   : full cleaned logon dataframe (Logon + Logoff)
        df_logon : only Logon events (used for feature extraction)
    """
    SEPARATOR = "-" * 50
    print(f"\n{SEPARATOR}")
    print("  Preprocessing: logon.csv")
    print(SEPARATOR)

    REQUIRED_COLS = ["id", "date", "user", "pc", "activity"]

    df = load_csv(path, "logon.csv")
    validate_columns(df, REQUIRED_COLS, "logon.csv")
    df = drop_duplicates(df, "logon.csv")
    df = drop_missing_critical(df, ["user", "pc", "activity"], "logon.csv")
    df = parse_dates(df, "logon.csv")
    df = standardize_text(df, ["user", "pc", "activity"], "logon.csv")
    df = add_time_features(df, "logon.csv")

    # Separate logon-only events for feature engineering
    df_logon = df[df["activity"] == "Logon"].copy()

    print(f"\n  📊 Final logon.csv stats:")
    print(f"     Total clean rows  : {len(df):,}")
    print(f"     Logon events only : {len(df_logon):,}")
    print(f"     Unique users      : {df['user'].nunique():,}")
    print(f"     Unique PCs        : {df['pc'].nunique():,}")

    return df, df_logon


def preprocess_device(path: str = DEVICE_PATH) -> pd.DataFrame:
    """
    Full preprocessing pipeline for device.csv.

    Steps:
    1. Load CSV
    2. Validate columns
    3. Drop duplicates
    4. Drop missing critical values
    5. Parse dates
    6. Standardize text
    7. Add time features
    8. Filter to Connect events only

    Returns:
        df_all     : full cleaned device dataframe
        df_connect : only Connect events (used for feature extraction)
    """
    SEPARATOR = "-" * 50
    print(f"\n{SEPARATOR}")
    print("  Preprocessing: device.csv")
    print(SEPARATOR)

    REQUIRED_COLS = ["id", "date", "user", "pc", "activity"]

    df = load_csv(path, "device.csv")
    validate_columns(df, REQUIRED_COLS, "device.csv")
    df = drop_duplicates(df, "device.csv")
    df = drop_missing_critical(df, ["user", "pc", "activity"], "device.csv")
    df = parse_dates(df, "device.csv")
    df = standardize_text(df, ["user", "pc", "activity"], "device.csv")
    df = add_time_features(df, "device.csv")

    # Separate connect-only events for feature engineering
    df_connect = df[df["activity"] == "Connect"].copy()

    print(f"\n  📊 Final device.csv stats:")
    print(f"     Total clean rows   : {len(df):,}")
    print(f"     Connect events only: {len(df_connect):,}")
    print(f"     Unique users       : {df['user'].nunique():,}")
    print(f"     Unique PCs         : {df['pc'].nunique():,}")

    return df, df_connect


# ─────────────────────────────────────────────
# QUICK SELF-TEST  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  STAGE 3 — Data Preprocessing Self-Test")
    print("=" * 60)

    # Run both pipelines
    logon_all, logon_events   = preprocess_logon()
    device_all, device_events = preprocess_device()

    # Show a sample of the processed data
    print("\n\n  Sample of cleaned logon data (first 3 rows):")
    print(logon_all[["user", "pc", "activity",
                      "hour", "is_weekend", "is_off_hours"]].head(3).to_string(index=False))

    print("\n  Sample of cleaned device data (first 3 rows):")
    print(device_all[["user", "pc", "activity",
                       "hour", "is_weekend", "is_off_hours"]].head(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("  ✅ Preprocessing complete — data is ready for Stage 4")
    print("=" * 60)