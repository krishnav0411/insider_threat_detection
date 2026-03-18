# notebooks/explore_data.py
# Stage 2: Dataset Exploration
# Run this from your project root:  python notebooks/explore_data.py

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
LOGON_PATH  = "data/logon.csv"
DEVICE_PATH = "data/device.csv"

SEPARATOR = "=" * 60


def section(title):
    """Print a clean section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ─────────────────────────────────────────────
# STEP 1 — FILE CHECK
# ─────────────────────────────────────────────
section("STEP 1 — Checking File Availability")

for path in [LOGON_PATH, DEVICE_PATH]:
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✅ Found: {path}  ({size_kb:.1f} KB)")
    else:
        print(f"  ❌ MISSING: {path}")
        print("     → Place your CSV files inside the /data folder and re-run.")


# ─────────────────────────────────────────────
# STEP 2 — LOAD DATASETS
# ─────────────────────────────────────────────
section("STEP 2 — Loading Datasets")

logon_df  = pd.read_csv(LOGON_PATH)
device_df = pd.read_csv(DEVICE_PATH)

print(f"  logon.csv  → {logon_df.shape[0]:,} rows  |  {logon_df.shape[1]} columns")
print(f"  device.csv → {device_df.shape[0]:,} rows  |  {device_df.shape[1]} columns")


# ─────────────────────────────────────────────
# STEP 3 — COLUMN NAMES & DATA TYPES
# ─────────────────────────────────────────────
section("STEP 3 — Column Names & Data Types")

print("\n  [ logon.csv ]")
print(logon_df.dtypes.to_string())

print("\n  [ device.csv ]")
print(device_df.dtypes.to_string())


# ─────────────────────────────────────────────
# STEP 4 — FIRST FEW ROWS (PREVIEW)
# ─────────────────────────────────────────────
section("STEP 4 — Data Preview (First 5 Rows)")

print("\n  [ logon.csv ]")
print(logon_df.head().to_string(index=False))

print("\n  [ device.csv ]")
print(device_df.head().to_string(index=False))


# ─────────────────────────────────────────────
# STEP 5 — MISSING VALUES CHECK
# ─────────────────────────────────────────────
section("STEP 5 — Missing Values Check")

logon_missing  = logon_df.isnull().sum()
device_missing = device_df.isnull().sum()

print("\n  [ logon.csv — Missing Values ]")
print(logon_missing.to_string())

print("\n  [ device.csv — Missing Values ]")
print(device_missing.to_string())

total_missing = logon_missing.sum() + device_missing.sum()
if total_missing == 0:
    print("\n  ✅ No missing values detected in either dataset.")
else:
    print(f"\n  ⚠️  Total missing values found: {total_missing} — will handle in Stage 3.")


# ─────────────────────────────────────────────
# STEP 6 — UNIQUE ACTIVITY TYPES
# ─────────────────────────────────────────────
section("STEP 6 — Unique Activity Types")

print("\n  [ logon.csv — activity values ]")
print(" ", logon_df["activity"].value_counts().to_string())

print("\n  [ device.csv — activity values ]")
print(" ", device_df["activity"].value_counts().to_string())


# ─────────────────────────────────────────────
# STEP 7 — USER STATISTICS
# ─────────────────────────────────────────────
section("STEP 7 — User Statistics")

logon_users  = logon_df["user"].nunique()
device_users = device_df["user"].nunique()

print(f"\n  Unique users in logon.csv  : {logon_users:,}")
print(f"  Unique users in device.csv : {device_users:,}")

# Top 10 most active users by logon count
print("\n  Top 10 Most Active Users (logon.csv):")
top_logon = logon_df["user"].value_counts().head(10)
for rank, (user, count) in enumerate(top_logon.items(), 1):
    print(f"    {rank:>2}. {user:<15} → {count:,} events")


# ─────────────────────────────────────────────
# STEP 8 — PC STATISTICS
# ─────────────────────────────────────────────
section("STEP 8 — PC (Workstation) Statistics")

logon_pcs  = logon_df["pc"].nunique()
device_pcs = device_df["pc"].nunique()

print(f"\n  Unique PCs in logon.csv  : {logon_pcs:,}")
print(f"  Unique PCs in device.csv : {device_pcs:,}")


# ─────────────────────────────────────────────
# STEP 9 — DATE RANGE
# ─────────────────────────────────────────────
section("STEP 9 — Date Range of Data")

# Convert date column to datetime for both datasets
logon_df["date"]  = pd.to_datetime(logon_df["date"],  errors="coerce")
device_df["date"] = pd.to_datetime(device_df["date"], errors="coerce")

print("\n  [ logon.csv ]")
print(f"    Earliest : {logon_df['date'].min()}")
print(f"    Latest   : {logon_df['date'].max()}")
print(f"    Span     : {(logon_df['date'].max() - logon_df['date'].min()).days} days")

print("\n  [ device.csv ]")
print(f"    Earliest : {device_df['date'].min()}")
print(f"    Latest   : {device_df['date'].max()}")
print(f"    Span     : {(device_df['date'].max() - device_df['date'].min()).days} days")


# ─────────────────────────────────────────────
# STEP 10 — DUPLICATE ROWS CHECK
# ─────────────────────────────────────────────
section("STEP 10 — Duplicate Rows Check")

logon_dupes  = logon_df.duplicated().sum()
device_dupes = device_df.duplicated().sum()

print(f"\n  Duplicate rows in logon.csv  : {logon_dupes:,}")
print(f"  Duplicate rows in device.csv : {device_dupes:,}")

if logon_dupes + device_dupes == 0:
    print("  ✅ No duplicates found.")
else:
    print("  ⚠️  Duplicates detected — will drop in Stage 3.")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
section("EXPLORATION SUMMARY")

print("""
  ✅ Both files loaded successfully
  ✅ Column structure confirmed: id | date | user | pc | activity
  ✅ Activity types identified
  ✅ User and PC counts recorded
  ✅ Date range verified
  ✅ Missing values and duplicates checked

  → Ready for Stage 3: Data Preprocessing
""")
