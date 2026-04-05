# Stage 6: Risk Scoring
# Converts anomaly scores into Low / Medium / High risk levels
# Adds per-user risk explanations and generates a summary report

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import run_model_pipeline

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SCORED_PATH      = "outputs/user_scores.csv"
RISK_REPORT_PATH = "outputs/risk_report.csv"

# Risk level thresholds
THRESHOLD_HIGH   = 70.0
THRESHOLD_MEDIUM = 40.0

# Behaviour thresholds for explanation generation
# (values above these are flagged as suspicious)
FLAG_LOGIN_COUNT        = 80     # logins above this are excessive
FLAG_OFF_HOUR_LOGINS    = 15     # off-hour logins above this are suspicious
FLAG_OFF_HOUR_RATIO     = 0.25   # more than 25% off-hour logins
FLAG_WEEKEND_LOGINS     = 10     # weekend logins above this
FLAG_WEEKEND_RATIO      = 0.20   # more than 20% weekend logins
FLAG_UNIQUE_PCS_LOGON   = 3      # logging into 3+ different PCs
FLAG_DEVICE_CONNECTIONS = 30     # device connections above this
FLAG_UNIQUE_PCS_DEVICE  = 3      # connecting devices to 3+ PCs


# ─────────────────────────────────────────────
# RISK LEVEL ASSIGNMENT
# ─────────────────────────────────────────────

def assign_risk_level(score: float) -> str:
    """
    Convert a numeric risk score (0–100) into a categorical risk level.

    Returns:
        'High'   if score >= THRESHOLD_HIGH
        'Medium' if score >= THRESHOLD_MEDIUM
        'Low'    otherwise
    """
    if score >= THRESHOLD_HIGH:
        return "High"
    elif score >= THRESHOLD_MEDIUM:
        return "Medium"
    else:
        return "Low"


def add_risk_levels(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply risk level labels to all users in the scored DataFrame.
    """
    scored_df = scored_df.copy()
    scored_df["risk_level"] = scored_df["risk_score"].apply(assign_risk_level)

    counts = scored_df["risk_level"].value_counts()
    total  = len(scored_df)

    print(f"\n  Risk Level Distribution:")
    for level in ["High", "Medium", "Low"]:
        count = counts.get(level, 0)
        pct   = count / total * 100
        icon  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[level]
        bar   = "█" * int(pct / 2)
        print(f"    {icon} {level:<8} : {count:>4} users  ({pct:5.1f}%)  {bar}")

    return scored_df


# ─────────────────────────────────────────────
# RISK EXPLANATION GENERATION
# ─────────────────────────────────────────────

def generate_explanation(row: pd.Series) -> str:
    """
    Generate a human-readable explanation of WHY a user is flagged.
    Checks each feature against suspicious thresholds and builds
    a plain-English description of the anomalous behaviors.

    Returns:
        A string summarising the suspicious behaviours found,
        or 'No significant anomalies detected.' for normal users.
    """
    flags = []

    # ── Logon-based flags
    if row["login_count"] > FLAG_LOGIN_COUNT:
        flags.append(
            f"Excessive logins ({int(row['login_count'])} total)"
        )

    if row["off_hour_logins"] > FLAG_OFF_HOUR_LOGINS:
        flags.append(
            f"High off-hours activity ({int(row['off_hour_logins'])} logins outside 8AM–6PM)"
        )
    elif row["off_hour_ratio"] > FLAG_OFF_HOUR_RATIO:
        flags.append(
            f"Elevated off-hours ratio ({row['off_hour_ratio']*100:.0f}% of logins are off-hours)"
        )

    if row["weekend_logins"] > FLAG_WEEKEND_LOGINS:
        flags.append(
            f"Frequent weekend logins ({int(row['weekend_logins'])} weekend events)"
        )
    elif row["weekend_ratio"] > FLAG_WEEKEND_RATIO:
        flags.append(
            f"Elevated weekend login ratio ({row['weekend_ratio']*100:.0f}% of logins on weekends)"
        )

    if row["unique_pcs_logon"] >= FLAG_UNIQUE_PCS_LOGON:
        flags.append(
            f"Logged into multiple PCs ({int(row['unique_pcs_logon'])} distinct machines)"
        )

    # ── Device-based flags
    if row["device_connections"] > FLAG_DEVICE_CONNECTIONS:
        flags.append(
            f"Excessive device connections ({int(row['device_connections'])} total)"
        )

    if row["unique_pcs_device"] >= FLAG_UNIQUE_PCS_DEVICE:
        flags.append(
            f"Connected devices across multiple PCs ({int(row['unique_pcs_device'])} machines)"
        )

    if flags:
        return " | ".join(flags)
    else:
        return "No significant anomalies detected."


def add_explanations(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply explanation generation to every user row.
    """
    scored_df = scored_df.copy()
    scored_df["risk_explanation"] = scored_df.apply(generate_explanation, axis=1)
    explained = (scored_df["risk_explanation"] != "No significant anomalies detected.").sum()
    print(f"\n  ✅ Risk explanations generated.")
    print(f"     Users with flagged behaviours: {explained:,}")
    return scored_df


# ─────────────────────────────────────────────
# RISK PROFILE BUILDER
# ─────────────────────────────────────────────

def build_risk_profile(row: pd.Series) -> dict:
    """
    Build a detailed risk profile dictionary for a single user.
    Used by the dashboard to display per-user detail cards.

    Returns a dict with all key fields for display.
    """
    return {
        "user"               : row["user"],
        "risk_level"         : row["risk_level"],
        "risk_score"         : row["risk_score"],
        "is_anomaly"         : bool(row["is_anomaly"]),
        "login_count"        : int(row["login_count"]),
        "off_hour_logins"    : int(row["off_hour_logins"]),
        "weekend_logins"     : int(row["weekend_logins"]),
        "unique_pcs_logon"   : int(row["unique_pcs_logon"]),
        "off_hour_ratio"     : float(row["off_hour_ratio"]),
        "weekend_ratio"      : float(row["weekend_ratio"]),
        "device_connections" : int(row["device_connections"]),
        "unique_pcs_device"  : int(row["unique_pcs_device"]),
        "risk_explanation"   : row["risk_explanation"]
    }


# ─────────────────────────────────────────────
# SAVE RISK REPORT
# ─────────────────────────────────────────────

def save_risk_report(risk_df: pd.DataFrame, path: str = RISK_REPORT_PATH):
    """
    Save the complete risk report to CSV.
    Sorted by risk_score descending so highest threats appear first.
    """
    os.makedirs("outputs", exist_ok=True)
    risk_df_sorted = risk_df.sort_values("risk_score", ascending=False)
    risk_df_sorted.to_csv(path, index=False)
    print(f"\n  💾 Risk report saved → {path}")


# ─────────────────────────────────────────────
# PRINT RISK REPORT SUMMARY
# ─────────────────────────────────────────────

def print_risk_report(risk_df: pd.DataFrame):
    """
    Print a formatted risk report to the terminal.
    Shows High risk users in full detail,
    and a brief summary for Medium and Low.
    """
    SEPARATOR = "=" * 60

    print(f"\n{SEPARATOR}")
    print("  FULL RISK REPORT")
    print(SEPARATOR)

    # ── High Risk Users
    high = risk_df[risk_df["risk_level"] == "High"].sort_values(
        "risk_score", ascending=False
    )
    print(f"\n  🔴 HIGH RISK USERS ({len(high)} total)")
    print("-" * 60)

    if len(high) == 0:
        print("  None detected.")
    else:
        for _, row in high.iterrows():
            print(f"\n  👤 User        : {row['user']}")
            print(f"     Risk Score  : {row['risk_score']:.1f} / 100")
            print(f"     Logins      : {int(row['login_count'])}  "
                  f"(Off-hours: {int(row['off_hour_logins'])}, "
                  f"Weekend: {int(row['weekend_logins'])})")
            print(f"     Unique PCs  : {int(row['unique_pcs_logon'])}  |  "
                  f"Device connections: {int(row['device_connections'])}")
            print(f"     ⚠️  Flags    : {row['risk_explanation']}")

    # ── Medium Risk Summary
    medium = risk_df[risk_df["risk_level"] == "Medium"].sort_values(
        "risk_score", ascending=False
    )
    print(f"\n\n  🟡 MEDIUM RISK USERS — Top 5 of {len(medium)} total")
    print("-" * 60)

    if len(medium) == 0:
        print("  None detected.")
    else:
        display_cols = ["user", "risk_score", "login_count",
                        "off_hour_logins", "device_connections"]
        print(medium[display_cols].head(5).to_string(index=False))

    # ── Low Risk Summary
    low = risk_df[risk_df["risk_level"] == "Low"]
    print(f"\n\n  🟢 LOW RISK USERS — {len(low)} users flagged as normal")
    print("-" * 60)
    print(f"  These users show typical behavioral patterns.")
    print(f"  Average login count : "
          f"{low['login_count'].mean():.1f}")
    print(f"  Average off-hour %  : "
          f"{low['off_hour_ratio'].mean()*100:.1f}%")


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_risk_scoring() -> pd.DataFrame:
    """
    Master function that runs the full risk scoring pipeline.
    Called by Stage 7 (attack simulation) and the dashboard.

    Returns:
        risk_df : DataFrame with risk_level and risk_explanation per user
    """
    SEPARATOR = "=" * 60

    print(f"\n{SEPARATOR}")
    print("  STAGE 6 — Risk Scoring Pipeline")
    print(SEPARATOR)

    # Step 1 — Load or generate scored data
    print("\n  [1/4] Loading anomaly scores...")
    if os.path.exists(SCORED_PATH):
        scored_df = pd.read_csv(SCORED_PATH)
        print(f"  ✅ Loaded scored data: {len(scored_df):,} users")
    else:
        print("  ⚠️  Scored file not found — running model pipeline first...")
        scored_df = run_model_pipeline()

    # Step 2 — Assign risk levels
    print("\n  [2/4] Assigning risk levels...")
    risk_df = add_risk_levels(scored_df)

    # Step 3 — Generate explanations
    print("\n  [3/4] Generating risk explanations...")
    risk_df = add_explanations(risk_df)

    # Step 4 — Save report
    print("\n  [4/4] Saving risk report...")
    save_risk_report(risk_df)

    return risk_df


# ─────────────────────────────────────────────
# SELF-TEST  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    risk_df = run_risk_scoring()

    print_risk_report(risk_df)

    # Show one full risk profile as a sample
    top_user = risk_df.nlargest(1, "risk_score").iloc[0]
    profile  = build_risk_profile(top_user)

    print("\n" + "=" * 60)
    print("  SAMPLE RISK PROFILE (Highest Risk User)")
    print("=" * 60)
    for key, value in profile.items():
        print(f"  {key:<22} : {value}")

    print("\n" + "=" * 60)
    print("  ✅ Stage 6 complete — risk report ready for Stage 7")
    print("=" * 60)
