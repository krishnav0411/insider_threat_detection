# src/risk_scoring.py
"""
File: risk_scoring.py
Purpose: Stage 5 — Risk Scoring. Converts anomaly scores into
         Low / Medium / High risk levels, generates human-readable
         explanations with population context, and saves the risk report.
Inputs:  config.yaml — thresholds and flag settings
         outputs/user_scores.csv (or regenerates via run_model_pipeline)
Outputs: outputs/risk_report.csv
Dependencies: pandas, numpy, logging; src.config
"""

import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    SCORED_PATH, RISK_REPORT_PATH,
    THRESHOLD_HIGH, THRESHOLD_MEDIUM, FLAGS,
)

logger = logging.getLogger(__name__)


# Backward-compatible CONFIG dict used by attack_simulation and dashboard.
CONFIG: Dict[str, Any] = {
    "threshold_high": THRESHOLD_HIGH,
    "threshold_medium": THRESHOLD_MEDIUM,
}
CONFIG.update({f"flag_{k}": v for k, v in FLAGS.items()})


# ─────────────────────────────────────────────
# RISK LEVEL ASSIGNMENT
# ─────────────────────────────────────────────

def assign_risk_level(score: float) -> str:
    """
    Convert a numeric risk score (0-100) into a categorical risk level.

    Thresholds (from config.yaml):
        >= 70: "High"
        >= 40: "Medium"
        else:  "Low"

    Args:
        score: Numeric risk score in [0, 100].

    Returns:
        "High", "Medium", or "Low".

    Example:
        >>> assign_risk_level(85.3)
        'High'
    """
    if score >= THRESHOLD_HIGH:
        return "High"
    if score >= THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"


def add_risk_levels(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply risk level labels to all users in the scored DataFrame.

    Args:
        scored_df: DataFrame with a 'risk_score' column.

    Returns:
        DataFrame with added 'risk_level' column.

    Example:
        >>> rdf = add_risk_levels(scored_df)
    """
    scored_df = scored_df.copy()
    scored_df["risk_level"] = scored_df["risk_score"].apply(assign_risk_level)

    counts = scored_df["risk_level"].value_counts()
    total = len(scored_df)

    logger.info("")
    logger.info("  Risk Level Distribution:")
    for level in ["High", "Medium", "Low"]:
        count = counts.get(level, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        logger.info("  %s %s : %s users  (%.1f%%)  %s",
                    {"High": "\U0001f534", "Medium": "\U0001f7e1", "Low": "\U0001f7e2"}[level],
                    f"{level:<8}", f"{count:>4}", pct, bar)
    return scored_df


# ─────────────────────────────────────────────
# RISK EXPLANATION GENERATION
# ─────────────────────────────────────────────

_population_means: Dict[str, float] = {}


def _load_population_means(scored_df: pd.DataFrame) -> None:
    """
    Compute population-level statistics from the scored DataFrame for
    contextual explanations (e.g. "above the X.X user average").

    Args:
        scored_df: DataFrame containing feature columns and risk data.

    Example:
        >>> _load_population_means(risk_df)
    """
    global _population_means
    numeric_cols: List[str] = [
        "login_count", "off_hour_logins", "weekend_logins", "late_night_logins",
        "unique_pcs_logon", "off_hour_ratio", "weekend_ratio", "pc_diversity_score",
        "device_connections", "unique_pcs_device", "device_to_login_ratio", "avg_session_gap",
    ]
    existing = [c for c in numeric_cols if c in scored_df.columns]
    _population_means = scored_df[existing].mean().to_dict()


def generate_explanation(row: pd.Series) -> str:
    """
    Generate a human-readable explanation of why a user is flagged.

    Checks each feature against CONFIG thresholds and builds a plain-English
    description that includes deviation from the population average.

    Args:
        row: pd.Series representing one user with feature columns.

    Returns:
        Pipe-separated string of flagged behaviours, or
        'No significant anomalies detected.' if no flags triggered.

    Example:
        >>> explanation = generate_explanation(user_row)
    """
    flags: List[str] = []
    pm = _population_means

    def _int(val: Any) -> int:
        return int(val) if not pd.isna(val) else 0

    def _flt(val: Any) -> float:
        return float(val) if not pd.isna(val) else 0.0

    lc = row.get("login_count", 0)
    ohl = row.get("off_hour_logins", 0)
    ohr = row.get("off_hour_ratio", 0.0)
    wl = row.get("weekend_logins", 0)
    wr = row.get("weekend_ratio", 0.0)
    upc = row.get("unique_pcs_logon", 0)
    lnl = row.get("late_night_logins", 0)
    dc = row.get("device_connections", 0)
    upd = row.get("unique_pcs_device", 0)
    dlr = row.get("device_to_login_ratio", 0.0)
    pcd = row.get("pc_diversity_score", 0.0)
    asg = row.get("avg_session_gap", 0.0)

    if lc > FLAGS["login_count"]:
        avg = pm.get("login_count", 0)
        flags.append(f"Excessive logins ({_int(lc)} total — {_int(lc - avg):+,} above the {avg:.1f} user average)")

    if ohl > FLAGS["off_hour_logins"]:
        avg = pm.get("off_hour_logins", 0)
        pct_str = f" ({_flt(ohl) / _flt(lc) * 100:.0f}% of total activity)" if lc > 0 else ""
        flags.append(f"{_int(ohl)} logins outside business hours{pct_str} — significantly above the {avg:.1f} user average")
    elif ohr > FLAGS["off_hour_ratio"]:
        avg = pm.get("off_hour_ratio", 0)
        flags.append(f"Elevated off-hours ratio ({ohl}%, — {ohr - avg:+.1%} vs the {avg:.1%} user average)")

    if wl > FLAGS["weekend_logins"]:
        avg = pm.get("weekend_logins", 0)
        flags.append(f"Frequent weekend logins ({_int(wl)} events — {_int(wl - avg):+,} above the {avg:.1f} user average)")
    elif wr > FLAGS["weekend_ratio"]:
        avg = pm.get("weekend_ratio", 0)
        flags.append(f"Elevated weekend login ratio ({wr * 100:.0f}% of logins — {wr - avg:+.1%} vs the {avg:.1%} user average)")

    if upc >= FLAGS["unique_pcs_logon"]:
        avg = pm.get("unique_pcs_logon", 0)
        flags.append(f"Accessed {_int(upc)} distinct machines — {_int(upc - avg):+,} above the {avg:.1f} user average")

    if lnl > FLAGS["late_night_logins"]:
        avg = pm.get("late_night_logins", 0)
        flags.append(f"{_int(lnl)} late-night logins (10PM-4AM) — {_int(lnl - avg):+,} above the {avg:.1f} user average")

    if dc > FLAGS["device_connections"]:
        avg = pm.get("device_connections", 0)
        flags.append(f"Excessive device connections ({_int(dc)} total — {_int(dc - avg):+,} above the {avg:.1f} user average")

    if upd >= FLAGS["unique_pcs_device"]:
        avg = pm.get("unique_pcs_device", 0)
        flags.append(f"Connected devices across {_int(upd)} machines — {_int(upd - avg):+,} above the {avg:.1f} user average")

    if dlr > FLAGS["device_to_login_ratio"]:
        avg = pm.get("device_to_login_ratio", 0)
        flags.append(f"High device-to-login ratio ({dlr:.2f} — {dlr - avg:+.2f} vs the {avg:.2f} user average)")

    if pcd > FLAGS["pc_diversity_score"]:
        avg = pm.get("pc_diversity_score", 0)
        flags.append(f"High PC diversity score ({pcd:.3f} — {pcd - avg:+.3f} vs the {avg:.3f} user average)")

    if asg < FLAGS["avg_session_gap_sec"]:
        avg = pm.get("avg_session_gap", 0)
        flags.append(f"Very frequent logins (avg gap of {asg:.0f}s between sessions — {avg - asg:+.0f}s below the {avg:.0f}s user average)")

    return " | ".join(flags) if flags else "No significant anomalies detected."


def add_explanations(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply explanation generation to every user row.

    Pre-computes population means once for efficient per-row access.

    Args:
        scored_df: DataFrame with feature columns and 'risk_score'.

    Returns:
        DataFrame with added 'risk_explanation' column.

    Example:
        >>> rdf = add_explanations(scored_df)
    """
    scored_df = scored_df.copy()
    _load_population_means(scored_df)
    scored_df["risk_explanation"] = scored_df.apply(generate_explanation, axis=1)
    explained = (scored_df["risk_explanation"] != "No significant anomalies detected.").sum()
    logger.info("")
    logger.info("  Explanations generated. Users with flagged behaviours: %s", f"{explained:,}")
    return scored_df


# ─────────────────────────────────────────────
# RISK PROFILE BUILDER
# ─────────────────────────────────────────────

def build_risk_profile(row: pd.Series) -> Dict[str, Any]:
    """
    Build a detailed risk profile dictionary for a single user.

    Args:
        row: pd.Series with feature and risk columns.

    Returns:
        Dictionary with all key fields for display.

    Example:
        >>> profile = build_risk_profile(user_row)
    """
    return {
        "user": row["user"],
        "risk_level": row["risk_level"],
        "risk_score": row["risk_score"],
        "is_anomaly": bool(row["is_anomaly"]),
        "login_count": int(row.get("login_count", 0)),
        "off_hour_logins": int(row.get("off_hour_logins", 0)),
        "weekend_logins": int(row.get("weekend_logins", 0)),
        "late_night_logins": int(row.get("late_night_logins", 0)),
        "unique_pcs_logon": int(row.get("unique_pcs_logon", 0)),
        "off_hour_ratio": float(row.get("off_hour_ratio", 0)),
        "weekend_ratio": float(row.get("weekend_ratio", 0)),
        "pc_diversity_score": float(row.get("pc_diversity_score", 0)),
        "device_connections": int(row.get("device_connections", 0)),
        "unique_pcs_device": int(row.get("unique_pcs_device", 0)),
        "device_to_login_ratio": float(row.get("device_to_login_ratio", 0)),
        "avg_session_gap": float(row.get("avg_session_gap", 0)),
        "risk_explanation": row["risk_explanation"],
    }


# ─────────────────────────────────────────────
# RISK SUMMARY
# ─────────────────────────────────────────────

def risk_summary(risk_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregate statistics about the risk distribution.

    Args:
        risk_df: DataFrame with risk_level, risk_score, and is_anomaly columns.

    Returns:
        Dictionary with counts, percentages, and score statistics.

    Example:
        >>> summary = risk_summary(risk_df)
    """
    total = len(risk_df)
    high = (risk_df["risk_level"] == "High").sum()
    med = (risk_df["risk_level"] == "Medium").sum()
    low = (risk_df["risk_level"] == "Low").sum()
    anomalies = risk_df["is_anomaly"].sum()

    return {
        "total_users": total,
        "high_risk": int(high),
        "high_risk_pct": round(high / total * 100, 2) if total > 0 else 0,
        "medium_risk": int(med),
        "medium_risk_pct": round(med / total * 100, 2) if total > 0 else 0,
        "low_risk": int(low),
        "low_risk_pct": round(low / total * 100, 2) if total > 0 else 0,
        "anomalies": int(anomalies),
        "anomaly_pct": round(anomalies / total * 100, 1) if total > 0 else 0,
        "mean_risk_score": round(risk_df["risk_score"].mean(), 2),
        "median_risk_score": round(risk_df["risk_score"].median(), 2),
        "max_risk_score": round(risk_df["risk_score"].max(), 2),
    }


# ─────────────────────────────────────────────
# SAVE RISK REPORT
# ─────────────────────────────────────────────

def save_risk_report(risk_df: pd.DataFrame, path: str = RISK_REPORT_PATH) -> None:
    """
    Save the risk report to CSV, sorted by risk_score descending.

    Args:
        risk_df: DataFrame with risk_level and risk_explanation.
        path:    Destination path. Defaults to config path.

    Example:
        >>> save_risk_report(risk_df)
    """
    os.makedirs("outputs", exist_ok=True)
    risk_df_sorted = risk_df.sort_values("risk_score", ascending=False)
    risk_df_sorted.to_csv(path, index=False)
    logger.info("  Risk report saved -> %s", path)


# ─────────────────────────────────────────────
# PRINT RISK REPORT
# ─────────────────────────────────────────────

def print_risk_report(risk_df: pd.DataFrame) -> None:
    """
    Print a formatted risk report to stdout.

    Shows full details for High risk users, summaries for Medium and Low.

    Args:
        risk_df: DataFrame with all risk columns.

    Example:
        >>> print_risk_report(risk_df)
    """
    _s = "=" * 60
    logger.info("")
    logger.info(_s)
    logger.info("  FULL RISK REPORT")
    logger.info(_s)

    high = risk_df[risk_df["risk_level"] == "High"].sort_values("risk_score", ascending=False)
    logger.info("")
    logger.info("  HIGH RISK USERS (%s total)", len(high))
    print("-" * 60)

    if len(high) == 0:
        logger.info("  None detected.")
    else:
        for _, row in high.iterrows():
            logger.info("")
            logger.info("  User        : %s", row['user'])
            logger.info("     Risk Score  : %.1f / 100", row['risk_score'])
            logger.info("     Logins      : %s  (Off-hours: %s, Weekend: %s)",
                        f"{int(row['login_count'])}", f"{int(row['off_hour_logins'])}",
                        f"{int(row['weekend_logins'])}")
            logger.info("     Unique PCs  : %s  |  Device connections: %s",
                        f"{int(row['unique_pcs_logon'])}", f"{int(row['device_connections'])}")
            logger.info("     Flags       : %s", row['risk_explanation'])

    medium = risk_df[risk_df["risk_level"] == "Medium"].sort_values("risk_score", ascending=False)
    logger.info("")
    logger.info("  MEDIUM RISK USERS — Top 5 of %s total", len(medium))
    print("-" * 60)
    if len(medium) > 0:
        display_cols = ["user", "risk_score", "login_count", "off_hour_logins", "device_connections"]
        print(medium[display_cols].head(5).to_string(index=False))

    low = risk_df[risk_df["risk_level"] == "Low"]
    logger.info("")
    logger.info("  LOW RISK USERS — %s users flagged as normal", len(low))
    print("-" * 60)
    logger.info("  Average login count : %.1f", low['login_count'].mean())
    logger.info("  Average off-hour %%  : %.1f%%", low['off_hour_ratio'].mean() * 100)


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_risk_scoring() -> pd.DataFrame:
    """
    Execute the full risk scoring pipeline.

    Returns:
        DataFrame with risk_level and risk_explanation per user.

    Example:
        >>> risk_df = run_risk_scoring()
    """
    _s = "=" * 60
    logger.info("")
    logger.info(_s)
    logger.info("  STAGE 5 — Risk Scoring Pipeline")
    logger.info(_s)

    logger.info("")
    logger.info("  [1/4] Loading anomaly scores...")
    if os.path.exists(SCORED_PATH):
        scored_df = pd.read_csv(SCORED_PATH)
        logger.info("  Loaded scored data: %s users", f"{len(scored_df):,}")
    else:
        logger.warning("  Scored file not found — running model pipeline first...")
        from src.model import run_model_pipeline as _rmp
        scored_df = _rmp()

    logger.info("")
    logger.info("  [2/4] Assigning risk levels...")
    risk_df = add_risk_levels(scored_df)

    logger.info("")
    logger.info("  [3/4] Generating risk explanations...")
    risk_df = add_explanations(risk_df)

    logger.info("")
    logger.info("  [4/4] Saving risk report...")
    save_risk_report(risk_df)

    return risk_df


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    risk_df = run_risk_scoring()
    print_risk_report(risk_df)

    top_user = risk_df.nlargest(1, "risk_score").iloc[0]
    profile = build_risk_profile(top_user)

    print("\n" + "=" * 60)
    print("  SAMPLE RISK PROFILE (Highest Risk User)")
    print("=" * 60)
    for key, value in profile.items():
        print(f"  {key:<22} : {value}")

    summary = risk_summary(risk_df)
    print(f"\n{'=' * 60}")
    print("  RISK SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    for key, value in summary.items():
        print(f"  {key:<20} : {value}")
