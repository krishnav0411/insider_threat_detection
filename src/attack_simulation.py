# src/attack_simulation.py
"""
File: attack_simulation.py
Purpose: Stage 6 — Attack Simulation. Injects synthetic insider threat
         personas, scores them via ML + rule-based logic, and validates
         detection performance.
Inputs:  config.yaml — paths, rule weights
         outputs/{user_features, user_scores, model, scaler}
Outputs: outputs/simulated_results.csv, outputs/simulation_report.txt
Dependencies: pandas, numpy, joblib, logging; sklearn; src.risk_scoring, src.features
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    FEATURE_COLS, FEATURE_PATH, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    SIM_OUTPUT_PATH, SIM_REPORT_PATH, RULE_WEIGHTS, TOTAL_RULE_WEIGHT,
)
from src.risk_scoring import assign_risk_level, generate_explanation, CONFIG, _load_population_means

logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


# ─────────────────────────────────────────────
# RULE-BASED SCORING
# ─────────────────────────────────────────────

def compute_rule_score(row: pd.Series) -> float:
    """
    Compute a rule-based risk score (0-100) by checking each feature
    against config thresholds. Each triggered flag contributes its
    weight to a cumulative sum normalised to 0-100.

    Args:
        row: pd.Series with feature columns (must match FEATURE_COLS).

    Returns:
        Float in [0, 100] representing rule-based risk level.

    Example:
        >>> score = compute_rule_score(user_row)
    """
    triggered: float = 0.0
    cfg = CONFIG
    rw = RULE_WEIGHTS

    lc = row.get("login_count", 0)
    ohl = row.get("off_hour_logins", 0)
    ohr = row.get("off_hour_ratio", 0.0)
    wl = row.get("weekend_logins", 0)
    wr = row.get("weekend_ratio", 0.0)
    upc = row.get("unique_pcs_logon", 0)
    dc = row.get("device_connections", 0)
    upd = row.get("unique_pcs_device", 0)
    lnl = row.get("late_night_logins", 0)
    dlr = row.get("device_to_login_ratio", 0.0)
    pcd = row.get("pc_diversity_score", 0.0)
    asg = row.get("avg_session_gap", 0.0)

    if lc > cfg["flag_login_count"]:
        triggered += rw["excessive_logins"]
    if ohl > cfg["flag_off_hour_logins"]:
        triggered += rw["off_hour_logins"]
    elif ohr > cfg["flag_off_hour_ratio"]:
        triggered += rw["off_hour_ratio"]
    if ohr > 0.85 and lc < 100:
        triggered += rw["stealth_off_hour"]
    if wl > cfg["flag_weekend_logins"]:
        triggered += rw["weekend_logins"]
    elif wr > cfg["flag_weekend_ratio"]:
        triggered += rw["weekend_ratio"]
    if upc >= cfg["flag_unique_pcs_logon"]:
        triggered += rw["unique_pcs_logon"]
    if dc > cfg["flag_device_connections"]:
        triggered += rw["device_connections"]
    if upd >= cfg["flag_unique_pcs_device"]:
        triggered += rw["unique_pcs_device"]
    if lnl > cfg["flag_late_night_logins"]:
        triggered += rw["late_night_logins"]
    if dlr > cfg["flag_device_to_login_ratio"]:
        triggered += rw["device_to_login_ratio"]
    if pcd > cfg["flag_pc_diversity_score"]:
        triggered += rw["pc_diversity_score"]
    if asg < cfg["flag_avg_session_gap_sec"]:
        triggered += rw["avg_session_gap"]

    return round(min(triggered / TOTAL_RULE_WEIGHT * 100, 100), 2)


# ─────────────────────────────────────────────
# ATTACK PERSONA DEFINITIONS
# ─────────────────────────────────────────────

def build_attack_personas() -> pd.DataFrame:
    """
    Define 7 synthetic insider threat personas.

    Each persona includes all 12 behavioural features and a description.
    Designed to test ML sensitivity and rule-based detection logic.

    Returns:
        DataFrame with 7 rows, one per persona, and 14 columns.

    Example:
        >>> personas = build_attack_personas()
    """
    personas: List[Dict[str, Any]] = [
        {
            "user": "SIM_NightOwl",
            "login_count": 95,
            "off_hour_logins": 88,
            "weekend_logins": 12,
            "late_night_logins": 80,
            "unique_pcs_logon": 2,
            "off_hour_ratio": 0.93,
            "weekend_ratio": 0.13,
            "pc_diversity_score": 0.02,
            "device_connections": 8,
            "unique_pcs_device": 1,
            "device_to_login_ratio": 0.08,
            "avg_session_gap": 7200,
            "persona_description": (
                "Logs in almost exclusively between midnight and 5AM. "
                "Suggests unauthorized remote access or credential theft."
            ),
        },
        {
            "user": "SIM_PCHopper",
            "login_count": 120,
            "off_hour_logins": 18,
            "weekend_logins": 14,
            "late_night_logins": 5,
            "unique_pcs_logon": 14,
            "off_hour_ratio": 0.15,
            "weekend_ratio": 0.12,
            "pc_diversity_score": 0.12,
            "device_connections": 12,
            "unique_pcs_device": 11,
            "device_to_login_ratio": 0.10,
            "avg_session_gap": 3600,
            "persona_description": (
                "Accesses 14 unique PCs across the organization. "
                "Classic lateral movement pattern."
            ),
        },
        {
            "user": "SIM_DataMule",
            "login_count": 55,
            "off_hour_logins": 22,
            "weekend_logins": 19,
            "late_night_logins": 15,
            "unique_pcs_logon": 4,
            "off_hour_ratio": 0.40,
            "weekend_ratio": 0.35,
            "pc_diversity_score": 0.07,
            "device_connections": 112,
            "unique_pcs_device": 9,
            "device_to_login_ratio": 2.04,
            "avg_session_gap": 5400,
            "persona_description": (
                "Connects external devices 112 times across 9 machines. "
                "High weekend and off-hours activity. "
                "Strong indicator of bulk data exfiltration."
            ),
        },
        {
            "user": "SIM_GhostUser",
            "login_count": 18,
            "off_hour_logins": 16,
            "weekend_logins": 10,
            "late_night_logins": 14,
            "unique_pcs_logon": 5,
            "off_hour_ratio": 0.89,
            "weekend_ratio": 0.56,
            "pc_diversity_score": 0.28,
            "device_connections": 4,
            "unique_pcs_device": 4,
            "device_to_login_ratio": 0.22,
            "avg_session_gap": 86400,
            "persona_description": (
                "Minimal login activity but 89% off-hours. "
                "Accesses 5 machines. Stealth behavior."
            ),
        },
        {
            "user": "SIM_FullThreat",
            "login_count": 210,
            "off_hour_logins": 175,
            "weekend_logins": 68,
            "late_night_logins": 140,
            "unique_pcs_logon": 18,
            "off_hour_ratio": 0.83,
            "weekend_ratio": 0.32,
            "pc_diversity_score": 0.09,
            "device_connections": 145,
            "unique_pcs_device": 15,
            "device_to_login_ratio": 0.69,
            "avg_session_gap": 300,
            "persona_description": (
                "Maximum threat profile — combines all suspicious behaviors. "
                "Should always be flagged High risk."
            ),
        },
        {
            "user": "SIM_EmailThief",
            "login_count": 78,
            "off_hour_logins": 45,
            "weekend_logins": 22,
            "late_night_logins": 38,
            "unique_pcs_logon": 6,
            "off_hour_ratio": 0.58,
            "weekend_ratio": 0.28,
            "pc_diversity_score": 0.08,
            "device_connections": 15,
            "unique_pcs_device": 5,
            "device_to_login_ratio": 0.19,
            "avg_session_gap": 45,
            "persona_description": (
                "Exfiltrates data via email — 6 systems off-hours, "
                "45s session gaps, moderate device usage."
            ),
        },
        {
            "user": "SIM_Saboteur",
            "login_count": 42,
            "off_hour_logins": 39,
            "weekend_logins": 28,
            "late_night_logins": 35,
            "unique_pcs_logon": 9,
            "off_hour_ratio": 0.93,
            "weekend_ratio": 0.67,
            "pc_diversity_score": 0.21,
            "device_connections": 67,
            "unique_pcs_device": 8,
            "device_to_login_ratio": 1.60,
            "avg_session_gap": 180,
            "persona_description": (
                "Sabotage pattern — 3AM device connects across 8 machines. "
                "Extreme dev/login ratio (1.60), 3min session gaps."
            ),
        },
    ]

    df = pd.DataFrame(personas)
    logger.info("  Built %s attack personas.", len(df))
    return df


# ─────────────────────────────────────────────
# LOAD MODEL AND SCALER
# ─────────────────────────────────────────────

def load_model_artifacts() -> Tuple[Any, StandardScaler]:
    """
    Load the trained Isolation Forest model and StandardScaler from disk.

    Returns:
        Tuple of (model, scaler).

    Raises:
        FileNotFoundError: If model or scaler pkl files are missing.

    Example:
        >>> model, scaler = load_model_artifacts()
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at '{MODEL_SAVE_PATH}'. Run python src/model.py first.")
    if not os.path.exists(SCALER_SAVE_PATH):
        raise FileNotFoundError(f"Scaler not found at '{SCALER_SAVE_PATH}'. Run python src/model.py first.")

    model = joblib.load(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    logger.info("  Model and scaler loaded successfully.")
    return model, scaler


# ─────────────────────────────────────────────
# SCORE SIMULATED PERSONAS
# ─────────────────────────────────────────────

def score_personas(personas_df: pd.DataFrame, model: Any, scaler: StandardScaler) -> pd.DataFrame:
    """
    Run simulated personas through the trained model.

    Produces ML anomaly scores, rule-based scores, and a composite (60/40 blend).
    Scores are calibrated against the real user population for consistency.

    Args:
        personas_df: DataFrame of persona definitions (from build_attack_personas).
        model:       Trained IsolationForest.
        scaler:      Fitted StandardScaler from training.

    Returns:
        DataFrame with ML, rule, and composite scores plus risk levels.

    Example:
        >>> scored = score_personas(personas, model, scaler)
    """
    X = personas_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    raw_scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    # Normalise against real user population
    if os.path.exists("outputs/user_scores.csv"):
        real_scores = pd.read_csv("outputs/user_scores.csv")
        real_raw = model.decision_function(scaler.transform(real_scores[FEATURE_COLS].values))
        all_raw = np.concatenate([real_raw, raw_scores])
        score_min = (-1 * all_raw).min()
        score_max = (-1 * all_raw).max()
    else:
        score_min = (-1 * raw_scores).min()
        score_max = (-1 * raw_scores).max()

    inverted = -1 * raw_scores
    ml_scores = np.where(
        score_max > score_min,
        ((inverted - score_min) / (score_max - score_min)) * 100,
        50.0,
    ).round(2)

    result = personas_df.copy()
    result["raw_anomaly_score"] = np.round(raw_scores, 6)
    result["ml_score"] = ml_scores
    result["is_anomaly"] = (predictions == -1).astype(int)

    result["rule_score"] = result.apply(compute_rule_score, axis=1)
    result["composite_score"] = (0.6 * result["ml_score"] + 0.4 * result["rule_score"]).round(2)

    # Load population means for contextual explanations
    if os.path.exists("outputs/user_scores.csv"):
        _load_population_means(pd.read_csv("outputs/user_scores.csv"))
    elif os.path.exists(FEATURE_PATH):
        _load_population_means(pd.read_csv(FEATURE_PATH))

    result["risk_score"] = result["composite_score"]
    result["risk_level"] = result["risk_score"].apply(assign_risk_level)
    result["risk_explanation"] = result.apply(generate_explanation, axis=1)

    return result


# ─────────────────────────────────────────────
# COMBINE WITH REAL DATA
# ─────────────────────────────────────────────

def combine_with_real_data(scored_personas: pd.DataFrame) -> pd.DataFrame:
    """
    Merge scored personas into the real user dataset for the dashboard.

    Args:
        scored_personas: DataFrame output from score_personas().

    Returns:
        Combined DataFrame sorted by risk_score descending, with is_simulated flag.

    Example:
        >>> combined = combine_with_real_data(scored_personas)
    """
    real_df = pd.read_csv("outputs/user_scores.csv")
    real_df["risk_level"] = real_df["risk_score"].apply(assign_risk_level)
    real_df["risk_explanation"] = real_df.apply(generate_explanation, axis=1)
    real_df["is_simulated"] = 0
    real_df["persona_description"] = ""

    sim_df = scored_personas.copy()
    sim_df["is_simulated"] = 1

    combined = pd.concat([real_df, sim_df], ignore_index=True)
    combined = combined.sort_values("risk_score", ascending=False)

    logger.info("  Combined dataset: %s total users (%s real + %s simulated)",
                f"{len(combined):,}", f"{len(real_df):,}", len(sim_df))
    return combined


# ─────────────────────────────────────────────
# SAVE SIMULATION REPORT
# ─────────────────────────────────────────────

def save_simulation_report(scored_personas: pd.DataFrame) -> None:
    """
    Save a human-readable simulation report to outputs/simulation_report.txt
    and print it to stdout.

    Args:
        scored_personas: DataFrame output from score_personas().

    Example:
        >>> save_simulation_report(scored_personas)
    """
    passed = (scored_personas["risk_level"] == "High").sum()
    total = len(scored_personas)
    pct = passed / total * 100
    all_passed = passed == total
    missed: List[str] = scored_personas[scored_personas["risk_level"] != "High"]["user"].tolist() if not all_passed else []

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  INSIDER THREAT DETECTION — ATTACK SIMULATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for _, row in scored_personas.iterrows():
        detected = row["risk_level"] == "High"
        status = "[DETECTED]" if detected else "[MISSED]"
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"  Persona  : {row['user']}")
        lines.append(f"  Status   : {status}")
        lines.append(f"  Risk     : {row['risk_level']:>6}  |  "
                      f"ML: {row['ml_score']:>6.1f}  |  "
                      f"Rule: {row['rule_score']:>6.1f}  |  "
                      f"Composite: {row['risk_score']:>6.1f}  |  "
                      f"Anomaly: {'Yes' if row['is_anomaly'] else 'No'}")
        lines.append("-" * 70)

        feat_cols = [c for c in FEATURE_COLS if c in row]
        max_nl = max(len(c) for c in feat_cols)
        for c in feat_cols:
            val = row[c]
            lines.append(f"    {c:<{max_nl}} : {val:>12.4f}" if isinstance(val, float)
                         else f"    {c:<{max_nl}} : {str(val):>12}")
        lines.append("")
        lines.append(f"  Description: {row['persona_description']}")
        lines.append(f"  Flags      : {row['risk_explanation']}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  OVERALL VALIDATION RESULT")
    lines.append("=" * 70)
    lines.append(f"  Personas detected as High Risk : {passed} / {total}  ({pct:.0f}%)")
    if all_passed:
        lines.append("  [PASS] All attack personas correctly detected.")
    else:
        lines.append(f"  [WARN] Missed: {missed}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    os.makedirs("outputs", exist_ok=True)
    with open(SIM_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("  Simulation report saved -> %s", SIM_REPORT_PATH)
    print(report_text)


# ─────────────────────────────────────────────
# PRINT SIMULATION VALIDATION REPORT
# ─────────────────────────────────────────────

def print_simulation_report(scored_personas: pd.DataFrame) -> None:
    """
    Print a compact validation report for each simulated persona.

    Args:
        scored_personas: DataFrame output from score_personas().

    Example:
        >>> print_simulation_report(scored_personas)
    """
    logger.info("")
    logger.info(SEPARATOR)
    logger.info("  ATTACK SIMULATION VALIDATION REPORT")
    logger.info(SEPARATOR)

    all_passed = True
    for _, row in scored_personas.iterrows():
        detected = row["risk_level"] == "High"
        status = "[DETECTED]" if detected else "[MISSED]"
        all_passed = all_passed and detected

        print(f"\n  Persona       : {row['user']}")
        print(f"  Status        : {status}")
        print(f"  ML Score      : {row['ml_score']:.1f} / 100")
        print(f"  Rule Score    : {row['rule_score']:.1f} / 100")
        print(f"  Composite     : {row['risk_score']:.1f} / 100")
        print(f"  Risk Level    : {row['risk_level']}")
        print(f"  Is Anomaly    : {'Yes' if row['is_anomaly'] else 'No'}")
        print(f"  Description   : {row['persona_description']}")
        print(f"  Flags         : {row['risk_explanation']}")
        print(f"  {'-' * 55}")

    passed = scored_personas[scored_personas["risk_level"] == "High"].shape[0]
    total = len(scored_personas)
    pct = passed / total * 100

    logger.info("")
    logger.info(SEPARATOR)
    logger.info("  OVERALL VALIDATION RESULT")
    logger.info(SEPARATOR)
    logger.info("")
    logger.info("  Personas detected as High Risk : %s / %s  (%d%%)",
                f"{passed}", f"{total}", int(pct))

    if all_passed:
        logger.info("  [PASS] All attack personas correctly detected.")
    else:
        missed = scored_personas[scored_personas["risk_level"] != "High"]["user"].tolist()
        logger.warning("  [WARN] Some personas not detected as High Risk: %s", missed)


# ─────────────────────────────────────────────
# SAVE SIMULATION RESULTS
# ─────────────────────────────────────────────

def save_simulation_results(combined_df: pd.DataFrame) -> None:
    """
    Save combined real + simulated results to CSV.

    Args:
        combined_df: DataFrame with is_simulated flag.

    Example:
        >>> save_simulation_results(combined)
    """
    os.makedirs("outputs", exist_ok=True)
    combined_df.to_csv(SIM_OUTPUT_PATH, index=False)
    logger.info("  Simulation results saved -> %s", SIM_OUTPUT_PATH)


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_attack_simulation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full attack simulation pipeline: build, load, score, combine, save.

    Returns:
        Tuple of (scored_personas, combined_df).

    Example:
        >>> sp, comb = run_attack_simulation()
    """
    logger.info("")
    logger.info(SEPARATOR)
    logger.info("  STAGE 6 — Attack Simulation Pipeline")
    logger.info(SEPARATOR)

    logger.info("")
    logger.info("  [1/6] Building attack personas...")
    personas_df = build_attack_personas()

    logger.info("")
    logger.info("  [2/6] Loading trained model and scaler...")
    model, scaler = load_model_artifacts()

    logger.info("")
    logger.info("  [3/6] Scoring simulated personas...")
    scored_personas = score_personas(personas_df, model, scaler)

    logger.info("")
    logger.info("  [4/6] Combining with real user data...")
    combined_df = combine_with_real_data(scored_personas)

    logger.info("")
    logger.info("  [5/6] Saving simulation report...")
    save_simulation_report(scored_personas)

    logger.info("")
    logger.info("  [6/6] Saving simulation results...")
    save_simulation_results(combined_df)

    return scored_personas, combined_df


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    scored_personas, combined_df = run_attack_simulation()
    print_simulation_report(scored_personas)

    print(f"\n{SEPARATOR}")
    print("  TOP 10 USERS IN COMBINED DATASET")
    print(SEPARATOR)
    top10 = combined_df.nlargest(10, "risk_score")[["user", "risk_score", "risk_level",
                                                     "is_anomaly", "is_simulated"]]
    print(top10.to_string(index=False))
