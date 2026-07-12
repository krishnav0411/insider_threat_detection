#!/usr/bin/env python3
"""
File: run_pipeline.py
Purpose: Stage 9 — Master orchestrator. Runs the complete insider threat
         detection pipeline end-to-end through all 8 stages.
Usage:   python run_pipeline.py
Inputs:  data/logon.csv, data/device.csv
Outputs: All files in outputs/ directory
Dependencies: All src modules, pandas, numpy, json
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

SEPARATOR     = "=" * 65
SUB_SEPARATOR = "-" * 65
START_TIME    = time.time()

FEATURE_LABELS = {
    "login_count": "Total Logins",
    "off_hour_logins": "Off-Hour Logins",
    "weekend_logins": "Weekend Logins",
    "late_night_logins": "Late-Night Logins",
    "unique_pcs_logon": "Unique PCs (Logon)",
    "off_hour_ratio": "Off-Hour Ratio",
    "weekend_ratio": "Weekend Ratio",
    "pc_diversity_score": "PC Diversity Score",
    "device_connections": "Device Connections",
    "unique_pcs_device": "Unique PCs (Device)",
    "device_to_login_ratio": "Device/Login Ratio",
    "avg_session_gap": "Avg Session Gap",
}

FEATURE_COLS = list(FEATURE_LABELS.keys())


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def section(title: str):
    """Print a section header with separator lines."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def step(number: int, total: int, label: str):
    """Print a step progress indicator."""
    print(f"\n  [{number}/{total}] {label}...")


def success(message: str):
    """Print a PASS message."""
    print(f"  [PASS] {message}")


def warning(message: str):
    """Print a WARN message."""
    print(f"  [WARN] {message}")


def failure(message: str):
    """Print a FAIL message and exit."""
    print(f"  [FAIL] {message}")
    print("\n  Pipeline stopped. Fix the error above and re-run.")
    sys.exit(1)


def elapsed() -> str:
    """Return elapsed time since START_TIME as a formatted string."""
    secs = time.time() - START_TIME
    return f"{secs:.1f}s"


def pretty_time(secs: float) -> str:
    """Format seconds into a human-readable time string."""
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    return f"{secs//60}m {secs%60}s"


# ─────────────────────────────────────────────
# STAGE 1 — ENVIRONMENT CHECK
# ─────────────────────────────────────────────

def check_environment():
    """
    Verify that required libraries and data files exist.
    Exits with error if anything is missing.
    """
    section("STAGE 1 — Environment & File Check")

    step(1, 3, "Checking required libraries")
    required_libs = ["pandas", "numpy", "sklearn", "streamlit", "plotly", "joblib"]
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        failure(f"Missing libraries: {missing}\n  -> Run: pip install -r requirements.txt")
    success(f"All {len(required_libs)} required libraries found.")

    step(2, 3, "Checking data files")
    required_files = ["data/logon.csv", "data/device.csv"]
    for path in required_files:
        if not os.path.exists(path):
            failure(f"Data file not found: '{path}'\n  -> Place logon.csv and device.csv in the /data folder.")
    success("Both data files found: logon.csv | device.csv")

    step(3, 3, "Checking output directory")
    os.makedirs("outputs", exist_ok=True)
    success("Output directory ready.")


# ─────────────────────────────────────────────
# STAGE 2 — PREPROCESSING
# ─────────────────────────────────────────────

def run_preprocessing() -> tuple:
    """
    Run data preprocessing on both logon.csv and device.csv.

    Returns:
        tuple: (logon_events, device_events)
            logon_events  — DataFrame of Logon events only.
            device_events — DataFrame of Connect events only.
    """
    section("STAGE 2 — Data Preprocessing")

    step(1, 2, "Preprocessing logon.csv")
    try:
        from src.preprocess import preprocess_logon
        logon_all, logon_events = preprocess_logon()
        if len(logon_events) == 0:
            failure("No Logon events found after preprocessing.")
        success(
            f"logon.csv cleaned: {len(logon_all):,} rows | "
            f"{len(logon_events):,} Logon events | "
            f"{logon_all['user'].nunique():,} users"
        )
    except Exception as e:
        failure(f"Preprocessing logon.csv failed:\n  {e}")

    step(2, 2, "Preprocessing device.csv")
    try:
        from src.preprocess import preprocess_device
        device_all, device_events = preprocess_device()
        if len(device_events) == 0:
            failure("No Connect events found after preprocessing.")
        success(
            f"device.csv cleaned: {len(device_all):,} rows | "
            f"{len(device_events):,} Connect events | "
            f"{device_all['user'].nunique():,} users"
        )
    except Exception as e:
        failure(f"Preprocessing device.csv failed:\n  {e}")

    return logon_events, device_events


# ─────────────────────────────────────────────
# STAGE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def run_feature_engineering(logon_events: pd.DataFrame, device_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 12 per-user features from preprocessed event data.

    Returns:
        DataFrame with one row per user and 12 feature columns.
    """
    section("STAGE 3 — Feature Engineering")

    step(1, 3, "Extracting logon features")
    try:
        from src.features import extract_logon_features
        logon_features = extract_logon_features(logon_events)
        success(f"Logon features extracted: {len(logon_features):,} users | "
                f"{len(logon_features.columns) - 1} features")
    except Exception as e:
        failure(f"Logon feature extraction failed:\n  {e}")

    step(2, 3, "Extracting device features")
    try:
        from src.features import extract_device_features
        device_features = extract_device_features(device_events)
        success(f"Device features extracted: {len(device_features):,} users | "
                f"{len(device_features.columns) - 1} features")
    except Exception as e:
        failure(f"Device feature extraction failed:\n  {e}")

    step(3, 3, "Building unified feature table")
    try:
        from src.features import build_feature_table, save_feature_table
        feature_table = build_feature_table(logon_features, device_features)
        save_feature_table(feature_table)

        missing_cols = [c for c in FEATURE_COLS if c not in feature_table.columns]
        if missing_cols:
            failure(f"Feature table missing columns: {missing_cols}")

        for col in ["off_hour_ratio", "weekend_ratio", "pc_diversity_score"]:
            if col in feature_table.columns and feature_table[col].max() > 1.0:
                warning(f"{col} has values > 1.0 — check feature logic.")

        success(f"Feature table built: {len(feature_table):,} users | "
                f"{len(FEATURE_COLS)} features | Saved to outputs/user_features.csv")
    except Exception as e:
        failure(f"Feature table build failed:\n  {e}")

    return feature_table


# ─────────────────────────────────────────────
# STAGE 4 — MODEL TRAINING
# ─────────────────────────────────────────────

def run_model_training(feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features, select contamination, train Isolation Forest, save artifacts.

    Returns:
        DataFrame with risk_score and is_anomaly for every user.
    """
    section("STAGE 4 — Isolation Forest Model Training")

    step(1, 4, "Scaling features")
    try:
        from src.model import scale_features
        X_scaled, scaler = scale_features(feature_table)
        success(f"Features scaled: matrix shape {X_scaled.shape}")
    except Exception as e:
        failure(f"Feature scaling failed:\n  {e}")

    step(2, 4, "Selecting best contamination via hyperparameter search")
    try:
        from src.model import select_contamination
        best_cont, hp_results, best_sep = select_contamination(X_scaled)
        success(f"Selected contamination={best_cont} (separation={best_sep:.4f})")
    except Exception as e:
        failure(f"Hyperparameter selection failed:\n  {e}")

    step(3, 4, "Training Isolation Forest")
    try:
        from src.model import train_isolation_forest
        model = train_isolation_forest(X_scaled, contamination=best_cont)
        success(f"Model trained: {model.n_estimators} trees | contamination={model.contamination}")
    except Exception as e:
        failure(f"Model training failed:\n  {e}")

    step(4, 4, "Generating anomaly scores and saving artifacts")
    try:
        from src.model import (generate_scores, save_model_artifacts,
                               save_model_metadata, save_scored_results)
        scored_df = generate_scores(model, X_scaled, feature_table)
        save_model_artifacts(model, scaler)
        save_model_metadata(model, scaler, len(feature_table), best_sep, best_cont)
        save_scored_results(scored_df)

        anomaly_count = scored_df["is_anomaly"].sum()
        anomaly_pct   = anomaly_count / len(scored_df) * 100
        if anomaly_count == 0:
            warning("No anomalies detected — check contamination setting.")

        success(f"Scoring complete: {len(scored_df):,} users scored | "
                f"{anomaly_count} anomalies ({anomaly_pct:.1f}%) | "
                f"Score range: {scored_df['risk_score'].min():.1f}"
                f" — {scored_df['risk_score'].max():.1f}")
    except Exception as e:
        failure(f"Score generation failed:\n  {e}")

    return scored_df


# ─────────────────────────────────────────────
# STAGE 5 — RISK SCORING
# ─────────────────────────────────────────────

def run_risk_scoring(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign risk levels and generate human-readable explanations.

    Returns:
        DataFrame with risk_level and risk_explanation columns.
    """
    section("STAGE 5 — Risk Scoring & Explanation")

    step(1, 2, "Assigning risk levels")
    try:
        from src.risk_scoring import add_risk_levels, add_explanations
        risk_df = add_risk_levels(scored_df)
        high   = (risk_df["risk_level"] == "High").sum()
        medium = (risk_df["risk_level"] == "Medium").sum()
        low    = (risk_df["risk_level"] == "Low").sum()
        if high == 0:
            warning("No High risk users found — consider lowering THRESHOLD_HIGH.")
        success(f"Risk levels assigned: [HIGH] {high} | [MED] {medium} | [LOW] {low}")
    except Exception as e:
        failure(f"Risk level assignment failed:\n  {e}")

    step(2, 2, "Generating risk explanations and saving report")
    try:
        from src.risk_scoring import save_risk_report
        risk_df = add_explanations(risk_df)
        save_risk_report(risk_df)
        explained = (risk_df["risk_explanation"] != "No significant anomalies detected.").sum()
        success(f"Explanations generated: {explained:,} users flagged | "
                f"Report saved to outputs/risk_report.csv")
    except Exception as e:
        failure(f"Risk explanation generation failed:\n  {e}")

    return risk_df


# ─────────────────────────────────────────────
# STAGE 6 — ATTACK SIMULATION
# ─────────────────────────────────────────────

def run_attack_simulation() -> tuple:
    """
    Inject synthetic attack personas, score them, and validate detection.

    Returns:
        tuple: (scored_personas, combined_df)
    """
    section("STAGE 6 — Attack Simulation & Validation")

    step(1, 2, "Running attack simulation")
    try:
        from src.attack_simulation import run_attack_simulation as sim
        scored_personas, combined_df = sim()
        success(f"Simulation complete: {len(scored_personas)} personas injected | "
                f"{len(combined_df):,} total users in combined dataset")
    except Exception as e:
        failure(f"Attack simulation failed:\n  {e}")

    step(2, 2, "Validating detection results")
    try:
        detected   = (scored_personas["risk_level"] == "High").sum()
        total_sims = len(scored_personas)
        detection_rate = detected / total_sims * 100

        print("\n    Per-persona breakdown:")
        for _, row in scored_personas.iterrows():
            tag = "[OK]" if row["risk_level"] == "High" else "[MISS]"
            print(f"    {tag} {row['user']:<18} ML: {row['ml_score']:>5.1f}  "
                  f"Rule: {row['rule_score']:>5.1f}  "
                  f"Score: {row['risk_score']:>5.1f}  Level: {row['risk_level']}")

        if detected == total_sims:
            success(f"Detection rate: {detected}/{total_sims} ({detection_rate:.0f}%) — ALL THREATS DETECTED")
        elif detected >= total_sims * 0.8:
            warning(f"Detection rate: {detected}/{total_sims} ({detection_rate:.0f}%) — Good.")
        else:
            warning(f"Detection rate: {detected}/{total_sims} ({detection_rate:.0f}%) — Low.")
    except Exception as e:
        failure(f"Simulation validation failed:\n  {e}")

    return scored_personas, combined_df


# ─────────────────────────────────────────────
# STAGE 7 — DATA QUALITY & MODEL VALIDATION
# ─────────────────────────────────────────────

def run_data_validation(feature_table: pd.DataFrame, scored_df: pd.DataFrame,
                         scored_personas: pd.DataFrame):
    """
    Run 4 quality checks: feature variance, correlations, score separation,
    and persona detection rate.
    """
    section("STAGE 7 — Data Quality & Model Validation")

    checks_passed = 0
    checks_total  = 4

    step(1, checks_total, "Checking feature variance (all 12 features)")
    try:
        numeric = feature_table[FEATURE_COLS]
        zero_var = [c for c in FEATURE_COLS if numeric[c].std() == 0]
        if zero_var:
            warning(f"Features with zero variance: {zero_var}")
        else:
            success(f"All {len(FEATURE_COLS)} features have non-zero variance.")
            var_df = numeric.var().sort_values(ascending=False)
            for i, (col, v) in enumerate(var_df.items()):
                tag = " <-- HIGHEST" if i == 0 else (" <-- LOWEST" if i == len(var_df)-1 else "")
                print(f"       {FEATURE_LABELS.get(col, col):<28} variance={v:>12.2f}{tag}")
            checks_passed += 1
    except Exception as e:
        warning(f"Variance check failed: {e}")

    step(2, checks_total, "Checking feature correlations (threshold=0.95)")
    try:
        corr_matrix = numeric.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_pairs = [
            (col, row, round(upper_tri[col][row], 3))
            for col in upper_tri.columns for row in upper_tri.index
            if not pd.isna(upper_tri[col][row]) and upper_tri[col][row] > 0.95
        ]
        if high_pairs:
            warning(f"Found {len(high_pairs)} highly correlated feature pair(s) (>0.95):")
            for col, row, val in high_pairs:
                print(f"       {col} <-> {row} = {val}")
        else:
            success("No feature pairs exceed correlation threshold of 0.95.")
            checks_passed += 1
    except Exception as e:
        warning(f"Correlation check failed: {e}")

    step(3, checks_total, "Checking model score separation (threshold=0.15)")
    try:
        raw = scored_df["raw_anomaly_score"].values
        anom_mask = scored_df["is_anomaly"] == 1
        if anom_mask.sum() > 0 and anom_mask.sum() < len(raw):
            sep = float(raw[~anom_mask].mean() - raw[anom_mask].mean())
        else:
            sep = 0.0
        if sep < 0.15:
            warning(f"Score separation is {sep:.4f} — below 0.15 threshold.")
        else:
            success(f"Score separation is {sep:.4f} — model is confidently discriminating.")
            checks_passed += 1
    except Exception as e:
        warning(f"Separation check failed: {e}")

    step(4, checks_total, "Checking attack persona detection (all 7 should be High)")
    try:
        high_count = (scored_personas["risk_level"] == "High").sum()
        if high_count == len(scored_personas):
            success(f"All {len(scored_personas)} personas detected as High risk.")
            checks_passed += 1
        else:
            missed = scored_personas[scored_personas["risk_level"] != "High"]["user"].tolist()
            warning(f"{high_count}/{len(scored_personas)} personas detected as High risk. Missed: {missed}")
            print(f"       Median composite score of personas: {scored_personas['risk_score'].median():.1f}")
            print("       High threshold: 70.0")
    except Exception as e:
        warning(f"Persona detection check failed: {e}")

    print(f"\n  Validation complete: {checks_passed}/{checks_total} checks passed.")


# ─────────────────────────────────────────────
# STAGE 8 — OUTPUT FILE VALIDATION
# ─────────────────────────────────────────────

def validate_outputs():
    """
    Verify that all expected output files exist and report their sizes.
    """
    section("STAGE 8 — Output File Validation")

    expected_outputs = {
        "outputs/user_features.csv"          : "Feature table",
        "outputs/user_scores.csv"            : "Anomaly scores",
        "outputs/risk_report.csv"            : "Risk report",
        "outputs/simulated_results.csv"      : "Simulation results",
        "outputs/simulation_report.txt"      : "Simulation report (txt)",
        "outputs/model_metadata.json"        : "Model metadata",
        "outputs/isolation_forest_model.pkl" : "Trained model",
        "outputs/scaler.pkl"                 : "Feature scaler",
    }

    all_present = True
    for path, label in expected_outputs.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            success(f"{label:<30} -> {path}  ({size_kb:>8.1f} KB)")
        else:
            warning(f"{label:<30} -> MISSING: {path}")
            all_present = False

    if all_present:
        print(f"\n  All {len(expected_outputs)} output files present.")
    else:
        print("\n  Some output files are missing. Re-run to regenerate.")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

def print_final_summary(feature_table: pd.DataFrame, risk_df: pd.DataFrame,
                         scored_personas: pd.DataFrame):
    """
    Print a comprehensive summary of pipeline results including model metadata,
    top features, risk distribution, detection breakdown, and best/worst users.
    """
    section("FINAL SYSTEM SUMMARY")

    total   = len(risk_df)
    high    = (risk_df["risk_level"] == "High").sum()
    medium  = (risk_df["risk_level"] == "Medium").sum()
    low     = (risk_df["risk_level"] == "Low").sum()
    detected = (scored_personas["risk_level"] == "High").sum()
    total_sim = len(scored_personas)
    runtime = elapsed()

    meta = {}
    if os.path.exists("outputs/model_metadata.json"):
        try:
            with open("outputs/model_metadata.json", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    contamination_used = meta.get("contamination", "?")
    separation = meta.get("score_separation", "?")
    trained_date = meta.get("training_date", "?")

    # Top 3 most influential features (by abs correlation with risk_score)
    print("\n  TOP 3 MOST INFLUENTIAL FEATURES")
    print(SUB_SEPARATOR)
    try:
        available = [c for c in FEATURE_COLS if c in risk_df.columns]
        corrs = risk_df[available].corrwith(risk_df["risk_score"]).abs().sort_values(ascending=False)
        for i, (col, val) in enumerate(corrs.head(3).items()):
            label = FEATURE_LABELS.get(col, col)
            print(f"  {i+1}. {label:<28} correlation={val:.3f}")
    except Exception:
        print("  (could not compute)")

    date_range_str = "?"
    try:
        dates = pd.to_datetime(pd.read_csv("data/logon.csv")["date"])
        date_range_str = f"{dates.iloc[0]} to {dates.iloc[-1]}"
    except Exception:
        pass

    print(f"""
  DATASET
  {"Users analysed":<30}: {total:,}
  {"Date range":<30}: {date_range_str}

  MODEL
  {"Algorithm":<30}: Isolation Forest
  {"Features used":<30}: {len(FEATURE_COLS)} behavioral features
  {"Contamination":<30}: {contamination_used}
  {"Score separation":<30}: {separation}
  {"Training date":<30}: {trained_date}

  RISK SCORING
  {"High risk users":<30}: {high:,}  ({high/total*100:.1f}%)
  {"Medium risk users":<30}: {medium:,}  ({medium/total*100:.1f}%)
  {"Low risk users":<30}: {low:,}  ({low/total*100:.1f}%)
  {"Peak risk score":<30}: {risk_df['risk_score'].max():.1f} / 100
  {"Mean risk score":<30}: {risk_df['risk_score'].mean():.2f}
  {"Median risk score":<30}: {risk_df['risk_score'].median():.2f}

  ATTACK SIMULATION
  {"Personas injected":<30}: {total_sim}
  {"Detected as High Risk":<30}: {detected} / {total_sim}
  {"Detection rate":<30}: {detected/total_sim*100:.0f}%
    """)

    # Per-persona detection breakdown
    print(SUB_SEPARATOR)
    print("  PER-PERSONA DETECTION BREAKDOWN")
    print(SUB_SEPARATOR)
    for _, row in scored_personas.iterrows():
        icon = "[DETECTED]" if row["risk_level"] == "High" else "[MISSED]"
        print(f"  {icon} {row['user']:<18}  "
              f"ML: {row['ml_score']:>6.1f}  "
              f"Rule: {row['rule_score']:>6.1f}  "
              f"Composite: {row['risk_score']:>6.1f}  "
              f"Level: {row['risk_level']}")

    # Best and worst scoring users
    print(f"\n{SUB_SEPARATOR}")
    print("  BEST & WORST SCORING USERS")
    print(SUB_SEPARATOR)

    worst = risk_df.nlargest(1, "risk_score").iloc[0]
    best  = risk_df.nsmallest(1, "risk_score").iloc[0]

    for label, row in [("MOST ANOMALOUS", worst), ("MOST NORMAL", best)]:
        print(f"\n  [{label}] {row['user']}")
        print(f"     Risk Score   : {row['risk_score']:.1f} / 100  |  Level: {row['risk_level']}")
        print(f"     Logins       : {int(row['login_count']):>5,}  |  "
              f"Off-Hour: {int(row.get('off_hour_logins', 0)):>5,}  |  "
              f"Late-Night: {int(row.get('late_night_logins', 0)):>4}")
        print(f"     Unique PCs   : {int(row.get('unique_pcs_logon', 0)):>5,}  |  "
              f"Device Conn: {int(row.get('device_connections', 0)):>5,}  |  "
              f"Dev PCs: {int(row.get('unique_pcs_device', 0)):>4}")
        print(f"     PC Diversity : {row.get('pc_diversity_score', 0):.3f}  |  "
              f"Dev/Login Ratio: {row.get('device_to_login_ratio', 0):.2f}  |  "
              f"Avg Gap: {row.get('avg_session_gap', 0):.0f}s")

    # Simulation report summary
    if os.path.exists("outputs/simulation_report.txt"):
        print(f"\n{SUB_SEPARATOR}")
        print("  SIMULATION REPORT SUMMARY")
        print(SUB_SEPARATOR)
        try:
            with open("outputs/simulation_report.txt", encoding="utf-8") as f:
                content = f.read()
            if "OVERALL VALIDATION RESULT" in content:
                summary_section = content.split("OVERALL VALIDATION RESULT")[1]
                for line in summary_section.strip().split("\n"):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("="):
                        print(f"  {stripped}")
        except Exception:
            print("  (could not read simulation_report.txt)")

    # Performance
    print(f"\n{SUB_SEPARATOR}")
    print("  PERFORMANCE")
    print(SUB_SEPARATOR)
    print(f"  Total runtime: {runtime}")

    print(f"\n{SEPARATOR}")
    print("  SYSTEM STATUS: READY")
    print("  All outputs saved to /outputs folder.")
    print("  Launch dashboard: streamlit run dashboard/app.py")
    print(SEPARATOR)


# ─────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────

def main():
    """
    Master orchestrator: runs all 8 stages in sequence.
    """
    print(f"\n{SEPARATOR}")
    print("  AI-DRIVEN INSIDER THREAT DETECTION SYSTEM")
    print("  Full Pipeline Runner  |  CERT Dataset  |  Stage 9")
    print(SEPARATOR)
    print(f"\n  Starting pipeline at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    check_environment()
    logon_events, device_events = run_preprocessing()
    feature_table               = run_feature_engineering(logon_events, device_events)
    scored_df                   = run_model_training(feature_table)
    risk_df                     = run_risk_scoring(scored_df)
    scored_personas, combined   = run_attack_simulation()
    run_data_validation(feature_table, scored_df, scored_personas)
    validate_outputs()
    print_final_summary(feature_table, risk_df, scored_personas)


if __name__ == "__main__":
    main()
