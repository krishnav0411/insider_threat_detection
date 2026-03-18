# run_pipeline.py
# Stage 9: Final System Integration
# Runs the complete insider threat detection pipeline end-to-end
# Usage: python run_pipeline.py

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

SEPARATOR     = "=" * 65
SUB_SEPARATOR = "-" * 65
START_TIME    = time.time()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def step(number: int, total: int, label: str):
    print(f"\n  [{number}/{total}] {label}...")


def success(message: str):
    print(f"  [PASS] {message}")


def warning(message: str):
    print(f"  [WARN] {message}")


def failure(message: str):
    print(f"  [FAIL] {message}")
    print(f"\n  Pipeline stopped. Fix the error above and re-run.")
    sys.exit(1)


def elapsed() -> str:
    secs = time.time() - START_TIME
    return f"{secs:.1f}s"


# ─────────────────────────────────────────────
# STAGE 1 — ENVIRONMENT CHECK
# ─────────────────────────────────────────────

def check_environment():
    section("STAGE 1 — Environment & File Check")

    # Check required libraries
    step(1, 3, "Checking required libraries")
    required_libs = [
        "pandas", "numpy", "sklearn",
        "streamlit", "plotly", "joblib"
    ]
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        failure(f"Missing libraries: {missing}\n"
                f"  -> Run: pip install -r requirements.txt")
    success(f"All {len(required_libs)} required libraries found.")

    # Check data files
    step(2, 3, "Checking data files")
    required_files = ["data/logon.csv", "data/device.csv"]
    for path in required_files:
        if not os.path.exists(path):
            failure(
                f"Data file not found: '{path}'\n"
                f"  -> Place logon.csv and device.csv in the /data folder."
            )
    success("Both data files found: logon.csv | device.csv")

    # Check output directory
    step(3, 3, "Checking output directory")
    os.makedirs("outputs", exist_ok=True)
    success("Output directory ready.")


# ─────────────────────────────────────────────
# STAGE 2 — PREPROCESSING
# ─────────────────────────────────────────────

def run_preprocessing():
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

def run_feature_engineering(logon_events, device_events):
    section("STAGE 3 — Feature Engineering")

    step(1, 3, "Extracting logon features")
    try:
        from src.features import extract_logon_features
        logon_features = extract_logon_features(logon_events)
        success(
            f"Logon features extracted: {len(logon_features):,} users | "
            f"{len(logon_features.columns) - 1} features"
        )
    except Exception as e:
        failure(f"Logon feature extraction failed:\n  {e}")

    step(2, 3, "Extracting device features")
    try:
        from src.features import extract_device_features
        device_features = extract_device_features(device_events)
        success(
            f"Device features extracted: {len(device_features):,} users | "
            f"{len(device_features.columns) - 1} features"
        )
    except Exception as e:
        failure(f"Device feature extraction failed:\n  {e}")

    step(3, 3, "Building unified feature table")
    try:
        from src.features import build_feature_table, save_feature_table
        feature_table = build_feature_table(logon_features, device_features)
        save_feature_table(feature_table)

        # Validate feature table
        expected_cols = [
            "user", "login_count", "off_hour_logins",
            "weekend_logins", "unique_pcs_logon",
            "off_hour_ratio", "weekend_ratio",
            "device_connections", "unique_pcs_device"
        ]
        missing_cols = [
            c for c in expected_cols if c not in feature_table.columns
        ]
        if missing_cols:
            failure(f"Feature table missing columns: {missing_cols}")

        if feature_table["off_hour_ratio"].max() > 1.0:
            warning("off_hour_ratio has values > 1.0 — check feature logic.")

        success(
            f"Feature table built: {len(feature_table):,} users | "
            f"8 features | Saved to outputs/user_features.csv"
        )
    except Exception as e:
        failure(f"Feature table build failed:\n  {e}")

    return feature_table


# ─────────────────────────────────────────────
# STAGE 4 — MODEL TRAINING
# ─────────────────────────────────────────────

def run_model_training(feature_table):
    section("STAGE 4 — Isolation Forest Model Training")

    step(1, 3, "Scaling features")
    try:
        from src.model import scale_features
        X_scaled, scaler = scale_features(feature_table)
        success(
            f"Features scaled: matrix shape {X_scaled.shape} | "
            f"StandardScaler applied"
        )
    except Exception as e:
        failure(f"Feature scaling failed:\n  {e}")

    step(2, 3, "Training Isolation Forest")
    try:
        from src.model import train_isolation_forest
        model = train_isolation_forest(X_scaled)
        success(
            f"Model trained: {model.n_estimators} trees | "
            f"contamination={model.contamination}"
        )
    except Exception as e:
        failure(f"Model training failed:\n  {e}")

    step(3, 3, "Generating anomaly scores and saving model")
    try:
        from src.model import (
            generate_scores, save_model_artifacts, save_scored_results
        )
        scored_df = generate_scores(model, X_scaled, feature_table)
        save_model_artifacts(model, scaler)
        save_scored_results(scored_df)

        anomaly_count = scored_df["is_anomaly"].sum()
        anomaly_pct   = anomaly_count / len(scored_df) * 100

        if anomaly_count == 0:
            warning("No anomalies detected — check contamination setting.")

        success(
            f"Scoring complete: {len(scored_df):,} users scored | "
            f"{anomaly_count} anomalies ({anomaly_pct:.1f}%) | "
            f"Score range: {scored_df['risk_score'].min():.1f}"
            f" - {scored_df['risk_score'].max():.1f}"
        )
    except Exception as e:
        failure(f"Score generation failed:\n  {e}")

    return scored_df


# ─────────────────────────────────────────────
# STAGE 5 — RISK SCORING
# ─────────────────────────────────────────────

def run_risk_scoring(scored_df):
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

        success(
            f"Risk levels assigned: "
            f"[HIGH] {high} | [MED] {medium} | [LOW] {low}"
        )
    except Exception as e:
        failure(f"Risk level assignment failed:\n  {e}")

    step(2, 2, "Generating risk explanations and saving report")
    try:
        from src.risk_scoring import save_risk_report
        risk_df = add_explanations(risk_df)
        save_risk_report(risk_df)

        explained = (
            risk_df["risk_explanation"] !=
            "No significant anomalies detected."
        ).sum()

        success(
            f"Explanations generated: {explained:,} users flagged | "
            f"Report saved to outputs/risk_report.csv"
        )
    except Exception as e:
        failure(f"Risk explanation generation failed:\n  {e}")

    return risk_df


# ─────────────────────────────────────────────
# STAGE 6 — ATTACK SIMULATION
# ─────────────────────────────────────────────

def run_attack_simulation():
    section("STAGE 6 — Attack Simulation & Validation")

    step(1, 2, "Running attack simulation")
    try:
        from src.attack_simulation import run_attack_simulation as sim
        scored_personas, combined_df = sim()
        success(
            f"Simulation complete: {len(scored_personas)} personas injected | "
            f"{len(combined_df):,} total users in combined dataset"
        )
    except Exception as e:
        failure(f"Attack simulation failed:\n  {e}")

    step(2, 2, "Validating detection results")
    try:
        detected   = (scored_personas["risk_level"] == "High").sum()
        total_sims = len(scored_personas)
        detection_rate = detected / total_sims * 100

        if detected == total_sims:
            success(
                f"Detection rate: {detected}/{total_sims} personas "
                f"({detection_rate:.0f}%) — ALL THREATS DETECTED"
            )
        elif detected >= total_sims * 0.8:
            warning(
                f"Detection rate: {detected}/{total_sims} personas "
                f"({detection_rate:.0f}%) — Consider adjusting thresholds."
            )
        else:
            warning(
                f"Detection rate: {detected}/{total_sims} "
                f"({detection_rate:.0f}%) — Low. Retrain with higher contamination."
            )
    except Exception as e:
        failure(f"Simulation validation failed:\n  {e}")

    return scored_personas, combined_df


# ─────────────────────────────────────────────
# STAGE 7 — OUTPUT FILE VALIDATION
# ─────────────────────────────────────────────

def validate_outputs():
    section("STAGE 7 — Output File Validation")

    expected_outputs = {
        "outputs/user_features.csv"          : "Feature table",
        "outputs/user_scores.csv"            : "Anomaly scores",
        "outputs/risk_report.csv"            : "Risk report",
        "outputs/simulated_results.csv"      : "Simulation results",
        "outputs/isolation_forest_model.pkl" : "Trained model",
        "outputs/scaler.pkl"                 : "Feature scaler",
    }

    all_present = True
    for path, label in expected_outputs.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            success(f"{label:<28} -> {path}  ({size_kb:.1f} KB)")
        else:
            warning(f"{label:<28} -> MISSING: {path}")
            all_present = False

    if all_present:
        print(f"\n  All {len(expected_outputs)} output files present.")
    else:
        print(f"\n  Some output files are missing.")
        print(f"  Re-run the pipeline to regenerate them.")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

def print_final_summary(risk_df, scored_personas):
    section("FINAL SYSTEM SUMMARY")

    total   = len(risk_df)
    high    = (risk_df["risk_level"] == "High").sum()
    medium  = (risk_df["risk_level"] == "Medium").sum()
    low     = (risk_df["risk_level"] == "Low").sum()
    detected = (scored_personas["risk_level"] == "High").sum()
    total_sim = len(scored_personas)
    runtime = elapsed()

    print(f"""
  DATASET
  {"Users analysed":<30}: {total:,}
  {"Date range":<30}: {
    pd.read_csv("data/logon.csv")["date"].iloc[0]
  } to {
    pd.read_csv("data/logon.csv")["date"].iloc[-1]
  }

  MODEL
  {"Algorithm":<30}: Isolation Forest
  {"Features used":<30}: 8 behavioral features
  {"Contamination":<30}: 5%

  RISK SCORING
  {"High risk users":<30}: {high:,}  ({high/total*100:.1f}%)
  {"Medium risk users":<30}: {medium:,}  ({medium/total*100:.1f}%)
  {"Low risk users":<30}: {low:,}  ({low/total*100:.1f}%)
  {"Peak risk score":<30}: {risk_df['risk_score'].max():.1f} / 100

  ATTACK SIMULATION
  {"Personas injected":<30}: {total_sim}
  {"Detected as High Risk":<30}: {detected} / {total_sim}
  {"Detection rate":<30}: {detected/total_sim*100:.0f}%

  PERFORMANCE
  {"Total runtime":<30}: {runtime}
    """)

    print(SUB_SEPARATOR)
    print("  TOP 5 HIGHEST RISK USERS")
    print(SUB_SEPARATOR)
    top5 = risk_df.nlargest(5, "risk_score")[
        ["user", "risk_level", "risk_score", "risk_explanation"]
    ]
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        lvl_tag = f"[{row['risk_level'].upper()[:3]}]"
        print(
            f"  {i}. {row['user']:<15} "
            f"{lvl_tag:<7} "
            f"Score: {row['risk_score']:>6.1f}  |  "
            f"{row['risk_explanation'][:55]}..."
        )

    print(f"\n{SEPARATOR}")
    print("  SYSTEM STATUS: READY")
    print(f"  All outputs saved to /outputs folder.")
    print(f"  Launch dashboard: streamlit run dashboard/app.py")
    print(SEPARATOR)


# ─────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────

def main():

    print(f"\n{SEPARATOR}")
    print("  AI-DRIVEN INSIDER THREAT DETECTION SYSTEM")
    print("  Full Pipeline Runner  |  CERT Dataset  |  Stage 9")
    print(SEPARATOR)
    print(f"\n  Starting pipeline at: "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all stages in order
    check_environment()
    logon_events, device_events = run_preprocessing()
    feature_table               = run_feature_engineering(
                                    logon_events, device_events
                                  )
    scored_df                   = run_model_training(feature_table)
    risk_df                     = run_risk_scoring(scored_df)
    scored_personas, combined   = run_attack_simulation()
    validate_outputs()
    print_final_summary(risk_df, scored_personas)


if __name__ == "__main__":
    main()