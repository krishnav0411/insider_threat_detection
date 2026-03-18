# src/attack_simulation.py
# Stage 7: Attack Simulation Module
# Injects synthetic insider threat personas and validates detection

import pandas as pd
import numpy as np
import os
import sys
import joblib

from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.risk_scoring import assign_risk_level, generate_explanation

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

FEATURE_PATH      = "outputs/user_features.csv"
MODEL_PATH        = "outputs/isolation_forest_model.pkl"
SCALER_PATH       = "outputs/scaler.pkl"
SIM_OUTPUT_PATH   = "outputs/simulated_results.csv"

FEATURE_COLS = [
    "login_count",
    "off_hour_logins",
    "weekend_logins",
    "unique_pcs_logon",
    "off_hour_ratio",
    "weekend_ratio",
    "device_connections",
    "unique_pcs_device"
]

SEPARATOR = "=" * 60


# ─────────────────────────────────────────────
# ATTACK PERSONA DEFINITIONS
# ─────────────────────────────────────────────

def build_attack_personas() -> pd.DataFrame:
    """
    Define synthetic insider threat personas.
    Each persona represents a realistic attack pattern.

    Feature values are deliberately extreme compared to
    the normal user population to test detection sensitivity.

    Returns:
        DataFrame of simulated users with feature values
    """
    personas = [

        # ── Persona 1: Night Owl
        # Logs in frequently but almost always after midnight
        # Suggests unauthorized access during off-hours
        {
            "user"               : "SIM_NightOwl",
            "login_count"        : 95,
            "off_hour_logins"    : 88,    # 93% of logins are off-hours
            "weekend_logins"     : 12,
            "unique_pcs_logon"   : 2,
            "off_hour_ratio"     : 0.93,
            "weekend_ratio"      : 0.13,
            "device_connections" : 8,
            "unique_pcs_device"  : 1,
            "persona_description": (
                "Logs in almost exclusively between midnight and 5AM. "
                "Suggests unauthorized remote access or credential theft."
            )
        },

        # ── Persona 2: PC Hopper
        # Logs into many different machines across the organization
        # Suggests lateral movement — trying to access multiple systems
        {
            "user"               : "SIM_PCHopper",
            "login_count"        : 120,
            "off_hour_logins"    : 18,
            "weekend_logins"     : 14,
            "unique_pcs_logon"   : 14,   # Uses 14 different machines
            "off_hour_ratio"     : 0.15,
            "weekend_ratio"      : 0.12,
            "device_connections" : 12,
            "unique_pcs_device"  : 11,   # Plugs devices into 11 machines
            "persona_description": (
                "Accesses 14 unique PCs across the organization. "
                "Classic lateral movement pattern — searching for data or escalating privileges."
            )
        },

        # ── Persona 3: Data Mule
        # Connects USB/external devices to many machines
        # Suggests bulk data exfiltration via removable media
        {
            "user"               : "SIM_DataMule",
            "login_count"        : 55,
            "off_hour_logins"    : 22,
            "weekend_logins"     : 19,
            "unique_pcs_logon"   : 4,
            "off_hour_ratio"     : 0.40,
            "weekend_ratio"      : 0.35,
            "device_connections" : 112,  # Extreme device usage
            "unique_pcs_device"  : 9,
            "persona_description": (
                "Connects external devices 112 times across 9 machines. "
                "High weekend and off-hours activity. "
                "Strong indicator of bulk data exfiltration via removable media."
            )
        },

        # ── Persona 4: Ghost User
        # Rarely logs in but every login is suspicious
        # Low volume makes them harder to detect — tests model sensitivity
        {
            "user"               : "SIM_GhostUser",
            "login_count"        : 18,   # Very few logins
            "off_hour_logins"    : 16,   # But nearly all are off-hours
            "weekend_logins"     : 10,
            "unique_pcs_logon"   : 5,
            "off_hour_ratio"     : 0.89,
            "weekend_ratio"      : 0.56,
            "device_connections" : 4,
            "unique_pcs_device"  : 4,
            "persona_description": (
                "Minimal login activity but 89% of logins are off-hours. "
                "Accesses 5 machines despite low total usage. "
                "Stealth behavior — may be using stolen credentials sparingly."
            )
        },

        # ── Persona 5: Full Threat
        # All suspicious behaviors combined at maximum intensity
        # Should always score near 100 — the ultimate validation test
        {
            "user"               : "SIM_FullThreat",
            "login_count"        : 210,
            "off_hour_logins"    : 175,
            "weekend_logins"     : 68,
            "unique_pcs_logon"   : 18,
            "off_hour_ratio"     : 0.83,
            "weekend_ratio"      : 0.32,
            "device_connections" : 145,
            "unique_pcs_device"  : 15,
            "persona_description": (
                "Maximum threat profile — combines all suspicious behaviors. "
                "Extremely high logins, off-hours activity, lateral movement, "
                "and device exfiltration. Should always be flagged High risk."
            )
        },
    ]

    df = pd.DataFrame(personas)
    print(f"  [OK] Built {len(df)} attack personas.")
    return df


# ─────────────────────────────────────────────
# LOAD MODEL AND SCALER
# ─────────────────────────────────────────────

def load_model_artifacts() -> tuple:
    """
    Load the trained Isolation Forest model and StandardScaler
    saved during Stage 5. These must exist before running simulation.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'\n"
            f"-> Run python src/model.py first to train and save the model."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'\n"
            f"-> Run python src/model.py first."
        )

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("  [OK] Model and scaler loaded successfully.")
    return model, scaler


# ─────────────────────────────────────────────
# SCORE SIMULATED PERSONAS
# ─────────────────────────────────────────────

def score_personas(
    personas_df : pd.DataFrame,
    model,
    scaler      : StandardScaler
) -> pd.DataFrame:
    """
    Run simulated personas through the trained model.

    Steps:
    1. Extract feature matrix from persona definitions
    2. Scale using the SAME scaler fitted on real data (critical!)
    3. Get anomaly scores and predictions from model
    4. Convert to 0-100 risk scores
    5. Assign risk levels and explanations

    Returns:
        scored_personas : DataFrame with full risk scoring
    """
    X = personas_df[FEATURE_COLS].values

    # Use the SAME scaler from training — do NOT refit
    X_scaled = scaler.transform(X)

    # Get raw anomaly scores
    raw_scores  = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    # Convert to 0-100 risk score
    # Load real data scores to use same normalization range
    if os.path.exists("outputs/user_scores.csv"):
        real_scores = pd.read_csv("outputs/user_scores.csv")
        real_raw    = model.decision_function(
            scaler.transform(real_scores[FEATURE_COLS].values)
        )
        all_raw     = np.concatenate([real_raw, raw_scores])
        score_min   = (-1 * all_raw).min()
        score_max   = (-1 * all_raw).max()
    else:
        score_min = (-1 * raw_scores).min()
        score_max = (-1 * raw_scores).max()

    inverted   = -1 * raw_scores
    risk_scores = np.where(
        score_max > score_min,
        ((inverted - score_min) / (score_max - score_min)) * 100,
        50.0
    ).round(2)

    # Build result DataFrame
    result = personas_df.copy()
    result["raw_anomaly_score"] = np.round(raw_scores, 6)
    result["risk_score"]        = risk_scores
    result["is_anomaly"]        = (predictions == -1).astype(int)
    result["risk_level"]        = result["risk_score"].apply(assign_risk_level)
    result["risk_explanation"]  = result.apply(generate_explanation, axis=1)

    return result


# ─────────────────────────────────────────────
# COMBINE WITH REAL DATA
# ─────────────────────────────────────────────

def combine_with_real_data(
    scored_personas : pd.DataFrame
) -> pd.DataFrame:
    """
    Merge simulated personas into the real user feature table.
    Adds a 'is_simulated' flag column to distinguish them.
    Used by the dashboard to show both real and simulated users.
    """
    real_df = pd.read_csv("outputs/user_scores.csv")
    real_df["risk_level"]       = real_df["risk_score"].apply(assign_risk_level)
    real_df["risk_explanation"] = real_df.apply(generate_explanation, axis=1)
    real_df["is_simulated"]     = 0
    real_df["persona_description"] = ""

    sim_df = scored_personas.copy()
    sim_df["is_simulated"] = 1

    combined = pd.concat([real_df, sim_df], ignore_index=True)
    combined = combined.sort_values("risk_score", ascending=False)

    print(f"  [OK] Combined dataset: {len(combined):,} total users "
          f"({len(real_df):,} real + {len(sim_df)} simulated)")
    return combined


# ─────────────────────────────────────────────
# PRINT SIMULATION VALIDATION REPORT
# ─────────────────────────────────────────────

def print_simulation_report(scored_personas: pd.DataFrame):
    """
    Print a detailed validation report for each simulated persona.
    Checks whether the model correctly detected each threat.
    """
    print(f"\n{SEPARATOR}")
    print("  ATTACK SIMULATION VALIDATION REPORT")
    print(SEPARATOR)

    all_passed = True

    for _, row in scored_personas.iterrows():
        detected    = row["risk_level"] == "High"
        status      = "[DETECTED]" if detected else "[MISSED]"
        all_passed  = all_passed and detected

        print(f"\n  Persona       : {row['user']}")
        print(f"  Status        : {status}")
        print(f"  Risk Score    : {row['risk_score']:.1f} / 100")
        print(f"  Risk Level    : {row['risk_level']}")
        print(f"  Is Anomaly    : {'Yes' if row['is_anomaly'] else 'No'}")
        print(f"  Description   : {row['persona_description']}")
        print(f"  Flags         : {row['risk_explanation']}")
        print(f"  {'-' * 55}")

    # Overall result
    print(f"\n{SEPARATOR}")
    print("  OVERALL VALIDATION RESULT")
    print(SEPARATOR)

    passed = scored_personas[scored_personas["risk_level"] == "High"].shape[0]
    total  = len(scored_personas)
    pct    = passed / total * 100

    print(f"\n  Personas detected as High Risk : {passed} / {total}  ({pct:.0f}%)")

    if all_passed:
        print("  [PASS] All attack personas correctly detected.")
        print("         The system is working as expected.")
    else:
        missed = scored_personas[scored_personas["risk_level"] != "High"]["user"].tolist()
        print(f"  [WARN] Some personas were not detected as High Risk: {missed}")
        print("         Consider adjusting THRESHOLD_HIGH or CONTAMINATION.")


# ─────────────────────────────────────────────
# SAVE SIMULATION RESULTS
# ─────────────────────────────────────────────

def save_simulation_results(combined_df: pd.DataFrame):
    """
    Save combined real + simulated results to CSV.
    """
    os.makedirs("outputs", exist_ok=True)
    combined_df.to_csv(SIM_OUTPUT_PATH, index=False)
    print(f"\n  [SAVE] Simulation results saved -> {SIM_OUTPUT_PATH}")


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_attack_simulation() -> tuple:
    """
    Master function for the full attack simulation pipeline.
    Called by the dashboard.

    Returns:
        scored_personas : DataFrame of scored simulated users only
        combined_df     : DataFrame of real + simulated users combined
    """
    print(f"\n{SEPARATOR}")
    print("  STAGE 7 - Attack Simulation Pipeline")
    print(SEPARATOR)

    # Step 1 - Build personas
    print("\n  [1/5] Building attack personas...")
    personas_df = build_attack_personas()

    # Step 2 - Load model
    print("\n  [2/5] Loading trained model and scaler...")
    model, scaler = load_model_artifacts()

    # Step 3 - Score personas
    print("\n  [3/5] Scoring simulated personas...")
    scored_personas = score_personas(personas_df, model, scaler)

    # Step 4 - Combine with real data
    print("\n  [4/5] Combining with real user data...")
    combined_df = combine_with_real_data(scored_personas)

    # Step 5 - Save results
    print("\n  [5/5] Saving simulation results...")
    save_simulation_results(combined_df)

    return scored_personas, combined_df


# ─────────────────────────────────────────────
# SELF-TEST  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    scored_personas, combined_df = run_attack_simulation()

    print_simulation_report(scored_personas)

    print(f"\n{SEPARATOR}")
    print("  TOP 10 USERS IN COMBINED DATASET")
    print(SEPARATOR)

    top10 = combined_df.nlargest(10, "risk_score")[[
        "user", "risk_score", "risk_level",
        "is_anomaly", "is_simulated"
    ]]
    print(top10.to_string(index=False))

    print(f"\n{SEPARATOR}")
    print("  [OK] Stage 7 complete - ready for dashboard in Stage 8")
    print(SEPARATOR)