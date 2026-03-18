# src/model.py
# Stage 5: Anomaly Detection Model
# Trains Isolation Forest and generates risk scores per user

import pandas as pd
import numpy as np
import os
import sys
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import run_feature_engineering

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

FEATURE_PATH    = "outputs/user_features.csv"
SCORED_PATH     = "outputs/user_scores.csv"
MODEL_SAVE_PATH = "outputs/isolation_forest_model.pkl"
SCALER_SAVE_PATH= "outputs/scaler.pkl"

# Isolation Forest parameters
CONTAMINATION   = 0.05    # Expect ~5% of users to be anomalous
N_ESTIMATORS    = 100     # Number of trees
RANDOM_STATE    = 42      # For reproducibility

# Feature columns used for training (exclude 'user' — it's just an ID)
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


# ─────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────

def load_features(path: str = FEATURE_PATH) -> pd.DataFrame:
    """
    Load the user feature table from CSV.
    Falls back to regenerating it if not found.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ Loaded feature table: {len(df):,} users | {len(df.columns)-1} features")
        return df
    else:
        print("  ⚠️  Feature table not found — regenerating from raw data...")
        return run_feature_engineering()


# ─────────────────────────────────────────────
# SCALE FEATURES
# ─────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple:
    """
    Standardize feature values using StandardScaler.

    Why scale?
    - login_count might range 1–200
    - off_hour_ratio ranges 0–1
    Without scaling, large-range features dominate the model unfairly.

    Returns:
        X_scaled : numpy array of scaled features
        scaler   : fitted scaler (saved for use on new/simulated data)
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  ✅ Features scaled using StandardScaler.")
    return X_scaled, scaler


# ─────────────────────────────────────────────
# TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────

def train_isolation_forest(X_scaled: np.ndarray) -> IsolationForest:
    """
    Train Isolation Forest on the scaled feature matrix.

    Parameters:
    - contamination : expected proportion of anomalies (5%)
    - n_estimators  : number of isolation trees (100 is standard)
    - random_state  : fixed seed for reproducibility

    Returns:
        Trained IsolationForest model
    """
    print(f"\n  Training Isolation Forest...")
    print(f"     Contamination : {CONTAMINATION} ({int(CONTAMINATION*100)}% of users flagged)")
    print(f"     Trees         : {N_ESTIMATORS}")
    print(f"     Random seed   : {RANDOM_STATE}")

    model = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        random_state  = RANDOM_STATE,
        n_jobs        = -1    # Use all CPU cores for speed
    )

    model.fit(X_scaled)
    print(f"  ✅ Model trained successfully.")
    return model


# ─────────────────────────────────────────────
# GENERATE ANOMALY SCORES
# ─────────────────────────────────────────────

def generate_scores(
    model: IsolationForest,
    X_scaled: np.ndarray,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Use the trained model to score every user.

    Isolation Forest returns:
    - decision_function() → raw anomaly score (higher = more normal)
    - predict()           → +1 (normal) or -1 (anomaly)

    We convert to a 0–100 risk score where:
    - 100 = most anomalous (highest threat)
    - 0   = most normal    (lowest threat)

    Steps:
    1. Get raw scores from decision_function
    2. Invert so that anomalies have HIGH values
    3. Normalize to 0–100 range using min-max scaling
    """
    print("\n  Generating anomaly scores...")

    # Raw scores: more negative = more anomalous
    raw_scores = model.decision_function(X_scaled)

    # Invert: multiply by -1 so anomalies become high positive values
    inverted = -1 * raw_scores

    # Normalize to 0–100
    score_min = inverted.min()
    score_max = inverted.max()

    if score_max > score_min:
        risk_score = ((inverted - score_min) / (score_max - score_min)) * 100
    else:
        risk_score = np.zeros(len(inverted))

    risk_score = np.round(risk_score, 2)

    # Model prediction: -1 = anomaly, +1 = normal
    predictions = model.predict(X_scaled)

    # Build scored DataFrame
    scored_df = df.copy()
    scored_df["raw_anomaly_score"] = np.round(raw_scores, 6)
    scored_df["risk_score"]        = risk_score
    scored_df["is_anomaly"]        = (predictions == -1).astype(int)

    anomaly_count = scored_df["is_anomaly"].sum()
    print(f"  ✅ Scores generated for {len(scored_df):,} users.")
    print(f"  🚨 Anomalies detected : {anomaly_count} users "
          f"({anomaly_count/len(scored_df)*100:.1f}% of total)")

    return scored_df


# ─────────────────────────────────────────────
# SAVE MODEL AND SCALER
# ─────────────────────────────────────────────

def save_model_artifacts(model: IsolationForest, scaler: StandardScaler):
    """
    Save trained model and scaler to disk using joblib.
    These will be loaded by the dashboard and attack simulation module.
    """
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model,  MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\n  💾 Model  saved → {MODEL_SAVE_PATH}")
    print(f"  💾 Scaler saved → {SCALER_SAVE_PATH}")


# ─────────────────────────────────────────────
# SAVE SCORED RESULTS
# ─────────────────────────────────────────────

def save_scored_results(scored_df: pd.DataFrame, path: str = SCORED_PATH):
    """
    Save the fully scored user table to CSV.
    """
    scored_df.to_csv(path, index=False)
    print(f"  💾 Scored results saved → {path}")


# ─────────────────────────────────────────────
# SCORE SUMMARY REPORT
# ─────────────────────────────────────────────

def print_score_summary(scored_df: pd.DataFrame):
    """
    Print a clean summary of the scoring results.
    """
    SEPARATOR = "=" * 60

    print(f"\n{SEPARATOR}")
    print("  ANOMALY SCORE SUMMARY")
    print(SEPARATOR)

    total      = len(scored_df)
    anomalies  = scored_df["is_anomaly"].sum()
    normal     = total - anomalies

    print(f"\n  Total users scored : {total:,}")
    print(f"  Normal users       : {normal:,}  ({normal/total*100:.1f}%)")
    print(f"  Anomalous users    : {anomalies:,}  ({anomalies/total*100:.1f}%)")

    print(f"\n  Risk Score Distribution:")
    print(f"     Min    : {scored_df['risk_score'].min():.2f}")
    print(f"     Mean   : {scored_df['risk_score'].mean():.2f}")
    print(f"     Median : {scored_df['risk_score'].median():.2f}")
    print(f"     Max    : {scored_df['risk_score'].max():.2f}")

    print(f"\n{SEPARATOR}")
    print("  TOP 10 HIGHEST RISK USERS")
    print(SEPARATOR)

    top10 = scored_df.nlargest(10, "risk_score")[[
        "user", "login_count", "off_hour_logins",
        "unique_pcs_logon", "device_connections", "risk_score", "is_anomaly"
    ]]

    print(top10.to_string(index=False))

    print(f"\n{SEPARATOR}")
    print("  BOTTOM 5 LOWEST RISK USERS (most normal)")
    print(SEPARATOR)

    bottom5 = scored_df.nsmallest(5, "risk_score")[[
        "user", "login_count", "off_hour_logins", "risk_score"
    ]]
    print(bottom5.to_string(index=False))


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_model_pipeline() -> pd.DataFrame:
    """
    Master function that runs the full modelling pipeline.
    Called by Stage 6 (risk scoring) and the dashboard.

    Returns:
        scored_df : DataFrame with risk_score and is_anomaly per user
    """
    SEPARATOR = "=" * 60

    print(f"\n{SEPARATOR}")
    print("  STAGE 5 — Isolation Forest Model Pipeline")
    print(SEPARATOR)

    # Step 1 — Load features
    print("\n  [1/5] Loading feature table...")
    df = load_features()

    # Step 2 — Scale features
    print("\n  [2/5] Scaling features...")
    X_scaled, scaler = scale_features(df)

    # Step 3 — Train model
    print("\n  [3/5] Training Isolation Forest...")
    model = train_isolation_forest(X_scaled)

    # Step 4 — Score users
    print("\n  [4/5] Scoring all users...")
    scored_df = generate_scores(model, X_scaled, df)

    # Step 5 — Save everything
    print("\n  [5/5] Saving model, scaler and results...")
    save_model_artifacts(model, scaler)
    save_scored_results(scored_df)

    return scored_df


# ─────────────────────────────────────────────
# SELF-TEST  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    scored_df = run_model_pipeline()
    print_score_summary(scored_df)

    print("\n" + "=" * 60)
    print("  ✅ Stage 5 complete — anomaly scores ready for Stage 6")
    print("=" * 60)