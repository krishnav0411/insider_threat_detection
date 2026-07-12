# src/model.py
"""
File: model.py
Purpose: Stage 4 — Anomaly Detection Model. Trains an Isolation Forest with
         automatic hyperparameter selection, generates 0-100 risk scores,
         and saves model artifacts.
Inputs:  config.yaml — paths, hyperparameters, feature columns
         outputs/user_features.csv (or regenerates via run_feature_engineering)
Outputs: outputs/isolation_forest_model.pkl, outputs/scaler.pkl,
         outputs/model_metadata.json, outputs/user_scores.csv
Dependencies: sklearn, pandas, numpy, joblib, logging; src.config, src.features
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    FEATURE_COLS, FEATURE_PATH, SCORED_PATH, MODEL_SAVE_PATH,
    SCALER_SAVE_PATH, METADATA_PATH,
    CONTAMINATION_DEFAULT, CONTAMINATION_CANDIDATES, N_ESTIMATORS, RANDOM_STATE,
)
from src.features import run_feature_engineering

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────

def load_features(path: str = FEATURE_PATH) -> pd.DataFrame:
    """
    Load the feature table from CSV, or regenerate if the file is missing.

    Args:
        path: Path to the feature table CSV. Defaults to config path.

    Returns:
        DataFrame with 'user' column and 12 feature columns.

    Example:
        >>> df = load_features()
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info("  Loaded feature table: %s users | %s features",
                    f"{len(df):,}", len(df.columns) - 1)
        return df
    logger.warning("  Feature table not found — regenerating from raw data...")
    return run_feature_engineering()


# ─────────────────────────────────────────────
# SCALE FEATURES
# ─────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardise feature values using a fitted StandardScaler.

    Args:
        df: Feature table DataFrame (must contain FEATURE_COLS columns).

    Returns:
        Tuple of (X_scaled, scaler):
            X_scaled — numpy array of shape (n_users, 12).
            scaler   — fitted StandardScaler for transforming new data.

    Example:
        >>> Xs, sc = scale_features(feature_table)
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("  Features scaled using StandardScaler (%s features).", len(FEATURE_COLS))
    return X_scaled, scaler


# ─────────────────────────────────────────────
# HYPERPARAMETER SELECTION
# ─────────────────────────────────────────────

def select_contamination(
    X_scaled: np.ndarray,
    candidates: Optional[List[float]] = None
) -> Tuple[float, List[Tuple[float, int, float, float]], float]:
    """
    Evaluate multiple contamination values and select the one that maximises
    score separation between flagged and normal users.

    Separation = mean(decision_function[normals]) - mean(decision_function[anomalies]).

    Args:
        X_scaled:  Scaled feature matrix.
        candidates: Contamination ratios to test. Defaults to config values.

    Returns:
        Tuple of (best_contamination, results_list, best_separation).
            results_list — list of (cont, n_anom, pct, sep) tuples.

    Example:
        >>> best_cont, hp_res, best_sep = select_contamination(Xs)
    """
    if candidates is None:
        candidates = CONTAMINATION_CANDIDATES

    logger.info("")
    logger.info("  Hyperparameter selection — testing contamination values:")
    logger.info("  %14s  %10s  %7s  %12s", "Contamination", "Anomalies", "Flag%", "Separation")

    results: List[Tuple[float, int, float, float]] = []
    best_cont: float = candidates[0]
    best_sep: float = -np.inf

    for cont in candidates:
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=cont,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        raw = model.decision_function(X_scaled)
        preds = model.predict(X_scaled)

        anomaly_mask = preds == -1
        n_anom = anomaly_mask.sum()
        pct = n_anom / len(preds) * 100
        sep = float(raw[~anomaly_mask].mean() - raw[anomaly_mask].mean()) if n_anom > 0 and n_anom < len(preds) else 0.0

        results.append((cont, int(n_anom), float(pct), round(sep, 4)))
        logger.info("  %14.2f  %10s  %6.2f%%  %12.4f",
                    cont, f"{n_anom:,}", pct, sep)

        if sep > best_sep:
            best_sep = sep
            best_cont = cont

    logger.info("")
    logger.info("  Selected contamination: %s (best separation = %.4f)", best_cont, best_sep)
    return best_cont, results, best_sep


# ─────────────────────────────────────────────
# TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────

def train_isolation_forest(
    X_scaled: np.ndarray,
    contamination: Optional[float] = None,
) -> IsolationForest:
    """
    Train an Isolation Forest model on the scaled feature matrix.

    Args:
        X_scaled:      Scaled feature matrix (n_samples, n_features).
        contamination: Expected proportion of anomalies. Defaults to config value.

    Returns:
        Trained IsolationForest instance.

    Example:
        >>> model = train_isolation_forest(Xs, contamination=0.03)
    """
    if contamination is None:
        contamination = CONTAMINATION_DEFAULT

    logger.info("")
    logger.info("  Training Isolation Forest...")
    logger.info("     Contamination : %s (%s%% of users flagged)", contamination, int(contamination * 100))
    logger.info("     Trees         : %s", N_ESTIMATORS)
    logger.info("     Random seed   : %s", RANDOM_STATE)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    logger.info("  Model trained successfully.")
    return model


# ─────────────────────────────────────────────
# GENERATE ANOMALY SCORES
# ─────────────────────────────────────────────

def generate_scores(
    model: IsolationForest,
    X_scaled: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score every user using the trained Isolation Forest and convert raw
    anomaly scores into 0-100 risk scores.

    Args:
        model:    Trained IsolationForest.
        X_scaled: Scaled feature matrix for all users.
        df:       Original feature table (for retaining user metadata).

    Returns:
        DataFrame with added columns: raw_anomaly_score, risk_score, is_anomaly.

    Example:
        >>> scored = generate_scores(model, Xs, feature_table)
    """
    logger.info("")
    logger.info("  Generating anomaly scores...")

    raw_scores = model.decision_function(X_scaled)
    inverted = -1 * raw_scores

    score_min = inverted.min()
    score_max = inverted.max()

    if score_max > score_min:
        risk_score = ((inverted - score_min) / (score_max - score_min)) * 100
    else:
        risk_score = np.zeros(len(inverted))

    risk_score = np.round(risk_score, 2)
    predictions = model.predict(X_scaled)

    scored_df = df.copy()
    scored_df["raw_anomaly_score"] = np.round(raw_scores, 6)
    scored_df["risk_score"] = risk_score
    scored_df["is_anomaly"] = (predictions == -1).astype(int)

    anomaly_count = scored_df["is_anomaly"].sum()
    logger.info("  Scores generated for %s users.", f"{len(scored_df):,}")
    logger.info("  Anomalies detected: %s users (%.1f%% of total)",
                f"{anomaly_count:,}", anomaly_count / len(scored_df) * 100)
    return scored_df


# ─────────────────────────────────────────────
# SAVE MODEL METADATA
# ─────────────────────────────────────────────

def save_model_metadata(
    model: IsolationForest,
    scaler: StandardScaler,
    n_users: int,
    separation: float,
    contamination_used: float,
) -> None:
    """
    Save training metadata as a JSON file alongside the model artifacts.

    Args:
        model:              Trained IsolationForest.
        scaler:             Fitted StandardScaler.
        n_users:            Number of users in the training set.
        separation:         Achieved score separation.
        contamination_used: Contamination ratio selected.

    Example:
        >>> save_model_metadata(model, scaler, 1000, 0.2878, 0.03)
    """
    metadata: Dict[str, Any] = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "IsolationForest",
        "n_estimators": N_ESTIMATORS,
        "contamination": contamination_used,
        "random_state": RANDOM_STATE,
        "n_users_trained": n_users,
        "n_features": len(FEATURE_COLS),
        "features_used": FEATURE_COLS,
        "score_separation": round(separation, 4),
        "model_file": os.path.basename(MODEL_SAVE_PATH),
        "scaler_file": os.path.basename(SCALER_SAVE_PATH),
    }

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Model metadata saved -> %s", METADATA_PATH)


# ─────────────────────────────────────────────
# SAVE MODEL AND SCALER
# ─────────────────────────────────────────────

def save_model_artifacts(model: IsolationForest, scaler: StandardScaler) -> None:
    """
    Serialise the trained model and scaler to disk using joblib.

    Args:
        model:  Trained IsolationForest.
        scaler: Fitted StandardScaler.

    Example:
        >>> save_model_artifacts(model, scaler)
    """
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info("")
    logger.info("  Model  saved -> %s", MODEL_SAVE_PATH)
    logger.info("  Scaler saved -> %s", SCALER_SAVE_PATH)


# ─────────────────────────────────────────────
# SAVE SCORED RESULTS
# ─────────────────────────────────────────────

def save_scored_results(scored_df: pd.DataFrame, path: str = SCORED_PATH) -> None:
    """
    Save the scored user DataFrame to CSV.

    Args:
        scored_df: DataFrame with risk_score and is_anomaly columns.
        path:      Destination path. Defaults to config path.

    Example:
        >>> save_scored_results(scored_df)
    """
    scored_df.to_csv(path, index=False)
    logger.info("  Scored results saved -> %s", path)


# ─────────────────────────────────────────────
# SCORE SUMMARY REPORT
# ─────────────────────────────────────────────

def print_score_summary(scored_df: pd.DataFrame) -> None:
    """
    Print a summary of model scoring results: counts, distribution, top users.

    Args:
        scored_df: DataFrame output from generate_scores().

    Example:
        >>> print_score_summary(scored_df)
    """
    _s = "=" * 60
    logger.info("")
    logger.info(_s)
    logger.info("  ANOMALY SCORE SUMMARY")
    logger.info(_s)

    total = len(scored_df)
    anomalies = scored_df["is_anomaly"].sum()
    normal = total - anomalies

    logger.info("")
    logger.info("  Total users scored : %s", f"{total:,}")
    logger.info("  Normal users       : %s  (%.1f%%)", f"{normal:,}", normal / total * 100)
    logger.info("  Anomalous users    : %s  (%.1f%%)", f"{anomalies:,}", anomalies / total * 100)

    logger.info("")
    logger.info("  Risk Score Distribution:")
    logger.info("     Min    : %.2f", scored_df['risk_score'].min())
    logger.info("     Mean   : %.2f", scored_df['risk_score'].mean())
    logger.info("     Median : %.2f", scored_df['risk_score'].median())
    logger.info("     Max    : %.2f", scored_df['risk_score'].max())

    logger.info("")
    logger.info(_s)
    logger.info("  TOP 10 HIGHEST RISK USERS")
    logger.info(_s)
    top10 = scored_df.nlargest(10, "risk_score")[[
        "user", "login_count", "off_hour_logins",
        "unique_pcs_logon", "device_connections", "risk_score", "is_anomaly",
    ]]
    print(top10.to_string(index=False))

    logger.info("")
    logger.info(_s)
    logger.info("  BOTTOM 5 LOWEST RISK USERS (most normal)")
    logger.info(_s)
    bottom5 = scored_df.nsmallest(5, "risk_score")[[
        "user", "login_count", "off_hour_logins", "risk_score",
    ]]
    print(bottom5.to_string(index=False))


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_model_pipeline() -> pd.DataFrame:
    """
    Execute the full model pipeline: load, scale, select contamination, train, score, save.

    Returns:
        DataFrame with risk_score and is_anomaly for every user.

    Example:
        >>> scored_df = run_model_pipeline()
    """
    _s = "=" * 60
    logger.info("")
    logger.info(_s)
    logger.info("  STAGE 4 — Isolation Forest Model Pipeline")
    logger.info(_s)

    logger.info("")
    logger.info("  [1/6] Loading feature table...")
    df = load_features()

    logger.info("")
    logger.info("  [2/6] Scaling features...")
    X_scaled, scaler = scale_features(df)

    logger.info("")
    logger.info("  [3/6] Selecting best contamination value...")
    best_cont, _, best_sep = select_contamination(X_scaled)

    logger.info("")
    logger.info("  [4/6] Training Isolation Forest with selected contamination...")
    model = train_isolation_forest(X_scaled, contamination=best_cont)

    logger.info("")
    logger.info("  [5/6] Scoring all users...")
    scored_df = generate_scores(model, X_scaled, df)

    logger.info("")
    logger.info("  [6/6] Saving model, scaler, metadata and results...")
    save_model_artifacts(model, scaler)
    save_model_metadata(model, scaler, len(df), best_sep, best_cont)
    save_scored_results(scored_df)

    return scored_df


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time
    logging.getLogger().setLevel(logging.INFO)
    t0 = time.time()

    scored_df = run_model_pipeline()
    elapsed = time.time() - t0
    print_score_summary(scored_df)

    logger.info("")
    logger.info("  Pipeline completed in %.1fs", elapsed)
    logger.info("=" * 60)
    logger.info("  Stage 4 complete.")
    logger.info("=" * 60)
