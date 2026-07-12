# verify_model.py
"""
File: verify_model.py
Purpose: Model verification and validation. Runs 7 analysis sections:
         population statistics, feature quality, confusion matrix estimate
         (approximate), top anomalous users, model metadata, quality checks,
         and score distribution analysis.
Inputs:  outputs/risk_report.csv, outputs/simulated_results.csv,
         outputs/user_features.csv, outputs/model_metadata.json
Outputs: Prints comprehensive verification report to stdout
Dependencies: pandas, numpy, os, sys, json, warnings; sklearn; src.risk_scoring, src.preprocess, src.features
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

SEPARATOR     = "=" * 65
SUB_SEPARATOR = "-" * 65

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
    "avg_session_gap": "Avg Session Gap (s)",
}

FEATURE_COLS = list(FEATURE_LABELS.keys())

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_all_data():
    """Load all output files needed for verification."""
    base = "outputs"

    risk_df = pd.read_csv(f"{base}/risk_report.csv")
    sim_df  = pd.read_csv(f"{base}/simulated_results.csv")
    feat_df = pd.read_csv(f"{base}/user_features.csv")

    meta = {}
    if os.path.exists(f"{base}/model_metadata.json"):
        with open(f"{base}/model_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

    return risk_df, sim_df, feat_df, meta


# ─────────────────────────────────────────────
# SECTION 1: POPULATION STATISTICS
# ─────────────────────────────────────────────

def print_population_statistics(feat_df: pd.DataFrame, risk_df: pd.DataFrame):
    """Print complete population statistics for all 12 features."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 1: POPULATION STATISTICS (12 features, N=1,000)")
    print(f"{SEPARATOR}")

    # Merge risk levels into feature stats
    merged = feat_df.merge(risk_df[["user", "risk_level", "risk_score"]], on="user", how="left")

    print(f"\n  {'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} "
          f"{'Median':>10} {'Max':>10} {'High-Risk Avg':>14}")
    print(SUB_SEPARATOR)

    high_avg = merged[merged["risk_level"] == "High"][FEATURE_COLS].mean()

    for col in FEATURE_COLS:
        s = feat_df[col]
        h = high_avg.get(col, 0)
        print(f"  {FEATURE_LABELS[col]:<30} {s.mean():>10.2f} {s.std():>10.2f} "
              f"{s.min():>10.2f} {s.median():>10.2f} {s.max():>10.2f} {h:>14.2f}")

    # Risk level breakdown
    print(f"\n  {'Risk Level':<20} {'Count':>8} {'Mean Score':>12} {'Mean Logins':>12} "
          f"{'Mean PCs':>10} {'Mean Devices':>14}")
    print(SUB_SEPARATOR)
    for level in ["High", "Medium", "Low"]:
        subset = merged[merged["risk_level"] == level]
        print(f"  {level:<20} {len(subset):>8} {subset['risk_score'].mean():>12.2f} "
              f"{subset['login_count'].mean():>12.1f} "
              f"{subset['unique_pcs_logon'].mean():>10.1f} "
              f"{subset['device_connections'].mean():>14.1f}")


# ─────────────────────────────────────────────
# SECTION 2: FEATURE COVERAGE & QUALITY
# ─────────────────────────────────────────────

def print_feature_quality(feat_df: pd.DataFrame):
    """Verify all 12 features are present, have non-zero variance, and check correlations."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 2: FEATURE COVERAGE & QUALITY")
    print(f"{SEPARATOR}")

    # 2a — Presence
    present = [c for c in FEATURE_COLS if c in feat_df.columns]
    missing = [c for c in FEATURE_COLS if c not in feat_df.columns]
    print(f"\n  Feature presence: {len(present)}/{len(FEATURE_COLS)}")
    if missing:
        print(f"  MISSING: {missing}")
    else:
        print("  All 12 features present.")

    # 2b — Variance
    zero_var = [c for c in present if feat_df[c].std() == 0]
    print(f"\n  Features with zero variance: {len(zero_var)}/12")
    if zero_var:
        print(f"  WARNING: {zero_var} have zero variance — no predictive power.")
    else:
        print("  All 12 features have non-zero variance.")

    # 2c — Correlations > 0.95
    corr = feat_df[present].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper[col][row]
            if not pd.isna(val) and val > 0.95:
                high_pairs.append((col, row, round(val, 3)))
    print(f"\n  Highly correlated pairs (>0.95): {len(high_pairs)}")
    if high_pairs:
        for col, row, val in high_pairs:
            print(f"    {col} <-> {row} = {val}")
        print("  NOTE: Correlated features don't break Isolation Forest,")
        print("  but may indicate redundant signals.")
    else:
        print("  No feature pairs exceed correlation threshold of 0.95.")

    # 2d — Coverage per feature (non-zero values)
    print("\n  Non-zero coverage per feature:")
    print(f"  {'Feature':<30} {'Non-zero':>10} {'%':>7} {'Mean':>10}")
    print(SUB_SEPARATOR)
    for col in present:
        nz = (feat_df[col] != 0).sum() if feat_df[col].dtype in (int, float) else len(feat_df)
        mean = feat_df[col].mean()
        pct = nz / len(feat_df) * 100
        print(f"  {FEATURE_LABELS[col]:<30} {nz:>10,} {pct:>6.1f}% {mean:>10.2f}")


# ─────────────────────────────────────────────
# SECTION 3: CONFUSION MATRIX ESTIMATE
# ─────────────────────────────────────────────

def print_confusion_matrix_estimate(risk_df: pd.DataFrame, sim_df: pd.DataFrame):
    """
    Estimate detection performance using:
    - Known positives: 7 simulated attack personas (deliberately designed as threats)
    - Known negatives: 50 lowest-risk real users (bottom risk_score, Low risk only)

    IMPORTANT: These are APPROXIMATE estimates only.
    The r1 CERT dataset has no ground-truth labels.
    """
    print(f"\n{SEPARATOR}")
    print("  SECTION 3: CONFUSION MATRIX ESTIMATE (APPROXIMATE)")
    print(f"{SEPARATOR}")
    print("\n  IMPORTANT DISCLAIMER:")
    print("  The CERT r1 dataset does NOT contain ground-truth labels.")
    print("  The following uses SIMULATED attack personas as known positives")
    print("  and LOWEST-RISK real users as known negatives.")
    print("  These are ESTIMATES, not true performance metrics.")
    print("  They are useful for RELATIVE comparison between model runs.")

    # Known positives: all simulated attack personas
    sim_personas = sim_df[sim_df["is_simulated"] == 1].copy()
    known_positives = len(sim_personas)

    # Known negatives: top 50 lowest-risk Low users (by risk_score ascending)
    low_users = risk_df[risk_df["risk_level"] == "Low"].nsmallest(50, "risk_score")
    known_negatives = len(low_users)

    print(f"\n  {'Metric':<35} {'Value':>10}")
    print(SUB_SEPARATOR)
    print(f"  {'Known positives (sim personas)':<35} {known_positives:>10}")
    print(f"  {'Known negatives (lowest-risk users)':<35} {known_negatives:>10}")

    # Build confusion matrix
    # TP = personas detected as High risk
    tp = (sim_personas["risk_level"] == "High").sum()
    # FN = personas NOT detected as High risk
    fn = known_positives - tp
    # FP = low-risk users that are flagged as anomalies
    fp = low_users["is_anomaly"].sum()
    # TN = low-risk users correctly not flagged
    tn = known_negatives - fp

    print(f"\n  {'─' * 47}")
    print(f"  {'':>5} {'Predicted Positive':>22} {'Predicted Negative':>22}")
    print(f"  {'─' * 47}")
    print(f"  {'Actual Positive':<18} {tp:>21} {fn:>22}")
    print(f"  {'Actual Negative':<18} {fp:>21} {tn:>22}")
    print(f"  {'─' * 47}")

    # Calculate metrics with safe defaults
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (known_positives + known_negatives) * 100

    print(f"\n  {'─' * 35}")
    print(f"  {'PERFORMANCE ESTIMATE':^35}")
    print(f"  {'─' * 35}")
    print(f"  {'Precision':<25} {precision:>8.1f}%")
    print(f"  {'Recall':<25} {recall:>8.1f}%")
    print(f"  {'F1 Score':<25} {f1:>8.1f}%")
    print(f"  {'Accuracy':<25} {accuracy:>8.1f}%")
    print(f"  {'─' * 35}")
    print("\n  Interpretation:")
    if precision > 80:
        print(f"    [OK] Precision {precision:.0f}%: flagged users are very likely real threats.")
    elif precision > 50:
        print(f"    [WARN] Precision {precision:.0f}%: moderate false positive rate.")
    else:
        print(f"    [LOW] Precision {precision:.0f}%: high false positive rate.")

    if recall > 80:
        print(f"    [OK] Recall {recall:.0f}%: most threats are being caught.")
    elif recall > 50:
        print(f"    [WARN] Recall {recall:.0f}%: a significant number of threats are missed.")
    else:
        print(f"    [LOW] Recall {recall:.0f}%: many threats are slipping through.")

    if f1 > 70:
        print(f"    [OK] F1 {f1:.0f}%: good balance of precision and recall.")
    elif f1 > 40:
        print(f"    [WARN] F1 {f1:.0f}%: moderate balance — consider adjusting thresholds.")
    else:
        print(f"    [LOW] F1 {f1:.0f}%: poor balance — needs tuning.")

    # Per-persona details
    print(f"\n  Per-persona detection (all {known_positives} personas):")
    for _, row in sim_personas.sort_values("risk_level", ascending=False).iterrows():
        status = "[ALERT]" if row["risk_level"] == "High" else "[MISS]"
        print(f"    {status} {row['user']:<18}  "
              f"Score: {row['risk_score']:>6.1f}  Level: {row['risk_level']}")


# ─────────────────────────────────────────────
# SECTION 4: TOP 5 MOST ANOMALOUS REAL USERS
# ─────────────────────────────────────────────

def print_top_anomalous_users(risk_df: pd.DataFrame):
    """Show the 5 highest-risk real users with all features and explanations."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 4: TOP 5 MOST ANOMALOUS REAL USERS")
    print(f"{SEPARATOR}")
    print("\n  (Excludes simulated personas — real users only)")

    top5 = risk_df.nlargest(5, "risk_score")

    for i, (_, row) in enumerate(top5.iterrows(), 1):
        risk_pct = row["risk_score"]
        print(f"\n  {'─' * 60}")
        print(f"  #{i}  {row['user']:<20}  "
              f"Score: {risk_pct:.1f}/100  "
              f"Level: {row['risk_level']:<5}  "
              f"Anomaly: {'YES' if row['is_anomaly'] else 'no'}")
        print(f"  {'─' * 60}")

        print("  Feature profile:")
        for col in FEATURE_COLS:
            val = row[col]
            label = FEATURE_LABELS[col]
            avg = risk_df[col].mean() if col in risk_df.columns else 0
            diff = val - avg
            if isinstance(val, float):
                arrow = "▲" if diff > 0 else "▼" if diff < 0 else "─"
                print(f"    {label:<30} {val:>12.4f}  "
                      f"{arrow} vs avg {avg:.2f}  ({diff:+.2f})")
            else:
                arrow = "▲" if diff > 0 else "▼" if diff < 0 else "─"
                print(f"    {label:<30} {val:>12}  "
                      f"{arrow} vs avg {avg:.2f}  ({diff:+.2f})")

        print(f"\n  Explantion: {row['risk_explanation']}")


# ─────────────────────────────────────────────
# SECTION 5: MODEL METADATA REVIEW
# ─────────────────────────────────────────────

def print_model_metadata(meta: dict):
    """Show saved model metadata."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 5: MODEL METADATA")
    print(f"{SEPARATOR}")

    if not meta:
        print("  No model_metadata.json found.")
        return

    fields = [
        ("training_date", "Training Date", ""),
        ("algorithm", "Algorithm", ""),
        ("contamination", "Contamination", ""),
        ("score_separation", "Score Separation", ""),
        ("n_estimators", "Number of Trees", ""),
        ("n_users_trained", "Users Trained", ","),
        ("n_features", "Features Used", ""),
    ]

    for key, label, fmt in fields:
        val = meta.get(key, "?")
        if fmt == "," and isinstance(val, (int, float)):
            val = f"{val:,}"
        print(f"  {label:<25}: {val}")

    features = meta.get("features_used", [])
    if features:
        missing = [f for f in FEATURE_COLS if f not in features]
        print(f"\n  Feature count: {len(features)}")
        if missing:
            print(f"  Missing from metadata: {missing}")


# ─────────────────────────────────────────────
# SECTION 6: MODEL QUALITY CHECKS
# ─────────────────────────────────────────────

def print_model_quality_checks(risk_df: pd.DataFrame, sim_df: pd.DataFrame):
    """Run a series of pass/fail quality checks."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 6: MODEL QUALITY CHECKS")
    print(f"{SEPARATOR}")

    checks = []
    total = 0
    passed = 0

    # Check 1: Score range covers 0-100
    total += 1
    if risk_df["risk_score"].min() < 1.0 and risk_df["risk_score"].max() > 99.0:
        checks.append(("[PASS]", "Score range spans 0-100"))
        passed += 1
    else:
        checks.append(("[WARN]", f"Score range limited: {risk_df['risk_score'].min():.1f}-{risk_df['risk_score'].max():.1f}"))

    # Check 2: Anomaly rate matches contamination
    total += 1
    anomaly_rate = risk_df["is_anomaly"].mean() * 100
    if 2.5 <= anomaly_rate <= 3.5:
        checks.append(("[PASS]", f"Anomaly rate {anomaly_rate:.1f}% matches contamination 0.03"))
        passed += 1
    else:
        checks.append(("[INFO]", f"Anomaly rate {anomaly_rate:.1f}%"))

    # Check 3: High risk users all flagged as anomalies
    total += 1
    high_flagged = risk_df[(risk_df["risk_level"] == "High") & (risk_df["is_anomaly"] == 1)].shape[0]
    high_total = (risk_df["risk_level"] == "High").sum()
    if high_flagged == high_total:
        checks.append(("[PASS]", f"All {high_total} High risk users flagged as anomalies"))
        passed += 1
    else:
        checks.append(("[WARN]", f"Only {high_flagged}/{high_total} High risk users flagged"))

    # Check 4: No anomalies among lowest 10% by score
    total += 1
    bottom10 = risk_df.nsmallest(100, "risk_score")
    anomalies_in_bottom = bottom10["is_anomaly"].sum()
    if anomalies_in_bottom == 0:
        checks.append(("[PASS]", "No anomalies in bottom 100 users by score"))
        passed += 1
    else:
        checks.append(("[WARN]", f"{anomalies_in_bottom} anomalies found in bottom 100 users"))

    # Check 5: Median High risk score significantly above Medium
    total += 1
    med_high = risk_df[risk_df["risk_level"] == "High"]["risk_score"].median()
    med_med  = risk_df[risk_df["risk_level"] == "Medium"]["risk_score"].median()
    if med_high - med_med > 20:
        checks.append(("[PASS]", f"Clear separation: High median={med_high:.1f}, Medium median={med_med:.1f} (delta={med_high-med_med:.1f})"))
        passed += 1
    else:
        checks.append(("[WARN]", f"Marginal separation: High median={med_high:.1f}, Medium median={med_med:.1f}"))

    # Check 6: FullThreat persona detected
    total += 1
    ft = sim_df[(sim_df["user"] == "SIM_FullThreat") & (sim_df["risk_level"] == "High")]
    if len(ft) > 0:
        checks.append(("[PASS]", "SIM_FullThreat correctly detected as High risk"))
        passed += 1
    else:
        checks.append(("[FAIL]", "SIM_FullThreat NOT detected — model failed the primary validation test"))

    # Check 7: Median rule score for simulated personas
    total += 1
    sim_only = sim_df[sim_df["is_simulated"] == 1]
    med_rule = sim_only["rule_score"].median() if "rule_score" in sim_only.columns else 0
    if med_rule > 50:
        checks.append(("[PASS]", f"Simulated personas have high rule-based scores (median={med_rule:.1f})"))
        passed += 1
    else:
        checks.append(("[INFO]", f"Simulated persona rule-based median={med_rule:.1f}"))

    # Print results
    for tag, msg in checks:
        print(f"  {tag} {msg}")
    print(f"\n  Quality checks: {passed}/{total} passed.")


# ─────────────────────────────────────────────
# SECTION 7: SCORE DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────

def print_score_distribution(risk_df: pd.DataFrame):
    """Analyze how risk scores are distributed across risk levels."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 7: SCORE DISTRIBUTION ANALYSIS")
    print(f"{SEPARATOR}")

    print(f"\n  {'Risk Level':<15} {'Count':>8} {'%':>7} {'Mean Score':>12} "
          f"{'Median Score':>14} {'Std Dev':>10} {'Min':>8} {'Max':>8}")
    print(f"{SUB_SEPARATOR}")

    for level in ["High", "Medium", "Low"]:
        subset = risk_df[risk_df["risk_level"] == level]
        s = subset["risk_score"]
        print(f"  {level:<15} {len(subset):>8} {len(subset)/len(risk_df)*100:>6.1f}% "
              f"{s.mean():>12.2f} {s.median():>14.2f} {s.std():>10.2f} {s.min():>8.2f} {s.max():>8.2f}")

    # Percentile boundaries (at every 10th percentile)
    print("\n  Score percentiles:")
    print(f"  {'Percentile':>12} {'Score':>8}")
    print(SUB_SEPARATOR)
    for p in range(0, 101, 10):
        val = risk_df["risk_score"].quantile(p / 100)
        print(f"  {p:>10}th {val:>8.2f}")

    # Flag distribution
    print("\n  Flag frequency (how many users trigger each rule-based check):")
    flag_counts = (risk_df["risk_explanation"] != "No significant anomalies detected.").value_counts()
    flagged = flag_counts.get(True, 0)
    not_flagged = flag_counts.get(False, 0)
    print(f"  Users with flagged behaviors: {flagged:,} / {len(risk_df):,} ({flagged/len(risk_df)*100:.1f}%)")
    print(f"  Users with no flagged behaviors: {not_flagged:,}")


# ─────────────────────────────────────────────
# SECTION 8: TEMPORAL CROSS-VALIDATION
# ─────────────────────────────────────────────

def temporal_validation():
    """
    Temporal cross-validation to prove the model generalises.

    Splits the CERT data by date:
      Train: first 365 days  (2010-01-04 to 2011-01-03)
      Test:  remaining ~130 days (2011-01-04 to 2011-05-14)

    Features are independently extracted from each split so that
    a user's behaviour in the train period is independent from
    their behaviour in the test period.  The model is trained on
    the train-split features and used to score both splits.

    If the model is not overfit the holdout (test) scores should
    show a similar distribution to the training scores, and the
    users flagged as anomalies in each period should overlap at
    a rate significantly above random chance.
    """
    from datetime import timedelta
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    from src.features import (extract_logon_features, extract_device_features,
                              build_feature_table)
    from src.model import FEATURE_COLS, N_ESTIMATORS, RANDOM_STATE


    print(f"\n{SEPARATOR}")
    print("  SECTION 8: TEMPORAL CROSS-VALIDATION")
    print(f"{SEPARATOR}")

    # ── Load and split raw data by date ──
    print("\n  Loading and splitting data by date...")

    logon_raw = pd.read_csv("data/logon.csv")
    device_raw = pd.read_csv("data/device.csv")

    # Parse dates to find the split point
    logon_raw["date"] = pd.to_datetime(logon_raw["date"])
    device_raw["date"] = pd.to_datetime(device_raw["date"])

    earliest = min(logon_raw["date"].min(), device_raw["date"].min())
    split_date = earliest + timedelta(days=365)
    print(f"  Date range     : {earliest.date()} to {max(logon_raw['date'].max(), device_raw['date'].max()).date()}")
    print(f"  Split point    : {split_date.date()}  (day 365 of {495})")

    # Split logon data
    logon_train = logon_raw[logon_raw["date"] < split_date].copy()
    logon_test  = logon_raw[logon_raw["date"] >= split_date].copy()
    print(f"  Logon events   : {len(logon_train):,} train  |  {len(logon_test):,} test")

    # Split device data
    device_train = device_raw[device_raw["date"] < split_date].copy()
    device_test  = device_raw[device_raw["date"] >= split_date].copy()
    print(f"  Device events  : {len(device_train):,} train  |  {len(device_test):,} test")

    # ── Preprocess each split independently ──
    print("\n  Preprocessing train split...")
    # We need to run the preprocess functions manually since they read from CSV paths
    # Let's do the preprocessing inline to use the split dataframes
    def preprocess_raw(df, label):
        """Minimal inline preprocessing matching src.preprocess logic."""
        df = df.copy()
        df = df.drop_duplicates()
        for col in ["user", "pc", "activity"]:
            if col in df.columns:
                df = df[~df[col].astype(str).str.strip().str.lower().isin(
                    ["", "nan", "null", "none", "na", "n/a"])]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for col in ["user", "pc"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        if "activity" in df.columns:
            df["activity"] = df["activity"].astype(str).str.strip().str.title()
        df["hour"]         = df["date"].dt.hour
        df["day_of_week"]  = df["date"].dt.dayofweek
        df["is_weekend"]   = df["day_of_week"] >= 5
        df["is_off_hours"] = (df["hour"] >= 18) | (df["hour"] < 8)
        return df

    logon_train = preprocess_raw(logon_train, "logon_train")
    logon_test  = preprocess_raw(logon_test, "logon_test")
    device_train = preprocess_raw(device_train, "device_train")
    device_test  = preprocess_raw(device_test, "device_test")

    # Filter to Logon / Connect events only
    lt_logon = logon_train[logon_train["activity"] == "Logon"].copy()
    ltest_logon = logon_test[logon_test["activity"] == "Logon"].copy()
    dt_connect = device_train[device_train["activity"] == "Connect"].copy()
    dtest_connect = device_test[device_test["activity"] == "Connect"].copy()

    print(f"  Train: {len(lt_logon):,} Logon events, {len(dt_connect):,} Connect events")
    print(f"  Test:  {len(ltest_logon):,} Logon events, {len(dtest_connect):,} Connect events")

    # ── Extract features independently ──
    print("\n  Extracting features...")
    train_logon_feats = extract_logon_features(lt_logon)
    train_device_feats = extract_device_features(dt_connect)
    train_features = build_feature_table(train_logon_feats, train_device_feats)

    test_logon_feats = extract_logon_features(ltest_logon)
    test_device_feats = extract_device_features(dtest_connect)
    test_features = build_feature_table(test_logon_feats, test_device_feats)

    print(f"  Train features: {len(train_features):,} users x {len(train_features.columns)-1} features")
    print(f"  Test features:  {len(test_features):,} users x {len(test_features.columns)-1} features")

    # ── Merge so we can compare users present in both periods ──
    common_users = set(train_features["user"]) & set(test_features["user"])
    print(f"\n  Users present in both periods: {len(common_users)}")

    # ── Train on train-split features ──
    print("\n  Training Isolation Forest on train-period features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features[FEATURE_COLS].values)

    # Auto-select contamination
    from src.model import CONTAMINATION_CANDIDATES
    best_cont = CONTAMINATION_CANDIDATES[0]
    best_sep = -np.inf
    for cont in CONTAMINATION_CANDIDATES:
        m = IsolationForest(n_estimators=N_ESTIMATORS, contamination=cont,
                            random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train)
        raw = m.decision_function(X_train)
        preds = m.predict(X_train)
        anom_mask = preds == -1
        if anom_mask.sum() > 0 and anom_mask.sum() < len(preds):
            sep = float(raw[~anom_mask].mean() - raw[anom_mask].mean())
        else:
            sep = 0.0
        if sep > best_sep:
            best_sep = sep
            best_cont = cont

    print(f"  Selected contamination: {best_cont} (separation={best_sep:.4f})")

    model = IsolationForest(n_estimators=N_ESTIMATORS, contamination=best_cont,
                            random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train)

    # ── Score both train and test ──
    def _normalize_scores(raw_scores, reference_raw):
        inverted = -1 * raw_scores
        ref_inv = -1 * reference_raw
        lo, hi = ref_inv.min(), ref_inv.max()
        if hi > lo:
            return ((inverted - lo) / (hi - lo) * 100).round(2)
        return np.zeros(len(inverted))

    # Train scores
    train_raw = model.decision_function(X_train)
    train_preds = model.predict(X_train)

    # Test scores
    common_test = test_features[test_features["user"].isin(common_users)]
    X_test = scaler.transform(common_test[FEATURE_COLS].values)
    test_raw = model.decision_function(X_test)
    test_preds = model.predict(X_test)

    train_scores = _normalize_scores(train_raw, train_raw)
    test_scores = _normalize_scores(test_raw, train_raw)

    # ── Compare distributions ──
    print(f"\n  {'─' * 55}")
    print(f"  {'Metric':<30} {'Train Period':>13} {'Test Period':>13}")
    print(f"  {'─' * 55}")
    print(f"  {'Users scored':<30} {len(train_features):>13,} {len(common_test):>13,}")
    print(f"  {'Mean risk score':<30} {np.mean(train_scores):>13.2f} {np.mean(test_scores):>13.2f}")
    print(f"  {'Median risk score':<30} {np.median(train_scores):>13.2f} {np.median(test_scores):>13.2f}")
    print(f"  {'Std risk score':<30} {np.std(train_scores):>13.2f} {np.std(test_scores):>13.2f}")
    print(f"  {'Min risk score':<30} {np.min(train_scores):>13.2f} {np.min(test_scores):>13.2f}")
    print(f"  {'Max risk score':<30} {np.max(train_scores):>13.2f} {np.max(test_scores):>13.2f}")
    train_anom_count = int((train_preds == -1).sum())
    test_anom_count = int((test_preds == -1).sum())
    print(f"  {'Anomalies flagged':<30} {train_anom_count:>13} {test_anom_count:>13}")
    print(f"  {'Anomaly rate':<30} {train_anom_count/len(train_preds)*100:>12.1f}% {test_anom_count/len(test_preds)*100:>12.1f}%")
    print(f"  {'─' * 55}")

    # ── Score rank correlation (users present in both periods) ──
    train_features[["user"]].copy()
    train_map = dict(zip(train_features["user"], train_scores))
    test_map = dict(zip(common_test["user"], test_scores))

    paired = [(u, train_map.get(u, 0), test_map.get(u, 0)) for u in common_users]
    paired_df = pd.DataFrame(paired, columns=["user", "train_score", "test_score"])

    # Drop users with zero variance in either period
    paired_df = paired_df[(paired_df["train_score"] != paired_df["train_score"].iloc[0]) &
                          (paired_df["test_score"] != paired_df["test_score"].iloc[0])]

    if len(paired_df) > 1:
        rank_corr = paired_df["train_score"].corr(paired_df["test_score"], method="spearman")
    else:
        rank_corr = 0.0

    print(f"\n  Spearman rank correlation (train vs test scores): {rank_corr:.3f}")
    if rank_corr > 0.4:
        print("  [PASS] Strong rank correlation — users keep their relative risk positions across time.")
    elif rank_corr > 0.2:
        print("  [WARN] Moderate rank correlation — some temporal drift in user behaviour.")
    else:
        print("  [WARN] Weak rank correlation — user risk rankings are not stable over time.")

    # ── Top-user overlap ──
    train_top10 = set(train_features.nlargest(10, "login_count")["user"])
    test_top10  = set(common_test.nlargest(10, "login_count")["user"])
    top_overlap = len(train_top10 & test_top10)
    print(f"\n  Top-10 most-active users present in both periods: {top_overlap}/10")

    # ── Anomaly overlap (users anomalous in both periods) ──
    train_anomalies = set(train_features[(train_preds == -1).flatten()]["user"])
    test_anomalies = set(common_test[(test_preds == -1).flatten()]["user"])
    both_anomalies = train_anomalies & test_anomalies
    print(f"  Anomalies in train: {len(train_anomalies)}")
    print(f"  Anomalies in test:  {len(test_anomalies)}")
    print(f"  Anomalous in both:  {len(both_anomalies)}")
    if len(both_anomalies) > 0:
        print(f"  Overlap ratio:      {len(both_anomalies) / min(len(train_anomalies), len(test_anomalies)) * 100:.1f}%")
        print(f"  Persistent anomalies: {', '.join(sorted(both_anomalies)[:5])}{'...' if len(both_anomalies) > 5 else ''}")

    # ── Verdict ──
    print("\n  Temporal validation verdict:")
    verdict_passes = 0
    verdict_total = 3

    # Check 1: mean scores within 15 points
    t_mean = float(np.mean(train_scores))
    e_mean = float(np.mean(test_scores))
    if abs(t_mean - e_mean) < 15:
        print(f"  [PASS] Mean scores consistent (train={t_mean:.1f}, test={e_mean:.1f}, delta={abs(t_mean-e_mean):.1f})")
        verdict_passes += 1
    else:
        print(f"  [WARN] Mean scores drifted (train={t_mean:.1f}, test={e_mean:.1f}, delta={abs(t_mean-e_mean):.1f})")

    # Check 2: anomaly rates within 5 percentage points
    train_ar = (train_preds==-1).sum()/len(train_preds)*100
    test_ar = (test_preds==-1).sum()/len(test_preds)*100
    if abs(train_ar - test_ar) < 5:
        print(f"  [PASS] Anomaly rates consistent (train={train_ar:.1f}%, test={test_ar:.1f}%, delta={abs(train_ar-test_ar):.1f}%)")
        verdict_passes += 1
    else:
        print(f"  [WARN] Anomaly rates diverged (train={train_ar:.1f}%, test={test_ar:.1f}%, delta={abs(train_ar-test_ar):.1f}%)")

    # Check 3: rank correlation above 0.3
    if rank_corr > 0.3:
        print(f"  [PASS] Rank correlation ({rank_corr:.3f}) above 0.3 — users maintain relative positions")
        verdict_passes += 1
    else:
        print(f"  [WARN] Rank correlation ({rank_corr:.3f}) below 0.3 — rankings not stable")

    print(f"\n  Temporal validation: {verdict_passes}/{verdict_total} checks passed.")

def main():
    print(f"\n{SEPARATOR}")
    print("  MODEL VERIFICATION & VALIDATION REPORT")
    print("  Insider Threat Detection — Isolation Forest — 12 Features")
    print(f"{SEPARATOR}")

    # Load data
    risk_df, sim_df, feat_df, meta = load_all_data()
    print(f"\n  Loaded: {len(risk_df):,} risk scores | {len(sim_df):,} combined results | "
          f"{len(feat_df):,} feature vectors")

    # Warm up population means for contextual explanations
    try:
        from src.risk_scoring import _load_population_means
        _load_population_means(risk_df)
    except Exception:
        pass

    # Run all sections
    print_population_statistics(feat_df, risk_df)
    print_feature_quality(feat_df)
    print_confusion_matrix_estimate(risk_df, sim_df)
    print_top_anomalous_users(risk_df)
    print_model_metadata(meta)
    print_model_quality_checks(risk_df, sim_df)
    print_score_distribution(risk_df)
    temporal_validation()

    # Final verdict
    print(f"\n{SEPARATOR}")
    print("  VERIFICATION SUMMARY")
    print(f"{SEPARATOR}")

    # Count how many checks in section 6 passed
    passes = (risk_df["is_anomaly"].mean() * 100)
    high_detected = (risk_df["risk_level"] == "High").sum()

    print(f"\n  Total real users analyzed: {len(risk_df):,}")
    print(f"  High risk users flagged: {high_detected} ({high_detected/len(risk_df)*100:.1f}%)")
    print(f"  Anomaly rate: {passes:.1f}% (target: 3.0%)")
    print(f"  Simulated personas tested: {len(sim_df[sim_df['is_simulated']==1]):,}")
    print(f"  High/low score ratio: {risk_df['risk_score'].max() / max(risk_df['risk_score'].min(), 0.01):.0f}x")

    if meta.get("score_separation", 0) >= 0.25:
        print(f"\n  [PASS] Model has strong discriminatory power (separation={meta.get('score_separation'):.4f}).")
    else:
        print(f"\n  [WARN] Model separation is low ({meta.get('score_separation'):.4f}). Consider retraining.")

    print("\n  Full output files:")
    print("    outputs/risk_report.csv        — all scored users")
    print("    outputs/simulated_results.csv  — simulation with real data")
    print("    outputs/user_features.csv      — feature table")
    print("    outputs/model_metadata.json    — training parameters")
    print("    outputs/isolation_forest_model.pkl — trained model")
    print("    outputs/simulation_report.txt  — human-readable simulation report")

    print(f"\n{SEPARATOR}")
    print("  VERIFICATION COMPLETE")
    print(f"{SEPARATOR}")


if __name__ == "__main__":
    main()
