# dashboard/app.py
"""
File: app.py
Purpose: Streamlit dashboard for the Insider Threat Detection system.
         Provides 5 interactive tabs: Overview, Risk Analysis, User
         Investigation, Live Attack Simulation, What-If Analyzer.
Inputs:  outputs/risk_report.csv, outputs/simulated_results.csv,
         outputs/isolation_forest_model.pkl, outputs/scaler.pkl,
         outputs/model_metadata.json
Dependencies: streamlit, pandas, numpy, plotly, os, sys, time, json, joblib;
              src.features, src.risk_scoring, src.attack_simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
import json
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FEATURE_COLS
from src.risk_scoring import assign_risk_level, generate_explanation, _load_population_means, CONFIG as RS_CONFIG
from src.attack_simulation import compute_rule_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="⛨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

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

FEATURE_UNITS = {
    "login_count": "",
    "off_hour_logins": "",
    "weekend_logins": "",
    "late_night_logins": "",
    "unique_pcs_logon": "",
    "off_hour_ratio": "",
    "weekend_ratio": "",
    "pc_diversity_score": "",
    "device_connections": "",
    "unique_pcs_device": "",
    "device_to_login_ratio": "",
    "avg_session_gap": "s",
}

SLIDER_RANGES = {
    "login_count": (0, 3000, 1),
    "off_hour_logins": (0, 2000, 1),
    "weekend_logins": (0, 100, 1),
    "late_night_logins": (0, 600, 1),
    "unique_pcs_logon": (1, 1000, 1),
    "off_hour_ratio": (0.0, 1.0, 0.01),
    "weekend_ratio": (0.0, 1.0, 0.01),
    "pc_diversity_score": (0.0, 1.0, 0.01),
    "device_connections": (0, 400, 1),
    "unique_pcs_device": (0, 20, 1),
    "device_to_login_ratio": (0.0, 3.0, 0.01),
    "avg_session_gap": (0, 200000, 100),
}

RADAR_FEATURES = [
    "login_count",
    "off_hour_logins",
    "weekend_logins",
    "late_night_logins",
    "unique_pcs_logon",
    "off_hour_ratio",
    "weekend_ratio",
    "pc_diversity_score",
    "device_connections",
    "unique_pcs_device",
    "device_to_login_ratio",
    "avg_session_gap",
]

RADAR_LABELS = [
    "Logins",
    "Off-Hour",
    "Weekend",
    "Late Night",
    "PCs",
    "Off-Hour\nRatio",
    "Weekend\nRatio",
    "PC\nDiversity",
    "Devices",
    "Dev PCs",
    "Dev/Login\nRatio",
    "Session\nGap",
]


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    [data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
    }
    .alert-box {
        background-color: rgba(255,75,75,0.12);
        border-left: 4px solid #ff4b4b;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .alert-medium {
        background-color: rgba(255,170,0,0.12);
        border-left: 4px solid #ffaa00;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .alert-low {
        background-color: rgba(0,200,100,0.1);
        border-left: 4px solid #00c864;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .persona-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .section-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1rem 0 0.6rem 0;
        border-bottom: 1px solid #2d3250;
        padding-bottom: 0.4rem;
    }
    .live-result-box {
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
    }
    .detected-box { background: rgba(255,75,75,0.15); border: 2px solid #ff4b4b; }
    .safe-box { background: rgba(0,200,100,0.12); border: 2px solid #00c864; }
    #MainMenu {visibility:hidden;}
    footer    {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_color(level):
    return {"High": "#ff4b4b", "Medium": "#ffaa00", "Low": "#00c864"}.get(
        level, "#888"
    )


def risk_alert_class(level):
    return {"High": "alert-box", "Medium": "alert-medium", "Low": "alert-low"}.get(
        level, "alert-low"
    )


def data_exists():
    return all(
        os.path.exists(p)
        for p in [
            "outputs/risk_report.csv",
            "outputs/isolation_forest_model.pkl",
            "outputs/scaler.pkl",
        ]
    )


def safe_load_csv(path):
    """Load a CSV and return None instead of crashing."""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────

@st.cache_resource
def load_model_and_scaler():
    try:
        if not os.path.exists("outputs/isolation_forest_model.pkl"):
            return None, None
        model = joblib.load("outputs/isolation_forest_model.pkl")
        scaler = joblib.load("outputs/scaler.pkl")
        return model, scaler
    except Exception:
        return None, None


@st.cache_data
def load_risk_report():
    df = safe_load_csv("outputs/risk_report.csv")
    if df is not None and "risk_level" not in df.columns:
        try:
            from src.risk_scoring import assign_risk_level

            df["risk_level"] = df["risk_score"].apply(assign_risk_level)
        except Exception:
            pass
    return df


@st.cache_data
def load_sim_results():
    df = safe_load_csv("outputs/simulated_results.csv")
    if df is not None and "risk_level" not in df.columns:
        try:
            from src.risk_scoring import assign_risk_level

            df["risk_level"] = df["risk_score"].apply(assign_risk_level)
        except Exception:
            pass
    return df


@st.cache_data
def load_model_metadata():
    try:
        if os.path.exists("outputs/model_metadata.json"):
            with open("outputs/model_metadata.json", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# LIVE SCORE FUNCTION
# ─────────────────────────────────────────────

def score_single_user_live(features: dict, model, scaler, risk_df) -> dict:
    """
    Score a single user dictionary in real time.
    Returns ML score, rule score, composite, risk level, explanation, percentile.
    """
    try:
        row_df = pd.DataFrame([features])[FEATURE_COLS]
        X = scaler.transform(row_df.values)

        raw = model.decision_function(X)[0]
        prediction = model.predict(X)[0]

        # Normalise ML score against real population
        all_raw = model.decision_function(
            scaler.transform(risk_df[FEATURE_COLS].values)
        )
        inv_all = -1 * np.append(all_raw, raw)
        score_min, score_max = inv_all.min(), inv_all.max()
        ml_score = (
            float(((-1 * raw) - score_min) / (score_max - score_min) * 100)
            if score_max > score_min
            else 50.0
        )
        ml_score = round(min(max(ml_score, 0), 100), 2)

        # Rule score
        rule_score = compute_rule_score(pd.Series(features))

        # Composite
        composite = round(0.6 * ml_score + 0.4 * rule_score, 2)

        level = assign_risk_level(composite)

        # Build a full row for explanation
        full_row = pd.Series(
            {**features, "risk_score": composite, "is_anomaly": int(prediction == -1)}
        )
        explanation = generate_explanation(full_row)

        percentile = float((risk_df["risk_score"] < composite).mean() * 100)

        return {
            "ml_score": ml_score,
            "rule_score": rule_score,
            "risk_score": composite,
            "risk_level": level,
            "is_anomaly": prediction == -1,
            "explanation": explanation,
            "percentile": percentile,
            "raw_score": round(raw, 6),
        }
    except Exception as e:
        return {
            "ml_score": 0,
            "rule_score": 0,
            "risk_score": 0,
            "risk_level": "Low",
            "is_anomaly": False,
            "explanation": f"Scoring error: {e}",
            "percentile": 0,
            "raw_score": 0,
        }


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (correlation-based proxy)
# ─────────────────────────────────────────────

@st.cache_data
def compute_feature_importance(risk_df):
    try:
        available = [c for c in FEATURE_COLS if c in risk_df.columns]
        if not available or "risk_score" not in risk_df.columns:
            return pd.Series(dtype=float)
        corrs = risk_df[available].corrwith(risk_df["risk_score"]).abs()
        total = corrs.sum()
        if total > 0:
            imp = (corrs / total * 100).round(1)
        else:
            imp = pd.Series(0.0, index=available)
        return imp.sort_values(ascending=True)
    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# COMPUTE PEER MEANS
# ─────────────────────────────────────────────

def get_peer_means(risk_df):
    result = {}
    for level in ["High", "Medium", "Low"]:
        subset = risk_df[risk_df["risk_level"] == level]
        if len(subset) > 0:
            means = {}
            for c in FEATURE_COLS:
                if c in subset.columns:
                    means[c] = subset[c].mean()
            result[level] = means
        else:
            result[level] = {}
    return result


# ─────────────────────────────────────────────
# COMPUTE PER-FEATURE RANK
# ─────────────────────────────────────────────

def get_feature_ranks(user_row, risk_df):
    ranks = {}
    for c in FEATURE_COLS:
        if c in risk_df.columns and c in user_row:
            val = user_row[c]
            pct = (risk_df[c] < val).mean() * 100
            ranks[c] = round(pct, 1)
    return ranks


# ─────────────────────────────────────────────
# FIND SAFETY SUGGESTIONS
# ─────────────────────────────────────────────

def get_safety_suggestions(features: dict) -> list:
    """Tell user which values to reduce to stop triggering rule-based flags."""
    suggestions = []
    cfg = RS_CONFIG

    checks = [
        ("login_count", cfg["flag_login_count"], "below", "Reduce Total Logins"),
        ("off_hour_logins", cfg["flag_off_hour_logins"], "below", "Reduce Off-Hour Logins"),
        ("weekend_logins", cfg["flag_weekend_logins"], "below", "Reduce Weekend Logins"),
        ("late_night_logins", cfg["flag_late_night_logins"], "below", "Reduce Late-Night Logins"),
        ("unique_pcs_logon", cfg["flag_unique_pcs_logon"] - 1, "below", "Use fewer PCs"),
        ("device_connections", cfg["flag_device_connections"], "below", "Reduce Device Connections"),
        ("unique_pcs_device", cfg["flag_unique_pcs_device"] - 1, "below", "Use fewer Device PCs"),
        ("device_to_login_ratio", cfg["flag_device_to_login_ratio"], "below", "Lower Device-to-Login Ratio"),
        ("pc_diversity_score", cfg["flag_pc_diversity_score"], "below", "Lower PC Diversity Score"),
    ]
    # avg_session_gap is inverted (too low is bad)
    checks.append(
        ("avg_session_gap", cfg["flag_avg_session_gap_sec"], "above", "Increase Session Gap (log in less frequently)")
    )

    for col, threshold, direction, label in checks:
        if col in features:
            val = features[col]
            if direction == "below" and val > threshold:
                suggestions.append((label, f"{val:.1f} -> below {threshold}", threshold))
            elif direction == "above" and val < threshold:
                suggestions.append((label, f"{val:.1f}s -> above {threshold}s", threshold))

    return suggestions


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_full_pipeline_live():
    st.markdown("---")
    prog = st.progress(0)
    state = st.empty()
    log_el = st.empty()
    lines = []

    def add_log(msg):
        lines.append(msg)
        log_el.markdown(
            "<div style='background:#1e2130;border:1px solid #2d3250;"
            "border-radius:8px;padding:12px;font-family:monospace;'>"
            + "".join(
                f"<div style='color:#a0aec0;font-size:0.82rem;margin:2px 0;'>{ln}</div>"
                for ln in lines[-10:]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    try:
        state.info("Stage 1/5 - Preprocessing...")
        prog.progress(10)
        from src.preprocess import preprocess_logon, preprocess_device

        _, logon_ev = preprocess_logon()
        _, device_ev = preprocess_device()
        add_log(f"[OK] Logon: {len(logon_ev):,} | Device: {len(device_ev):,}")

        state.info("Stage 2/5 - Feature engineering (12 features)...")
        prog.progress(28)
        from src.features import (
            extract_logon_features,
            extract_device_features,
            build_feature_table,
            save_feature_table,
        )

        ft = build_feature_table(
            extract_logon_features(logon_ev), extract_device_features(device_ev)
        )
        save_feature_table(ft)
        add_log(f"[OK] Features: {len(ft):,} users | {len(ft.columns)-1} features")

        state.info("Stage 3/5 - Training Isolation Forest...")
        prog.progress(50)
        from src.model import (
            scale_features,
            select_contamination,
            train_isolation_forest,
            generate_scores,
            save_model_artifacts,
            save_model_metadata,
            save_scored_results,
        )

        Xs, sc = scale_features(ft)
        best_cont, _, best_sep = select_contamination(Xs)
        mdl = train_isolation_forest(Xs, contamination=best_cont)
        scored = generate_scores(mdl, Xs, ft)
        save_model_artifacts(mdl, sc)
        save_model_metadata(mdl, sc, len(ft), best_sep, best_cont)
        save_scored_results(scored)
        add_log(f"[OK] Anomalies: {scored['is_anomaly'].sum()}/{len(scored):,}")

        state.info("Stage 4/5 - Risk scoring...")
        prog.progress(72)
        from src.risk_scoring import add_risk_levels, add_explanations, save_risk_report

        rdf = add_explanations(add_risk_levels(scored))
        save_risk_report(rdf)
        h = (rdf["risk_level"] == "High").sum()
        m = (rdf["risk_level"] == "Medium").sum()
        add_log(f"[OK] High: {h} | Medium: {m} | Low: {(rdf['risk_level']=='Low').sum()}")

        state.info("Stage 5/5 - Attack simulation...")
        prog.progress(90)
        from src.attack_simulation import run_attack_simulation

        sp, comb = run_attack_simulation()
        det = (sp["risk_level"] == "High").sum()
        add_log(f"[OK] Simulation: {det}/{len(sp)} detected")

        prog.progress(100)
        state.success(
            f"Pipeline complete - {len(rdf):,} users | {h} High | {det}/{len(sp)} threats detected"
        )
        add_log("--- DONE ---")
        st.cache_data.clear()
        st.cache_resource.clear()
        return True
    except Exception as e:
        state.error(f"Pipeline failed: {e}")
        add_log(f"[FAIL] {e}")
        return False


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown(
    """
<div style='text-align:center;padding:1.2rem 0 0.4rem;'>
    <h1 style='color:#e2e8f0;font-size:1.9rem;font-weight:700;margin:0;'>
        AI-Driven Insider Threat Detection
    </h1>
    <p style='color:#718096;font-size:0.9rem;margin-top:0.3rem;'>
        CERT Dataset &nbsp;|&nbsp; Isolation Forest &nbsp;|&nbsp;
        12 Behavioral Features &nbsp;|&nbsp; Live Attack Simulation
    </p>
</div>
<hr style='border-color:#2d3250;margin-bottom:0.8rem;'>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONTROL PANEL
# ─────────────────────────────────────────────

st.markdown("<p class='section-header'>System Control</p>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    run_btn = st.button("Run Full Pipeline", type="primary", width="stretch")
with c2:
    clear_btn = st.button("Clear Cache & Retrain", width="stretch")
with c3:
    if data_exists():
        st.success("Pipeline ready. Use tabs below for analysis.")
    else:
        st.warning("No data found. Click Run Full Pipeline first.")

if run_btn:
    if run_full_pipeline_live():
        time.sleep(0.5)
        st.rerun()

if clear_btn:
    import shutil

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs")
    st.cache_data.clear()
    st.cache_resource.clear()
    st.info("Cache cleared. Click Run Full Pipeline to retrain.")
    st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

risk_df = load_risk_report()
sim_df = load_sim_results()
model, scaler = load_model_and_scaler()
meta = load_model_metadata()

if risk_df is None:
    st.markdown(
        """
    <div style='text-align:center;padding:5rem 0;'>
        <h2 style='color:#4a9eff;'>No Data Found</h2>
        <p style='color:#718096;'>
            Click <strong>Run Full Pipeline</strong> to get started.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

# Populate population means for contextual explanations
try:
    _load_population_means(risk_df)
except Exception:
    pass

total = len(risk_df)
high = (risk_df["risk_level"] == "High").sum()
medium = (risk_df["risk_level"] == "Medium").sum()
low = (risk_df["risk_level"] == "Low").sum()

sim_only = pd.DataFrame()
if sim_df is not None and "is_simulated" in sim_df.columns:
    sim_only = sim_df[sim_df["is_simulated"] == 1].copy()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    st.markdown("---")
    show_high = st.checkbox("High Risk", value=True)
    show_medium = st.checkbox("Medium Risk", value=True)
    show_low = st.checkbox("Low Risk", value=False)
    selected_levels = (
        (["High"] if show_high else [])
        + (["Medium"] if show_medium else [])
        + (["Low"] if show_low else [])
    )
    score_range = st.slider("Score Range", 0, 100, (0, 100), step=5)
    st.markdown("---")
    st.metric("Total Users", f"{total:,}")
    st.metric("High Risk", high, delta=f"{high/total*100:.1f}%")
    st.metric("Medium Risk", medium)
    st.metric("Low Risk", low)
    if meta:
        st.markdown("---")
        st.markdown(
            f"<p style='color:#718096;font-size:0.75rem;'>"
            f"Trained: {meta.get('training_date','?')}<br>"
            f"Contamination: {meta.get('contamination','?')}<br>"
            f"Separation: {meta.get('score_separation','?')}<br>"
            f"Users: {meta.get('n_users_trained','?'):,}</p>",
            unsafe_allow_html=True,
        )

filtered_df = risk_df[
    (risk_df["risk_level"].isin(selected_levels))
    & (risk_df["risk_score"] >= score_range[0])
    & (risk_df["risk_score"] <= score_range[1])
].copy()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Risk Analysis",
        "User Investigation",
        "Live Attack Simulation",
        "What-If Analyzer",
    ]
)


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab1:
    st.markdown("<p class='section-header'>Security Posture Overview</p>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users Monitored", f"{total:,}")
    c2.metric("High Risk", high, delta=f"{high/total*100:.1f}%", delta_color="inverse")
    c3.metric("Medium Risk", medium, delta=f"{medium/total*100:.1f}%", delta_color="off")
    c4.metric("Avg Risk Score", f"{risk_df['risk_score'].mean():.1f}")
    c5.metric("Peak Risk Score", f"{risk_df['risk_score'].max():.1f}")

    # Model info card
    if meta:
        st.markdown("<p class='section-header'>Model Information</p>", unsafe_allow_html=True)
        mi1, mi2, mi3, mi4, mi5 = st.columns(5)
        mi1.metric("Training Date", meta.get("training_date", "?"))
        mi2.metric("Algorithm", meta.get("algorithm", "?"))
        mi3.metric("Contamination", f"{meta.get('contamination', '?'):.0%}")
        mi4.metric("Users Trained", f"{meta.get('n_users_trained', '?'):,}")
        mi5.metric("Score Separation", f"{meta.get('score_separation', '?'):.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("<p class='section-header'>Risk Distribution</p>", unsafe_allow_html=True)
        try:
            rc = risk_df["risk_level"].value_counts().reset_index()
            rc.columns = ["Risk Level", "Count"]
            fig_pie = px.pie(
                rc,
                values="Count",
                names="Risk Level",
                color="Risk Level",
                color_discrete_map={"High": "#ff4b4b", "Medium": "#ffaa00", "Low": "#00c864"},
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0",
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, width="stretch")
        except Exception:
            st.info("Could not render pie chart.")

    with col_r:
        st.markdown("<p class='section-header'>Risk Score Distribution</p>", unsafe_allow_html=True)
        try:
            fig_hist = px.histogram(
                risk_df, x="risk_score", nbins=40, color_discrete_sequence=["#4a9eff"]
            )
            for val, lbl, clr in [(70, "High Threshold", "#ff4b4b"), (40, "Medium Threshold", "#ffaa00")]:
                fig_hist.add_vline(
                    x=val,
                    line_dash="dash",
                    line_color=clr,
                    annotation_text=lbl,
                    annotation_font_color=clr,
                )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0",
                height=280,
                xaxis=dict(gridcolor="#2d3250"),
                yaxis=dict(gridcolor="#2d3250"),
                margin=dict(t=10, b=30, l=30, r=10),
            )
            st.plotly_chart(fig_hist, width="stretch")
        except Exception:
            st.info("Could not render histogram.")

    # Active alerts
    st.markdown("<p class='section-header'>Active High Risk Alerts</p>", unsafe_allow_html=True)
    try:
        high_users = (
            risk_df[risk_df["risk_level"] == "High"]
            .sort_values("risk_score", ascending=False)
            .head(10)
        )
        for _, row in high_users.iterrows():
            st.markdown(
                f"""
            <div class='alert-box'>
                <strong style='color:#ff4b4b;'>[ALERT]</strong>
                <strong style='color:#e2e8f0;'> {row['user']}</strong>
                <span style='color:#718096;'> — Score: </span>
                <strong style='color:#ff4b4b;'>{row['risk_score']:.1f}</strong>
                <span style='color:#718096;'> | </span>
                <span style='color:#a0aec0;font-size:0.88rem;'>{row['risk_explanation']}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
    except Exception:
        st.info("No alerts to display.")


# ══════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════

with tab2:
    st.markdown("<p class='section-header'>Behavioral Scatter Analysis</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for col, xc, yc, ttl in [
        (col1, "login_count", "off_hour_logins", "Login Count vs Off-Hour Logins"),
        (col2, "device_connections", "unique_pcs_device", "Device Connections vs Unique PCs"),
    ]:
        with col:
            try:
                fig = px.scatter(
                    risk_df,
                    x=xc,
                    y=yc,
                    color="risk_level",
                    size="risk_score",
                    hover_data=["user", "risk_score"],
                    color_discrete_map={
                        "High": "#ff4b4b",
                        "Medium": "#ffaa00",
                        "Low": "#00c864",
                    },
                    title=ttl,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0",
                    height=360,
                    xaxis=dict(gridcolor="#2d3250"),
                    yaxis=dict(gridcolor="#2d3250"),
                )
                st.plotly_chart(fig, width="stretch")
            except Exception:
                st.info(f"Could not render {ttl}.")

    # Feature importance
    st.markdown("<p class='section-header'>Feature Importance (correlation with risk score)</p>", unsafe_allow_html=True)
    try:
        imp = compute_feature_importance(risk_df)
        if len(imp) > 0:
            imp_df = imp.reset_index()
            imp_df.columns = ["Feature", "Importance (%)"]
            imp_df["Label"] = imp_df["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            fig_imp = px.bar(
                imp_df,
                x="Importance (%)",
                y="Label",
                orientation="h",
                color="Importance (%)",
                color_continuous_scale="Reds",
                text="Importance (%)",
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0",
                height=400,
                xaxis=dict(gridcolor="#2d3250"),
                yaxis=dict(gridcolor="#2d3250", autorange="reversed"),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            fig_imp.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_imp, width="stretch")
        else:
            st.info("Feature importance not available.")
    except Exception:
        st.info("Could not compute feature importance.")

    # Percentile distribution
    st.markdown("<p class='section-header'>Percentile Distribution by Risk Level</p>", unsafe_allow_html=True)
    try:
        risk_df["percentile_rank"] = risk_df["risk_score"].rank(pct=True) * 100
        fig_pct = px.box(
            risk_df,
            x="risk_level",
            y="percentile_rank",
            color="risk_level",
            color_discrete_map={"High": "#ff4b4b", "Medium": "#ffaa00", "Low": "#00c864"},
            title="Score percentile spread by risk level",
        )
        fig_pct.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=360,
            xaxis=dict(gridcolor="#2d3250"),
            yaxis=dict(gridcolor="#2d3250"),
            showlegend=False,
        )
        st.plotly_chart(fig_pct, width="stretch")
    except Exception:
        st.info("Could not render percentile chart.")

    # Feature correlation heatmap
    st.markdown("<p class='section-header'>Feature Correlation Heatmap</p>", unsafe_allow_html=True)
    try:
        available_feats = [c for c in FEATURE_COLS if c in risk_df.columns]
        heat_cols = available_feats + ["risk_score"]
        corr = risk_df[heat_cols].corr().round(2)
        fig_hm = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=True,
            aspect="auto",
        )
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=500,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hm, width="stretch")
    except Exception:
        st.info("Could not render heatmap.")

    # Top 20 bar
    st.markdown("<p class='section-header'>Top 20 by Risk Score</p>", unsafe_allow_html=True)
    try:
        fig_bar = px.bar(
            risk_df.nlargest(20, "risk_score"),
            x="risk_score",
            y="user",
            orientation="h",
            color="risk_level",
            color_discrete_map={"High": "#ff4b4b", "Medium": "#ffaa00", "Low": "#00c864"},
            hover_data=["login_count", "off_hour_logins", "device_connections"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=480,
            yaxis=dict(autorange="reversed", gridcolor="#2d3250"),
            xaxis=dict(gridcolor="#2d3250"),
        )
        st.plotly_chart(fig_bar, width="stretch")
    except Exception:
        st.info("Could not render bar chart.")

    # Full risk table (searchable)
    st.markdown("<p class='section-header'>Full Risk Table</p>", unsafe_allow_html=True)
    try:
        search = st.text_input("Search by username:", "", key="risk_table_search")
        dcols = [c for c in ["user", "risk_level", "risk_score", "login_count",
                              "off_hour_logins", "weekend_logins", "late_night_logins",
                              "unique_pcs_logon", "device_connections", "off_hour_ratio",
                              "device_to_login_ratio", "avg_session_gap", "risk_explanation"]
                 if c in filtered_df.columns]
        table_df = filtered_df[dcols].sort_values("risk_score", ascending=False).reset_index(drop=True)
        if search:
            table_df = table_df[table_df["user"].str.contains(search, case=False, na=False)]
        st.dataframe(
            table_df,
            width="stretch",
            height=380,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=100, format="%.1f"
                )
            },
        )
        st.caption(f"Showing {len(table_df):,} users.")
    except Exception:
        st.info("Could not render risk table.")


# ══════════════════════════════════════════════
# TAB 3 — USER INVESTIGATION
# ══════════════════════════════════════════════

with tab3:
    st.markdown("<p class='section-header'>Individual User Deep Dive</p>", unsafe_allow_html=True)
    try:
        selected_user = st.selectbox(
            "Select user",
            options=risk_df.sort_values("risk_score", ascending=False)["user"].tolist(),
            index=0,
        )
    except Exception:
        st.warning("No users available.")
        st.stop()

    try:
        user_row = risk_df[risk_df["user"] == selected_user].iloc[0]
    except (IndexError, KeyError):
        st.warning("Selected user not found.")
        st.stop()

    level = user_row["risk_level"]
    color = risk_color(level)

    st.markdown(
        f"""
    <div style='background:#1e2130;border:2px solid {color};
                border-radius:12px;padding:20px;margin:10px 0;'>
        <h2 style='color:{color};margin:0;'>{selected_user}</h2>
        <p style='color:#718096;margin:6px 0;'>
            Risk Level: <strong style='color:{color}'>{level}</strong>
            &nbsp;|&nbsp;
            Score: <strong style='color:{color}'>{user_row['risk_score']:.1f}/100</strong>
            &nbsp;|&nbsp;
            Anomaly: <strong style='color:{color}'>{'Yes' if user_row['is_anomaly'] else 'No'}</strong>
        </p>
        <p style='color:#a0aec0;font-size:0.88rem;margin-top:8px;'>{user_row['risk_explanation']}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Top 8 metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Logins", int(user_row.get("login_count", 0)))
    m2.metric("Off-Hour Logins", int(user_row.get("off_hour_logins", 0)),
              delta=f"{user_row.get('off_hour_ratio', 0)*100:.0f}%")
    m3.metric("Late-Night Logins", int(user_row.get("late_night_logins", 0)))
    m4.metric("Unique PCs", int(user_row.get("unique_pcs_logon", 0)))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Device Connections", int(user_row.get("device_connections", 0)))
    m6.metric("Device PCs", int(user_row.get("unique_pcs_device", 0)))
    m7.metric("Device/Login Ratio", f"{user_row.get('device_to_login_ratio', 0):.2f}")
    m8.metric("Avg Session Gap", f"{user_row.get('avg_session_gap', 0):.0f}s")

    # 12-feature radar chart
    st.markdown("<p class='section-header'>Behavioral Radar (12 features, normalised 0-1)</p>", unsafe_allow_html=True)
    try:
        available_radar = [f for f in RADAR_FEATURES if f in risk_df.columns]
        fmax = risk_df[available_radar].max()
        uvals = [user_row[f] / fmax[f] if fmax[f] > 0 else 0 for f in available_radar]
        avals = [risk_df[f].mean() / fmax[f] if fmax[f] > 0 else 0 for f in available_radar]
        radar_labels = []
        for f in available_radar:
            idx = RADAR_FEATURES.index(f)
            radar_labels.append(RADAR_LABELS[idx] if idx < len(RADAR_LABELS) else f)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=uvals + [uvals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name=selected_user, line_color=color,
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=avals + [avals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name="Avg User", line_color="#4a9eff",
            fillcolor="rgba(74,158,255,0.1)",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2d3250", color="#718096"),
                angularaxis=dict(color="#a0aec0"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=450,
            showlegend=True,
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_r, width="stretch")
    except Exception:
        st.info("Could not render radar chart.")

    # Peer comparison
    st.markdown("<p class='section-header'>Peer Comparison (user vs High/Medium/Low averages)</p>", unsafe_allow_html=True)
    try:
        peers = get_peer_means(risk_df)
        available_comp = [c for c in FEATURE_COLS if c in risk_df.columns]
        comp_data = {"Feature": []}
        for lvl in ["High", "Medium", "Low"]:
            comp_data[lvl] = []
        comp_data["Selected User"] = []

        for c in available_comp:
            label = FEATURE_LABELS.get(c, c)
            comp_data["Feature"].append(label)
            for lvl in ["High", "Medium", "Low"]:
                comp_data[lvl].append(round(peers.get(lvl, {}).get(c, 0), 2))
            comp_data["Selected User"].append(round(user_row.get(c, 0), 2))

        comp_df = pd.DataFrame(comp_data)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="High Avg", x=comp_df["Feature"], y=comp_df["High"],
                                  marker_color="rgba(255,75,75,0.6)"))
        fig_comp.add_trace(go.Bar(name="Medium Avg", x=comp_df["Feature"], y=comp_df["Medium"],
                                  marker_color="rgba(255,170,0,0.6)"))
        fig_comp.add_trace(go.Bar(name="Low Avg", x=comp_df["Feature"], y=comp_df["Low"],
                                  marker_color="rgba(0,200,100,0.6)"))
        fig_comp.add_trace(go.Bar(name=selected_user, x=comp_df["Feature"], y=comp_df["Selected User"],
                                  marker_color=color))

        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=400,
            xaxis=dict(gridcolor="#2d3250"),
            yaxis=dict(gridcolor="#2d3250"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=80, l=10, r=10),
        )
        st.plotly_chart(fig_comp, width="stretch")
    except Exception:
        st.info("Could not render peer comparison.")

    # Per-feature rank
    st.markdown("<p class='section-header'>Per-Feature Percentile Rank</p>", unsafe_allow_html=True)
    try:
        ranks = get_feature_ranks(user_row, risk_df)
        rank_items = [(FEATURE_LABELS.get(c, c), ranks[c], c) for c in FEATURE_COLS if c in ranks]
        rank_items.sort(key=lambda x: x[1], reverse=True)
        rcols = st.columns(3)
        for i, (label, pct, col_name) in enumerate(rank_items):
            with rcols[i % 3]:
                st.metric(label, f"Top {100-pct:.0f}%" if pct > 50 else f"Bottom {pct:.0f}%",
                          delta=f"p{pct:.0f}")
    except Exception:
        st.info("Could not compute per-feature ranks.")

    # Overall rank
    try:
        rank = int(
            risk_df["risk_score"]
            .rank(ascending=False, method="min")[risk_df["user"] == selected_user]
            .values[0]
        )
        st.info(f"**{selected_user}** ranks **#{rank}** out of {total:,} users.")
    except Exception:
        pass


# ══════════════════════════════════════════════
# TAB 4 — LIVE ATTACK SIMULATION
# ══════════════════════════════════════════════

with tab4:
    st.markdown(
        """
    <div style='background:#1e2130;border:1px solid #2d3250;
                border-radius:10px;padding:16px;margin-bottom:1rem;'>
        <h4 style='color:#e2e8f0;margin:0 0 6px 0;'>Custom Attack Builder</h4>
        <p style='color:#718096;margin:0;font-size:0.88rem;'>
            Design a threat actor using the sliders below. Hit <strong style='color:#4a9eff;'>Launch Attack</strong>
            to score them live against the trained model. Shows ML, Rule, and Composite scores.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    sl_col, pre_col = st.columns([3, 1])

    with pre_col:
        st.markdown("<p class='section-header'>Quick Presets</p>", unsafe_allow_html=True)
        preset = st.radio(
            "Load a preset",
            ["Custom", "Night Owl", "PC Hopper", "Data Mule", "Ghost User", "Full Threat",
             "Email Thief", "Saboteur"],
            index=0,
        )

    # Preset values (all 12 features)
    presets = {
        "Custom": dict(lc=40, ohl=5, wl=3, lnl=2, upc=1, ohr=0.12, wr=0.07,
                       pcd=0.02, dc=10, upd=1, dlr=0.10, asg=36000),
        "Night Owl": dict(lc=95, ohl=88, wl=12, lnl=80, upc=2, ohr=0.93, wr=0.13,
                          pcd=0.02, dc=8, upd=1, dlr=0.08, asg=7200),
        "PC Hopper": dict(lc=120, ohl=18, wl=14, lnl=5, upc=14, ohr=0.15, wr=0.12,
                          pcd=0.12, dc=12, upd=11, dlr=0.10, asg=3600),
        "Data Mule": dict(lc=55, ohl=22, wl=19, lnl=15, upc=4, ohr=0.40, wr=0.35,
                          pcd=0.07, dc=112, upd=9, dlr=2.04, asg=5400),
        "Ghost User": dict(lc=18, ohl=16, wl=10, lnl=14, upc=5, ohr=0.89, wr=0.56,
                           pcd=0.28, dc=4, upd=4, dlr=0.22, asg=86400),
        "Full Threat": dict(lc=210, ohl=175, wl=68, lnl=140, upc=18, ohr=0.83, wr=0.32,
                            pcd=0.09, dc=145, upd=15, dlr=0.69, asg=300),
        "Email Thief": dict(lc=78, ohl=45, wl=22, lnl=38, upc=6, ohr=0.58, wr=0.28,
                            pcd=0.08, dc=15, upd=5, dlr=0.19, asg=45),
        "Saboteur": dict(lc=42, ohl=39, wl=28, lnl=35, upc=9, ohr=0.93, wr=0.67,
                         pcd=0.21, dc=67, upd=8, dlr=1.60, asg=180),
    }
    pv = presets.get(preset, presets["Custom"])

    with sl_col:
        st.markdown("<p class='section-header'>Behavior Parameters (12 features)</p>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            login_count = st.slider("Total Logins", 0, 2500, pv["lc"], step=1)
            off_hour_logins = st.slider("Off-Hour Logins", 0, 2000, pv["ohl"], step=1)
            weekend_logins = st.slider("Weekend Logins", 0, 100, pv["wl"], step=1)
            late_night_logins = st.slider("Late-Night Logins (10PM-4AM)", 0, 600, pv["lnl"], step=1)
            unique_pcs_logon = st.slider("Unique PCs (Logon)", 1, 1000, pv["upc"], step=1)
            off_hour_ratio = st.slider("Off-Hour Ratio", 0.0, 1.0, pv["ohr"], step=0.01)
        with s2:
            weekend_ratio = st.slider("Weekend Ratio", 0.0, 1.0, pv["wr"], step=0.01)
            pc_diversity_score = st.slider("PC Diversity Score", 0.0, 1.0, pv["pcd"], step=0.01)
            device_connections = st.slider("Device Connections", 0, 400, pv["dc"], step=1)
            unique_pcs_device = st.slider("Unique PCs (Device)", 0, 20, pv["upd"], step=1)
            device_to_login_ratio = st.slider("Device/Login Ratio", 0.0, 3.0, pv["dlr"], step=0.01)
            avg_session_gap = st.slider("Avg Session Gap (s)", 0, 200000, pv["asg"], step=100)

    attacker_name = st.text_input(
        "Attacker Label",
        value=f"LIVE_{preset.replace(' ','_').upper()}" if preset != "Custom" else "LIVE_CUSTOM_ATTACKER",
    )

    st.markdown("---")

    launch_btn = st.button("Launch Attack", type="primary", width="stretch")

    if launch_btn:
        if model is None or scaler is None:
            st.error("No trained model found. Run the full pipeline first.")
        else:
            features = {
                "login_count": login_count,
                "off_hour_logins": off_hour_logins,
                "weekend_logins": weekend_logins,
                "late_night_logins": late_night_logins,
                "unique_pcs_logon": unique_pcs_logon,
                "off_hour_ratio": off_hour_ratio,
                "weekend_ratio": weekend_ratio,
                "pc_diversity_score": pc_diversity_score,
                "device_connections": device_connections,
                "unique_pcs_device": unique_pcs_device,
                "device_to_login_ratio": device_to_login_ratio,
                "avg_session_gap": avg_session_gap,
            }

            result = score_single_user_live(features, model, scaler, risk_df)

            lvl = result["risk_level"]
            score = result["risk_score"]
            clr = risk_color(lvl)
            box_cls = "detected-box" if lvl == "High" else ("safe-box" if lvl == "Low" else "")

            verdict = ("THREAT DETECTED" if lvl == "High" else
                       "SUSPICIOUS ACTIVITY" if lvl == "Medium" else "NO THREAT DETECTED")
            verdict_icon = "[ALERT]" if lvl == "High" else ("[WARN]" if lvl == "Medium" else "[CLEAR]")

            st.markdown(
                f"""
            <div class='live-result-box {box_cls}' style='border-color:{clr};background:rgba(0,0,0,0.3);'>
                <h1 style='color:{clr};font-size:2.5rem;margin:0;'>{verdict_icon}</h1>
                <h2 style='color:{clr};margin:8px 0;'>{verdict}</h2>
                <h3 style='color:#e2e8f0;margin:0;'>{attacker_name}</h3>
                <p style='color:#e2e8f0;font-size:1.4rem;font-weight:700;margin:12px 0 4px;'>
                    Composite: {score:.1f} / 100
                </p>
                <p style='color:#a0aec0;font-size:0.9rem;margin:0;'>
                    ML: {result['ml_score']:.1f}  |  Rule: {result['rule_score']:.1f}  |
                    Risk Level: <strong style='color:{clr};'>{lvl}</strong>  |
                    Anomaly: {'Yes' if result['is_anomaly'] else 'No'}  |
                    Riskier than {result['percentile']:.0f}% of users
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("ML Score", f"{result['ml_score']:.1f}")
            d2.metric("Rule Score", f"{result['rule_score']:.1f}")
            d3.metric("Composite", f"{result['risk_score']:.1f}")
            d4.metric("Percentile", f"Top {100-result['percentile']:.0f}%")

            if result["explanation"] != "No significant anomalies detected.":
                st.markdown(
                    f"""
                <div class='alert-box' style='border-color:{clr};'>
                    <strong style='color:{clr};'>Triggered Flags:</strong><br>
                    <span style='color:#e2e8f0;font-size:0.9rem;'>
                        {result['explanation'].replace(' | ', '<br>')}
                    </span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Population scatter
            st.markdown("<p class='section-header'>Attacker vs Real Population</p>", unsafe_allow_html=True)
            try:
                attacker_row = pd.DataFrame([{
                    "user": attacker_name, "risk_score": score, "risk_level": lvl,
                    "login_count": login_count, "off_hour_logins": off_hour_logins,
                    "is_simulated": 1,
                }])
                pop_sample = risk_df.sample(min(300, len(risk_df)), random_state=42).copy()
                pop_sample["is_simulated"] = 0
                plot_df = pd.concat([pop_sample, attacker_row], ignore_index=True)
                plot_df["Type"] = plot_df["is_simulated"].map({0: "Real User", 1: "Live Attacker"})
                fig_pop = px.scatter(
                    plot_df, x="login_count", y="off_hour_logins",
                    color="Type", size="risk_score",
                    hover_data=["user", "risk_score", "risk_level"],
                    color_discrete_map={"Real User": "#4a9eff", "Live Attacker": "#ff4b4b"},
                )
                fig_pop.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", height=400,
                    xaxis=dict(gridcolor="#2d3250"), yaxis=dict(gridcolor="#2d3250"),
                )
                st.plotly_chart(fig_pop, width="stretch")
            except Exception:
                pass

    # Batch simulation
    st.markdown("---")
    st.markdown("<p class='section-header'>Batch Simulation - Live Alert Feed</p>", unsafe_allow_html=True)
    batch_btn = st.button("Run Batch Simulation (All 7 Personas)", width="stretch")

    if batch_btn:
        if model is None or scaler is None:
            st.error("Run the full pipeline first.")
        else:
            from src.attack_simulation import build_attack_personas

            personas_df = build_attack_personas()
            alert_feed = st.empty()
            alert_lines = []
            batch_prog = st.progress(0)
            batch_status = st.empty()
            chart_placeholder = st.empty()

            all_results = []
            chart_data = []

            for i, (_, prow) in enumerate(personas_df.iterrows()):
                batch_status.info(f"Scoring {prow['user']} ({i+1}/{len(personas_df)})...")
                batch_prog.progress((i + 1) / len(personas_df))
                time.sleep(0.3)

                feats = {c: prow[c] for c in FEATURE_COLS}
                res = score_single_user_live(feats, model, scaler, risk_df)
                all_results.append({"user": prow["user"], **res})
                chart_data.append({"user": prow["user"], "composite": res["risk_score"],
                                   "ml": res["ml_score"], "rule": res["rule_score"]})

                lvl = res["risk_level"]
                clr = risk_color(lvl)
                icon = "[ALERT]" if lvl == "High" else ("[WARN]" if lvl == "Medium" else "[OK]")
                alert_lines.append(
                    f"<div class='{risk_alert_class(lvl)}'>"
                    f"<strong style='color:{clr};'>{icon}</strong>"
                    f" <strong style='color:#e2e8f0;'>{prow['user']}</strong>"
                    f"<span style='color:#718096;'> -- Score: </span>"
                    f"<strong style='color:{clr};'>{res['risk_score']:.1f}</strong>"
                    f"<span style='color:#718096;'> | ML: {res['ml_score']:.1f} Rule: {res['rule_score']:.1f} | Level: </span>"
                    f"<strong style='color:{clr};'>{lvl}</strong>"
                    f"<br><span style='color:#a0aec0;font-size:0.82rem;'>{res['explanation']}</span></div>"
                )
                alert_feed.markdown("".join(alert_lines), unsafe_allow_html=True)

                # Live chart
                cd = pd.DataFrame(chart_data)
                fig_batch = go.Figure()
                fig_batch.add_trace(go.Bar(name="ML Score", x=cd["user"], y=cd["ml"],
                                           marker_color="#4a9eff"))
                fig_batch.add_trace(go.Bar(name="Rule Score", x=cd["user"], y=cd["rule"],
                                           marker_color="#ffaa00"))
                fig_batch.add_trace(go.Bar(name="Composite", x=cd["user"], y=cd["composite"],
                                           marker_color="#ff4b4b"))
                fig_batch.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", height=300,
                    xaxis=dict(gridcolor="#2d3250"), yaxis=dict(gridcolor="#2d3250"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                chart_placeholder.plotly_chart(fig_batch, width="stretch")

            batch_prog.progress(1.0)
            detected = sum(1 for r in all_results if r["risk_level"] == "High")
            batch_status.success(f"Batch complete - {detected}/{len(all_results)} detected as High Risk")

            st.markdown("<p class='section-header'>Batch Results Summary</p>", unsafe_allow_html=True)
            try:
                batch_df = pd.DataFrame(all_results)[["user", "ml_score", "rule_score", "risk_score",
                                                       "risk_level", "is_anomaly", "percentile"]]
                batch_df["is_anomaly"] = batch_df["is_anomaly"].map({True: "Yes", False: "No"})
                batch_df["percentile"] = batch_df["percentile"].apply(lambda x: f"Top {100-x:.0f}%")
                st.dataframe(
                    batch_df.sort_values("risk_score", ascending=False).reset_index(drop=True),
                    width="stretch",
                    column_config={
                        "risk_score": st.column_config.ProgressColumn(
                            "Composite", min_value=0, max_value=100, format="%.1f"
                        )
                    },
                )
            except Exception:
                pass


# ══════════════════════════════════════════════
# TAB 5 — WHAT-IF ANALYZER
# ══════════════════════════════════════════════

with tab5:
    st.markdown(
        """
    <div style='background:#1e2130;border:1px solid #2d3250;
                border-radius:10px;padding:16px;margin-bottom:1rem;'>
        <h4 style='color:#e2e8f0;margin:0 0 6px 0;'>What-If Analyzer</h4>
        <p style='color:#718096;margin:0;font-size:0.88rem;'>
            Pick any real user. Modify their behavior using all 12 sliders.
            See their risk score update <strong style='color:#4a9eff;'>instantly</strong>.
            The system shows which flags changed and how to become safe.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Run the full pipeline first.")
        st.stop()

    try:
        wi_user = st.selectbox(
            "Base user",
            options=risk_df.sort_values("risk_score", ascending=False)["user"].tolist(),
            index=0,
            key="whatif_user",
        )
        base = risk_df[risk_df["user"] == wi_user].iloc[0]
    except Exception:
        st.warning("Could not load user data.")
        st.stop()

    # Check if the new feature columns exist in the base row
    def safe_get(row, col, default=0):
        return row[col] if col in row else default

    st.markdown("<p class='section-header'>Modify Behavior (all 12 features)</p>", unsafe_allow_html=True)
    wa, wb = st.columns(2)

    with wa:
        wi_lc = st.slider("Login Count", 0, 3000, int(safe_get(base, "login_count", 40)), key="wi_lc")
        wi_ohl = st.slider("Off-Hour Logins", 0, 2000, int(safe_get(base, "off_hour_logins", 5)), key="wi_ohl")
        wi_wl = st.slider("Weekend Logins", 0, 100, int(safe_get(base, "weekend_logins", 3)), key="wi_wl")
        wi_lnl = st.slider("Late-Night Logins", 0, 600, int(safe_get(base, "late_night_logins", 2)), key="wi_lnl")
        wi_upc = st.slider("Unique PCs (Logon)", 1, 1000, int(safe_get(base, "unique_pcs_logon", 1)), key="wi_upc")
        wi_ohr = st.slider("Off-Hour Ratio", 0.0, 1.0, float(safe_get(base, "off_hour_ratio", 0.12)), step=0.01, key="wi_ohr")
    with wb:
        wi_wr = st.slider("Weekend Ratio", 0.0, 1.0, float(safe_get(base, "weekend_ratio", 0.07)), step=0.01, key="wi_wr")
        wi_pcd = st.slider("PC Diversity Score", 0.0, 1.0, float(safe_get(base, "pc_diversity_score", 0.02)), step=0.01, key="wi_pcd")
        wi_dc = st.slider("Device Connections", 0, 400, int(safe_get(base, "device_connections", 10)), key="wi_dc")
        wi_upd = st.slider("Unique PCs (Device)", 0, 20, int(safe_get(base, "unique_pcs_device", 0)), key="wi_upd")
        wi_dlr = st.slider("Device/Login Ratio", 0.0, 3.0, float(safe_get(base, "device_to_login_ratio", 0.10)), step=0.01, key="wi_dlr")
        wi_asg = st.slider("Avg Session Gap (s)", 0, 200000, int(safe_get(base, "avg_session_gap", 36000)), step=100, key="wi_asg")

    wi_features = {
        "login_count": wi_lc,
        "off_hour_logins": wi_ohl,
        "weekend_logins": wi_wl,
        "late_night_logins": wi_lnl,
        "unique_pcs_logon": wi_upc,
        "off_hour_ratio": wi_ohr,
        "weekend_ratio": wi_wr,
        "pc_diversity_score": wi_pcd,
        "device_connections": wi_dc,
        "unique_pcs_device": wi_upd,
        "device_to_login_ratio": wi_dlr,
        "avg_session_gap": wi_asg,
    }

    wi_res = score_single_user_live(wi_features, model, scaler, risk_df)
    orig_risk_score = float(safe_get(base, "risk_score", 0))
    orig_risk_level = str(base.get("risk_level", "Low")) if "risk_level" in base else "Low"

    wi_lvl = wi_res["risk_level"]
    wi_clr = risk_color(wi_lvl)
    orig_clr = risk_color(orig_risk_level)
    delta = wi_res["risk_score"] - orig_risk_score

    st.markdown("---")
    st.markdown("<p class='section-header'>Live Result</p>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Original Score", f"{orig_risk_score:.1f}", delta=orig_risk_level)
    r2.metric("Modified Score", f"{wi_res['risk_score']:.1f}", delta=f"{delta:+.1f}",
              delta_color="inverse")
    r3.metric("Risk Level", wi_lvl)
    r4.metric("Anomaly", "YES" if wi_res["is_anomaly"] else "NO")

    # Sub-metrics
    dm1, dm2, dm3 = st.columns(3)
    dm1.metric("ML Score", f"{wi_res['ml_score']:.1f}")
    dm2.metric("Rule Score", f"{wi_res['rule_score']:.1f}")
    dm3.metric("Percentile", f"Top {100-wi_res['percentile']:.0f}%")

    # Flag change detection
    st.markdown("<p class='section-header'>Flag Changes</p>", unsafe_allow_html=True)
    try:
        original_explanation = str(base.get("risk_explanation", ""))
        modified_explanation = wi_res["explanation"]

        if original_explanation != modified_explanation and modified_explanation != "No significant anomalies detected.":
            st.markdown(
                f"<div class='alert-box' style='border-color:{wi_clr};'>"
                f"<strong style='color:{wi_clr};'>Active Threat Flags:</strong><br>"
                f"<span style='color:#e2e8f0;font-size:0.9rem;'>"
                f"{modified_explanation.replace(' | ', '<br>')}</span></div>",
                unsafe_allow_html=True,
            )
        elif modified_explanation == "No significant anomalies detected.":
            st.markdown(
                "<div class='alert-low'><strong style='color:#00c864;'>[CLEAR]</strong>"
                "<span style='color:#a0aec0;'>No suspicious behavioral flags detected.</span></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # Safety suggestions
    st.markdown("<p class='section-header'>How to Make This User Safe</p>", unsafe_allow_html=True)
    try:
        suggestions = get_safety_suggestions(wi_features)
        if suggestions:
            for label, change, _ in suggestions:
                st.markdown(
                    f"<div class='alert-medium'>"
                    f"<strong style='color:#ffaa00;'>Suggested:</strong> "
                    f"<span style='color:#e2e8f0;'>{label}: {change}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='alert-low'><strong style='color:#00c864;'>[OK]</strong>"
                "<span style='color:#a0aec0;'>No rule-based flags triggered. User is within normal thresholds.</span></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # Gauge
    st.markdown("<p class='section-header'>Risk Gauge</p>", unsafe_allow_html=True)
    try:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=wi_res["risk_score"],
            delta={"reference": orig_risk_score,
                   "increasing": {"color": "#ff4b4b"}, "decreasing": {"color": "#00c864"}},
            title={"text": f"{wi_user} - Modified", "font": {"color": "#a0aec0"}},
            number={"font": {"color": wi_clr, "size": 48}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#718096"},
                "bar": {"color": wi_clr},
                "bgcolor": "#1e2130",
                "bordercolor": "#2d3250",
                "steps": [
                    {"range": [0, 40], "color": "rgba(0,200,100,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,170,0,0.15)"},
                    {"range": [70, 100], "color": "rgba(255,75,75,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ff4b4b", "width": 3},
                    "thickness": 0.8,
                    "value": 70,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
            height=320, margin=dict(t=40, b=20, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, width="stretch")
    except Exception:
        pass

    # Before vs After comparison bar
    st.markdown("<p class='section-header'>Before vs After Comparison</p>", unsafe_allow_html=True)
    try:
        compare_vals = pd.DataFrame({
            "Feature": ["Logins", "Off-Hour", "Weekend", "Late Night", "Unique PCs",
                        "Devices", "Dev PCs", "Dev/Login"],
            "Original": [
                safe_get(base, "login_count", 0),
                safe_get(base, "off_hour_logins", 0),
                safe_get(base, "weekend_logins", 0),
                safe_get(base, "late_night_logins", 0),
                safe_get(base, "unique_pcs_logon", 0),
                safe_get(base, "device_connections", 0),
                safe_get(base, "unique_pcs_device", 0),
                safe_get(base, "device_to_login_ratio", 0),
            ],
            "Modified": [
                wi_lc, wi_ohl, wi_wl, wi_lnl, wi_upc, wi_dc, wi_upd, wi_dlr,
            ],
        })
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Original", x=compare_vals["Feature"],
                                  y=compare_vals["Original"], marker_color="#4a9eff"))
        fig_comp.add_trace(go.Bar(name="Modified", x=compare_vals["Feature"],
                                  y=compare_vals["Modified"], marker_color=wi_clr))
        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=320,
            xaxis=dict(gridcolor="#2d3250"), yaxis=dict(gridcolor="#2d3250"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=30, l=30, r=10),
        )
        st.plotly_chart(fig_comp, width="stretch")
    except Exception:
        st.info("Could not render comparison chart.")
