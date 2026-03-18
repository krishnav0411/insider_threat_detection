# dashboard/app.py
# FULL UPGRADE: Live attack builder, real-time alerts, what-if analyzer

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, time, joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
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
        animation: fadeIn 0.4s ease-in;
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
    .detected-box {
        background: rgba(255,75,75,0.15);
        border: 2px solid #ff4b4b;
    }
    .safe-box {
        background: rgba(0,200,100,0.12);
        border: 2px solid #00c864;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateX(-10px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    #MainMenu {visibility:hidden;}
    footer    {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_color(level):
    return {"High":"#ff4b4b","Medium":"#ffaa00","Low":"#00c864"}.get(level,"#888")

def risk_alert_class(level):
    return {"High":"alert-box","Medium":"alert-medium","Low":"alert-low"}.get(
        level,"alert-low"
    )

def data_exists():
    return all(os.path.exists(p) for p in [
        "outputs/risk_report.csv",
        "outputs/isolation_forest_model.pkl",
        "outputs/scaler.pkl"
    ])

FEATURE_COLS = [
    "login_count","off_hour_logins","weekend_logins",
    "unique_pcs_logon","off_hour_ratio","weekend_ratio",
    "device_connections","unique_pcs_device"
]

# ── Load model once (cached by Streamlit session)
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists("outputs/isolation_forest_model.pkl"):
        return None, None
    model  = joblib.load("outputs/isolation_forest_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    return model, scaler

@st.cache_data
def load_risk_report():
    if not os.path.exists("outputs/risk_report.csv"):
        return None
    df = pd.read_csv("outputs/risk_report.csv")
    if "risk_level" not in df.columns:
        from src.risk_scoring import assign_risk_level
        df["risk_level"] = df["risk_score"].apply(assign_risk_level)
    return df

@st.cache_data
def load_sim_results():
    if not os.path.exists("outputs/simulated_results.csv"):
        return None
    df = pd.read_csv("outputs/simulated_results.csv")
    if "risk_level" not in df.columns:
        from src.risk_scoring import assign_risk_level
        df["risk_level"] = df["risk_score"].apply(assign_risk_level)
    return df


# ─────────────────────────────────────────────
# LIVE SCORE FUNCTION (core of all interactivity)
# ─────────────────────────────────────────────

def score_single_user_live(features: dict, model, scaler, risk_df) -> dict:
    """
    Score a single user dictionary in real time using
    the trained model and return a full result dict.
    """
    row = pd.DataFrame([features])[FEATURE_COLS]
    X   = scaler.transform(row.values)

    raw        = model.decision_function(X)[0]
    prediction = model.predict(X)[0]

    # Normalize against real population range
    all_raw    = model.decision_function(
        scaler.transform(risk_df[FEATURE_COLS].values)
    )
    inv        = -1 * np.append(all_raw, raw)
    score_min, score_max = inv.min(), inv.max()
    risk_score = float(
        (((-1*raw) - score_min) / (score_max - score_min)) * 100
    ) if score_max > score_min else 50.0
    risk_score = round(min(max(risk_score, 0), 100), 2)

    from src.risk_scoring import assign_risk_level, generate_explanation
    level = assign_risk_level(risk_score)

    full_row = pd.Series({**features,
                          "risk_score": risk_score,
                          "is_anomaly": int(prediction == -1)})
    explanation = generate_explanation(full_row)

    # Population percentile
    percentile = float(
        (risk_df["risk_score"] < risk_score).mean() * 100
    )

    return {
        "risk_score"     : risk_score,
        "risk_level"     : level,
        "is_anomaly"     : prediction == -1,
        "explanation"    : explanation,
        "percentile"     : percentile,
        "raw_score"      : round(raw, 6)
    }


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_full_pipeline_live():
    st.markdown("---")
    prog  = st.progress(0)
    state = st.empty()
    log   = st.empty()
    lines = []

    def add_log(msg):
        lines.append(msg)
        log.markdown(
            "<div style='background:#1e2130;border:1px solid #2d3250;"
            "border-radius:8px;padding:12px;font-family:monospace;'>"
            + "".join(
                f"<div style='color:#a0aec0;font-size:0.82rem;"
                f"margin:2px 0;'>{l}</div>"
                for l in lines[-10:]
            ) + "</div>",
            unsafe_allow_html=True
        )

    try:
        state.info("Stage 1/5 — Preprocessing...")
        prog.progress(10)
        from src.preprocess import preprocess_logon, preprocess_device
        _, logon_ev  = preprocess_logon()
        _, device_ev = preprocess_device()
        add_log(f"[OK] Logon events: {len(logon_ev):,} | "
                f"Device events: {len(device_ev):,}")

        state.info("Stage 2/5 — Feature engineering...")
        prog.progress(28)
        from src.features import (extract_logon_features,
            extract_device_features, build_feature_table, save_feature_table)
        ft = build_feature_table(
            extract_logon_features(logon_ev),
            extract_device_features(device_ev)
        )
        save_feature_table(ft)
        add_log(f"[OK] Feature table: {len(ft):,} users | 8 features")

        state.info("Stage 3/5 — Training Isolation Forest...")
        prog.progress(50)
        from src.model import (scale_features, train_isolation_forest,
            generate_scores, save_model_artifacts, save_scored_results)
        Xs, sc  = scale_features(ft)
        mdl     = train_isolation_forest(Xs)
        scored  = generate_scores(mdl, Xs, ft)
        save_model_artifacts(mdl, sc)
        save_scored_results(scored)
        add_log(f"[OK] Anomalies: {scored['is_anomaly'].sum()} / {len(scored):,}")

        state.info("Stage 4/5 — Risk scoring...")
        prog.progress(72)
        from src.risk_scoring import (add_risk_levels,
            add_explanations, save_risk_report)
        rdf = add_explanations(add_risk_levels(scored))
        save_risk_report(rdf)
        h = (rdf["risk_level"]=="High").sum()
        m = (rdf["risk_level"]=="Medium").sum()
        add_log(f"[OK] High: {h} | Medium: {m} | "
                f"Low: {(rdf['risk_level']=='Low').sum()}")

        state.info("Stage 5/5 — Attack simulation...")
        prog.progress(90)
        from src.attack_simulation import run_attack_simulation
        sp, comb = run_attack_simulation()
        det = (sp["risk_level"]=="High").sum()
        add_log(f"[OK] Simulation: {det}/{len(sp)} personas detected")

        prog.progress(100)
        state.success(
            f"Pipeline complete — {len(rdf):,} users scored | "
            f"{h} High Risk | {det}/{len(sp)} threats detected"
        )
        add_log("--- DONE ---")
        st.cache_data.clear()
        st.cache_resource.clear()
        return True
    except Exception as e:
        state.error(f"Failed: {e}")
        add_log(f"[FAIL] {e}")
        return False


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;padding:1.2rem 0 0.4rem;'>
    <h1 style='color:#e2e8f0;font-size:1.9rem;font-weight:700;margin:0;'>
        AI-Driven Insider Threat Detection
    </h1>
    <p style='color:#718096;font-size:0.9rem;margin-top:0.3rem;'>
        CERT Dataset &nbsp;|&nbsp; Isolation Forest &nbsp;|&nbsp;
        Live Attack Simulation
    </p>
</div>
<hr style='border-color:#2d3250;margin-bottom:0.8rem;'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONTROL PANEL
# ─────────────────────────────────────────────

st.markdown(
    "<p class='section-header'>System Control</p>",
    unsafe_allow_html=True
)
c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    run_btn = st.button(
        "Run Full Pipeline",
        type="primary",
        use_container_width=True
    )
with c2:
    clear_btn = st.button(
        "Clear Cache & Retrain",
        use_container_width=True
    )
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
sim_df  = load_sim_results()
model, scaler = load_model_and_scaler()

if risk_df is None:
    st.markdown("""
    <div style='text-align:center;padding:5rem 0;'>
        <h2 style='color:#4a9eff;'>No Data Found</h2>
        <p style='color:#718096;'>
            Click <strong>Run Full Pipeline</strong> to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

total  = len(risk_df)
high   = (risk_df["risk_level"] == "High").sum()
medium = (risk_df["risk_level"] == "Medium").sum()
low    = (risk_df["risk_level"] == "Low").sum()

sim_only = pd.DataFrame()
if sim_df is not None and "is_simulated" in sim_df.columns:
    sim_only = sim_df[sim_df["is_simulated"] == 1].copy()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    st.markdown("---")
    show_high   = st.checkbox("High Risk",   value=True)
    show_medium = st.checkbox("Medium Risk", value=True)
    show_low    = st.checkbox("Low Risk",    value=False)
    selected_levels = (
        (["High"]    if show_high   else []) +
        (["Medium"]  if show_medium else []) +
        (["Low"]     if show_low    else [])
    )
    score_range = st.slider("Score Range", 0, 100, (0,100), step=5)
    st.markdown("---")
    st.metric("Total Users",  f"{total:,}")
    st.metric("High Risk",    high,   delta=f"{high/total*100:.1f}%")
    st.metric("Medium Risk",  medium)
    st.metric("Low Risk",     low)
    st.markdown("---")
    st.markdown(
        "<p style='color:#718096;font-size:0.75rem;'>"
        "Algorithm: Isolation Forest<br>"
        "Dataset: CERT r1<br>"
        "Features: 8 behavioral<br>"
        "Contamination: 5%</p>",
        unsafe_allow_html=True
    )

filtered_df = risk_df[
    (risk_df["risk_level"].isin(selected_levels)) &
    (risk_df["risk_score"] >= score_range[0]) &
    (risk_df["risk_score"] <= score_range[1])
].copy()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Risk Analysis",
    "User Investigation",
    "Live Attack Simulation",
    "What-If Analyzer"
])


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab1:
    st.markdown(
        "<p class='section-header'>Security Posture Overview</p>",
        unsafe_allow_html=True
    )
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Users Monitored",  f"{total:,}")
    c2.metric("High Risk",        high,
              delta=f"{high/total*100:.1f}%", delta_color="inverse")
    c3.metric("Medium Risk",      medium,
              delta=f"{medium/total*100:.1f}%", delta_color="off")
    c4.metric("Avg Risk Score",   f"{risk_df['risk_score'].mean():.1f}")
    c5.metric("Peak Risk Score",  f"{risk_df['risk_score'].max():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1,2])

    with col_l:
        st.markdown(
            "<p class='section-header'>Risk Distribution</p>",
            unsafe_allow_html=True
        )
        rc = risk_df["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk Level","Count"]
        fig_pie = px.pie(
            rc, values="Count", names="Risk Level",
            color="Risk Level",
            color_discrete_map={
                "High":"#ff4b4b","Medium":"#ffaa00","Low":"#00c864"
            },
            hole=0.5
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=280,
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown(
            "<p class='section-header'>Risk Score Distribution</p>",
            unsafe_allow_html=True
        )
        fig_hist = px.histogram(
            risk_df, x="risk_score", nbins=40,
            color_discrete_sequence=["#4a9eff"]
        )
        for val, lbl, clr in [
            (70,"High Threshold","#ff4b4b"),
            (40,"Medium Threshold","#ffaa00")
        ]:
            fig_hist.add_vline(
                x=val, line_dash="dash", line_color=clr,
                annotation_text=lbl, annotation_font_color=clr
            )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=280,
            xaxis=dict(gridcolor="#2d3250"),
            yaxis=dict(gridcolor="#2d3250"),
            margin=dict(t=10,b=30,l=30,r=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        "<p class='section-header'>Active High Risk Alerts</p>",
        unsafe_allow_html=True
    )
    high_users = risk_df[risk_df["risk_level"]=="High"] \
                     .sort_values("risk_score",ascending=False).head(10)
    for _, row in high_users.iterrows():
        st.markdown(f"""
        <div class='alert-box'>
            <strong style='color:#ff4b4b;'>[ALERT]</strong>
            <strong style='color:#e2e8f0;'> {row['user']}</strong>
            <span style='color:#718096;'> — Score: </span>
            <strong style='color:#ff4b4b;'>{row['risk_score']:.1f}</strong>
            <span style='color:#718096;'> | </span>
            <span style='color:#a0aec0;font-size:0.88rem;'>
                {row['risk_explanation']}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════

with tab2:
    st.markdown(
        "<p class='section-header'>Behavioral Scatter Analysis</p>",
        unsafe_allow_html=True
    )
    col1,col2 = st.columns(2)
    for (col, xc, yc, ttl) in [
        (col1,"login_count","off_hour_logins",
         "Login Count vs Off-Hour Logins"),
        (col2,"device_connections","unique_pcs_device",
         "Device Connections vs Unique PCs")
    ]:
        with col:
            fig = px.scatter(
                risk_df, x=xc, y=yc,
                color="risk_level", size="risk_score",
                hover_data=["user","risk_score"],
                color_discrete_map={
                    "High":"#ff4b4b","Medium":"#ffaa00","Low":"#00c864"
                },
                title=ttl
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", height=360,
                xaxis=dict(gridcolor="#2d3250"),
                yaxis=dict(gridcolor="#2d3250")
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<p class='section-header'>Feature Correlation Heatmap</p>",
        unsafe_allow_html=True
    )
    fcols = FEATURE_COLS + ["risk_score"]
    corr  = risk_df[fcols].corr().round(2)
    fig_hm = px.imshow(
        corr, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, text_auto=True, aspect="auto"
    )
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=400,
        margin=dict(t=10,b=10,l=10,r=10)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown(
        "<p class='section-header'>Top 20 by Risk Score</p>",
        unsafe_allow_html=True
    )
    fig_bar = px.bar(
        risk_df.nlargest(20,"risk_score"),
        x="risk_score", y="user", orientation="h",
        color="risk_level",
        color_discrete_map={
            "High":"#ff4b4b","Medium":"#ffaa00","Low":"#00c864"
        },
        hover_data=["login_count","off_hour_logins","device_connections"]
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=480,
        yaxis=dict(autorange="reversed",gridcolor="#2d3250"),
        xaxis=dict(gridcolor="#2d3250")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        "<p class='section-header'>Full Risk Table</p>",
        unsafe_allow_html=True
    )
    dcols = [c for c in [
        "user","risk_level","risk_score","login_count",
        "off_hour_logins","weekend_logins","unique_pcs_logon",
        "device_connections","off_hour_ratio","risk_explanation"
    ] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[dcols].sort_values(
            "risk_score",ascending=False
        ).reset_index(drop=True),
        use_container_width=True, height=380,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%.1f"
            )
        }
    )
    st.caption(f"Showing {len(filtered_df):,} users.")


# ══════════════════════════════════════════════
# TAB 3 — USER INVESTIGATION
# ══════════════════════════════════════════════

with tab3:
    st.markdown(
        "<p class='section-header'>Individual User Deep Dive</p>",
        unsafe_allow_html=True
    )
    selected_user = st.selectbox(
        "Select user",
        options=risk_df.sort_values(
            "risk_score", ascending=False
        )["user"].tolist(),
        index=0
    )
    user_row = risk_df[risk_df["user"]==selected_user].iloc[0]
    level    = user_row["risk_level"]
    color    = risk_color(level)

    st.markdown(f"""
    <div style='background:#1e2130;border:2px solid {color};
                border-radius:12px;padding:20px;margin:10px 0;'>
        <h2 style='color:{color};margin:0;'>{selected_user}</h2>
        <p style='color:#718096;margin:6px 0;'>
            Risk Level: <strong style='color:{color}'>{level}</strong>
            &nbsp;|&nbsp;
            Score: <strong style='color:{color}'>
                {user_row['risk_score']:.1f}/100
            </strong>
            &nbsp;|&nbsp;
            Anomaly: <strong style='color:{color}'>
                {'Yes' if user_row['is_anomaly'] else 'No'}
            </strong>
        </p>
        <p style='color:#a0aec0;font-size:0.88rem;margin-top:8px;'>
            {user_row['risk_explanation']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Logins",      int(user_row["login_count"]))
    m2.metric("Off-Hour Logins",   int(user_row["off_hour_logins"]),
              delta=f"{user_row['off_hour_ratio']*100:.0f}%")
    m3.metric("Weekend Logins",    int(user_row["weekend_logins"]),
              delta=f"{user_row['weekend_ratio']*100:.0f}%")
    m4.metric("Unique PCs",        int(user_row["unique_pcs_logon"]))

    m5,m6,m7,m8 = st.columns(4)
    m5.metric("Device Connections", int(user_row["device_connections"]))
    m6.metric("Device PCs",         int(user_row["unique_pcs_device"]))
    m7.metric("Off-Hour Ratio",     f"{user_row['off_hour_ratio']*100:.1f}%")
    m8.metric("Weekend Ratio",      f"{user_row['weekend_ratio']*100:.1f}%")

    # Radar
    radar_f = [
        "login_count","off_hour_logins","weekend_logins",
        "unique_pcs_logon","device_connections","unique_pcs_device"
    ]
    radar_l = ["Logins","Off-Hour","Weekend","PCs","Devices","Dev PCs"]
    fmax    = risk_df[radar_f].max()
    uvals   = [user_row[f]/fmax[f] if fmax[f]>0 else 0 for f in radar_f]
    avals   = [risk_df[f].mean()/fmax[f] if fmax[f]>0 else 0 for f in radar_f]
    fig_r   = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=uvals+[uvals[0]], theta=radar_l+[radar_l[0]],
        fill="toself", name=selected_user, line_color=color
    ))
    fig_r.add_trace(go.Scatterpolar(
        r=avals+[avals[0]], theta=radar_l+[radar_l[0]],
        fill="toself", name="Avg User", line_color="#4a9eff",
        fillcolor="rgba(74,158,255,0.1)"
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0,1],
                gridcolor="#2d3250", color="#718096"
            ),
            angularaxis=dict(color="#a0aec0")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=400, showlegend=True,
        margin=dict(t=30,b=30,l=30,r=30)
    )
    st.plotly_chart(fig_r, use_container_width=True)

    rank = int(risk_df["risk_score"].rank(
        ascending=False, method="min"
    )[risk_df["user"]==selected_user].values[0])
    st.info(
        f"**{selected_user}** ranks **#{rank}** out of {total:,} users."
    )


# ══════════════════════════════════════════════
# TAB 4 — LIVE ATTACK SIMULATION
# ══════════════════════════════════════════════

with tab4:

    st.markdown("""
    <div style='background:#1e2130;border:1px solid #2d3250;
                border-radius:10px;padding:16px;margin-bottom:1rem;'>
        <h4 style='color:#e2e8f0;margin:0 0 6px 0;'>
            Custom Attack Builder
        </h4>
        <p style='color:#718096;margin:0;font-size:0.88rem;'>
            Design a threat actor using the sliders below.
            Hit <strong style='color:#4a9eff;'>Launch Attack</strong>
            to score them live against the trained model.
            Watch alerts fire in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: sliders | presets
    sl_col, pre_col = st.columns([3,1])

    with pre_col:
        st.markdown(
            "<p class='section-header'>Quick Presets</p>",
            unsafe_allow_html=True
        )
        preset = st.radio(
            "Load a preset",
            ["Custom","Night Owl","PC Hopper",
             "Data Mule","Ghost User","Full Threat"],
            index=0
        )

    # Preset values
    presets = {
        "Custom"     : dict(lc=40, ohl=5,  wl=3,  upc=1,
                            dc=10, upd=1,  ohr=0.12, wr=0.07),
        "Night Owl"  : dict(lc=95, ohl=88, wl=12, upc=2,
                            dc=8,  upd=1,  ohr=0.93, wr=0.13),
        "PC Hopper"  : dict(lc=120,ohl=18, wl=14, upc=14,
                            dc=12, upd=11, ohr=0.15, wr=0.12),
        "Data Mule"  : dict(lc=55, ohl=22, wl=19, upc=4,
                            dc=112,upd=9,  ohr=0.40, wr=0.35),
        "Ghost User" : dict(lc=18, ohl=16, wl=10, upc=5,
                            dc=4,  upd=4,  ohr=0.89, wr=0.56),
        "Full Threat": dict(lc=210,ohl=175,wl=68, upc=18,
                            dc=145,upd=15, ohr=0.83, wr=0.32),
    }
    pv = presets[preset]

    with sl_col:
        st.markdown(
            "<p class='section-header'>Behavior Parameters</p>",
            unsafe_allow_html=True
        )

        s1, s2 = st.columns(2)
        with s1:
            login_count = st.slider(
                "Total Logins",
                0, 250, pv["lc"], step=1,
                help="How many times does this user log in?"
            )
            off_hour_logins = st.slider(
                "Off-Hour Logins (outside 8AM-6PM)",
                0, 200, pv["ohl"], step=1,
                help="Logins before 8AM or after 6PM"
            )
            weekend_logins = st.slider(
                "Weekend Logins",
                0, 100, pv["wl"], step=1
            )
            unique_pcs_logon = st.slider(
                "Unique PCs Logged Into",
                1, 20, pv["upc"], step=1,
                help="How many different machines?"
            )
        with s2:
            device_connections = st.slider(
                "Device Connections",
                0, 200, pv["dc"], step=1,
                help="USB / external device connect events"
            )
            unique_pcs_device = st.slider(
                "Unique PCs for Device Connections",
                0, 20, pv["upd"], step=1
            )
            off_hour_ratio = st.slider(
                "Off-Hour Ratio (0.0 - 1.0)",
                0.0, 1.0, pv["ohr"], step=0.01,
                help="Fraction of logins that are off-hours"
            )
            weekend_ratio = st.slider(
                "Weekend Ratio (0.0 - 1.0)",
                0.0, 1.0, pv["wr"], step=0.01
            )

    # ── Attacker name input
    attacker_name = st.text_input(
        "Attacker Label (optional)",
        value=f"LIVE_{preset.replace(' ','_').upper()}"
        if preset != "Custom" else "LIVE_CUSTOM_ATTACKER"
    )

    st.markdown("---")

    # ── LAUNCH BUTTON
    launch_btn = st.button(
        "Launch Attack",
        type="primary",
        use_container_width=True
    )

    if launch_btn:
        if model is None or scaler is None:
            st.error(
                "No trained model found. "
                "Run the full pipeline first."
            )
        else:
            features = {
                "login_count"        : login_count,
                "off_hour_logins"    : off_hour_logins,
                "weekend_logins"     : weekend_logins,
                "unique_pcs_logon"   : unique_pcs_logon,
                "off_hour_ratio"     : off_hour_ratio,
                "weekend_ratio"      : weekend_ratio,
                "device_connections" : device_connections,
                "unique_pcs_device"  : unique_pcs_device
            }

            # ── Live progress animation
            result_placeholder = st.empty()
            prog_placeholder   = st.empty()
            alert_placeholder  = st.empty()

            steps = [
                ("Injecting user into system...",          0.20),
                ("Normalizing behavioral features...",     0.40),
                ("Running Isolation Forest scoring...",    0.65),
                ("Calculating risk level...",              0.85),
                ("Generating alert...",                    1.00),
            ]

            prog = prog_placeholder.progress(0.0)
            for msg, val in steps:
                result_placeholder.info(f"Processing: {msg}")
                prog.progress(val)
                time.sleep(0.4)

            prog_placeholder.empty()
            result_placeholder.empty()

            # ── Score the attacker live
            result = score_single_user_live(
                features, model, scaler, risk_df
            )

            lvl   = result["risk_level"]
            score = result["risk_score"]
            clr   = risk_color(lvl)
            box_cls = "detected-box" if lvl == "High" else (
                "safe-box" if lvl == "Low" else ""
            )
            verdict = (
                "THREAT DETECTED"   if lvl == "High"   else
                "SUSPICIOUS ACTIVITY" if lvl == "Medium" else
                "NO THREAT DETECTED"
            )
            verdict_icon = (
                "[ALERT]" if lvl == "High" else
                "[WARN]"  if lvl == "Medium" else
                "[CLEAR]"
            )

            # ── Flash alert banner
            alert_placeholder.markdown(f"""
            <div class='live-result-box {box_cls}'
                 style='border-color:{clr};
                        background:rgba(0,0,0,0.3);'>
                <h1 style='color:{clr};font-size:2.5rem;margin:0;'>
                    {verdict_icon}
                </h1>
                <h2 style='color:{clr};margin:8px 0;'>
                    {verdict}
                </h2>
                <h3 style='color:#e2e8f0;margin:0;'>
                    {attacker_name}
                </h3>
                <p style='color:{clr};font-size:2rem;
                          font-weight:700;margin:12px 0 4px;'>
                    Risk Score: {score:.1f} / 100
                </p>
                <p style='color:#a0aec0;font-size:0.9rem;margin:0;'>
                    Risk Level: <strong style='color:{clr};'>{lvl}</strong>
                    &nbsp;|&nbsp;
                    Anomaly: <strong style='color:{clr};'>
                        {'Yes' if result['is_anomaly'] else 'No'}
                    </strong>
                    &nbsp;|&nbsp;
                    Riskier than {result['percentile']:.0f}% of users
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Detailed breakdown
            st.markdown("---")
            st.markdown(
                "<p class='section-header'>Detection Breakdown</p>",
                unsafe_allow_html=True
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Risk Score",    f"{score:.1f} / 100")
            d2.metric("Risk Level",    lvl)
            d3.metric("Anomaly Flag",  "YES" if result["is_anomaly"] else "NO")
            d4.metric("Percentile",    f"Top {100-result['percentile']:.0f}%")

            # ── Explanation flags
            if result["explanation"] != "No significant anomalies detected.":
                st.markdown(f"""
                <div class='alert-box' style='border-color:{clr};'>
                    <strong style='color:{clr};'>Triggered Flags:</strong>
                    <br>
                    <span style='color:#e2e8f0;font-size:0.9rem;'>
                        {result['explanation'].replace(' | ','<br>• ')}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── Where does this attacker sit vs population?
            st.markdown(
                "<p class='section-header'>"
                "Attacker vs Real Population</p>",
                unsafe_allow_html=True
            )

            attacker_row = pd.DataFrame([{
                "user"        : attacker_name,
                "risk_score"  : score,
                "risk_level"  : lvl,
                "login_count" : login_count,
                "off_hour_logins": off_hour_logins,
                "is_simulated": 1
            }])
            pop_sample = risk_df.sample(
                min(300, len(risk_df)), random_state=42
            ).copy()
            pop_sample["is_simulated"] = 0
            plot_df = pd.concat([pop_sample, attacker_row],
                                ignore_index=True)
            plot_df["Type"] = plot_df["is_simulated"].map(
                {0:"Real User", 1:"Live Attacker"}
            )

            fig_pop = px.scatter(
                plot_df,
                x="login_count", y="off_hour_logins",
                color="Type", size="risk_score",
                hover_data=["user","risk_score","risk_level"],
                color_discrete_map={
                    "Real User"    : "#4a9eff",
                    "Live Attacker": "#ff4b4b"
                },
                title="Live Attacker Position in User Population"
            )
            fig_pop.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", height=400,
                xaxis=dict(gridcolor="#2d3250"),
                yaxis=dict(gridcolor="#2d3250")
            )
            st.plotly_chart(fig_pop, use_container_width=True)

    # ── Batch Simulation with Live Alert Feed
    st.markdown("---")
    st.markdown(
        "<p class='section-header'>Batch Simulation — Live Alert Feed</p>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <p style='color:#718096;font-size:0.85rem;margin-bottom:10px;'>
        Runs all 5 built-in threat personas through the model
        one-by-one. Alerts appear live as each is scored.
    </p>
    """, unsafe_allow_html=True)

    batch_btn = st.button(
        "Run Batch Simulation (All 5 Personas)",
        use_container_width=True
    )

    if batch_btn:
        if model is None or scaler is None:
            st.error("Run the full pipeline first.")
        else:
            from src.attack_simulation import build_attack_personas
            personas_df  = build_attack_personas()
            alert_feed   = st.empty()
            alert_lines  = []
            batch_prog   = st.progress(0)
            batch_status = st.empty()

            all_results = []

            for i, (_, prow) in enumerate(personas_df.iterrows()):
                batch_status.info(
                    f"Scoring {prow['user']} "
                    f"({i+1}/{len(personas_df)})..."
                )
                batch_prog.progress((i+1)/len(personas_df))
                time.sleep(0.6)

                feats = {c: prow[c] for c in FEATURE_COLS}
                res   = score_single_user_live(
                    feats, model, scaler, risk_df
                )
                all_results.append({"user": prow["user"], **res})
                lvl   = res["risk_level"]
                clr   = risk_color(lvl)
                icon  = (
                    "[ALERT]" if lvl=="High" else
                    "[WARN]"  if lvl=="Medium" else
                    "[OK]"
                )
                alert_lines.append(
                    f"<div class='{risk_alert_class(lvl)}'>"
                    f"<strong style='color:{clr};'>{icon}</strong>"
                    f" <strong style='color:#e2e8f0;'>{prow['user']}</strong>"
                    f"<span style='color:#718096;'> — Score: </span>"
                    f"<strong style='color:{clr};'>"
                    f"{res['risk_score']:.1f}</strong>"
                    f"<span style='color:#718096;'> | Level: </span>"
                    f"<strong style='color:{clr};'>{lvl}</strong>"
                    f"<br><span style='color:#a0aec0;"
                    f"font-size:0.82rem;'>"
                    f"{res['explanation']}</span></div>"
                )
                alert_feed.markdown(
                    "".join(alert_lines),
                    unsafe_allow_html=True
                )

            batch_prog.progress(1.0)
            detected = sum(
                1 for r in all_results if r["risk_level"]=="High"
            )
            batch_status.success(
                f"Batch complete — "
                f"{detected}/{len(all_results)} personas detected "
                f"as High Risk"
            )

            # Summary table
            st.markdown(
                "<p class='section-header'>Batch Results Summary</p>",
                unsafe_allow_html=True
            )
            batch_df = pd.DataFrame(all_results)[[
                "user","risk_score","risk_level","is_anomaly","percentile"
            ]]
            batch_df["is_anomaly"] = batch_df["is_anomaly"].map(
                {True:"Yes", False:"No"}
            )
            batch_df["percentile"] = batch_df["percentile"].apply(
                lambda x: f"Top {100-x:.0f}%"
            )
            st.dataframe(
                batch_df.sort_values(
                    "risk_score", ascending=False
                ).reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score", min_value=0,
                        max_value=100, format="%.1f"
                    )
                }
            )


# ══════════════════════════════════════════════
# TAB 5 — WHAT-IF ANALYZER
# ══════════════════════════════════════════════

with tab5:

    st.markdown("""
    <div style='background:#1e2130;border:1px solid #2d3250;
                border-radius:10px;padding:16px;margin-bottom:1rem;'>
        <h4 style='color:#e2e8f0;margin:0 0 6px 0;'>
            What-If Analyzer
        </h4>
        <p style='color:#718096;margin:0;font-size:0.88rem;'>
            Pick any real user. Modify their behavior using sliders.
            See their risk score update <strong style='color:#4a9eff;'>
            instantly</strong> — no button needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Run the full pipeline first.")
        st.stop()

    # ── Pick a real user to modify
    wi_user = st.selectbox(
        "Base user (start from their real values)",
        options=risk_df.sort_values(
            "risk_score", ascending=False
        )["user"].tolist(),
        index=0,
        key="whatif_user"
    )

    base = risk_df[risk_df["user"]==wi_user].iloc[0]

    st.markdown(
        "<p class='section-header'>Modify Behavior</p>",
        unsafe_allow_html=True
    )

    wa, wb = st.columns(2)

    with wa:
        wi_lc  = st.slider("Login Count",
            0, 250, int(base["login_count"]),   key="wi_lc")
        wi_ohl = st.slider("Off-Hour Logins",
            0, 200, int(base["off_hour_logins"]),key="wi_ohl")
        wi_wl  = st.slider("Weekend Logins",
            0, 100, int(base["weekend_logins"]), key="wi_wl")
        wi_upc = st.slider("Unique PCs (Logon)",
            1, 20,  int(base["unique_pcs_logon"]),key="wi_upc")
    with wb:
        wi_dc  = st.slider("Device Connections",
            0, 200, int(base["device_connections"]),key="wi_dc")
        wi_upd = st.slider("Unique PCs (Device)",
            0, 20,  int(base["unique_pcs_device"]), key="wi_upd")
        wi_ohr = st.slider("Off-Hour Ratio",
            0.0, 1.0, float(round(base["off_hour_ratio"],2)),
            step=0.01, key="wi_ohr")
        wi_wr  = st.slider("Weekend Ratio",
            0.0, 1.0, float(round(base["weekend_ratio"],2)),
            step=0.01, key="wi_wr")

    # ── LIVE score — recalculates on every slider move
    wi_features = {
        "login_count"        : wi_lc,
        "off_hour_logins"    : wi_ohl,
        "weekend_logins"     : wi_wl,
        "unique_pcs_logon"   : wi_upc,
        "off_hour_ratio"     : wi_ohr,
        "weekend_ratio"      : wi_wr,
        "device_connections" : wi_dc,
        "unique_pcs_device"  : wi_upd
    }
    wi_res   = score_single_user_live(wi_features, model, scaler, risk_df)
    orig_res = {
        "risk_score" : float(base["risk_score"]),
        "risk_level" : str(base["risk_level"])
    }

    wi_lvl   = wi_res["risk_level"]
    wi_clr   = risk_color(wi_lvl)
    orig_clr = risk_color(orig_res["risk_level"])
    delta    = wi_res["risk_score"] - orig_res["risk_score"]

    st.markdown("---")
    st.markdown(
        "<p class='section-header'>Live Result</p>",
        unsafe_allow_html=True
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Original Score",
        f"{orig_res['risk_score']:.1f}",
        delta=orig_res["risk_level"]
    )
    r2.metric(
        "Modified Score",
        f"{wi_res['risk_score']:.1f}",
        delta=f"{delta:+.1f}",
        delta_color="inverse"
    )
    r3.metric("Risk Level",   wi_lvl)
    r4.metric(
        "Anomaly",
        "YES" if wi_res["is_anomaly"] else "NO"
    )

    # ── Live risk gauge
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = wi_res["risk_score"],
        delta = {
            "reference" : orig_res["risk_score"],
            "increasing": {"color":"#ff4b4b"},
            "decreasing": {"color":"#00c864"}
        },
        title = {"text": f"{wi_user} — Modified Risk Score",
                 "font": {"color":"#a0aec0"}},
        number= {"font":{"color": wi_clr, "size":48}},
        gauge = {
            "axis" : {"range":[0,100], "tickcolor":"#718096"},
            "bar"  : {"color": wi_clr},
            "bgcolor": "#1e2130",
            "bordercolor":"#2d3250",
            "steps": [
                {"range":[0,40],  "color":"rgba(0,200,100,0.15)"},
                {"range":[40,70], "color":"rgba(255,170,0,0.15)"},
                {"range":[70,100],"color":"rgba(255,75,75,0.15)"}
            ],
            "threshold": {
                "line" :{"color":"#ff4b4b","width":3},
                "thickness":0.8,
                "value": 70
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0",
        height=320,
        margin=dict(t=40,b=20,l=30,r=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Explanation
    if wi_res["explanation"] != "No significant anomalies detected.":
        st.markdown(f"""
        <div class='alert-box' style='border-color:{wi_clr};'>
            <strong style='color:{wi_clr};'>Active Threat Flags:</strong>
            <br>
            <span style='color:#e2e8f0;font-size:0.88rem;'>
                • {wi_res['explanation'].replace(' | ','<br>• ')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='alert-low'>
            <strong style='color:#00c864;'>[CLEAR]</strong>
            <span style='color:#a0aec0;'>
                No suspicious behavioral flags detected
                with current settings.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Before vs After comparison bar
    st.markdown(
        "<p class='section-header'>Before vs After Comparison</p>",
        unsafe_allow_html=True
    )
    compare_vals = pd.DataFrame({
        "Feature" : [
            "Logins","Off-Hour Logins","Weekend Logins",
            "Unique PCs","Device Conn.","Device PCs"
        ],
        "Original": [
            base["login_count"], base["off_hour_logins"],
            base["weekend_logins"], base["unique_pcs_logon"],
            base["device_connections"], base["unique_pcs_device"]
        ],
        "Modified": [
            wi_lc, wi_ohl, wi_wl,
            wi_upc, wi_dc, wi_upd
        ]
    })
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Original", x=compare_vals["Feature"],
        y=compare_vals["Original"],
        marker_color="#4a9eff"
    ))
    fig_comp.add_trace(go.Bar(
        name="Modified", x=compare_vals["Feature"],
        y=compare_vals["Modified"],
        marker_color=wi_clr
    ))
    fig_comp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0",
        height=320,
        xaxis=dict(gridcolor="#2d3250"),
        yaxis=dict(gridcolor="#2d3250"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10,b=30,l=30,r=10)
    )
    st.plotly_chart(fig_comp, use_container_width=True)git init
    