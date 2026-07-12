# Insider Threat Detection System

An AI-driven system that detects anomalous user behavior in enterprise environments using the CERT r1 insider-threat dataset. The system uses an **Isolation Forest** model combined with **rule-based scoring** to identify suspicious activities such as off-hours access, lateral movement, and data exfiltration.

---

## Architecture

```
data/logon.csv ──┐
                 │
data/device.csv ┐├──> preprocess.py ──> features.py ──> model.py ──> risk_scoring.py
                ││     (clean +         (12 per-user    (Isolation      (risk levels +
                ││      time features)   features)       Forest +        explanations)
                ││                                        scores)
                ││                                          │
                ││                                   attack_simulation.py
                ││                                   (7 synthetic personas)
                ││                                          │
                ││                                   run_pipeline.py
                ││                                   (orchestrates all)
                ││                                          │
                ││                                   dashboard/app.py
                ││                                   (Streamlit UI)
                │└──────────────────────────────────────────┘
                └──────────────────────────────────────────┘

Output files (outputs/):
  user_features.csv          — 1,000 users × 12 features
  user_scores.csv            — ML anomaly scores (0-100)
  risk_report.csv            — Risk levels + explanations
  simulated_results.csv      — Real + simulated users combined
  simulation_report.txt      — Human-readable persona report
  model_metadata.json        — Training params and metrics
  isolation_forest_model.pkl — Trained sklearn model
  scaler.pkl                 — Fitted StandardScaler
```

---

## Setup

### Prerequisites
- Python 3.10+
- [CERT r1 Insider Threat Dataset](https://www.cmu.edu/cer/insider-threat-dataset/) — place `logon.csv` and `device.csv` in `data/`

### Installation

```bash
# Clone / navigate to the project
cd insider-threat-detection

# Install dependencies
pip install -r requirements.txt

# Verify data files exist
ls data/
# logon.csv    device.csv
```

---

## Running the System

### 1. Full Pipeline (recommended)

```bash
python run_pipeline.py
```

Runs all stages: preprocessing -> features -> model -> risk scoring -> attack simulation -> validation. Takes ~40 seconds. All outputs land in `outputs/`.

### 2. Individual Stages

```bash
python src/features.py        # Stage 3: feature engineering only
python src/model.py           # Stage 4: train model + score
python src/risk_scoring.py    # Stage 5: risk levels + explanations
python src/attack_simulation.py  # Stage 6: simulate attacks
```

### 3. Verify Model

```bash
python verify_model.py
```

Generates a comprehensive verification report: population statistics, feature quality, confusion matrix estimate, top anomalous users, quality checks, score distribution.

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Opens a browser UI with 5 tabs: Overview, Risk Analysis, User Investigation, Live Attack Simulation, What-If Analyzer.

---

## 12 Behavioral Features

| # | Feature | Description | Why It Matters |
|---|---------|-------------|----------------|
| 1 | `login_count` | Total login events per user | Baseline activity volume |
| 2 | `off_hour_logins` | Logins outside 8AM-6PM | Unauthorized after-hours access |
| 3 | `weekend_logins` | Logins on Saturday or Sunday | Unusual weekend activity |
| 4 | `late_night_logins` | Logins between 10PM-4AM | Highly suspicious time window |
| 5 | `unique_pcs_logon` | Distinct PCs a user logs into | Lateral movement indicator |
| 6 | `off_hour_ratio` | Off-hour logins / total | Proportion of abnormal timing |
| 7 | `weekend_ratio` | Weekend logins / total | Proportion of weekend activity |
| 8 | `pc_diversity_score` | Unique PCs / logins | How spread out across machines |
| 9 | `device_connections` | USB/device connect events | Data exfiltration via removable media |
| 10 | `unique_pcs_device` | Distinct PCs for device connects | Device lateral movement |
| 11 | `device_to_login_ratio` | Device connects / logins | Exfiltration intensity indicator |
| 12 | `avg_session_gap` | Mean seconds between logins | Compressed sessions = suspicious |

---

## Risk Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| **Low** | 0-39 | Normal behavioral patterns |
| **Medium** | 40-69 | Some anomalous signals detected; investigate |
| **High** | 70-100 | Strong anomaly indicators; immediate attention needed |

---

## Scoring System

The system uses **two independent scoring methods** combined into a composite:

- **ML Score** (60% weight): Isolation Forest detects statistical outliers across all 12 features. The contamination parameter is auto-selected (typically 0.03) by maximising separation between flagged and normal users.

- **Rule Score** (40% weight): 12 rule-based checks against configured thresholds (e.g. `unique_pcs_logon >= 3` = +20 points). Each rule has a different weight. Raw sum is normalised to 0-100.

---

## Attack Personas

7 synthetic insider threat personas are injected to validate detection:

| Persona | Pattern | Expected Detection |
|---------|---------|-------------------|
| Night Owl | 93% off-hours, midnight access | Medium (rule score low) |
| PC Hopper | 14 PCs, lateral movement | Medium (rule score low) |
| Data Mule | 112 device connections, ratio 2.04 | High |
| Ghost User | Minimal logins, 89% off-hours | Medium (stealth) |
| Full Threat | All signals maxed | High (89.0 composite) |
| Email Thief | 45s session gaps, 6 PCs | High |
| Saboteur | 1.60 dev/login ratio, 3AM | High (88.0 composite) |

---

## Interpreting Results

### Key Metrics

- **Score Separation**: >0.15 means the model confidently distinguishes anomalies from normal users. Current system achieves 0.2878.
- **Anomaly Rate**: Matches the auto-selected contamination (typically 3%).
- **Feature Importance**: Correlation between each feature and the final risk score. Top features: PC Diversity Score (0.685), Unique PCs Device (0.667), Avg Session Gap (0.655).

### Typical Anomalous Users

The most high-risk users (e.g. DTAA/DSM0990, score 100.0) show extreme patterns:
- 2,000+ logins (vs 470 avg)
- 850+ unique PCs (vs 22 avg)
- 500+ late-night logins (vs 8 avg)

These are likely compromised credentials being used for lateral movement across the network.

---

## Project Structure

```
├── data/
│   ├── logon.csv              # CERT r1 logon events
│   └── device.csv             # CERT r1 device events
├── src/
│   ├── preprocess.py          # Stage 2: data cleaning
│   ├── features.py            # Stage 3: 12 feature extraction
│   ├── model.py               # Stage 4: Isolation Forest training
│   ├── risk_scoring.py        # Stage 5: risk levels & explanations
│   └── attack_simulation.py   # Stage 6: persona injection & scoring
├── dashboard/
│   └── app.py                 # Streamlit dashboard (5 tabs)
├── outputs/                   # Generated artifacts (see above)
├── run_pipeline.py            # Stage 9: end-to-end orchestrator
├── verify_model.py            # Verification & validation report
├── requirements.txt
├── README.md
```

---

## Configuration

Key thresholds in `src/risk_scoring.py` (`CONFIG` dict):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `threshold_high` | 70.0 | Risk score for High level |
| `threshold_medium` | 40.0 | Risk score for Medium level |
| `flag_login_count` | 80 | Excessive login threshold |
| `flag_off_hour_logins` | 15 | Off-hour alert threshold |
| `flag_device_connections` | 30 | Device connection alert |
| `flag_late_night_logins` | 10 | Late-night alert threshold |

Contamination candidates in `src/model.py`: `[0.03, 0.05, 0.08, 0.10]`
