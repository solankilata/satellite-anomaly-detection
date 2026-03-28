# Satellite Telemetry Anomaly Detection

**Comparative Analysis of Statistical and Time Series Methods for Anomaly Detection in Spacecraft Subsystem Telemetry**

A term project for BSc Honours in Data Science and Artificial Intelligence — Indian Institute of Technology Guwahati — Trimester 7

---

## Project Overview

This project implements and compares two anomaly detection methods on real sensor telemetry data from the SKAB benchmark dataset. The dataset contains 44,534 sensor readings across 33 experiments with ground truth anomaly labels — physically equivalent to spacecraft subsystem sensors.

**Two methods are implemented and compared:**
- Z-score statistical threshold detection — DA206 Statistical Inferencing
- ARIMA residual detection — DA210 Time Series Analysis

Results are visualized through an interactive Streamlit dashboard.

---

## Key Results

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Z-score (threshold=0.5) | 0.2756 | 0.9998 | 0.4321 |
| ARIMA (1,1,1) | 0.4838 | 0.2743 | 0.3501 |

**Key finding:** Z-score achieves near-perfect recall — best for safety-critical systems. ARIMA achieves higher precision — best for operational systems where false alarms are costly.

---

## Repository Structure
```
satellite-anomaly-detection/
├── data/                          # Dataset files (SKAB)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data loading and EDA
│   ├── 02_preprocessing.ipynb     # Normalization and feature engineering
│   ├── 03_zscore_model.ipynb      # Z-score anomaly detection
│   ├── 04_arima_model.ipynb       # ARIMA anomaly detection
│   └── 05_evaluation.ipynb        # Comparative evaluation
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── report/                        # Charts and report figures
├── requirements.txt
└── README.md
```

---

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/solankilata/satellite-anomaly-detection.git
cd satellite-anomaly-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
cd dashboard
streamlit run app.py
```

---

## Dataset

**SKAB — Skoltech Anomaly Benchmark**
- 33 CSV files across 4 folders
- 44,534 total sensor readings
- 8 sensor channels — accelerometers, pressure, temperature, flow rate, voltage, current
- Ground truth anomaly labels
- Source: https://github.com/waico/SKAB

---

## Courses Referenced

- DA102 — Data Analysis Basics
- DA206 — Statistical Inferencing
- DA210 — Time Series Analysis and Forecasting
- DA209 — Data Modeling and Visualization

---

## Author

**Lata Solanki**
Roll Number: 23035010508
BSc Honours Data Science and AI — IIT Guwahati
s.lata@op.iitg.ac.in
