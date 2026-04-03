import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Satellite Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
    color: #ffffff;
}

.stApp { background-color: #0d0d0d; }

section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222222;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }

div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333333 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

@keyframes orbit {
    0% { transform: rotate(0deg) translateX(28px) rotate(0deg); }
    100% { transform: rotate(360deg) translateX(28px) rotate(-360deg); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.7); }
}

.header-container {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 20px 0 24px 0;
    border-bottom: 1px solid #222222;
    margin-bottom: 28px;
}
.satellite-container {
    position: relative;
    width: 80px;
    height: 80px;
    flex-shrink: 0;
}
.orbit-ring {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 60px;
    height: 60px;
    border: 1px solid #333333;
    border-radius: 50%;
    transform: translate(-50%, -50%);
}
.planet {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 18px;
    height: 18px;
    background: #3b82f6;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 12px rgba(59,130,246,0.5);
}
.satellite {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 10px;
    height: 10px;
    margin-top: -5px;
    margin-left: -5px;
    animation: orbit 3s linear infinite;
    transform-origin: 0 0;
}
.satellite-body {
    width: 10px;
    height: 10px;
    background: #ffffff;
    border-radius: 2px;
    position: relative;
}
.satellite-body::before,
.satellite-body::after {
    content: '';
    position: absolute;
    top: 50%;
    width: 7px;
    height: 3px;
    background: #93c5fd;
    transform: translateY(-50%);
}
.satellite-body::before { right: 10px; }
.satellite-body::after { left: 10px; }

.header-text { flex: 1; }
.header-title {
    font-size: 30px;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 500;
    color: #059669;
    background: rgba(5,150,105,0.1);
    border: 1px solid rgba(5,150,105,0.3);
    border-radius: 20px;
    padding: 3px 10px;
}
.live-dot {
    width: 6px;
    height: 6px;
    background: #059669;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
    display: inline-block;
}
.header-sub {
    font-size: 13px;
    color: #888888;
    font-weight: 400;
}

.metric-card {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 10px;
    padding: 18px 20px;
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    color: #666666;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 26px;
    font-weight: 600;
    color: #ffffff;
    line-height: 1;
}
.metric-value.green { color: #10b981; }
.metric-value.blue { color: #3b82f6; }
.metric-value.amber { color: #f59e0b; }

.section-title {
    font-size: 11px;
    font-weight: 600;
    color: #555555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f1f1f;
}

.method-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 24px;
}
.method-card {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 10px;
    padding: 20px;
}
.method-title {
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1f1f1f;
    display: flex;
    align-items: center;
    gap: 8px;
}
.course-tag {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
}
.course-tag.blue {
    background: rgba(59,130,246,0.15);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.2);
}
.course-tag.green {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.2);
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1a1a1a;
}
.stat-row:last-child { border-bottom: none; }
.stat-name { font-size: 12px; color: #666666; }
.stat-val { font-size: 13px; font-weight: 500; color: #ffffff; }
.stat-val.green { color: #10b981; }
.stat-val.blue { color: #3b82f6; }

.detection-count {
    font-size: 12px;
    color: #666666;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('../data/preprocessed_data.csv',
                     parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

@st.cache_data
def load_results():
    with open('../data/zscore_results.json', 'r') as f:
        zscore = json.load(f)
    with open('../data/arima_results.json', 'r') as f:
        arima = json.load(f)
    with open('../data/project_summary.json', 'r') as f:
        summary = json.load(f)
    return zscore, arima, summary

df = load_data()
zscore_results, arima_results, summary = load_results()

sensor_cols = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
               'Pressure', 'Temperature', 'Thermocouple',
               'Voltage', 'Volume Flow RateRMS']

with st.sidebar:
    st.markdown("""
    <div style='font-size: 11px; font-weight: 600; color: #888888;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 16px; padding-bottom: 12px;
    border-bottom: 1px solid #222222;'>
    🛰️ Controls
    </div>
    """, unsafe_allow_html=True)

    selected_sensor = st.selectbox("Sensor channel", sensor_cols, index=7)
    selected_method = st.selectbox("Detection method",
                                   ["Z-score", "ARIMA", "Both"], index=0)
    selected_source = st.multiselect(
        "Anomaly source",
        ["anomaly-free", "valve1", "valve2", "other"],
        default=["valve1", "valve2", "other"]
    )
st.markdown(f"""
<div class="header-container">
    <div class="satellite-container">
        <div class="orbit-ring"></div>
        <div class="planet"></div>
        <div class="satellite"><div class="satellite-body"></div></div>
    </div>
    <div class="header-text">
        <div class="header-title">
            Satellite Telemetry Anomaly Detection
            <span class="live-badge">
                <span class="live-dot"></span>
                Monitoring Active
            </span>
        </div>
        <div class="header-sub">
            Comparative Analysis of Statistical and Time Series Methods —
            SKAB Benchmark Dataset
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Readings</div>
        <div class="metric-value">{summary['total_rows']:,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Anomalies</div>
        <div class="metric-value amber">{summary['total_anomalies']:,}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Anomaly Rate</div>
        <div class="metric-value">{summary['anomaly_percentage']}%</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Z-score F1</div>
        <div class="metric-value green">{summary['zscore_f1']}</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ARIMA Precision</div>
        <div class="metric-value blue">{summary['arima_precision']}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Sensor readings — {selected_sensor}</div>',
            unsafe_allow_html=True)

filtered_df = df[df['anomaly_source'].isin(selected_source)].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df['datetime'],
    y=filtered_df[selected_sensor],
    mode='lines',
    name='Sensor reading',
    line=dict(color='#475569', width=0.8),
))

actual_anomalies = filtered_df[filtered_df['anomaly'] == 1.0]
fig.add_trace(go.Scatter(
    x=actual_anomalies['datetime'],
    y=actual_anomalies[selected_sensor],
    mode='markers',
    name='Actual anomaly',
    marker=dict(color='#ef4444', size=3),
))

if selected_method == "Z-score":
    detected = filtered_df[filtered_df['zscore_prediction'] == 1]
    fig.add_trace(go.Scatter(
        x=detected['datetime'],
        y=detected[selected_sensor],
        mode='markers',
        name='Z-score detection',
        marker=dict(color='#f59e0b', size=4, symbol='x'),
    ))
elif selected_method == "ARIMA":
    detected = filtered_df[filtered_df['arima_prediction'] == 1]
    fig.add_trace(go.Scatter(
        x=detected['datetime'],
        y=detected[selected_sensor],
        mode='markers',
        name='ARIMA detection',
        marker=dict(color='#10b981', size=4, symbol='x'),
    ))
else:
    detected_z = filtered_df[filtered_df['zscore_prediction'] == 1]
    detected_a = filtered_df[filtered_df['arima_prediction'] == 1]
    fig.add_trace(go.Scatter(
        x=detected_z['datetime'],
        y=detected_z[selected_sensor],
        mode='markers',
        name='Z-score',
        marker=dict(color='#f59e0b', size=4, symbol='x'),
    ))
    fig.add_trace(go.Scatter(
        x=detected_a['datetime'],
        y=detected_a[selected_sensor],
        mode='markers',
        name='ARIMA',
        marker=dict(color='#10b981', size=4, symbol='x'),
    ))

fig.update_layout(
    paper_bgcolor='#111111',
    plot_bgcolor='#111111',
    height=400,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(
        gridcolor='#1a1a1a',
        tickfont=dict(size=10, color='#555555'),
        showline=True,
        linecolor='#222222',
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='#1a1a1a',
        tickfont=dict(size=10, color='#555555'),
        showline=True,
        linecolor='#222222',
        zeroline=False
    ),
   legend=dict(
        font=dict(size=11, color='#aaaaaa'),
        bgcolor='rgba(17,17,17,1)',
        bordercolor='#333333',
        borderwidth=1,
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0
    ),
   hoverlabel=dict(
        bgcolor='#111111',
        font_color='#ffffff',
        bordercolor='#444444',
        font_size=12,
        namelength=-1
    ),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">Model comparison</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="method-grid">
    <div class="method-card">
        <div class="method-title">
            Z-score Detector
        </div>
        <div class="stat-row">
            <span class="stat-name">Precision</span>
            <span class="stat-val">{zscore_results['precision']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">Recall</span>
            <span class="stat-val green">{zscore_results['recall']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">F1 Score</span>
            <span class="stat-val">{zscore_results['f1']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">True Positives</span>
            <span class="stat-val">{zscore_results['true_positives']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">False Positives</span>
            <span class="stat-val">{zscore_results['false_positives']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">Best threshold</span>
            <span class="stat-val">0.5 sigma</span>
        </div>
    </div>
    <div class="method-card">
        <div class="method-title">
            ARIMA Detector
        </div>
        <div class="stat-row">
            <span class="stat-name">Precision</span>
            <span class="stat-val green">{arima_results['precision']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">Recall</span>
            <span class="stat-val">{arima_results['recall']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">F1 Score</span>
            <span class="stat-val">{arima_results['f1']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">True Positives</span>
            <span class="stat-val">{arima_results['true_positives']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">False Positives</span>
            <span class="stat-val">{arima_results['false_positives']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-name">Model order</span>
            <span class="stat-val">ARIMA(1,1,1)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

metrics = ['Precision', 'Recall', 'F1 Score']
zscore_vals = [zscore_results['precision'],
               zscore_results['recall'],
               zscore_results['f1']]
arima_vals = [arima_results['precision'],
              arima_results['recall'],
              arima_results['f1']]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name='Z-score',
    x=metrics,
    y=zscore_vals,
    marker_color='#3b82f6',
    marker_line_width=0,
    text=[f'{v:.3f}' for v in zscore_vals],
    textposition='outside',
    textfont=dict(size=11, color='#aaaaaa')
))
fig2.add_trace(go.Bar(
    name='ARIMA',
    x=metrics,
    y=arima_vals,
    marker_color='#f97316',
    marker_line_width=0,
    text=[f'{v:.3f}' for v in arima_vals],
    textposition='outside',
    textfont=dict(size=11, color='#aaaaaa')
))
fig2.update_layout(
    barmode='group',
    paper_bgcolor='#111111',
    plot_bgcolor='#111111',
    height=320,
    margin=dict(l=0, r=0, t=60, b=0),
    yaxis=dict(
        range=[0, 1.3],
        gridcolor='#1a1a1a',
        tickfont=dict(size=10, color='#555555'),
        zeroline=False
    ),
    xaxis=dict(
        tickfont=dict(size=12, color='#aaaaaa'),
        zeroline=False
    ),
    legend=dict(
        font=dict(size=12, color='#aaaaaa'),
        bgcolor='rgba(17,17,17,1)',
        bordercolor='#333333',
        borderwidth=1,
        orientation='v',
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99
    ),
    bargap=0.3,
    bargroupgap=0.05,
    hoverlabel=dict(
        bgcolor='#1a1a1a',
        font_color='#ffffff',
        bordercolor='#333333',
        font_size=12
    ),
    title=dict(
        text='Z-score vs ARIMA — precision, recall, F1',
        font=dict(size=13, color='#aaaaaa'),
        x=0,
        xanchor='left'
    )
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section-title">Detected anomalies</div>',
            unsafe_allow_html=True)

if selected_method == "Z-score":
    anomaly_table = filtered_df[
        filtered_df['zscore_prediction'] == 1
    ][['datetime', 'anomaly_source', 'filename',
       selected_sensor, 'anomaly']].copy()
elif selected_method == "ARIMA":
    anomaly_table = filtered_df[
        filtered_df['arima_prediction'] == 1
    ][['datetime', 'anomaly_source', 'filename',
       selected_sensor, 'anomaly']].copy()
else:
    anomaly_table = filtered_df[
        (filtered_df['zscore_prediction'] == 1) |
        (filtered_df['arima_prediction'] == 1)
    ][['datetime', 'anomaly_source', 'filename',
       selected_sensor, 'anomaly']].copy()

anomaly_table.columns = ['Timestamp', 'Source',
                          'File', 'Sensor Value', 'Actual Anomaly']
anomaly_table['Sensor Value'] = anomaly_table['Sensor Value'].round(4)
anomaly_table['Actual Anomaly'] = anomaly_table['Actual Anomaly'].map(
    {0.0: 'Normal', 1.0: 'Anomaly'})

st.markdown(f"""
<div class="detection-count">{len(anomaly_table):,} detections found</div>
""", unsafe_allow_html=True)

st.dataframe(anomaly_table, use_container_width=True, height=250)