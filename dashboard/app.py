import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Satellite Telemetry Anomaly Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("Satellite Telemetry Anomaly Detection Dashboard")
st.markdown("**Comparative Analysis of Statistical and Time Series Methods**")
st.markdown("Dataset: SKAB — Skoltech Anomaly Benchmark | Methods: Z-score (DA206) + ARIMA (DA210)")
st.markdown("---")
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
st.sidebar.title("Controls")
st.sidebar.markdown("---")

sensor_cols = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
               'Pressure', 'Temperature', 'Thermocouple',
               'Voltage', 'Volume Flow RateRMS']

selected_sensor = st.sidebar.selectbox(
    "Select sensor channel:",
    sensor_cols,
    index=7
)

selected_method = st.sidebar.selectbox(
    "Select detection method:",
    ["Z-score", "ARIMA", "Both"],
    index=0
)

selected_source = st.sidebar.multiselect(
    "Filter by anomaly source:",
    ["anomaly-free", "valve1", "valve2", "other"],
    default=["valve1", "valve2", "other"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project info**")
st.sidebar.markdown("Course: BSc Data Science and AI")
st.sidebar.markdown("Methods: Z-score (DA206), ARIMA (DA210)")
st.sidebar.markdown("Dataset: SKAB Benchmark (33 files, 44,534 readings)")
st.subheader("Dataset overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label="Total readings",
    value=f"{summary['total_rows']:,}"
)
col2.metric(
    label="Total anomalies",
    value=f"{summary['total_anomalies']:,}",
    delta=f"{summary['anomaly_percentage']}% of data"
)
col3.metric(
    label="Z-score F1",
    value=summary['zscore_f1']
)
col4.metric(
    label="ARIMA F1",
    value=summary['arima_f1']
)
col5.metric(
    label="ARIMA precision",
    value=summary['arima_precision'],
    delta="Better than Z-score"
)

st.markdown("---")
st.subheader(f"Sensor readings — {selected_sensor}")

filtered_df = df[df['anomaly_source'].isin(selected_source)].copy()

if selected_method == "Z-score":
    pred_col = 'zscore_prediction'
    pred_label = 'Z-score detection'
    detection_color = 'orange'
elif selected_method == "ARIMA":
    pred_col = 'arima_prediction'
    pred_label = 'ARIMA detection'
    detection_color = 'red'

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df['datetime'],
    y=filtered_df[selected_sensor],
    mode='lines',
    name='Sensor reading',
    line=dict(color='steelblue', width=0.8),
    opacity=0.8
))

actual_anomalies = filtered_df[filtered_df['anomaly'] == 1.0]
fig.add_trace(go.Scatter(
    x=actual_anomalies['datetime'],
    y=actual_anomalies[selected_sensor],
    mode='markers',
    name='Actual anomaly',
    marker=dict(color='red', size=3, symbol='circle'),
    opacity=0.6
))

if selected_method != "Both":
    detected = filtered_df[filtered_df[pred_col] == 1]
    fig.add_trace(go.Scatter(
        x=detected['datetime'],
        y=detected[selected_sensor],
        mode='markers',
        name=pred_label,
        marker=dict(color=detection_color, size=4, 
                   symbol='x'),
        opacity=0.8
    ))

fig.update_layout(
    height=400,
    xaxis_title='Time',
    yaxis_title='Normalized value',
    legend=dict(orientation="h", yanchor="bottom", 
                y=1.02, xanchor="right", x=1),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.subheader("Model comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Z-score detector (DA206)**")
    st.metric("Precision", zscore_results['precision'])
    st.metric("Recall", zscore_results['recall'])
    st.metric("F1 Score", zscore_results['f1'])
    st.metric("False Positives", f"{zscore_results['false_positives']:,}")
    st.metric("True Positives", f"{zscore_results['true_positives']:,}")

with col2:
    st.markdown("**ARIMA detector (DA210)**")
    st.metric("Precision", arima_results['precision'])
    st.metric("Recall", arima_results['recall'])
    st.metric("F1 Score", arima_results['f1'])
    st.metric("False Positives", f"{arima_results['false_positives']:,}")
    st.metric("True Positives", f"{arima_results['true_positives']:,}")

st.markdown("---")

metrics = ['Precision', 'Recall', 'F1 Score']
zscore_vals = [zscore_results['precision'], 
               zscore_results['recall'], 
               zscore_results['f1']]
arima_vals = [arima_results['precision'], 
              arima_results['recall'], 
              arima_results['f1']]

fig_comparison = go.Figure()
fig_comparison.add_trace(go.Bar(
    name='Z-score',
    x=metrics,
    y=zscore_vals,
    marker_color='steelblue',
    text=[f'{v:.3f}' for v in zscore_vals],
    textposition='outside'
))
fig_comparison.add_trace(go.Bar(
    name='ARIMA',
    x=metrics,
    y=arima_vals,
    marker_color='coral',
    text=[f'{v:.3f}' for v in arima_vals],
    textposition='outside'
))
fig_comparison.update_layout(
    barmode='group',
    height=350,
    yaxis=dict(range=[0, 1.2]),
    title='Z-score vs ARIMA — precision, recall, F1'
)
st.plotly_chart(fig_comparison, use_container_width=True)
st.markdown("---")
st.subheader("Detected anomalies table")

if selected_method == "Z-score":
    anomaly_table = filtered_df[
        filtered_df['zscore_prediction'] == 1
    ][['datetime', 'anomaly_source', 'filename', 
       selected_sensor, 'anomaly', 'zscore_prediction']].copy()
    anomaly_table.columns = ['Timestamp', 'Source', 'File', 
                              'Sensor Value', 'Actual Anomaly', 
                              'Predicted']
elif selected_method == "ARIMA":
    anomaly_table = filtered_df[
        filtered_df['arima_prediction'] == 1
    ][['datetime', 'anomaly_source', 'filename',
       selected_sensor, 'anomaly', 'arima_prediction']].copy()
    anomaly_table.columns = ['Timestamp', 'Source', 'File',
                              'Sensor Value', 'Actual Anomaly',
                              'Predicted']
else:
    anomaly_table = filtered_df[
        (filtered_df['zscore_prediction'] == 1) | 
        (filtered_df['arima_prediction'] == 1)
    ][['datetime', 'anomaly_source', 'filename',
       selected_sensor, 'anomaly']].copy()
    anomaly_table.columns = ['Timestamp', 'Source', 'File',
                              'Sensor Value', 'Actual Anomaly']

anomaly_table['Sensor Value'] = anomaly_table['Sensor Value'].round(4)
anomaly_table['Actual Anomaly'] = anomaly_table['Actual Anomaly'].map(
    {0.0: 'Normal', 1.0: 'Anomaly'}
)

st.write(f"Showing {len(anomaly_table):,} detected rows")
st.dataframe(anomaly_table, use_container_width=True, height=300)

st.markdown("---")
st.markdown("**Project:** Comparative Analysis of Statistical and Time Series Methods for Anomaly Detection in Spacecraft Subsystem Telemetry")
st.markdown("**Dataset:** SKAB — Skoltech Anomaly Benchmark | **Courses:** DA206, DA210, DA209")