import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data, add_rul
from features import run_feature_pipeline
from models.xgboost_model import predict_test as xgb_predict, load_model as xgb_load
from models.lstm_model import LSTMModel, predict_test as lstm_predict
from models.transformer_model import TransformerModel, predict_test as tf_predict
from evaluate import rmse, mae, nasa_score

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance — RUL Prediction",
    page_icon="⚙️",
    layout="wide"
)

# ── Load data and models (cached so they don't reload every interaction) ──
@st.cache_data
def load_all_data():
    train_df, test_df, rul_df = load_data(data_dir='data/raw/CMaps')
    train_df = add_rul(train_df)
    train_df, test_df, scaler = run_feature_pipeline(train_df, test_df)
    return train_df, test_df, rul_df

@st.cache_resource
def load_all_models(input_size):
    models = {}

    if os.path.exists('models/xgboost_rul.pkl'):
        models['XGBoost'] = xgb_load('models/xgboost_rul.pkl')

    if os.path.exists('models/lstm_rul.pt'):
        lstm = LSTMModel(input_size=input_size)
        lstm.load_state_dict(torch.load('models/lstm_rul.pt', map_location='cpu'))
        lstm.eval()
        models['LSTM'] = lstm

    if os.path.exists('models/transformer_rul.pt'):
        tf = TransformerModel(input_size=input_size)
        tf.load_state_dict(torch.load('models/transformer_rul.pt', map_location='cpu'))
        tf.eval()
        models['Transformer'] = tf

    return models

# ── Header ────────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance Dashboard")
st.markdown("**Remaining Useful Life (RUL) Prediction** — NASA CMAPSS Turbofan Engine Dataset")
st.markdown("---")

# ── Load everything ───────────────────────────────────────────
with st.spinner("Loading data and models..."):
    train_df, test_df, rul_df = load_all_data()
    feature_cols = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL']]
    models = load_all_models(input_size=len(feature_cols))

actuals = rul_df['RUL'].values

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Controls")
selected_models = st.sidebar.multiselect(
    "Select models to compare",
    options=list(models.keys()),
    default=list(models.keys())
)

engine_ids = sorted(test_df['engine_id'].unique())
selected_engine = st.sidebar.selectbox("Select test engine", engine_ids)

# ── Section 1: Model Comparison ───────────────────────────────
st.header("📊 Model Comparison")

results = {}
all_preds = {}

for name in selected_models:
    model = models[name]
    if name == 'XGBoost':
        preds, _ = xgb_predict(model, test_df, rul_df)
    elif name == 'LSTM':
        preds = lstm_predict(model, test_df)
    else:
        preds = tf_predict(model, test_df)

    all_preds[name] = preds
    results[name] = {
        'RMSE': round(rmse(actuals, preds), 2),
        'MAE': round(mae(actuals, preds), 2),
        'NASA Score': round(nasa_score(actuals, preds), 2)
    }

results_df = pd.DataFrame(results).T
st.dataframe(results_df, use_container_width=True)

# ── Bar charts ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
metrics = ['RMSE', 'MAE', 'NASA Score']
cols = [col1, col2, col3]
colors = ['#4C72B0', '#DD8452', '#55A868']

for col, metric in zip(cols, metrics):
    with col:
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(results_df.index, results_df[metric], color=colors[:len(results_df)])
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Score (lower = better)')
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(val), ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# ── Section 2: Single Engine Deep Dive ────────────────────────
st.header(f"🔍 Engine {selected_engine} — RUL Prediction")

# Get actual RUL for selected engine
actual_rul = actuals[selected_engine - 1]

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Actual RUL", f"{actual_rul} cycles")
    for name in selected_models:
        idx = selected_engine - 1
        pred = round(float(all_preds[name][idx]), 1)
        error = round(float(all_preds[name][idx]) - actual_rul, 1)
        st.metric(
            label=f"{name} Prediction",
            value=f"{pred} cycles",
            delta=f"{error:+.1f} cycles error",
            delta_color="inverse"  # red = bad (large error), green = good
        )

with col2:
    # Show sensor degradation trend for selected engine
    engine_data = test_df[test_df['engine_id'] == selected_engine]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot a few key sensors to show degradation
    key_sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor11']
    key_sensors = [s for s in key_sensors if s in engine_data.columns]

    for sensor in key_sensors:
        ax.plot(engine_data['cycle'], engine_data[sensor], label=sensor, alpha=0.8)

    ax.set_xlabel('Cycle')
    ax.set_ylabel('Normalized Sensor Value')
    ax.set_title(f'Engine {selected_engine} — Sensor Degradation Trend')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Section 3: Prediction vs Actual scatter ───────────────────
st.header("🎯 Predictions vs Actual RUL")

fig, axes = plt.subplots(1, len(selected_models), figsize=(5 * len(selected_models), 4))
if len(selected_models) == 1:
    axes = [axes]

for ax, name in zip(axes, selected_models):
    preds = all_preds[name]
    ax.scatter(actuals, preds, alpha=0.4, s=15, color='steelblue')
    ax.plot([0, 125], [0, 125], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built by **Aniruddh Mukherjee** | MSc INFOTECH, Universität Stuttgart | [GitHub](https://github.com/AniruddhMukherjee)")