import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.lstm_model import MultivariateLSTM
from models.rnn_model import SimpleRNN
from models.train_engine import train_model, autoregressive_forecast
from data_pipeline.data_loader import fetch_and_verify_market_data, cross_verify_price
from data_pipeline.feature_engineering import add_technical_indicators, fetch_vix_index, generate_signal
from data_pipeline.finbert_analyzer import load_finbert, analyze_news_sentiment
from data_pipeline.alpaca_execution import execute_paper_trade
from mlops.continuous_learning import load_saved_weights, save_model_weights
from mlops.db_logger import init_db, log_prediction, get_recent_logs

init_db()

st.set_page_config(page_title="AlphaTrade", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body { font-family: 'Inter', sans-serif; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    
    h1, h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
    
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        width: 100%;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1.25rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .empty-state {
        border: 1px dashed rgba(128, 128, 128, 0.4);
        border-radius: 8px;
        padding: 4rem 2rem;
        text-align: center;
        background-color: var(--secondary-background-color);
        margin-top: 2rem;
    }
    
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: var(--background-color);
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        padding: 0.75rem 1rem;
        text-align: right;
        opacity: 0.7;
        font-size: 0.75rem;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='font-size: 1.5rem; margin-bottom: 0.25rem;'>AlphaTrade Workspace</h1>
    <p style='opacity: 0.7; font-size: 0.875rem; margin: 0;'>Quantitative Forecasting & Execution Suite</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='font-size: 0.875rem; text-transform: uppercase; opacity: 0.7; letter-spacing: 0.05em; margin-bottom: 1rem;'>Configuration</h3>", unsafe_allow_html=True)
ticker_input = st.sidebar.text_input("Asset Ticker", value="AAPL")

is_indian = st.sidebar.checkbox("NSE Routing (Indian Equity)", value=False)
if is_indian and not ticker_input.endswith(".NS"):
    ticker = f"{ticker_input.upper()}.NS"
else:
    ticker = ticker_input.upper()

st.sidebar.markdown("<hr style='border: none; border-top: 1px solid rgba(128, 128, 128, 0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)

with st.sidebar.expander("Model Parameters", expanded=False):
    model_type = st.selectbox("Neural Architecture", ["LSTM", "RNN"], index=0)
    optimizer_type = st.selectbox("Optimizer", ["Adam", "SGD"], index=0)
    forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 30, 7)
    epochs = st.slider("Optimization Epochs", 5, 50, 15)
    seq_length = st.slider("Lookback Window", 10, 90, 60)
    hidden_size = st.selectbox("Hidden State Size", [32, 64, 128], index=1)

st.sidebar.markdown("<hr style='border: none; border-top: 1px solid rgba(128, 128, 128, 0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)
enable_trading = st.sidebar.checkbox("Enable Automated Execution", value=False)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.sidebar.button("Run Inference Pipeline")

if analyze_btn:
    if not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        with st.spinner(f"Loading Models & Ingesting Market Data for {ticker}..."):
            df, verified_price, feed_status = fetch_and_verify_market_data(ticker)
            current_vix = fetch_vix_index()
            sentiment_score, news_items = analyze_news_sentiment(ticker)
            
        if df.empty or len(df) < seq_length + 20:
            st.error(f"Insufficient historical data for {ticker}.")
        else:
            df = add_technical_indicators(df)
            features = ['Close', 'SMA_20', 'RSI_14']
            data = df[features].values
            current_price = data[-1][0]
            current_rsi = data[-1][2]
            
            curr_symbol = "INR " if ticker.endswith((".NS", ".BO")) else "USD "
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(seq_length, len(scaled_data)):
                X.append(scaled_data[i-seq_length:i, :])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            if model_type == "RNN":
    pass
