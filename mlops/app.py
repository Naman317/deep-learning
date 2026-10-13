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
    pass
