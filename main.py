import argparse
import sys
import os

from models.lstm_model import MultivariateLSTM
from models.rnn_model import SimpleRNN
from models.train_engine import train_model, autoregressive_forecast
from data_pipeline.data_loader import fetch_and_verify_market_data
from data_pipeline.feature_engineering import add_technical_indicators, fetch_vix_index, generate_signal
from data_pipeline.finbert_analyzer import analyze_news_sentiment
from data_pipeline.alpaca_execution import execute_paper_trade
from mlops.continuous_learning import load_saved_weights, save_model_weights
from mlops.db_logger import log_prediction
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np


def run_pipeline(ticker="AAPL", horizon=7, model_type="LSTM", execute_trade=False):
    print("=" * 70)
    print(f"  Pipeline: {ticker} (Horizon: {horizon}D | Model: {model_type})")
    print("=" * 70)
    
    print("[1/5] Fetching and verifying market data...")
    df, verified_price, status = fetch_and_verify_market_data(ticker)
    if df.empty:
        print("Error: Could not retrieve market data.")
        return
        
    df = add_technical_indicators(df)
    current_vix = fetch_vix_index()
    features = ['Close', 'SMA_20', 'RSI_14']
    data = df[features].values
    current_price = data[-1][0]
    current_rsi = data[-1][2]
    
    print("[2/5] Preparing sequences and model weights...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    pass
