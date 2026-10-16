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
    
    seq_length = 60
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, :])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    if model_type.upper() == "RNN":
        model = SimpleRNN(input_size=len(features), hidden_size=64, num_layers=2, output_size=1)
    else:
        model = MultivariateLSTM(input_size=len(features), hidden_size=64, num_layers=2, output_size=1)
        
    model, is_loaded = load_saved_weights(model, ticker, model_type)
    if is_loaded:
        print(f"      -> Loaded saved weights for {ticker}. Fine-tuning...")
    else:
        print(f"      -> Initializing new weights for {ticker}...")
        
    print("[3/5] Training model...")
    model, losses = train_model(model, train_loader, epochs=15, optimizer_type="Adam")
    save_model_weights(model, ticker, model_type)
    
    print(f"[4/5] Generating {horizon}-day forecast...")
    predictions = autoregressive_forecast(model, scaled_data[-seq_length:], scaler, 
                                          seq_length=seq_length, num_features=len(features), 
                                          forecast_horizon=horizon)
    t1_pred = predictions[0]
    signal, reason = generate_signal(current_price, t1_pred, current_rsi)
    
    print("[5/5] Analyzing news sentiment...")
    sentiment_score, news_items = analyze_news_sentiment(ticker)
    
    # Adjust signal if negative sentiment detected
    if sentiment_score < -0.2 and "BUY" in signal:
        signal = "HOLD"
        reason += " (Adjusted due to negative sentiment)"
        
    log_prediction(ticker, current_price, t1_pred, signal)
    
    print("\n" + "=" * 70)
    print("  Forecast Summary")
    print("=" * 70)
    curr_sym = "INR " if ticker.endswith((".NS", ".BO")) else "USD "
    print(f"  Ticker            : {ticker}")
    print(f"  Current Price     : {curr_sym}{current_price:.2f}")
    print(f"  T+1 Forecast      : {curr_sym}{t1_pred:.2f} ({((t1_pred-current_price)/current_price)*100:+.2f}%)")
    print(f"  Trade Signal      : {signal} ({reason})")
    print(f"  Sentiment Score   : {sentiment_score:+.2f}")
    print(f"  VIX Index         : {current_vix:.1f}")
    print(f"  Roadmap           : {[f'{curr_sym}{p:.2f}' for p in predictions]}")
    
    if execute_trade:
        trade = execute_paper_trade(ticker, signal, current_price)
        print(f"  Order Execution   : {trade.get('message')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTrade CLI Inference Engine")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, RELIANCE.NS)")
    parser.add_argument("--horizon", type=int, default=7, help="Multi-day forecast horizon (1 to 30)")
    parser.add_argument("--model", type=str, default="LSTM", choices=["LSTM", "RNN"], help="Neural architecture")
    parser.add_argument("--trade", action="store_true", help="Enable paper trade order execution")
    args = parser.parse_args()
    
    run_pipeline(ticker=args.ticker, horizon=args.horizon, model_type=args.model, execute_trade=args.trade)
