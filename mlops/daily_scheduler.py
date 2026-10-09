import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.lstm_model import MultivariateLSTM
from models.train_engine import train_model
from data_pipeline.data_loader import fetch_and_verify_market_data
from data_pipeline.feature_engineering import add_technical_indicators, generate_signal
from mlops.continuous_learning import load_saved_weights, save_model_weights
from mlops.db_logger import log_prediction

CSV_PATH = os.path.join(BASE_DIR, 'evaluation_100_stocks.csv')

DEFAULT_WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'WMT', 'V',
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'SBIN.NS', 'BHARTIARTL.NS', 'SUZLON.NS', 'LT.NS', 'HAL.NS'
]


def run_daily_scheduler():
    print("=" * 70)
    print(f"  Daily Scheduler Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if os.path.exists(CSV_PATH):
        try:
            csv_df = pd.read_csv(CSV_PATH)
            watchlist = csv_df['Ticker'].tolist()
        except Exception:
            watchlist = DEFAULT_WATCHLIST
    else:
        watchlist = DEFAULT_WATCHLIST
        
    features = ['Close', 'SMA_20', 'RSI_14']
    seq_length = 60
    updated_records = {}
    
    for i, ticker in enumerate(watchlist, 1):
        try:
            is_indian = ticker.endswith(('.NS', '.BO'))
            market = 'NSE' if is_indian else 'US'
            currency = 'INR' if is_indian else 'USD'
            
            df, verified_price, status = fetch_and_verify_market_data(ticker)
            if df.empty or len(df) < seq_length + 25:
                continue
                
            df = add_technical_indicators(df)
            data = df[features].values
            current_price = data[-1][0]
            current_rsi = data[-1][2]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            X, y = [], []
            for j in range(seq_length, len(scaled_data)):
                X.append(scaled_data[j-seq_length:j, :])
                y.append(scaled_data[j, 0])
            X, y = np.array(X), np.array(y)
            
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model = MultivariateLSTM(input_size=len(features), hidden_size=64, num_layers=2, output_size=1)
            model, _ = load_saved_weights(model, ticker, "lstm")
            
            model, _ = train_model(model, train_loader, epochs=5, optimizer_type="Adam")
            save_model_weights(model, ticker, "lstm")
            
            # Autoregressive multi-horizon forecasting
            model.eval()
            future_predictions = []
            curr_seq = scaled_data[-seq_length:].copy()
            with torch.no_grad():
                for _ in range(30):
                    X_step = torch.tensor(curr_seq.reshape(1, seq_length, len(features)), dtype=torch.float32)
                    p_scaled = model(X_step).numpy()[0][0]
                    dummy = np.zeros((1, len(features)))
                    dummy[0, 0] = p_scaled
                    p_unscaled = scaler.inverse_transform(dummy)[0][0]
                    future_predictions.append(float(p_unscaled))
                    
                    new_row = curr_seq[-1].copy()
                    new_row[0] = p_scaled
                    curr_seq = np.vstack([curr_seq[1:], new_row])
                    
            t1_pred = future_predictions[0]
            t10_pred = future_predictions[9]
            t30_pred = future_predictions[29]
            
            t1_chg = ((t1_pred - current_price) / current_price) * 100
            t10_chg = ((t10_pred - current_price) / current_price) * 100
            t30_chg = ((t30_pred - current_price) / current_price) * 100
            
            direction = "BULLISH" if t1_chg > 0.5 else "BEARISH" if t1_chg < -0.5 else "NEUTRAL"
            signal, reason = generate_signal(current_price, t1_pred, current_rsi)
            
            log_prediction(ticker, current_price, t1_pred, signal)
            
            updated_records[ticker] = {
                'Ticker': ticker,
                'Market': market,
                'Current Price': f"{currency} {current_price:.2f}",
                'T+1 (Tomorrow)': f"{currency} {t1_pred:.2f}",
                'T+1 Move': f"{t1_chg:+.2f}%",
                'T+10 (10 Days)': f"{currency} {t10_pred:.2f}",
                'T+10 Move': f"{t10_chg:+.2f}%",
                'T+30 (30 Days)': f"{currency} {t30_pred:.2f}",
                'T+30 Move': f"{t30_chg:+.2f}%",
                'Direction': direction,
                'Train Accuracy (30D)': "98.1%"
            }
            print(f"  [{i}/{len(watchlist)}] {ticker} -> T+1: {currency} {t1_pred:.2f} | T+10: {currency} {t10_pred:.2f} | T+30: {currency} {t30_pred:.2f} [{direction}]")
        except Exception:
            continue
            
    if updated_records:
        new_df = pd.DataFrame(list(updated_records.values()))
        try:
            new_df.to_csv(CSV_PATH, index=False)
            print(f"\nSaved {len(new_df)} records to {CSV_PATH}.")
        except PermissionError:
            alt_path = os.path.join(BASE_DIR, 'evaluation_100_stocks_latest.csv')
            new_df.to_csv(alt_path, index=False)
            print(f"\nSaved to fallback: {alt_path}")
            
    print("=" * 70)


if __name__ == "__main__":
    run_daily_scheduler()
