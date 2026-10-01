import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import MultivariateLSTM
from models.rnn_model import SimpleRNN


def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df


def run_benchmark(ticker="AAPL", epochs=15, seq_length=60, hidden_size=64):
    print(f"Running benchmark on {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
        
    df = add_technical_indicators(df)
    features = ['Close', 'SMA_20', 'RSI_14']
    data = df[features].values
    current_price = data[-1][0]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, :])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.MSELoss()
    
    # Train LSTM
    print("Training Multivariate LSTM...")
    lstm = MultivariateLSTM(input_size=len(features), hidden_size=hidden_size, num_layers=2, output_size=1)
    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=0.001)
    t0 = time.time()
    for _ in range(epochs):
        lstm.train()
        for bx, by in train_loader:
            opt_lstm.zero_grad()
            loss = criterion(lstm(bx), by.unsqueeze(1))
            loss.backward()
            opt_lstm.step()
    t_lstm = time.time() - t0
    
    lstm.eval()
    with torch.no_grad():
        p_sc = lstm(torch.tensor(scaled_data[-seq_length:].reshape(1, seq_length, len(features)), dtype=torch.float32)).numpy()[0][0]
        dum = np.zeros((1, len(features)))
        dum[0, 0] = p_sc
        p_lstm = scaler.inverse_transform(dum)[0][0]
        
    # Train RNN
    print("Training Simple RNN baseline...")
    rnn = SimpleRNN(input_size=len(features), hidden_size=hidden_size, num_layers=2, output_size=1)
    opt_rnn = torch.optim.Adam(rnn.parameters(), lr=0.001)
    t0 = time.time()
    for _ in range(epochs):
        rnn.train()
        for bx, by in train_loader:
            opt_rnn.zero_grad()
            loss = criterion(rnn(bx), by.unsqueeze(1))
            loss.backward()
            opt_rnn.step()
    t_rnn = time.time() - t0
    
    rnn.eval()
    with torch.no_grad():
        p_sc = rnn(torch.tensor(scaled_data[-seq_length:].reshape(1, seq_length, len(features)), dtype=torch.float32)).numpy()[0][0]
        dum = np.zeros((1, len(features)))
        dum[0, 0] = p_sc
        p_rnn = scaler.inverse_transform(dum)[0][0]
        
    print("-" * 50)
    print(f"Current Price ({ticker}): ${current_price:.2f}")
    print(f"LSTM Prediction:  ${p_lstm:.2f} (Time: {t_lstm:.2f}s)")
    print(f"RNN Prediction:   ${p_rnn:.2f} (Time: {t_rnn:.2f}s)")
    print("-" * 50)


if __name__ == "__main__":
    run_benchmark("AAPL")
