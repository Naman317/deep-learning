import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def add_technical_indicators(df):
    df_copy = df.copy()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + rs))
    
    df_copy.dropna(inplace=True)
    return df_copy


def fetch_vix_index():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        vix_df = yf.download('^VIX', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df = vix_df.xs('^VIX', level=1, axis=1)
        if not vix_df.empty:
            return float(vix_df['Close'].iloc[-1])
    except Exception:
        pass
    return 20.0


def generate_signal(current_price, predicted_price, current_rsi, threshold=0.015):
    price_change = (predicted_price - current_price) / current_price
    
    if price_change > threshold and current_rsi < 70:
        signal = "STRONG BUY"
        reason = f"Expected gain of {price_change*100:+.2f}% with safe momentum (RSI: {current_rsi:.1f})"
    elif price_change > 0:
        signal = "WEAK BUY"
        reason = f"Slight gain expected ({price_change*100:+.2f}%)"
    elif price_change < -threshold and current_rsi > 30:
        signal = "STRONG SELL"
        reason = f"Expected drop of {abs(price_change)*100:.2f}% (RSI: {current_rsi:.1f})"
    elif price_change < 0:
        signal = "WEAK SELL"
        reason = f"Slight drop expected ({abs(price_change)*100:.2f}%)"
    else:
        signal = "HOLD"
        reason = "Projected movement is neutral"
        
    return signal, reason
