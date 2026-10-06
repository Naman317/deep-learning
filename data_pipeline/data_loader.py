import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta


def cross_verify_price(ticker, historical_close=None):
    channel_a = None
    channel_b = None
    
    # Fast quote check
    try:
        channel_a = float(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        pass
        
    # REST API fallback
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d', headers=headers, timeout=4)
        channel_b = float(r.json()['chart']['result'][0]['meta'].get('regularMarketPrice', 0))
        if channel_b <= 0:
            channel_b = None
    except Exception:
        pass
        
    candidates = [p for p in [channel_a, channel_b] if p is not None and p > 0]
    live_consensus = candidates[0] if candidates else None
    
    status = "VERIFIED_CONSISTENT"
    final_price = historical_close
    
    if historical_close is None or pd.isna(historical_close) or historical_close <= 0:
        if live_consensus:
            final_price = live_consensus
            status = "CORRECTED_FROM_LIVE_CONSENSUS"
    elif live_consensus:
        pct_diff = abs(historical_close - live_consensus) / live_consensus
        if pct_diff > 0.018:
            final_price = live_consensus
            status = "REANCHORED_TO_LIVE_FEED"
        else:
            final_price = historical_close
            status = "VERIFIED_CONSISTENT"
            
    return final_price, live_consensus, status


def fetch_and_verify_market_data(ticker, lookback_years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    if df.empty:
        return df, None, "EMPTY_DATA"
        
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
        
    raw_last_close = df['Close'].iloc[-1] if not df.empty else None
    verified_price, live_consensus, status = cross_verify_price(ticker, raw_last_close)
    
    if status in ["CORRECTED_FROM_LIVE_CONSENSUS", "REANCHORED_TO_LIVE_FEED"] and verified_price is not None:
        df.iloc[-1, df.columns.get_loc('Close')] = verified_price
        
    return df, verified_price, status
