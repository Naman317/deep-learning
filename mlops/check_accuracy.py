import os
import sys
import sqlite3
import pandas as pd
import yfinance as yf
import numpy as np
import warnings

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')


def audit_historical_predictions():
    if not os.path.exists(DB_PATH):
        return {"status": "NO_DATABASE", "message": "No database found."}

    conn = sqlite3.connect(DB_PATH)
    logs_df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp ASC", conn)
    conn.close()

    if logs_df.empty:
        return {"status": "EMPTY", "message": "No prediction logs found."}

    tickers = logs_df['ticker'].unique()
    hist_cache = {}
    for t in tickers:
        try:
            hist = yf.download(t, period='2mo', progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist = hist.xs(t, level=1, axis=1)
            hist_cache[t] = hist
    pass
