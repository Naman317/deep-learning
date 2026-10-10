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
        except Exception:
            pass

    evaluated = []
    for _, row in logs_df.iterrows():
        ticker = row['ticker']
        pred_time = row['timestamp']
        pred_date = pd.to_datetime(pred_time.split(' ')[0])
        curr_p = row['current_price']
        pred_p = row['predicted_price']

        if ticker not in hist_cache or hist_cache[ticker].empty:
            continue

        hist = hist_cache[ticker]
        dates_after = hist.index[hist.index > pred_date]

        if len(dates_after) > 0:
            actual_date = dates_after[0].strftime('%Y-%m-%d')
            actual_price = float(hist.loc[dates_after[0], 'Close'])
            err = abs(actual_price - pred_p) / actual_price * 100
            
            dir_pred = "UP" if pred_p > curr_p else "DOWN"
            dir_actual = "UP" if actual_price > curr_p else "DOWN"
            hit = (dir_pred == dir_actual)

            evaluated.append({
                'timestamp': pred_time,
                'ticker': ticker,
                'curr_price': curr_p,
                'pred_price': pred_p,
                'actual_date': actual_date,
                'actual_price': actual_price,
                'error_pct': err,
                'hit': hit
            })

    if not evaluated:
        return {"status": "AWAITING_SETTLEMENT", "count": len(logs_df)}

    df = pd.DataFrame(evaluated)
    mape = df['error_pct'].mean()
    hits = df['hit'].sum()
    total = len(df)
    precision = (hits / total) * 100

    return {
        "status": "SUCCESS",
        "total_evaluated": total,
        "mape": mape,
        "accuracy": 100 - mape,
        "precision": precision,
        "hits": hits,
        "records": df
    }


if __name__ == "__main__":
    res = audit_historical_predictions()
    if res.get('status') == 'SUCCESS':
        print(f"Accuracy: {res['accuracy']:.2f}% | Precision: {res['precision']:.1f}% ({res['hits']}/{res['total_evaluated']})")
    else:
        print(res.get('message', 'No settled records.'))
