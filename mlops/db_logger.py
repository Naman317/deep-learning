import os
import sqlite3
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, ticker TEXT, current_price REAL, predicted_price REAL, signal TEXT)''')
    conn.commit()
    conn.close()


def log_prediction(ticker, current_price, predicted_price, signal):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker, float(current_price), float(predicted_price), signal))
    conn.commit()
    conn.close()


def get_recent_logs(ticker=None, limit=10):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    if ticker:
        df = pd.read_sql_query("SELECT timestamp, current_price, predicted_price, signal FROM logs WHERE ticker=? ORDER BY timestamp DESC LIMIT ?", 
                               conn, params=(ticker, limit))
    else:
        df = pd.read_sql_query("SELECT timestamp, ticker, current_price, predicted_price, signal FROM logs ORDER BY timestamp DESC LIMIT ?", 
                               conn, params=(limit,))
    conn.close()
    return df
