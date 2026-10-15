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
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1.25rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .empty-state {
        border: 1px dashed rgba(128, 128, 128, 0.4);
        border-radius: 8px;
        padding: 4rem 2rem;
        text-align: center;
        background-color: var(--secondary-background-color);
        margin-top: 2rem;
    }
    
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: var(--background-color);
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        padding: 0.75rem 1rem;
        text-align: right;
        opacity: 0.7;
        font-size: 0.75rem;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='font-size: 1.5rem; margin-bottom: 0.25rem;'>AlphaTrade Workspace</h1>
    <p style='opacity: 0.7; font-size: 0.875rem; margin: 0;'>Quantitative Forecasting & Execution Suite</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='font-size: 0.875rem; text-transform: uppercase; opacity: 0.7; letter-spacing: 0.05em; margin-bottom: 1rem;'>Configuration</h3>", unsafe_allow_html=True)
ticker_input = st.sidebar.text_input("Asset Ticker", value="AAPL")

is_indian = st.sidebar.checkbox("NSE Routing (Indian Equity)", value=False)
if is_indian and not ticker_input.endswith(".NS"):
    ticker = f"{ticker_input.upper()}.NS"
else:
    ticker = ticker_input.upper()

st.sidebar.markdown("<hr style='border: none; border-top: 1px solid rgba(128, 128, 128, 0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)

with st.sidebar.expander("Model Parameters", expanded=False):
    model_type = st.selectbox("Neural Architecture", ["LSTM", "RNN"], index=0)
    optimizer_type = st.selectbox("Optimizer", ["Adam", "SGD"], index=0)
    forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 30, 7)
    epochs = st.slider("Optimization Epochs", 5, 50, 15)
    seq_length = st.slider("Lookback Window", 10, 90, 60)
    hidden_size = st.selectbox("Hidden State Size", [32, 64, 128], index=1)

st.sidebar.markdown("<hr style='border: none; border-top: 1px solid rgba(128, 128, 128, 0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)
enable_trading = st.sidebar.checkbox("Enable Automated Execution", value=False)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.sidebar.button("Run Inference Pipeline")

if analyze_btn:
    if not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        with st.spinner(f"Loading Models & Ingesting Market Data for {ticker}..."):
            df, verified_price, feed_status = fetch_and_verify_market_data(ticker)
            current_vix = fetch_vix_index()
            sentiment_score, news_items = analyze_news_sentiment(ticker)
            
        if df.empty or len(df) < seq_length + 20:
            st.error(f"Insufficient historical data for {ticker}.")
        else:
            df = add_technical_indicators(df)
            features = ['Close', 'SMA_20', 'RSI_14']
            data = df[features].values
            current_price = data[-1][0]
            current_rsi = data[-1][2]
            
            curr_symbol = "INR " if ticker.endswith((".NS", ".BO")) else "USD "
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(seq_length, len(scaled_data)):
                X.append(scaled_data[i-seq_length:i, :])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            if model_type == "RNN":
                model = SimpleRNN(input_size=len(features), hidden_size=hidden_size, num_layers=2, output_size=1)
            else:
                model = MultivariateLSTM(input_size=len(features), hidden_size=hidden_size, num_layers=2, output_size=1)
                
            model, is_fine_tuned = load_saved_weights(model, ticker, model_type)
            progress_msg = f"Fine-tuning {model_type} weights..." if is_fine_tuned else f"Training {model_type} model..."
            
            my_bar = st.progress(0, text=progress_msg)
            
            def progress_cb(ep, total_ep, loss):
                my_bar.progress(int((ep / total_ep) * 100), text=f"{progress_msg} ({ep}/{total_ep})")
                
            model, _ = train_model(model, train_loader, epochs=epochs, optimizer_type=optimizer_type, callback=progress_cb)
            time.sleep(0.2)
            my_bar.empty()
            
            save_model_weights(model, ticker, model_type)
            
            future_predictions = autoregressive_forecast(model, scaled_data[-seq_length:], scaler, 
                                                         seq_length=seq_length, num_features=len(features), 
                                                         forecast_horizon=forecast_horizon)
            t1_pred = future_predictions[0]
            
            signal, reason = generate_signal(current_price, t1_pred, current_rsi)
            
            # Risk & sentiment integration
            model_confidence = 100 - (abs((t1_pred - current_price)/current_price) * 1000) - (current_vix * 0.5)
            if sentiment_score < -0.2:
                model_confidence -= 10.0
                if "BUY" in signal:
                    signal = "HOLD"
                    reason += " (Downgraded due to negative news sentiment)"
            elif sentiment_score > 0.2:
                model_confidence += 5.0
            model_confidence = max(min(model_confidence, 99.9), 45.0)
            
            log_prediction(ticker, current_price, t1_pred, signal)
            
            st.markdown("<h3 style='font-size: 1rem; margin-bottom: 0.25rem;'>Inference Results</h3>", unsafe_allow_html=True)
            if feed_status in ["CORRECTED_FROM_LIVE_CONSENSUS", "REANCHORED_TO_LIVE_FEED"]:
                st.markdown(f"<p style='font-size: 0.75rem; color: #3B82F6; margin-bottom: 1rem;'>⚡ <b>Feed Auto-Reanchored:</b> Corrected to real-time consensus stream ({curr_symbol}{verified_price:.2f})</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='font-size: 0.75rem; color: #10B981; margin-bottom: 1rem;'>🟢 <b>Feed Verified:</b> Historical close cross-validated ({curr_symbol}{current_price:.2f})</p>", unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Current Asset", f"{curr_symbol}{current_price:.2f}")
            col2.metric("Forecast (T+1)", f"{curr_symbol}{t1_pred:.2f}", f"{(t1_pred-current_price)/current_price*100:+.2f}%")
            col3.metric("Sentiment", f"{sentiment_score:+.2f}")
            col4.metric("RSI (14D)", f"{current_rsi:.1f}")
            col5.metric("Confidence", f"{model_confidence:.1f}%", f"VIX: {current_vix:.1f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Signal:** {signal} — {reason}")
            
            if enable_trading:
                trade_res = execute_paper_trade(ticker, signal, current_price)
                if trade_res.get("status") == "SUCCESS":
                    st.success(f"Execution: {trade_res['message']}")
            
            st.markdown(f"<h3 style='font-size: 1rem; margin-top: 2rem; margin-bottom: 1rem;'>Price Trajectory ({forecast_horizon}-Day Horizon)</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(10, 3.2))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('#00000000')
            
            last_30_days = df['Close'].iloc[-30:].values
            ax.plot(range(len(last_30_days)), last_30_days, label='Historical (Last 30D)', color='#3B82F6', linewidth=1.5)
            
            forecast_x = range(len(last_30_days) - 1, len(last_30_days) + len(future_predictions))
            forecast_y = [last_30_days[-1]] + future_predictions
            ax.plot(forecast_x, forecast_y, marker='o', markersize=3, color='#10B981', linestyle='--', linewidth=1.5, label=f'Forecast (T+1 to T+{forecast_horizon})')
            
            ax.set_ylabel(f"Price ({curr_symbol.strip()})")
            ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
            legend = ax.legend(frameon=True, loc='upper left', fontsize='small')
            legend.get_frame().set_alpha(0.1)
            legend.get_frame().set_edgecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('gray')
            ax.spines['bottom'].set_alpha(0.3)
            ax.tick_params(colors='gray', bottom=False, left=False)
            st.pyplot(fig)
            
            st.markdown(f"<h3 style='font-size: 1rem; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Multi-Day Roadmap ({forecast_horizon} Days)</h3>", unsafe_allow_html=True)
            roadmap_data = []
            for d_idx, p_val in enumerate(future_predictions, 1):
                pct_chg = ((p_val - current_price) / current_price) * 100
                trend_icon = "Bullish" if pct_chg > 0.5 else "Bearish" if pct_chg < -0.5 else "Neutral"
                roadmap_data.append({
                    "Horizon": f"Day {d_idx} (T+{d_idx})",
                    "Projected Target": f"{curr_symbol}{p_val:.2f}",
                    "Expected Move": f"{pct_chg:+.2f}%",
                    "Trajectory": trend_icon
                })
            roadmap_df = pd.DataFrame(roadmap_data)
            
            if forecast_horizon > 7:
                milestone_indices = [1, 3, 5, 7, 10, 15, 20, 25, 30]
                milestone_df = roadmap_df[roadmap_df['Horizon'].apply(lambda x: int(x.split(' ')[1]) in milestone_indices)]
                st.dataframe(milestone_df, use_container_width=True, hide_index=True)
                with st.expander(f"View Full Day-by-Day Roadmap (All {forecast_horizon} Days)", expanded=False):
                    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
                
            st.markdown("<h3 style='font-size: 1rem; margin-top: 2rem; margin-bottom: 1rem;'>News & Sentiment (FinBERT)</h3>", unsafe_allow_html=True)
            if news_items:
                for item in news_items:
                    st.markdown(f"<a href='{item['url']}' style='color: inherit; opacity: 0.8; text-decoration: none; font-size: 0.875rem;'>• {item['title']} <span style='color: {item['color']}; font-size: 0.75rem; font-weight: 600; margin-left: 8px;'>[{item['label']}]</span></a>", unsafe_allow_html=True)
            else:
                st.write("No live news items found.")
                
            st.markdown("<h3 style='font-size: 1rem; margin-top: 2rem; margin-bottom: 1rem;'>Model Benchmark</h3>", unsafe_allow_html=True)
            with st.expander("Compare LSTM vs Simple RNN", expanded=False):
                if st.button("Run Comparison"):
                    with st.spinner(f"Training LSTM and RNN models for {ticker}..."):
                        rnn_comp = SimpleRNN(input_size=len(features), hidden_size=hidden_size, num_layers=2, output_size=1)
                        rnn_comp, _ = train_model(rnn_comp, train_loader, epochs=epochs, optimizer_type="Adam")
                        rnn_preds = autoregressive_forecast(rnn_comp, scaled_data[-seq_length:], scaler, 
                                                            seq_length=seq_length, num_features=len(features), forecast_horizon=1)
                        rnn_t1 = rnn_preds[0]
                        comp_summary = pd.DataFrame([
                            {"Architecture": "Multivariate LSTM", "T+1 Forecast Target": f"{curr_symbol}{t1_pred:.2f}", "Expected Move": f"{((t1_pred-current_price)/current_price)*100:+.2f}%"},
                            {"Architecture": "Simple RNN Baseline", "T+1 Forecast Target": f"{curr_symbol}{rnn_t1:.2f}", "Expected Move": f"{((rnn_t1-current_price)/current_price)*100:+.2f}%"}
                        ])
                        st.dataframe(comp_summary, use_container_width=True, hide_index=True)
                
            st.markdown("<h3 style='font-size: 1rem; margin-top: 3rem; margin-bottom: 1rem;'>Historical Prediction Logs</h3>", unsafe_allow_html=True)
            history_df = get_recent_logs(ticker=ticker, limit=5)
            if not history_df.empty:
                history_df.columns = ["Timestamp", "Price at Inference", "AI Forecast (T+1)", "Signal"]
                try:
                    history_df["Current Actual Price"] = float(current_price)
                    history_df["Error %"] = (abs(history_df["AI Forecast (T+1)"] - history_df["Current Actual Price"]) / history_df["Current Actual Price"]) * 100
                    history_df["Error %"] = history_df["Error %"].map("{:.2f}%".format)
                except Exception:
                    pass
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.write("No historical predictions logged for this asset.")

else:
    st.markdown("""
    <div class='empty-state'>
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 1rem; opacity: 0.5;">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        <h3 style="margin-bottom: 0.5rem;">System Standby</h3>
        <p style="opacity: 0.7; font-size: 0.875rem; max-width: 400px; margin: 0 auto;">
            Configure parameters in the sidebar and run the inference pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)
