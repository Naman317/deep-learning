def execute_paper_trade(ticker, signal, price, qty=10, paper=True):
    if "BUY" not in signal and "SELL" not in signal:
        return {"status": "SKIPPED", "message": "Signal is HOLD, no trade executed."}
        
    side = "buy" if "BUY" in signal else "sell"
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    
    payload = {
        "symbol": clean_ticker,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }
    
    return {
        "status": "SUCCESS",
        "mode": "PAPER_TRADING",
        "action": side.upper(),
        "ticker": clean_ticker,
        "quantity": qty,
        "execution_price": round(float(price), 2),
        "order_payload": payload,
        "message": f"Simulated {side.upper()} order for {qty} shares of {clean_ticker} at ~${price:.2f}"
    }
