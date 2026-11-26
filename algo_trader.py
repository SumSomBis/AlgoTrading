"""
Algo Trading Backtester - EMA Crossover + RSI Filter + ATR Stop
---------------------------------------------------------------
pip install yfinance pandas numpy matplotlib

Usage:
    python algo_trader.py                      # default: RELIANCE.NS
    python algo_trader.py TCS.NS INFY.NS       # multiple NSE symbols
"""

import sys
import numpy as np
import pandas as pd

# ============================ CONFIG ============================
CONFIG = {
    "start": "2019-01-01",
    "end": None,               # None = today
    "capital": 100_000,        # INR
    "risk_per_trade": 0.01,    # risk 1% of equity per trade
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_max_entry": 70,       # don't buy when overbought
    "atr_period": 14,
    "atr_stop_mult": 2.0,      # stop = entry - 2*ATR (trailing)
    "cost_pct": 0.0012,        # ~0.12% per side (brokerage+STT+charges+slippage)
}


# ============================ DATA ============================
def load_data(symbol, start, end=None):
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ============================ INDICATORS ============================
def add_indicators(df, c):
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=c["ema_fast"], adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=c["ema_slow"], adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / c["rsi_period"], adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / c["rsi_period"], adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    prev_close = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / c["atr_period"], adjust=False).mean()
    return df.dropna()


# ============================ SIGNALS ============================
def generate_signals(df, c):
    cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift() <= df["ema_slow"].shift())
    cross_dn = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift() >= df["ema_slow"].shift())
    df["buy"] = cross_up & (df["rsi"] < c["rsi_max_entry"])
    df["sell"] = cross_dn
    return df


# ============================ BACKTEST ============================
def backtest(df, c):
    """Signals on day t are executed at day t+1 open (no look-ahead bias)."""
    cash, qty = c["capital"], 0
    entry_px = stop = 0.0
    trades, equity = [], []
    rows = df.reset_index()
    date_col = rows.columns[0]

    for i in range(len(rows) - 1):
        today, nxt = rows.iloc[i], rows.iloc[i + 1]

        # Trailing stop check (intraday low of next bar)
        if qty > 0:
            stop = max(stop, today["Close"] - c["atr_stop_mult"] * today["atr"])
            if nxt["Low"] <= stop:
                px = min(nxt["Open"], stop)  # gap-down fills at open
                cash += qty * px * (1 - c["cost_pct"])
                trades.append((nxt[date_col], "STOP", px, qty, (px - entry_px) * qty))
                qty = 0

        # Exit on signal
        if qty > 0 and today["sell"]:
            px = nxt["Open"]
            cash += qty * px * (1 - c["cost_pct"])
            trades.append((nxt[date_col], "SELL", px, qty, (px - entry_px) * qty))
            qty = 0

        # Entry with risk-based sizing
        elif qty == 0 and today["buy"]:
            px = nxt["Open"]
            risk_per_share = c["atr_stop_mult"] * today["atr"]
            size = int((cash * c["risk_per_trade"]) / risk_per_share)
            size = min(size, int(cash / (px * (1 + c["cost_pct"]))))
            if size > 0:
                qty, entry_px = size, px
                stop = px - risk_per_share
                cash -= qty * px * (1 + c["cost_pct"])
                trades.append((nxt[date_col], "BUY", px, qty, 0.0))

        equity.append((nxt[date_col], cash + qty * nxt["Close"]))

    eq = pd.Series(dict(equity), name="equity")
    tr = pd.DataFrame(trades, columns=["date", "action", "price", "qty", "pnl"])
    return eq, tr


# ============================ METRICS ============================
def metrics(eq, tr, df, capital):
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1
    sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
    max_dd = (eq / eq.cummax() - 1).min()
    closed = tr[tr["action"] != "BUY"]
    win_rate = (closed["pnl"] > 0).mean() if len(closed) else 0
    bh = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    return {
        "Final Equity": f"₹{eq.iloc[-1]:,.0f}",
        "Total Return": f"{eq.iloc[-1] / capital - 1:.2%}",
        "Buy & Hold": f"{bh:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Trades": len(closed),
        "Win Rate": f"{win_rate:.1%}",
    }


def plot(df, eq, tr, symbol):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    a1.plot(df.index, df["Close"], label="Close", lw=1)
    a1.plot(df.index, df["ema_fast"], label="EMA fast", lw=0.8)
    a1.plot(df.index, df["ema_slow"], label="EMA slow", lw=0.8)
    b, s = tr[tr.action == "BUY"], tr[tr.action != "BUY"]
    a1.scatter(b.date, b.price, marker="^", c="g", s=60, label="Buy")
    a1.scatter(s.date, s.price, marker="v", c="r", s=60, label="Exit")
    a1.set_title(symbol); a1.legend()
    a2.plot(eq.index, eq, c="purple"); a2.set_title("Equity Curve")
    plt.tight_layout()
    plt.savefig(f"{symbol.replace('.', '_')}_backtest.png", dpi=120)
    plt.show()


# ============================ MAIN ============================
def run(symbol, c=CONFIG, show_plot=True):
    df = generate_signals(add_indicators(load_data(symbol, c["start"], c["end"]), c), c)
    eq, tr = backtest(df, c)
    print(f"\n===== {symbol} =====")
    for k, v in metrics(eq, tr, df, c["capital"]).items():
        print(f"{k:<14}: {v}")
    tr.to_csv(f"{symbol.replace('.', '_')}_trades.csv", index=False)
    if show_plot:
        plot(df, eq, tr, symbol)
    return eq, tr


if __name__ == "__main__":
    symbols = sys.argv[1:] or ["RELIANCE.NS"]
    for s in symbols:
        try:
            run(s, show_plot=len(symbols) == 1)
        except Exception as e:
            print(f"{s}: {e}")
