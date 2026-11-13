"""
Stock checklist + forecast tool.  Educational only, NOT financial advice.

Setup (VS Code terminal, venv active):
    pip install yfinance numpy pandas matplotlib

Run:
    python stock_analyser.py          # uses TICKER below
    python stock_analyser.py AMD      # or pass a ticker on the command line
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ====================== SETTINGS (edit these) ======================
TICKER = "NVDA"
HORIZON = 60               # trading days to forecast (~3 months)
N_SIMS = 5000              # Monte Carlo paths

ACCOUNT_SIZE = 10_000      # your account size in USD (used for position sizing)
RISK_PER_TRADE_PCT = 1.0   # max % of account you'd lose if the stop is hit
MAX_POSITION_PCT = 10.0    # cap on a single position, % of account (keep small as a beginner)
ATR_STOP_MULT = 2.0        # stop-loss = price - (this x ATR)

# Rough long-run sector P/E references, only a ballpark. Edit freely.
SECTOR_PE = {
    "Technology": 28, "Healthcare": 22, "Financial Services": 14,
    "Consumer Cyclical": 22, "Communication Services": 20,
    "Consumer Defensive": 20, "Energy": 12, "Industrials": 20,
    "Utilities": 17, "Basic Materials": 16, "Real Estate": 35,
}
# ===================================================================


# ---------------------------- data ----------------------------
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data found for '{ticker}'. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def get_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg": info.get("trailingPegRatio") or info.get("pegRatio"),
    }


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd.ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ------------------------ checklist items ------------------------
# Each check returns (score, explanation): +1 favourable, 0 neutral, -1 unfavourable.
def check_valuation(f: dict):
    pe = f["forward_pe"] or f["trailing_pe"]
    peg = f["peg"]
    sector = f["sector"]
    if pe is None and peg is None:
        return 0, "No P/E or PEG data (ETF, or company without earnings). Not scored."
    if pe is not None and pe <= 0:
        return -1, "Negative earnings, so P/E isn't meaningful."

    scores, notes = [], []
    if peg is not None and peg > 0:
        scores.append(1 if peg <= 1.5 else 0 if peg <= 2.5 else -1)
        notes.append(f"PEG {peg:.2f} (<=1.5 cheap-ish, >2.5 pricey)")
    if pe is not None and sector in SECTOR_PE:
        ratio = pe / SECTOR_PE[sector]
        scores.append(1 if ratio <= 1.0 else 0 if ratio <= 1.5 else -1)
        notes.append(f"P/E {pe:.1f} vs rough {sector} reference {SECTOR_PE[sector]} ({ratio:.1f}x)")
    elif pe is not None:
        notes.append(f"P/E {pe:.1f} (no sector reference)")

    if not scores:
        return 0, "; ".join(notes) + ". Not scored."
    total = sum(scores)
    return (1 if total > 0 else -1 if total < 0 else 0), "; ".join(notes)


def check_trend(df: pd.DataFrame):
    last = df.iloc[-1]
    sma50_slope = df["SMA50"].iloc[-1] - df["SMA50"].iloc[-21]
    if last["Close"] > last["SMA200"] and sma50_slope > 0:
        return 1, "Uptrend: price above 200-day SMA and 50-day SMA rising."
    change_10d = df["Close"].iloc[-1] / df["Close"].iloc[-11] - 1
    if change_10d > 0 and last["Close"] > last["EMA20"]:
        return 0, f"Stabilising: up {change_10d:.1%} over 10 days and above the 20-day EMA, but not a confirmed uptrend."
    return -1, "Downtrend: no confirmed uptrend and no stabilisation yet."


def check_moving_averages(df: pd.DataFrame):
    last, prev = df.iloc[-1], df.iloc[-2]
    levels = {"EMA20": last["EMA20"], "SMA50": last["SMA50"], "SMA200": last["SMA200"]}
    above = [k for k, v in levels.items() if last["Close"] > v]
    bounce = [k for k, v in levels.items()
              if abs(last["Close"] / v - 1) <= 0.02 and last["Close"] > prev["Close"]]
    txt = f"Above: {', '.join(above) or 'none'}."
    if len(above) == 3:
        return 1, txt
    if bounce:
        return 0, txt + f" Bouncing near {', '.join(bounce)}."
    return (0 if len(above) == 2 else -1), txt


def check_rsi(df: pd.DataFrame):
    rsi = df["RSI"]
    now, recent_min = rsi.iloc[-1], rsi.iloc[-11:-1].min()
    if now > 70:
        return -1, f"RSI {now:.0f}: overbought."
    if recent_min < 30 <= now:
        return 1, f"RSI {now:.0f}: just came out of oversold (was {recent_min:.0f} recently)."
    if now < 30:
        return 0, f"RSI {now:.0f}: still oversold, no turn yet."
    return 0, f"RSI {now:.0f}: neutral zone."


def check_macd_volume(df: pd.DataFrame):
    hist, last = df["MACD_HIST"], df.iloc[-1]
    momentum_up = hist.iloc[-1] > hist.iloc[-4]
    bullish = last["MACD"] > last["MACD_SIGNAL"]
    w = df.iloc[-10:]
    d = w["Close"].diff()
    vol_ok = w.loc[d > 0, "Volume"].sum() > w.loc[d < 0, "Volume"].sum()
    txt = (f"MACD histogram {'rising' if momentum_up else 'falling'}, "
           f"MACD {'above' if bullish else 'below'} signal; "
           f"volume {'confirms' if vol_ok else 'does not confirm'} (up-day vs down-day volume, 10d).")
    if momentum_up and bullish and vol_ok:
        return 1, txt
    return (0 if momentum_up else -1), txt


def risk_plan(df: pd.DataFrame, account_size: float, risk_pct: float, max_pos_pct: float) -> dict:
    price, atr = float(df["Close"].iloc[-1]), float(df["ATR"].iloc[-1])
    stop = price - ATR_STOP_MULT * atr
    risk_per_share = price - stop
    shares_by_risk = account_size * risk_pct / 100 / risk_per_share
    shares_by_cap = account_size * max_pos_pct / 100 / price
    shares = int(min(shares_by_risk, shares_by_cap))
    return {
        "price": price, "stop": stop, "stop_pct": risk_per_share / price,
        "shares": shares, "position_value": shares * price,
        "max_loss": shares * risk_per_share,
    }


def verdict(total: int) -> str:
    if total >= 3:
        return "Mostly bullish setup"
    if total >= 1:
        return "Leaning positive, mixed signals"
    if total == 0:
        return "Neutral / mixed, no clear edge"
    if total >= -2:
        return "Leaning negative, mixed signals"
    return "Mostly bearish setup"


# --------------------------- forecast ---------------------------
def monte_carlo(prices: pd.Series, horizon: int, n_sims: int, neutral_drift: bool) -> np.ndarray:
    r = np.log(prices).diff().dropna().values
    if neutral_drift:
        r = r - r.mean()  # removes the historical up/down trend so the median stays near today's price
    sampled = np.random.choice(r, size=(n_sims, horizon))
    return prices.iloc[-1] * np.exp(sampled.cumsum(axis=1))


# ---------------------------- analysis ----------------------------
def analyse(ticker: str, account_size: float = ACCOUNT_SIZE, risk_pct: float = RISK_PER_TRADE_PCT,
            max_pos_pct: float = MAX_POSITION_PCT, horizon: int = HORIZON, n_sims: int = N_SIMS) -> dict:
    df = add_indicators(load_data(ticker))
    fund = get_fundamentals(ticker)

    checks = {
        "Valuation": check_valuation(fund),
        "Trend": check_trend(df),
        "Moving averages": check_moving_averages(df),
        "RSI": check_rsi(df),
        "MACD + Volume": check_macd_volume(df),
    }
    total = sum(s for s, _ in checks.values())

    close = df["Close"]
    neutral = monte_carlo(close, horizon, n_sims, neutral_drift=True)
    hist_drift = monte_carlo(close, horizon, n_sims, neutral_drift=False)

    return {
        "ticker": ticker, "df": df, "fund": fund, "checks": checks, "total": total,
        "risk": risk_plan(df, account_size, risk_pct, max_pos_pct),
        "neutral": neutral, "hist_drift": hist_drift,
        "account_size": account_size, "horizon": horizon,
    }


def forecast_summary(res: dict) -> list:
    price = float(res["df"]["Close"].iloc[-1])
    rows = []
    for label, key in [("Neutral drift", "neutral"), ("Historical drift", "hist_drift")]:
        end = res[key][:, -1]
        p10, p50, p90 = np.percentile(end, [10, 50, 90])
        rows.append({"Scenario": label, "10th pct": p10, "Median": p50,
                     "90th pct": p90, "P(above today)": float((end > price).mean())})
    return rows


def report(res: dict):
    f, df, price = res["fund"], res["df"], float(res["df"]["Close"].iloc[-1])
    print(f"\n===== {res['ticker']} ({f['name']}) =====")
    print(f"Last close: ${price:,.2f}  ({df.index[-1].date()})\n")

    print("CHECKLIST")
    icons = {1: "✅", 0: "⚠️ ", -1: "❌"}
    for name, (score, text) in res["checks"].items():
        print(f"  {icons[score]} {name:<16} {text}")
    print(f"\n  Score: {res['total']:+d} / 5  ->  {verdict(res['total'])}")

    r = res["risk"]
    print("\nRISK PLAN (if you act on this setup)")
    print(f"  Stop-loss:      ${r['stop']:,.2f}  ({r['stop_pct']:.1%} below price, {ATR_STOP_MULT}x ATR)")
    acct = res["account_size"]
    print(f"  Position size:  {r['shares']} shares = ${r['position_value']:,.0f} "
          f"({r['position_value'] / acct:.1%} of account)")
    print(f"  Loss if stopped out: about ${r['max_loss']:,.0f} "
          f"({r['max_loss'] / acct:.2%} of account)")
    if r["shares"] == 0:
        print("  (Position cap or risk limit allows 0 shares at this price/account size.)")

    print(f"\nFORECAST ({res['horizon']} trading days ahead, Monte Carlo)")
    for row in forecast_summary(res):
        print(f"  {row['Scenario']:<16}: 10th ${row['10th pct']:,.2f} | median ${row['Median']:,.2f} | "
              f"90th ${row['90th pct']:,.2f} | P(above today) {row['P(above today)']:.0%}")

    print("\nThis is a rules-based checklist and a statistical simulation, not a prediction or advice.")


def plot(res: dict, show: bool = True):
    df, ticker = res["df"], res["ticker"]
    h = df.iloc[-250:]
    future = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=res["horizon"])
    p10, p50, p90 = np.percentile(res["neutral"], [10, 50, 90], axis=0)
    hist_median = np.percentile(res["hist_drift"], 50, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
    ax1.plot(h.index, h["Close"], color="black", label="Close")
    ax1.plot(h.index, h["EMA20"], lw=1, label="EMA20")
    ax1.plot(h.index, h["SMA50"], lw=1, label="SMA50")
    ax1.plot(h.index, h["SMA200"], lw=1, label="SMA200")
    ax1.plot(future, p50, color="tab:blue", label="Forecast median (neutral drift)")
    ax1.plot(future, hist_median, color="tab:blue", ls="--", lw=1, label="Median (historical drift)")
    ax1.fill_between(future, p10, p90, alpha=0.25, color="tab:blue", label="10–90% range")
    risk = res["risk"]
    ax1.axhline(risk["stop"], color="red", ls=":", lw=1, label=f"Stop ${risk['stop']:,.2f}")
    ax1.set_title(f"{ticker}: {verdict(res['total'])} (score {res['total']:+d}/5)  |  educational, not financial advice")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(h.index, h["RSI"], color="purple")
    ax2.axhline(70, color="red", ls="--", lw=0.8)
    ax2.axhline(30, color="green", ls="--", lw=0.8)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.3)

    ax3.bar(h.index, h["MACD_HIST"], color=np.where(h["MACD_HIST"] >= 0, "green", "red"))
    ax3.plot(h.index, h["MACD"], color="blue", lw=1)
    ax3.plot(h.index, h["MACD_SIGNAL"], color="orange", lw=1)
    ax3.set_ylabel("MACD")
    ax3.grid(alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(h.index[0], future[-1])

    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    ticker = (sys.argv[1] if len(sys.argv) > 1 else TICKER).upper()
    try:
        result = analyse(ticker)
    except ValueError as e:
        raise SystemExit(str(e))
    report(result)
    plot(result)
