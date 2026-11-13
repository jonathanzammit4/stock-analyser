# Stock Checklist & Forecast

A small Python project that scores a stock against a beginner-friendly technical checklist, computes a
risk-based stop-loss and position size, and simulates a range of possible future prices.

**Live demo:** _add your Streamlit link here after deploying_

> **Disclaimer:** This is an educational project, **not financial advice**. The checklist uses standard
> technical indicators and the forecast is a statistical simulation. Neither can predict what a stock will do.
> Data comes from Yahoo Finance via `yfinance` and may be delayed or incomplete.

## What it does

Enter a ticker and the app runs these checks, each scored favourable (+1), neutral (0) or unfavourable (-1):

| Check | Logic |
|---|---|
| Valuation | PEG ratio and P/E compared with a rough sector reference |
| Trend | Price above the 200-day SMA with the 50-day SMA rising (or at least stabilising) |
| Moving averages | Price vs EMA20, SMA50 and SMA200, plus bounces within 2% of one |
| RSI | Rewards a recent exit from oversold (<30), penalises overbought (>70) |
| MACD + Volume | MACD histogram rising, MACD above signal, up-day volume above down-day volume |

It also produces:

- **Risk plan:** ATR-based stop-loss and a position size from your account size, max risk per trade and a position cap.
- **Forecast:** a 20-120 trading day Monte Carlo simulation (bootstrapped historical returns) with a 10th-90th percentile range, shown with neutral and historical drift.

The overall score describes how many of the rules the stock currently meets. It is not a buy or sell signal.

## Run it locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # web app
python stock_analyser.py AMD     # or the command-line version
```

## Project structure

```
app.py              Streamlit web interface
stock_analyser.py   Data loading, indicators, checklist, risk plan, forecast, charts
requirements.txt    Dependencies
```

## Limitations

- Technical indicators describe past price behaviour; they have no knowledge of earnings, news or macro events.
- Sector P/E references are rough and hard-coded. Yahoo's PEG and P/E data is sometimes missing or inconsistent.
- The Monte Carlo model assumes future returns resemble past returns, so it understates tail risk.
- The checklist has not been backtested. Treat it as a structured way to look at a chart, not a strategy.

## Possible next steps

- Backtest whether higher checklist scores historically preceded better returns
- Compare a watchlist of tickers in one ranked table
- Add a pretrained time-series model (e.g. Chronos) alongside the Monte Carlo forecast

## Tech

Python, pandas, NumPy, matplotlib, yfinance, Streamlit

## Quick start (my Mac)

**First time only:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Every time after that:**
```bash
source .venv/bin/activate
streamlit run app.py
```

Stop the app with `Ctrl+C`. Open a second terminal if I need to run other commands while it's running.