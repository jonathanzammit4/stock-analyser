"""Streamlit web app: stock checklist + Monte Carlo forecast. Educational only."""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import stock_analyser as sa

st.set_page_config(page_title="Stock Checklist & Forecast", page_icon="📈", layout="wide")

st.title("📈 Stock Checklist & Forecast")
st.warning(
    "Educational project only, **not financial advice**. The checklist uses standard technical "
    "indicators and the forecast is a statistical simulation. Neither predicts what a stock will do."
)

# ---------------- sidebar inputs ----------------
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker symbol", value="NVDA").strip().upper()
    account = st.number_input("Account size (USD)", min_value=100, value=10_000, step=500)
    risk_pct = st.number_input("Max risk per trade (% of account)", 0.1, 5.0, 1.0, step=0.1)
    max_pos = st.number_input("Max position size (% of account)", 1.0, 100.0, 10.0, step=1.0)
    horizon = st.slider("Forecast horizon (trading days)", 20, 120, 60, step=10)
    run = st.button("Analyse", type="primary", width="stretch")


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(ticker, account, risk_pct, max_pos, horizon):
    return sa.analyse(ticker, account, risk_pct, max_pos, horizon)


if not run:
    st.info("Enter a ticker in the sidebar and press **Analyse**.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()

try:
    with st.spinner(f"Analysing {ticker}..."):
        res = run_analysis(ticker, account, risk_pct, max_pos, horizon)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load data for {ticker} (the data provider may be rate-limiting). Try again shortly. Details: {e}")
    st.stop()

df, fund, risk = res["df"], res["fund"], res["risk"]
price = float(df["Close"].iloc[-1])

# ---------------- headline ----------------
st.subheader(f"{ticker}: {fund['name']}")
c1, c2, c3 = st.columns(3)
c1.metric("Last close", f"${price:,.2f}")
c2.metric("Checklist score", f"{res['total']:+d} / 5")
c3.metric("Setup", sa.verdict(res["total"]))

# ---------------- checklist ----------------
st.markdown("### Checklist")
icons = {1: "✅", 0: "⚠️", -1: "❌"}
table = pd.DataFrame(
    [{"": icons[score], "Check": name, "Result": text} for name, (score, text) in res["checks"].items()]
)
st.dataframe(table, hide_index=True, width="stretch")

# ---------------- risk plan ----------------
st.markdown("### Risk plan")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Stop-loss", f"${risk['stop']:,.2f}", f"-{risk['stop_pct']:.1%}", delta_color="off")
r2.metric("Shares", f"{risk['shares']}")
r3.metric("Position value", f"${risk['position_value']:,.0f}")
r4.metric("Loss if stopped out", f"${risk['max_loss']:,.0f}")
if risk["shares"] == 0:
    st.caption("Your risk or position-size limit allows 0 shares at this price and account size.")

# ---------------- forecast ----------------
st.markdown(f"### Forecast ({res['horizon']} trading days, Monte Carlo)")
rows = sa.forecast_summary(res)
fc = pd.DataFrame(rows).set_index("Scenario")
st.dataframe(
    fc.style.format({"10th pct": "${:,.2f}", "Median": "${:,.2f}", "90th pct": "${:,.2f}", "P(above today)": "{:.0%}"}),
    width="stretch",
)
st.caption(
    "Neutral drift removes the stock's historical trend so the median stays near today's price. "
    "Historical drift assumes the past trend continues. The 10th-90th percentile range is the useful part."
)

fig = sa.plot(res, show=False)
st.pyplot(fig)
plt.close(fig)

st.caption("Data from Yahoo Finance via yfinance; may be delayed or incomplete. Not financial advice.")
