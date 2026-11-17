import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import load

HORIZON = 5  # forecast the next 5 trading days

st.set_page_config(page_title="Stock Risk Forecaster", layout="wide")
st.title("📈 Stock Risk Forecaster")
st.caption("Forecasts how much a stock is likely to swing over the next 5 trading days. Not a price prediction.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(ticker):
    return load(ticker)


def build_features(df):
    c = df["Close"]
    ret = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["vol5"] = ret.rolling(5).std()
    f["vol10"] = ret.rolling(10).std()
    f["vol20"] = ret.rolling(20).std()
    f["vol60"] = ret.rolling(60).std()
    f["abs_ret1"] = ret.abs()
    f["abs_ret2"] = ret.abs().shift(1)
    f["range_pct"] = (df["High"] - df["Low"]) / c
    f["vol_ratio"] = f["vol5"] / f["vol60"]
    f = f.replace([np.inf, -np.inf], np.nan)
    f["target"] = ret.rolling(HORIZON).std().shift(-HORIZON)
    return f


def forecast_vol(df):
    f = build_features(df)
    cols = [c for c in f.columns if c != "target"]
    train = f.dropna()
    latest = f[cols].iloc[[-1]]
    if len(train) < 200 or latest.isna().any().any():
        return None, f
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[cols], train["target"])
    pred = max(float(model.predict(latest)[0]), 0.0)
    return pred, f


def show_csv(path, title, caption):
    st.subheader(title)
    st.caption(caption)
    if os.path.exists(path):
        st.dataframe(pd.read_csv(path).round(2), hide_index=True)
    else:
        st.info(f"{path} not found. Run the matching experiment script first.")


ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA, INFY.NS, TCS.NS)", "AAPL").strip().upper()

tab1, tab2, tab3 = st.tabs(["📊 Forecast", "🔬 Research Results", "ℹ️ About"])

# ---------------- Tab 1: Forecast ----------------
with tab1:
    with st.spinner("Loading data..."):
        df = get_prices(ticker) if ticker else None

    if df is None or df.empty:
        st.error("Could not load data for this ticker. Check the symbol and your internet connection.")
    else:
        pred, f = forecast_vol(df)

        if pred is None:
            st.warning("Not enough history to build a forecast for this ticker.")
        else:
            annual = np.sqrt(252) * 100
            hist = f["vol5"].dropna()
            pct = float((hist < pred).mean() * 100)
            level = "Low" if pct < 33 else ("Medium" if pct < 67 else "High")
            icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[level]

            current_price = float(df["Close"].iloc[-1])
            recent_vol = float(f["vol5"].iloc[-1]) * annual
            forecast_annual = pred * annual

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest Close", f"{current_price:,.2f}")
            c2.metric("Recent Swing (last 5 days)", f"{recent_vol:.1f}%")
            c3.metric("Forecast Swing (next 5 days)", f"{forecast_annual:.1f}%")
            c4.metric("Risk Level", f"{icon} {level}")
            st.caption(
                f"Swing is shown as annualized volatility. Risk level compares the forecast with this "
                f"stock's own history over the loaded period (forecast is higher than {pct:.0f}% of past readings)."
            )

            st.subheader("Price and 20-day average")
            price_df = pd.DataFrame({"Close": df["Close"], "MA20": df["Close"].rolling(20).mean()})
            st.line_chart(price_df.tail(250))

            st.subheader("Recent swing over time (annualized %)")
            st.line_chart((f["vol5"] * annual).rename("5-day volatility %").tail(250))

            # Downloads for Power BI (no files written to disk, so no permission errors)
            export = pd.DataFrame(
                {
                    "Date": df.index.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Close": df["Close"].values,
                    "MA20": df["Close"].rolling(20).mean().values,
                    "Volatility5d_Annual_Pct": (f["vol5"] * annual).values,
                }
            )
            summary = pd.DataFrame(
                [
                    {
                        "Ticker": ticker,
                        "Latest_Close": current_price,
                        "Recent_Vol_Pct": recent_vol,
                        "Forecast_Vol_Pct": forecast_annual,
                        "Risk_Level": level,
                        "Percentile": pct,
                    }
                ]
            )
            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download price history (CSV)", export.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_history.csv", mime="text/csv",
            )
            d2.download_button(
                "⬇️ Download forecast summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_forecast.csv", mime="text/csv",
            )

# ---------------- Tab 2: Research Results ----------------
with tab2:
    st.write(
        "These tables come from the experiments behind this app. All use walk-forward testing: "
        "the model is trained on the past, tested on the following month, and the window rolls forward."
    )
    show_csv(
        "results/model_comparison.csv",
        "1. Can ML predict tomorrow's price?",
        "Result: no. Every model was slightly worse than the lazy guess 'tomorrow = today' (vs_Naive_% is negative).",
    )
    show_csv(
        "results/volatility_comparison.csv",
        "2. Can ML forecast how much a stock will swing?",
        "Result: yes, partly. Models cut forecast error versus the baselines (vs_Naive_% is positive). "
        "Forecast correlation is modest (about 0.1 to 0.3), so this is a small edge, not precise tracking.",
    )
    show_csv(
        "results/backtest_summary.csv",
        "3. Does using the risk forecast help a trading rule?",
        "Early result: mixed. Holding less when risk is high sometimes reduced big losses, but not consistently, "
        "and better forecasts did not always mean a better rule. These numbers have not been fully validated yet.",
    )

# ---------------- Tab 3: About ----------------
with tab3:
    st.markdown(
        """
**What it does:** downloads daily prices from Yahoo Finance (yfinance), and forecasts the stock's
volatility, meaning how much it is likely to swing, over the next 5 trading days.

**Method:** a Ridge regression trained on recent volatility levels, absolute returns and daily price range.
The risk level (Low / Medium / High) compares the forecast with the stock's own history.

**What we found:**
- Price changes are not predictable from past prices. No model beat the "tomorrow = today" baseline.
- Volatility is partly predictable, because calm and wild periods come in clusters.

**Limits:** results come from about 5 years of data on 5 stocks. This is a learning and research
project, not financial advice.
"""
    )