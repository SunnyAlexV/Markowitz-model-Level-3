import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Markowitz Portfolio Optimizer", layout="wide")

st.title("Markowitz Portfolio Optimizer")
st.caption("Exchange selector + company selector + automatic Yahoo Finance ticker mapping")

company_master = pd.DataFrame([
    {"Exchange": "NSE India", "Company": "State Bank of India", "Ticker": "SBIN.NS"},
    {"Exchange": "NSE India", "Company": "Reliance Industries", "Ticker": "RELIANCE.NS"},
    {"Exchange": "NSE India", "Company": "TCS", "Ticker": "TCS.NS"},
    {"Exchange": "NSE India", "Company": "HDFC Bank", "Ticker": "HDFCBANK.NS"},
    {"Exchange": "NSE India", "Company": "Laurus Labs", "Ticker": "LAURUSLABS.NS"},
    {"Exchange": "NSE India", "Company": "HCL Technologies", "Ticker": "HCLTECH.NS"},
    {"Exchange": "NSE India", "Company": "ICICI Bank", "Ticker": "ICICIBANK.NS"},
    {"Exchange": "NSE India", "Company": "Kotak Mahindra Bank", "Ticker": "KOTAKBANK.NS"},
    {"Exchange": "NASDAQ USA", "Company": "Apple", "Ticker": "AAPL"},
    {"Exchange": "NASDAQ USA", "Company": "Meta Platforms", "Ticker": "META"},
    {"Exchange": "NASDAQ USA", "Company": "Alphabet (Google) Class A", "Ticker": "GOOGL"},
    {"Exchange": "NASDAQ USA", "Company": "Alphabet (Google) Class C", "Ticker": "GOOG"},
    {"Exchange": "NASDAQ USA", "Company": "Microsoft", "Ticker": "MSFT"},
    {"Exchange": "NASDAQ USA", "Company": "Amazon", "Ticker": "AMZN"},
    {"Exchange": "NASDAQ USA", "Company": "NVIDIA", "Ticker": "NVDA"},
    {"Exchange": "NASDAQ USA", "Company": "Tesla", "Ticker": "TSLA"},
    {"Exchange": "NYSE USA", "Company": "JPMorgan Chase", "Ticker": "JPM"},
    {"Exchange": "NYSE USA", "Company": "Walmart", "Ticker": "WMT"},
    {"Exchange": "NYSE USA", "Company": "Coca-Cola", "Ticker": "KO"},
    {"Exchange": "NYSE USA", "Company": "Exxon Mobil", "Ticker": "XOM"},
    {"Exchange": "LSE UK", "Company": "Barclays", "Ticker": "BARC.L"},
    {"Exchange": "LSE UK", "Company": "HSBC", "Ticker": "HSBA.L"},
    {"Exchange": "LSE UK", "Company": "BP", "Ticker": "BP.L"},
    {"Exchange": "LSE UK", "Company": "Vodafone", "Ticker": "VOD.L"},
    {"Exchange": "TSX Canada", "Company": "Shopify", "Ticker": "SHOP.TO"},
    {"Exchange": "TSX Canada", "Company": "Royal Bank of Canada", "Ticker": "RY.TO"},
    {"Exchange": "TSX Canada", "Company": "Toronto-Dominion Bank", "Ticker": "TD.TO"},
])

with st.sidebar:
    st.header("Market selection")
    selected_exchange = st.selectbox(
        "Select stock exchange",
        sorted(company_master["Exchange"].unique())
    )

    available_companies = company_master.loc[
        company_master["Exchange"] == selected_exchange, "Company"
    ].tolist()

    selected_companies = st.multiselect(
        "Select company / companies",
        available_companies,
        default=available_companies[:3] if len(available_companies) >= 3 else available_companies
    )

    manual_tickers = st.text_input(
        "Optional extra tickers (comma separated)",
        "",
        help="Use this if your company is not in the dropdown list. Example: INFY, ULVR.L, 0700.HK"
    )

    start_date = st.date_input("Estimation window start date", pd.to_datetime("2024-01-01"))
    end_date = st.date_input("Estimation window end date", pd.to_datetime("2025-12-31"))
    total_capital = st.number_input("Total capital", min_value=0.0, value=100000.0, step=1000.0)
    target_return = st.number_input("Target annual return", value=0.09, step=0.01, format="%.4f")
    allow_shorts = st.checkbox("Allow short selling", value=True)
    run_button = st.button("Run optimization", use_container_width=True)

selected_df = company_master[
    (company_master["Exchange"] == selected_exchange) &
    (company_master["Company"].isin(selected_companies))
].copy()

auto_tickers = selected_df["Ticker"].tolist()
manual_list = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
tickers = list(dict.fromkeys(auto_tickers + manual_list))

st.subheader("Selected universe")
st.dataframe(selected_df[["Exchange", "Company", "Ticker"]], use_container_width=True)
if manual_list:
    st.write("Manual tickers added:", manual_list)

if run_button:
    if len(tickers) < 2:
        st.error("Please select at least two stocks for portfolio optimization.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    st.subheader("1. Downloading price data")
    data_raw = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if data_raw.empty:
        st.error("No raw data returned. Check selected tickers and dates.")
        st.stop()

    if isinstance(data_raw.columns, pd.MultiIndex):
        if "Close" in data_raw.columns.get_level_values(0):
            data = data_raw["Close"].copy()
        else:
            data = data_raw.xs(data_raw.columns.levels[0][0], axis=1, level=0).copy()
    else:
        data = data_raw[["Close"]].copy() if "Close" in data_raw.columns else data_raw.copy()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(how="all")
    available_tickers = data.columns.tolist()
    missing_tickers = [t for t in tickers if t not in available_tickers]

    if data.empty or len(available_tickers) < 2:
        st.error("Not enough price data returned for optimization. Try different stocks or dates.")
        if missing_tickers:
            st.write("Tickers with missing data:", missing_tickers)
        st.stop()

    if missing_tickers:
        st.warning(f"Dropped tickers with no usable data: {', '.join(missing_tickers)}")

    data = data[available_tickers].dropna()

    if data.empty or data.shape[0] < 2:
        st.error("Price data became empty after aligning dates across tickers.")
        st.stop()

    last_estimation_date = data.index[-1]
    st.write("Last estimation date used:", last_estimation_date.date())
    st.dataframe(data.tail(), use_container_width=True)

    st.subheader("2. Returns and risk")
    daily_returns = data.pct_change().dropna()

    mu = daily_returns.mean() * 252
    cov = daily_returns.cov() * 252
    sigma = np.sqrt(np.diag(cov))

    stats_df = pd.DataFrame({
        "Ticker": available_tickers,
        "Expected Return": mu.values,
        "Volatility": sigma
    })
    st.dataframe(stats_df.style.format({"Expected Return": "{:.4f}", "Volatility": "{:.4f}"}), use_container_width=True)

    st.subheader("3. Markowitz weights")
    mu_vec = mu.values.reshape(-1, 1)
    cov_mat = cov.values
    ones = np.ones((len(available_tickers), 1))

    try:
        cov_inv = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_mat)
        st.warning("Covariance matrix was singular; used pseudo-inverse instead of inverse.")

    if allow_shorts:
        A = ones.T @ cov_inv @ ones
        B = ones.T @ cov_inv @ mu_vec
        C = mu_vec.T @ cov_inv @ mu_vec
        D = A * C - B**2

        if np.isclose(D.item(), 0):
            st.error("Optimization failed because D is numerically close to zero.")
            st.stop()

        m = target_return
        Lambda = (A * m - B) / D
        Gamma = (C - B * m) / D
        weights_vec = cov_inv @ (Lambda * mu_vec + Gamma * ones)
        weights = weights_vec.flatten()
    else:
        raw = cov_inv @ mu_vec
        weights = raw.flatten() / raw.sum()

    weights_df = pd.DataFrame({
        "Ticker": available_tickers,
        "Weight": weights,
        "Weight %": weights * 100
    })
    st.dataframe(weights_df.style.format({"Weight": "{:.4f}", "Weight %": "{:.2f}"}), use_container_width=True)

    portfolio_return = float(weights @ mu.values)
    portfolio_vol = float(np.sqrt(weights @ cov_mat @ weights))

    col1, col2 = st.columns(2)
    col1.metric("Portfolio expected annual return", f"{portfolio_return:.2%}")
    col2.metric("Portfolio annual volatility", f"{portfolio_vol:.2%}")

    st.subheader("4. Execution prices and share sizing")
    execution_date = data.index[-1]
    prices_exec = data.loc[execution_date]
    amount_per_asset = weights * total_capital

    shares = {}
    for i, t in enumerate(available_tickers):
        price = float(prices_exec[t])
        amt = float(amount_per_asset[i])
        shares[t] = int(np.floor(abs(amt) / price)) * (1 if amt >= 0 else -1)

    alloc_df = pd.DataFrame({
        "Ticker": available_tickers,
        "Weight": weights,
        "Capital Allocation": amount_per_asset,
        "Execution Price": [float(prices_exec[t]) for t in available_tickers],
        "Shares": [shares[t] for t in available_tickers]
    })

    st.write(f"Execution date used: **{execution_date.date()}**")
    st.dataframe(
        alloc_df.style.format({
            "Weight": "{:.4f}",
            "Capital Allocation": "{:.2f}",
            "Execution Price": "{:.2f}",
            "Shares": "{:d}"
        }),
        use_container_width=True
    )

    st.success("Optimization and sizing completed.")
