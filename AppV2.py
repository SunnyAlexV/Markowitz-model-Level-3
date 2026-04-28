import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- CONFIG --------------------

st.set_page_config(page_title="Markowitz Portfolio Optimizer (FX-aware)", layout="wide")

st.title("Markowitz Portfolio Optimizer (FX-aware)")
st.caption("Mix tickers across exchanges, optimize in USD, size in local currencies")

BASE_CCY = "USD"  # base currency for optimization and user capital

# -------------------- MASTER UNIVERSE --------------------

company_master = pd.DataFrame([
    # NSE India (INR)
    {"Exchange": "NSE India", "Company": "State Bank of India", "Ticker": "SBIN.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "Reliance Industries", "Ticker": "RELIANCE.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "TCS", "Ticker": "TCS.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "HDFC Bank", "Ticker": "HDFCBANK.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "Laurus Labs", "Ticker": "LAURUSLABS.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "HCL Technologies", "Ticker": "HCLTECH.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "ICICI Bank", "Ticker": "ICICIBANK.NS", "Currency": "INR"},
    {"Exchange": "NSE India", "Company": "Kotak Mahindra Bank", "Ticker": "KOTAKBANK.NS", "Currency": "INR"},

    # NASDAQ USA (USD)
    {"Exchange": "NASDAQ USA", "Company": "Apple",          "Ticker": "AAPL",  "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "Meta Platforms", "Ticker": "META",  "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "Alphabet (Class A)", "Ticker": "GOOGL", "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "Alphabet (Class C)", "Ticker": "GOOG",  "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "Microsoft",      "Ticker": "MSFT",  "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "Amazon",         "Ticker": "AMZN",  "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "NVIDIA",         "Ticker": "NVDA",  "Currency": "USD"},
    {"Exchange": "NASDAQ USA", "Company": "Tesla",          "Ticker": "TSLA",  "Currency": "USD"},

    # NYSE USA (USD)
    {"Exchange": "NYSE USA", "Company": "JPMorgan Chase", "Ticker": "JPM", "Currency": "USD"},
    {"Exchange": "NYSE USA", "Company": "Walmart",        "Ticker": "WMT", "Currency": "USD"},
    {"Exchange": "NYSE USA", "Company": "Coca-Cola",      "Ticker": "KO",  "Currency": "USD"},
    {"Exchange": "NYSE USA", "Company": "Exxon Mobil",    "Ticker": "XOM", "Currency": "USD"},

    # LSE UK (GBP)
    {"Exchange": "LSE UK", "Company": "Barclays",  "Ticker": "BARC.L", "Currency": "GBP"},
    {"Exchange": "LSE UK", "Company": "HSBC",      "Ticker": "HSBA.L", "Currency": "GBP"},
    {"Exchange": "LSE UK", "Company": "BP",        "Ticker": "BP.L",   "Currency": "GBP"},
    {"Exchange": "LSE UK", "Company": "Vodafone",  "Ticker": "VOD.L",  "Currency": "GBP"},

    # TSX Canada (CAD)
    {"Exchange": "TSX Canada", "Company": "Shopify",                "Ticker": "SHOP.TO", "Currency": "CAD"},
    {"Exchange": "TSX Canada", "Company": "Royal Bank of Canada",  "Ticker": "RY.TO",   "Currency": "CAD"},
    {"Exchange": "TSX Canada", "Company": "Toronto-Dominion Bank", "Ticker": "TD.TO",   "Currency": "CAD"},
])

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("Universe & parameters")

    # Exchange just for browsing; user can still mix via manual tickers
    selected_exchange = st.selectbox(
        "Browse companies by exchange",
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
        help="You can mix exchanges, e.g. AAPL, SBIN.NS, BP.L, SHOP.TO"
    )

    start_date = st.date_input("Estimation window start date", pd.to_datetime("2024-01-01"))
    end_date   = st.date_input("Estimation window end date",   pd.to_datetime("2025-12-31"))

    total_capital_base = st.number_input(
        f"Total capital in {BASE_CCY}",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
    )

    target_return = st.number_input(
        "Target annual return (in base currency, e.g. 0.10 = 10%)",
        value=0.10,
        step=0.01,
        format="%.4f"
    )

    allow_shorts = st.checkbox("Allow short selling", value=True)

    run_button = st.button("Run optimization", use_container_width=True)

# -------------------- BUILD TICKER + CCY LISTS --------------------

selected_df = company_master[
    (company_master["Exchange"] == selected_exchange) &
    (company_master["Company"].isin(selected_companies))
].copy()

auto_tickers = selected_df["Ticker"].tolist()
auto_ccy     = selected_df["Currency"].tolist()

# Manual tickers – assume USD by default; you can extend by asking user for currency
manual_list = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
manual_ccy  = [BASE_CCY] * len(manual_list)

tickers    = auto_tickers + manual_list
currencies = auto_ccy     + manual_ccy

universe_df = pd.DataFrame({"Ticker": tickers, "Currency": currencies})

st.subheader("Selected universe (before data check)")
if not universe_df.empty:
    st.dataframe(universe_df, use_container_width=True)

# -------------------- FX HELPER --------------------

@st.cache_data
def get_fx_series(local_ccy_list, start, end, base_ccy=BASE_CCY):
    """
    For each non-base currency in local_ccy_list, download a Yahoo FX pair
    base_ccy + local, e.g. USDINR=X, USDGBP=X, and return a DataFrame
    with columns named by local currency.

    Convention: fx_rate[ccy] = how many units of local_ccy per 1 base_ccy.
    Then: price_in_base = price_in_local / fx_rate.
    """
    unique_ccy = sorted(set([c for c in local_ccy_list if c != base_ccy]))
    if not unique_ccy:
        return pd.DataFrame()

    fx_symbols = {ccy: f"{base_ccy}{ccy}=X" for ccy in unique_ccy}

    data_fx = yf.download(
        list(fx_symbols.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data_fx.empty:
        return pd.DataFrame()

    # Extract closing FX; handle MultiIndex or simple columns
    if isinstance(data_fx.columns, pd.MultiIndex):
        if "Adj Close" in data_fx.columns.get_level_values(0):
            fx_close = data_fx["Adj Close"].copy()
        else:
            fx_close = data_fx["Close"].copy()
    else:
        fx_close = data_fx.copy()

    fx_close = fx_close.dropna(how="all")

    # Map columns back to currency codes
    rename_map = {}
    for ccy, sym in fx_symbols.items():
        for col in fx_close.columns:
            if sym in str(col):
                rename_map[col] = ccy

    fx_close = fx_close.rename(columns=rename_map)

    # Keep only relevant currency columns
    kept_cols = [c for c in unique_ccy if c in fx_close.columns]
    fx_close = fx_close[kept_cols]

    return fx_close

# -------------------- MAIN WORKFLOW --------------------

if run_button:
    if len(tickers) < 2:
        st.error("Please select at least two stocks.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    st.subheader("1. Downloading local-currency price data")

    data_raw = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data_raw.empty:
        st.error("No raw data returned. Check tickers and dates.")
        st.stop()

    # Extract close prices: rows=dates, cols=tickers
    if isinstance(data_raw.columns, pd.MultiIndex):
        if "Close" in data_raw.columns.get_level_values(0):
            data_local = data_raw["Close"].copy()
        else:
            data_local = data_raw.xs(data_raw.columns.levels[0][0], axis=1, level=0).copy()
    else:
        data_local = data_raw[["Close"]].copy() if "Close" in data_raw.columns else data_raw.copy()

    if isinstance(data_local, pd.Series):
        data_local = data_local.to_frame()

    data_local = data_local.dropna(how="all")

    # Filter out tickers with no data
    available_tickers = data_local.columns.tolist()
    missing_tickers = [t for t in tickers if t not in available_tickers]

    if data_local.empty or len(available_tickers) < 2:
        st.error("Not enough price data for optimization.")
        if missing_tickers:
            st.write("Tickers with no data:", missing_tickers)
        st.stop()

    if missing_tickers:
        st.warning(f"Dropped tickers with no usable data: {', '.join(missing_tickers)}")

    # Align the universe to those tickers
    universe_df = universe_df[universe_df["Ticker"].isin(available_tickers)].copy()
    available_tickers = universe_df["Ticker"].tolist()
    data_local = data_local[available_tickers].dropna()

    if data_local.empty or data_local.shape[0] < 2:
        st.error("Data became empty after date alignment.")
        st.stop()

    st.write("Local-currency prices (tail):")
    st.dataframe(data_local.tail(), use_container_width=True)

    # -------------------- 2. FX CONVERSION: LOCAL -> USD --------------------

    st.subheader(f"2. FX conversion to {BASE_CCY} prices")

    # Currencies in same order as available_tickers
    local_ccy = universe_df.set_index("Ticker").loc[available_tickers, "Currency"].tolist()

    fx_df = get_fx_series(local_ccy, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # Build base-currency price DataFrame
    data_base = pd.DataFrame(index=data_local.index)

    for i, t in enumerate(available_tickers):
        ccy = local_ccy[i]
        series_local = data_local[t]

        if ccy == BASE_CCY:
            data_base[t] = series_local  # already in USD
        else:
            if fx_df.empty or ccy not in fx_df.columns:
                st.error(
                    f"Missing FX data for {ccy}. Cannot convert ticker {t} to {BASE_CCY}.\n"
                    f"Check that Yahoo Finance has the pair {BASE_CCY}{ccy}=X."
                )
                st.stop()
            # Align FX series to price dates
            fx_series = fx_df[ccy].reindex(series_local.index).ffill().bfill()
            # price in base = local / (local units per base)
            data_base[t] = series_local / fx_series

    data_base = data_base.dropna(how="all")
    if data_base.empty or data_base.shape[0] < 2:
        st.error("FX-converted price data insufficient.")
        st.stop()

    st.write(f"{BASE_CCY} prices (tail):")
    st.dataframe(data_base.tail(), use_container_width=True)

    # -------------------- 3. RETURNS, COVARIANCE, MARKOWITZ --------------------

    st.subheader(f"3. Markowitz weights in {BASE_CCY}")

    daily_returns = data_base.pct_change().dropna()

    mu = daily_returns.mean() * 252  # expected annual return in BASE_CCY
    cov = daily_returns.cov() * 252  # annual covariance matrix in BASE_CCY
    sigma = np.sqrt(np.diag(cov))

    stats_df = pd.DataFrame({
        "Ticker":   available_tickers,
        "Currency": local_ccy,
        "Exp Return (base)": mu.values,
        "Volatility (base)": sigma,
    })

    st.write("Asset statistics (in base currency):")
    st.dataframe(
        stats_df.style.format({"Exp Return (base)": "{:.4f}", "Volatility (base)": "{:.4f}"}),
        use_container_width=True
    )

    mu_vec = mu.values.reshape(-1, 1)
    cov_mat = cov.values
    ones = np.ones((len(available_tickers), 1))

    # Invert covariance (with pseudo-inverse fallback)
    try:
        cov_inv = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_mat)
        st.warning("Covariance matrix singular; used pseudo-inverse.")

    if allow_shorts:
        A = ones.T @ cov_inv @ ones
        B = ones.T @ cov_inv @ mu_vec
        C = mu_vec.T @ cov_inv @ mu_vec
        D = A * C - B**2

        if np.isclose(D.item(), 0):
            st.error("Optimization failed (D ≈ 0). Try different assets or dates.")
            st.stop()

        m = target_return
        Lambda = (A * m - B) / D
        Gamma  = (C - B * m) / D

        weights_vec = cov_inv @ (Lambda * mu_vec + Gamma * ones)
        weights = weights_vec.flatten()
    else:
        raw = cov_inv @ mu_vec
        weights = raw.flatten() / raw.sum()

    weights_df = pd.DataFrame({
        "Ticker":    available_tickers,
        "Currency":  local_ccy,
        "Weight":    weights,
        "Weight %":  weights * 100,
    })

    st.write("Optimal portfolio weights (can be negative if shorts allowed):")
    st.dataframe(
        weights_df.style.format({"Weight": "{:.4f}", "Weight %": "{:.2f}"}),
        use_container_width=True
    )

    portfolio_return = float(weights @ mu.values)
    portfolio_vol    = float(np.sqrt(weights @ cov_mat @ weights))

    col1, col2 = st.columns(2)
    col1.metric("Portfolio expected annual return", f"{portfolio_return:.2%}")
    col2.metric("Portfolio annual volatility",      f"{portfolio_vol:.2%}")

    # -------------------- 4. EXECUTION PRICES & SIZING --------------------

    st.subheader("4. Execution prices and share sizing (local currencies)")

    execution_date = data_local.index[-1]
    prices_exec_local = data_local.loc[execution_date]

    # Last FX rates for conversion at execution date
    if not fx_df.empty:
        fx_exec = fx_df.reindex([execution_date]).ffill().bfill().iloc[0]
    else:
        fx_exec = pd.Series(dtype=float)  # no FX needed if everything is in BASE_CCY

    # Capital per asset in base currency
    amount_per_asset_base = weights * total_capital_base

    alloc_rows = []
    for i, t in enumerate(available_tickers):
        ccy = local_ccy[i]
        price_local = float(prices_exec_local[t])
        capital_base = float(amount_per_asset_base[i])

        if ccy == BASE_CCY:
            capital_local = capital_base
        else:
            if ccy not in fx_exec.index:
                st.error(f"Missing execution FX rate for {ccy} at {execution_date.date()}.")
                st.stop()
            fx_rate = float(fx_exec[ccy])  # local per 1 base
            capital_local = capital_base * fx_rate

        shares = int(np.floor(abs(capital_local) / price_local)) * (1 if capital_local >= 0 else -1)

        alloc_rows.append({
            "Ticker":                 t,
            "Currency":               ccy,
            "Weight":                 weights[i],
            "Weight %":               weights[i] * 100,
            f"Capital ({BASE_CCY})":  capital_base,
            f"Capital ({ccy})":       capital_local,
            "Execution Price (local)": price_local,
            "Shares":                 shares,
        })

    alloc_df = pd.DataFrame(alloc_rows)

    st.write(f"Execution date used: **{execution_date.date()}**")
    st.dataframe(
        alloc_df.style.format({
            "Weight": "{:.4f}",
            "Weight %": "{:.2f}",
            f"Capital ({BASE_CCY})": "{:.2f}",
            "Capital (INR)": "{:.2f}",
            "Capital (GBP)": "{:.2f}",
            "Capital (CAD)": "{:.2f}",
            "Execution Price (local)": "{:.2f}",
            "Shares": "{:d}",
        }),
        use_container_width=True
    )

    st.success("Optimization and FX-aware sizing completed.")
