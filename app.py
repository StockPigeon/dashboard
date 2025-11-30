import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
import requests


# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Equity Valuation Dashboard",
    layout="wide",
)

# -------------------------
# DATA LOADING
# -------------------------
@st.cache_data
def load_data():
    sp500_raw = pd.read_csv("sp500_valuations_raw.csv")
    nasdaq_raw = pd.read_csv("nasdaq100_valuations_raw.csv")

    # Add index label (these weren't in the original CSVs)
    sp500_raw["index"] = "S&P 500"
    nasdaq_raw["index"] = "NASDAQ 100"

    all_raw = pd.concat([sp500_raw, nasdaq_raw], ignore_index=True)

    return sp500_raw, nasdaq_raw, all_raw


sp500_raw, nasdaq_raw, all_raw = load_data()

# Ensure consistent types
all_raw.columns = [c.strip() for c in all_raw.columns]


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def numericify(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_metric_columns(df):
    metrics = [
        "peRatioTTM",
        "priceToSalesRatioTTM",
        "priceToBookRatioTTM",
        "evToEbitdaTTM",
        "enterpriseValueOverEBITDATTM",
        "enterpriseValueMultiple",
        "freeCashFlowPerShareTTM",
        "roeTTM",
        "roaTTM",
        "grossMarginTTM",
        "operatingMarginTTM",
    ]
    return [m for m in metrics if m in df.columns]


metric_cols = get_metric_columns(all_raw)


def clean_pe(df, pe_col):
    df = df.copy()
    df[pe_col] = pd.to_numeric(df[pe_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[pe_col])
    df = df[df[pe_col] > 0]
    low, high = df[pe_col].quantile([0.01, 0.99])
    df[pe_col] = df[pe_col].clip(lower=low, upper=high)
    return df


def detect_pe_column(df):
    candidates = [
        c for c in df.columns
        if "peratiottm" in c.lower()
        or ("pe" in c.lower() and "ttm" in c.lower())
    ]
    return candidates[0] if candidates else None


def detect_ev_ebitda_column(df):
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "evtoebitda",
            "enterprisevalueoverebitda",
            "ev_ebitda",
            "enterprisevaluemultiple",
        ])
    ]
    if not candidates:
        return None
    for cand in candidates:
        if "evtoebitda" in cand.lower() or "enterprisevalueoverebitda" in cand.lower():
            return cand
    return candidates[0]


def compute_value_quality_scores(df):
    df = df.copy()
    num_cols = [
        "peRatioTTM",
        "priceToBookRatioTTM",
        "priceToSalesRatioTTM",
        "freeCashFlowPerShareTTM",
        "roeTTM",
        "roaTTM",
        "grossMarginTTM",
        "operatingMarginTTM",
    ]
    df = numericify(df, num_cols)

    # Value score: low P/E, P/B, P/S; high FCF/share
    value_components = []
    if "peRatioTTM" in df.columns:
        value_components.append(-df["peRatioTTM"])
    if "priceToBookRatioTTM" in df.columns:
        value_components.append(-df["priceToBookRatioTTM"])
    if "priceToSalesRatioTTM" in df.columns:
        value_components.append(-df["priceToSalesRatioTTM"])
    if "freeCashFlowPerShareTTM" in df.columns:
        value_components.append(df["freeCashFlowPerShareTTM"])

    if value_components:
        vmat = np.column_stack(value_components)
        v_z = (vmat - np.nanmean(vmat, axis=0)) / np.nanstd(vmat, axis=0)
        df["ValueScore"] = np.nanmean(v_z, axis=1)
    else:
        df["ValueScore"] = np.nan

    # Quality score: high ROE/ROA/margins
    quality_components = []
    for qcol in ["roeTTM", "roaTTM", "grossMarginTTM", "operatingMarginTTM"]:
        if qcol in df.columns:
            quality_components.append(df[qcol])

    if quality_components:
        qmat = np.column_stack(quality_components)
        q_z = (qmat - np.nanmean(qmat, axis=0)) / np.nanstd(qmat, axis=0)
        df["QualityScore"] = np.nanmean(q_z, axis=1)
    else:
        df["QualityScore"] = np.nan

    return df


def add_quadrant(df):
    def quad(row):
        if row["ValueScore"] >= 0 and row["QualityScore"] >= 0:
            return "Cheap & Quality"
        if row["ValueScore"] >= 0 and row["QualityScore"] < 0:
            return "Cheap but Low Quality"
        if row["ValueScore"] < 0 and row["QualityScore"] >= 0:
            return "Expensive but Quality"
        return "Expensive & Low Quality"

    df = df.copy()
    df["quadrant"] = df.apply(quad, axis=1)
    return df


# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Controls")

index_filter = st.sidebar.multiselect(
    "Index",
    options=["S&P 500", "NASDAQ 100"],
    default=["S&P 500", "NASDAQ 100"],
)

sector_options = sorted(all_raw["sector"].dropna().unique())
sector_filter = st.sidebar.multiselect(
    "Sector (optional filter)",
    options=sector_options,
    default=[],
)

show_cap_100 = st.sidebar.checkbox("Cap multiples at 100 (P/E & EV/EBITDA)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("CSV files must be in the same folder as app.py")

# Apply filters
df_filtered = all_raw.copy()
if index_filter:
    df_filtered = df_filtered[df_filtered["index"].isin(index_filter)]
if sector_filter:
    df_filtered = df_filtered[df_filtered["sector"].isin(sector_filter)]


# -------------------------
# MAIN LAYOUT
# -------------------------
st.title("Equity Valuation Dashboard")
st.caption("S&P 500 & NASDAQ 100 — sector valuations, spreads, and factor scores")

tab_overview, tab_heatmap, tab_vq, tab_pe_ev, tab_table = st.tabs(
    ["Overview", "Sector Heatmap", "Value vs Quality", "P/E vs EV/EBITDA", "Data Table"]
)

# -------------------------
# OVERVIEW TAB
# -------------------------
with tab_overview:
    st.subheader("Universe Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total companies", f"{len(df_filtered):,}")

    with col2:
        st.metric("Sectors", df_filtered["sector"].nunique())

    with col3:
        pe_col_det = detect_pe_column(df_filtered)
        if pe_col_det:
            dft = numericify(df_filtered, [pe_col_det])
            pe_median = dft[pe_col_det].replace([np.inf, -np.inf], np.nan).median()
            st.metric("Median P/E (overall)", f"{pe_median:.1f}")
        else:
            st.metric("Median P/E (overall)", "N/A")

    st.write("### Index breakdown")
    st.bar_chart(df_filtered["index"].value_counts())


# -------------------------
# SECTOR HEATMAP TAB
# -------------------------
with tab_heatmap:
    st.subheader("Sector P/E Heatmap")

    pe_col = detect_pe_column(df_filtered)
    if not pe_col:
        st.warning("No P/E column detected in the data.")
    else:
        df_pe = clean_pe(df_filtered, pe_col)
        # Sector x Index median P/E
        pivot = (
            df_pe.groupby(["sector", "index"])[pe_col]
            .median()
            .unstack("index")
            .sort_index()
        )

        # Apply cap if requested
        if show_cap_100:
            pivot = pivot.clip(upper=100)

        # Plotly heatmap (green → red)
        fig_hm = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            labels=dict(color="Median P/E"),
        )
        fig_hm.update_layout(
            xaxis_title="Index",
            yaxis_title="Sector",
            title="Median P/E by Sector & Index (outliers clipped)",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.dataframe(pivot.style.format("{:.1f}"))


# -------------------------
# VALUE vs QUALITY TAB
# -------------------------
with tab_vq:
    st.subheader("Value vs Quality Scatter")

    df_vq = compute_value_quality_scores(df_filtered)
    df_vq = df_vq.replace([np.inf, -np.inf], np.nan)
    df_vq = df_vq.dropna(subset=["ValueScore", "QualityScore"])

    if df_vq.empty:
        st.warning("Not enough data to compute Value & Quality scores.")
    else:
        df_vq = add_quadrant(df_vq)

        # Optional sector filter inside this tab
        quad_filter = st.multiselect(
            "Quadrant filter",
            options=["Cheap & Quality", "Cheap but Low Quality",
                     "Expensive but Quality", "Expensive & Low Quality"],
            default=[],
        )
        df_plot = df_vq.copy()
        if quad_filter:
            df_plot = df_plot[df_plot["quadrant"].isin(quad_filter)]

        fig_vq = px.scatter(
            df_plot,
            x="ValueScore",
            y="QualityScore",
            color="quadrant",
            symbol="index",
            hover_name="name",
            hover_data={
                "symbol": True,
                "sector": True,
                "index": True,
                "ValueScore": ":.2f",
                "QualityScore": ":.2f",
            },
            opacity=0.7,
            title="Value vs Quality – factor-style view",
        )

        fig_vq.add_vline(x=0, line_dash="dash", line_width=1)
        fig_vq.add_hline(y=0, line_dash="dash", line_width=1)

        fig_vq.update_layout(
            xaxis_title="ValueScore (higher = cheaper / better value)",
            yaxis_title="QualityScore (higher = better quality)",
            legend_title="Quadrant",
        )

        st.plotly_chart(fig_vq, use_container_width=True)

        st.write("#### Quadrant counts")
        st.dataframe(
            df_vq["quadrant"].value_counts()
            .rename_axis("Quadrant")
            .to_frame("Companies")
        )


# -------------------------
# P/E vs EV/EBITDA TAB
# -------------------------
with tab_pe_ev:
    st.subheader("P/E vs EV/EBITDA Scatter")

    pe_col = detect_pe_column(df_filtered)
    ev_col = detect_ev_ebitda_column(df_filtered)

    if not pe_col or not ev_col:
        st.warning(
            f"Could not detect both P/E and EV/EBITDA-like columns. "
            f"P/E detected: {pe_col}, EV/EBITDA detected: {ev_col}"
        )
    else:
        df_sc = df_filtered.copy()
        df_sc = numericify(df_sc, [pe_col, ev_col])
        df_sc = df_sc.replace([np.inf, -np.inf], np.nan)
        df_sc = df_sc.dropna(subset=[pe_col, ev_col])

        df_sc = df_sc[(df_sc[pe_col] > 0) & (df_sc[ev_col] > 0)]

        if show_cap_100:
            df_sc = df_sc[(df_sc[pe_col] <= 100) & (df_sc[ev_col] <= 100)]
        else:
            # clip at 99th percentile for readability
            pe_high = df_sc[pe_col].quantile(0.99)
            ev_high = df_sc[ev_col].quantile(0.99)
            df_sc = df_sc[(df_sc[pe_col] <= pe_high) & (df_sc[ev_col] <= ev_high)]

        fig_sc = px.scatter(
            df_sc,
            x=pe_col,
            y=ev_col,
            color="index",
            hover_name="name",
            hover_data={
                "symbol": True,
                "sector": True,
                "index": True,
                pe_col: ":.2f",
                ev_col: ":.2f",
            },
            opacity=0.7,
            title="P/E vs EV/EBITDA (interactive)",
        )
        fig_sc.update_layout(
            xaxis_title="P/E",
            yaxis_title="EV/EBITDA (or similar multiple)",
        )

        st.plotly_chart(fig_sc, use_container_width=True)



# -------------------------
# ROIC vs EARNINGS YIELD TAB
# -------------------------
tab_roic_ey = st.tabs(["ROIC vs Earnings Yield"])[0]

with tab_roic_ey:
    st.subheader("ROIC vs Earnings Yield (Outliers & negatives removed, % axes)")

    # Prepare data
    df_roic_ey = df_filtered.copy()
    df_roic_ey = numericify(df_roic_ey, ["roicTTM", "earningsYieldTTM"])
    df_roic_ey = df_roic_ey.dropna(subset=["roicTTM", "earningsYieldTTM"])
    df_roic_ey = df_roic_ey[(df_roic_ey["roicTTM"] >= 0) & (df_roic_ey["earningsYieldTTM"] >= 0)]

    # Outlier removal (IQR method)
    Q1_roic = df_roic_ey["roicTTM"].quantile(0.25)
    Q3_roic = df_roic_ey["roicTTM"].quantile(0.75)
    IQR_roic = Q3_roic - Q1_roic
    Q1_ey = df_roic_ey["earningsYieldTTM"].quantile(0.25)
    Q3_ey = df_roic_ey["earningsYieldTTM"].quantile(0.75)
    IQR_ey = Q3_ey - Q1_ey

    lower_roic = Q1_roic - 1.5 * IQR_roic
    upper_roic = Q3_roic + 1.5 * IQR_roic
    lower_ey = Q1_ey - 1.5 * IQR_ey
    upper_ey = Q3_ey + 1.5 * IQR_ey

    outliers = df_roic_ey[
        (df_roic_ey["roicTTM"] < lower_roic) | (df_roic_ey["roicTTM"] > upper_roic) |
        (df_roic_ey["earningsYieldTTM"] < lower_ey) | (df_roic_ey["earningsYieldTTM"] > upper_ey)
    ]
    df_roic_ey_clean = df_roic_ey[~df_roic_ey.index.isin(outliers.index)].copy()

    # Convert to percent
    df_roic_ey_clean["ROIC (%)"] = df_roic_ey_clean["roicTTM"] * 100
    df_roic_ey_clean["Earnings Yield (%)"] = df_roic_ey_clean["earningsYieldTTM"] * 100

    # Plot
    fig_roic_ey = px.scatter(
        df_roic_ey_clean,
        x="Earnings Yield (%)",
        y="ROIC (%)",
        hover_name="name",
        hover_data={
            "symbol": True,
            "sector": True,
            "index": True,
            "ROIC (%)": ":.2f",
            "Earnings Yield (%)": ":.2f",
        },
        color="sector" if "sector" in df_roic_ey_clean.columns else None,
        opacity=0.7,
        title="ROIC vs Earnings Yield (Negatives & Outliers Removed, % Axes)",
    )
    fig_roic_ey.update_layout(
        xaxis_title="Earnings Yield (%)",
        yaxis_title="ROIC (%)",
    )
    st.plotly_chart(fig_roic_ey, use_container_width=True)

    # Show outliers as a table (optional)
    if not outliers.empty:
        outliers_disp = outliers.copy()
        outliers_disp["ROIC (%)"] = outliers_disp["roicTTM"] * 100
        outliers_disp["Earnings Yield (%)"] = outliers_disp["earningsYieldTTM"] * 100
        st.write("#### Outliers (excluded from chart)")
        st.dataframe(
            outliers_disp[["symbol", "name", "ROIC (%)", "Earnings Yield (%)"]].sort_values("ROIC (%)", ascending=False)
        )



# ---- CONFIG ----
FMP_API_KEY = "3168acf93e1a4ca67ce62d850fdfc9bd"  # <-- Replace with your actual FMP API key

# ---- PORTFOLIO OPTIMIZER TAB ----
tab_portfolio = st.tabs(["Portfolio Optimizer"])[0]

with tab_portfolio:
    st.subheader("Portfolio Optimizer (Modern Portfolio Theory)")

    # --- User Inputs ---
    all_tickers = sorted(all_raw['symbol'].dropna().unique())
    selected_tickers = st.multiselect(
        "Select Holdings (tickers):",
        options=all_tickers,
        default=["AAPL", "MSFT", "GOOGL"]
    )

    goal = st.selectbox(
        "Optimization Goal:",
        options=[("Max Sharpe Ratio", "Sharpe"), ("Max Return", "Return"), ("Min Volatility", "Volatility")],
        format_func=lambda x: x[0]
    )

    risk_free = st.text_input("Risk-free rate (annual, decimal, e.g. 0.02):", value="0.02")
    years = st.slider("Lookback period (years):", min_value=1, max_value=10, value=3, step=1)
    freq = st.selectbox("Return Frequency:", options=[("Monthly (ME)", "ME"), ("Daily", "D")], format_func=lambda x: x[0])

    run_opt = st.button("Run Portfolio Optimizer")

    # --- Helper Functions ---
    def fetch_prices_fmp(tickers, start_date, end_date, api_key):
        price_data = {}
        for ticker in tickers:
            url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}'
            r = requests.get(url)
            data = r.json()
            if 'historical' in data:
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                price_data[ticker] = df['adjClose']
        return pd.DataFrame(price_data)

    def calc_returns_cov(prices, freq='ME'):
        if freq == 'ME':
            prices = prices.resample('ME').last()
        returns = np.log(prices / prices.shift(1)).dropna()
        exp_returns = returns.mean() * (12 if freq == 'ME' else 252)
        cov_matrix = returns.cov() * (12 if freq == 'ME' else 252)
        return exp_returns, cov_matrix

    def optimise_portfolio(returns, cov, risk_free, goal):
        n = len(returns)
        bounds = tuple((0, 1) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        def portfolio_perf(weights):
            port_return = np.dot(weights, returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            sharpe = (port_return - risk_free) / port_vol if port_vol > 0 else 0
            return port_return, port_vol, sharpe

        if goal == 'Sharpe':
            def neg_sharpe(weights): return -portfolio_perf(weights)[2]
            result = minimize(neg_sharpe, n*[1./n], bounds=bounds, constraints=constraints)
        elif goal == 'Return':
            def neg_return(weights): return -portfolio_perf(weights)[0]
            result = minimize(neg_return, n*[1./n], bounds=bounds, constraints=constraints)
        elif goal == 'Volatility':
            def port_vol(weights): return portfolio_perf(weights)[1]
            result = minimize(port_vol, n*[1./n], bounds=bounds, constraints=constraints)
        else:
            raise ValueError("Unknown goal")
        return result

    # --- Run Optimizer ---
    if run_opt:
        try:
            risk_free_val = float(risk_free)
        except Exception:
            st.error("Risk-free rate must be a valid number (e.g. 0.02).")
            st.stop()

        if not selected_tickers:
            st.warning("Please select at least one ticker.")
        else:
            st.info(f"Fetching price data for: {selected_tickers}")
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.Timedelta(days=365*years)
            prices = fetch_prices_fmp(selected_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), FMP_API_KEY)
            if prices.empty or prices.isnull().all().all():
                st.error("No price data found for selected tickers.")
            else:
                exp_returns, cov_matrix = calc_returns_cov(prices, freq=freq[1])
                # Check for NaNs or invalid covariance
                if np.any(np.isnan(exp_returns)) or np.any(np.isnan(cov_matrix)) or cov_matrix.shape[0] != len(selected_tickers):
                    st.error("Insufficient or invalid price data for optimization. Try different tickers or a longer lookback.")
                else:
                    result = optimise_portfolio(exp_returns.values, cov_matrix.values, risk_free_val, goal[1])
                    if not result.success:
                        st.error(f"Optimization failed: {result.message}")
                    else:
                        weights = result.x
                        st.success("Optimal Portfolio Allocation:")
                        alloc_df = pd.DataFrame({
                            "Ticker": selected_tickers,
                            "Weight": weights
                        })
                        st.dataframe(alloc_df.style.format({"Weight": "{:.2%}"}))
                        fig = px.pie(
                            alloc_df,
                            names="Ticker",
                            values="Weight",
                            title=f"Optimal Portfolio Allocation ({goal[0]})",
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)






# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Equity Valuation Dashboard",
    layout="wide",
)

# -------------------------
# DATA LOADING
# -------------------------
@st.cache_data
def load_data():
    sp500_raw = pd.read_csv("sp500_valuations_raw.csv")
    nasdaq_raw = pd.read_csv("nasdaq100_valuations_raw.csv")

    # Add index label (these weren't in the original CSVs)
    sp500_raw["index"] = "S&P 500"
    nasdaq_raw["index"] = "NASDAQ 100"

    all_raw = pd.concat([sp500_raw, nasdaq_raw], ignore_index=True)

    return sp500_raw, nasdaq_raw, all_raw

sp500_raw, nasdaq_raw, all_raw = load_data()
all_raw.columns = [c.strip() for c in all_raw.columns]

# -------------------------
# BUFFETT SCREEN FUNCTION
# -------------------------
def show_buffett_screen(df):
    cols = [
        'roicTTM',
        'returnOnTangibleAssetsTTM',
        'freeCashFlowPerShareTTM',
        'debtToEquityTTM',
        'netIncomePerShareTTM',
        'capexToOperatingCashFlowTTM'
    ]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols)

    criteria = {
        'ROIC (roicTTM)': '> 15%',
        'Return on Tangible Assets (returnOnTangibleAssetsTTM)': '> 10%',
        'Free Cash Flow per Share (freeCashFlowPerShareTTM)': f">= {df['freeCashFlowPerShareTTM'].quantile(0.75):.2f} (Top Quartile)",
        'Debt to Equity (debtToEquityTTM)': '<= 1.0',
        'Net Income per Share (netIncomePerShareTTM)': f">= {df['netIncomePerShareTTM'].quantile(0.75):.2f} (Top Quartile)",
        'Capex to Operating Cash Flow (capexToOperatingCashFlowTTM)': '<= 20%'
    }
    numeric_criteria = {
        'roicTTM': 0.15,
        'returnOnTangibleAssetsTTM': 0.10,
        'freeCashFlowPerShareTTM': df['freeCashFlowPerShareTTM'].quantile(0.75),
        'debtToEquityTTM': 1.0,
        'netIncomePerShareTTM': df['netIncomePerShareTTM'].quantile(0.75),
        'capexToOperatingCashFlowTTM': 0.2
    }

    filtered = df.copy()
    filtered = filtered[filtered['roicTTM'] >= numeric_criteria['roicTTM']]
    filtered = filtered[filtered['returnOnTangibleAssetsTTM'] >= numeric_criteria['returnOnTangibleAssetsTTM']]
    filtered = filtered[filtered['freeCashFlowPerShareTTM'] >= numeric_criteria['freeCashFlowPerShareTTM']]
    filtered = filtered[filtered['debtToEquityTTM'] <= numeric_criteria['debtToEquityTTM']]
    filtered = filtered[filtered['netIncomePerShareTTM'] >= numeric_criteria['netIncomePerShareTTM']]
    filtered = filtered[filtered['capexToOperatingCashFlowTTM'] <= numeric_criteria['capexToOperatingCashFlowTTM']]

    # Rank by composite score
    for col in ['roicTTM','returnOnTangibleAssetsTTM','freeCashFlowPerShareTTM','netIncomePerShareTTM']:
        filtered[col+'_score'] = (filtered[col]-filtered[col].min())/(filtered[col].max()-filtered[col].min()+1e-9)
    filtered['score'] = filtered[[c for c in filtered.columns if c.endswith('_score')]].sum(axis=1)
    filtered = filtered.sort_values('score',ascending=False)

    filtered['Buffett Description'] = filtered.apply(lambda row: (
        f"ROIC: {row['roicTTM']:.2%}, "
        f"Tangible ROA: {row['returnOnTangibleAssetsTTM']:.2%}, "
        f"FCF/Share: {row['freeCashFlowPerShareTTM']:.2f}, "
        f"Debt/Equity: {row['debtToEquityTTM']:.2f}, "
        f"Capex/OpCF: {row['capexToOperatingCashFlowTTM']:.2%}"
    ), axis=1)

    result = filtered[['symbol','name','sector','roicTTM','returnOnTangibleAssetsTTM','freeCashFlowPerShareTTM','debtToEquityTTM','capexToOperatingCashFlowTTM','Buffett Description']].head(5)

    st.markdown("### Buffett Screening Criteria and Hurdles")
    for metric, hurdle in criteria.items():
        st.markdown(f"- **{metric}**: {hurdle}")

    st.markdown("### Top 5 Buffett-style Candidates")
    st.dataframe(result.rename(columns={
        'symbol': 'Symbol',
        'name': 'Name',
        'sector': 'Sector',
        'roicTTM': 'ROIC',
        'returnOnTangibleAssetsTTM': 'Tangible ROA',
        'freeCashFlowPerShareTTM': 'FCF/Share',
        'debtToEquityTTM': 'Debt/Equity',
        'capexToOperatingCashFlowTTM': 'Capex/OpCF'
    }), use_container_width=True)



# --- BUFFETT SCREEN TAB (UPDATED) ---
tab_buffett = st.tabs(["Buffett Screen"])[0]

with tab_buffett:
    st.subheader("What Would Warren Want? (to buy) Stock Screening Tool for SP500")

    # --- Metrics and Default Hurdles ---
    metrics = {
        'ROE_min': ('Return on Equity (%)', 0.15),
        'Debt_to_Equity_max': ('Debt/Equity', 0.5),
        'Net_Margin_min': ('Net Margin (%)', 0.10),
        'PE_max': ('P/E Ratio', 20),
        'ROIC_min': ('ROIC (%)', 0.10),
        'Current_Ratio_min': ('Current Ratio', 1.5),
        'Interest_Coverage_min': ('Interest Coverage', 3),
        'Dividend_Yield_min': ('Dividend Yield (%)', 0.02),
        'Payout_Ratio_max': ('Payout Ratio (%)', 0.60),
        'Price_to_Book_max': ('Price/Book', 3),
        'Capex_to_OCF_max': ('Capex/OCF (%)', 0.20),
        'Positive_NI': ('Positive Net Income', True)
    }

    # --- Compute Derived Metrics ---
    df = df_filtered.copy()
    shares_outstanding = df['marketCapTTM'] / (df['priceToSalesRatioTTM'] * df['revenuePerShareTTM'])
    df['net_income'] = df['netIncomePerShareTTM'] * shares_outstanding
    df['revenue'] = df['revenuePerShareTTM'] * shares_outstanding
    df['equity'] = df['shareholdersEquityPerShareTTM'] * shares_outstanding
    df['debt'] = df['interestDebtPerShareTTM'] * shares_outstanding

    df['ROE_calc'] = df['net_income'] / df['equity']
    df['Debt_to_Equity'] = df['debt'] / df['equity']
    df['Net_Margin'] = df['net_income'] / df['revenue']
    df['PE'] = df['peRatioTTM']
    df['ROIC'] = df['roicTTM']
    df['Current_Ratio'] = df['currentRatioTTM']
    df['Interest_Coverage'] = df['interestCoverageTTM']
    df['Dividend_Yield'] = df['dividendYieldTTM']
    df['Payout_Ratio'] = df['payoutRatioTTM']
    df['Price_to_Book'] = df['pbRatioTTM']
    df['Capex_to_OCF'] = df['capexToOperatingCashFlowTTM']
    df['Positive_NI'] = df['net_income'] > 0

    # --- Sidebar/Top Controls ---
    st.markdown("#### Filter & Hurdle Controls")
    col_search, col_sector = st.columns([2, 2])
    with col_search:
        search_term = st.text_input("Search Symbol", "")
    with col_sector:
        sector_options = ['All'] + sorted(df['sector'].dropna().unique())
        sector_filter = st.selectbox("Sector", sector_options)

    # --- Metric Selection ---
    all_metric_keys = list(metrics.keys())
    metric_labels = [metrics[k][0] for k in all_metric_keys]
    metric_key_to_label = dict(zip(all_metric_keys, metric_labels))
    metric_label_to_key = dict(zip(metric_labels, all_metric_keys))

    selected_labels = st.multiselect(
        "Select metrics to display",
        options=metric_labels,
        default=metric_labels
    )
    selected_metrics = [metric_label_to_key[lbl] for lbl in selected_labels]

    # --- Hurdle Controls ---
    col1, col2, col3 = st.columns(3)
    with col1:
        roe_min = st.number_input("ROE min (%)", value=15.0)
        debt_max = st.number_input("Debt/Equity max", value=0.5)
        margin_min = st.number_input("Net Margin min (%)", value=10.0)
        pe_max = st.number_input("PE max", value=20.0)
    with col2:
        roic_min = st.number_input("ROIC min (%)", value=10.0)
        current_min = st.number_input("Current Ratio min", value=1.5)
        interest_min = st.number_input("Interest Coverage min", value=3.0)
        dividend_min = st.number_input("Dividend Yield min (%)", value=2.0)
    with col3:
        payout_max = st.number_input("Payout Ratio max (%)", value=60.0)
        pb_max = st.number_input("Price/Book max", value=3.0)
        capex_max = st.number_input("Capex/OCF max (%)", value=20.0)

    # --- Criteria Dict ---
    criteria = {
        'ROE_min': roe_min / 100,
        'Debt_to_Equity_max': debt_max,
        'Net_Margin_min': margin_min / 100,
        'PE_max': pe_max,
        'ROIC_min': roic_min / 100,
        'Current_Ratio_min': current_min,
        'Interest_Coverage_min': interest_min,
        'Dividend_Yield_min': dividend_min / 100,
        'Payout_Ratio_max': payout_max / 100,
        'Price_to_Book_max': pb_max,
        'Capex_to_OCF_max': capex_max / 100,
        'Positive_NI': True
    }

    # --- Filtering ---
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['symbol'].str.contains(search_term, case=False, na=False)]
    if sector_filter != 'All':
        filtered_df = filtered_df[filtered_df['sector'] == sector_filter]

    # --- Table Generation ---
    def create_html_table(criteria, filtered_df, selected_metrics):
        tick = '✅'
        cross = '❌'
        html = '<table style="border-collapse: collapse; width: 100%;">'
        html += '<tr style="background-color:#f2f2f2;">'
        html += '<th style="border:1px solid #ccc; padding:5px;">Company</th>'
        for metric in selected_metrics:
            html += f'<th style="border:1px solid #ccc; padding:5px;">{metrics[metric][0]}</th>'
        html += '<th style="border:1px solid #ccc; padding:5px;">Hurdles Met</th>'
        html += '</tr>'

        # Calculate pass counts
        pass_counts = []
        for i, row in filtered_df.iterrows():
            count = 0
            for metric in metrics.keys():
                if metric == 'ROE_min':
                    passed = row['ROE_calc'] >= criteria[metric]
                elif metric == 'Debt_to_Equity_max':
                    passed = row['Debt_to_Equity'] <= criteria[metric]
                elif metric == 'Net_Margin_min':
                    passed = row['Net_Margin'] >= criteria[metric]
                elif metric == 'PE_max':
                    passed = row['PE'] <= criteria[metric]
                elif metric == 'ROIC_min':
                    passed = row['ROIC'] >= criteria[metric]
                elif metric == 'Current_Ratio_min':
                    passed = row['Current_Ratio'] >= criteria[metric]
                elif metric == 'Interest_Coverage_min':
                    passed = row['Interest_Coverage'] >= criteria[metric]
                elif metric == 'Dividend_Yield_min':
                    passed = row['Dividend_Yield'] >= criteria[metric]
                elif metric == 'Payout_Ratio_max':
                    passed = row['Payout_Ratio'] <= criteria[metric]
                elif metric == 'Price_to_Book_max':
                    passed = row['Price_to_Book'] <= criteria[metric]
                elif metric == 'Capex_to_OCF_max':
                    passed = row['Capex_to_OCF'] <= criteria[metric]
                elif metric == 'Positive_NI':
                    passed = row['Positive_NI']
                if passed:
                    count += 1
            pass_counts.append(count)

        # Sort by pass count
        sorted_indices = sorted(range(len(pass_counts)), key=lambda i: pass_counts[i], reverse=True)

        for idx in sorted_indices:
            row = filtered_df.iloc[idx]
            html += '<tr>'
            html += f'<td style="border:1px solid #ccc; padding:5px;">{row["symbol"]}</td>'
            count = 0
            for metric in selected_metrics:
                if metric == 'ROE_min':
                    value = row['ROE_calc'] * 100
                    passed = row['ROE_calc'] >= criteria[metric]
                elif metric == 'Debt_to_Equity_max':
                    value = row['Debt_to_Equity']
                    passed = value <= criteria[metric]
                elif metric == 'Net_Margin_min':
                    value = row['Net_Margin'] * 100
                    passed = row['Net_Margin'] >= criteria[metric]
                elif metric == 'PE_max':
                    value = row['PE']
                    passed = value <= criteria[metric]
                elif metric == 'ROIC_min':
                    value = row['ROIC'] * 100
                    passed = row['ROIC'] >= criteria[metric]
                elif metric == 'Current_Ratio_min':
                    value = row['Current_Ratio']
                    passed = value >= criteria[metric]
                elif metric == 'Interest_Coverage_min':
                    value = row['Interest_Coverage']
                    passed = value >= criteria[metric]
                elif metric == 'Dividend_Yield_min':
                    value = row['Dividend_Yield'] * 100
                    passed = row['Dividend_Yield'] >= criteria[metric]
                elif metric == 'Payout_Ratio_max':
                    value = row['Payout_Ratio'] * 100
                    passed = row['Payout_Ratio'] <= criteria[metric]
                elif metric == 'Price_to_Book_max':
                    value = row['Price_to_Book']
                    passed = value <= criteria[metric]
                elif metric == 'Capex_to_OCF_max':
                    value = row['Capex_to_OCF'] * 100
                    passed = row['Capex_to_OCF'] <= criteria[metric]
                elif metric == 'Positive_NI':
                    value = row['net_income']
                    passed = row['Positive_NI']

                color = '#d4edda' if passed else '#f8d7da'
                symbol = '✅' if passed else '❌'
                tooltip = f"Metric value: {round(value,2)}"
                html += f'<td style="border:1px solid #ccc; padding:5px; background-color:{color};" title="{tooltip}">{symbol}</td>'
                if passed:
                    count += 1
            html += f'<td style="border:1px solid #ccc; padding:5px; font-weight:bold;">{count}</td>'
            html += '</tr>'
        html += '</table>'
        return html

    # --- Show Table ---
    st.markdown(
        create_html_table(criteria, filtered_df, selected_metrics),
        unsafe_allow_html=True
    )







# -------------------------
# MAIN LAYOUT WITH BUFFETT TAB
# -------------------------
tab_overview, tab_buffett, tab_heatmap, tab_vq, tab_pe_ev, tab_table = st.tabs(
    ["Overview", "Buffett Screen", "Sector Heatmap", "Value vs Quality", "P/E vs EV/EBITDA", "Data Table"]
)

with tab_overview:
    st.subheader("Universe Summary")
    # ... (your overview code here)

with tab_buffett:
    show_buffett_screen(sp500_raw)

with tab_heatmap:
    st.subheader("Sector P/E Heatmap")
    # ... (your heatmap code here)

with tab_vq:
    st.subheader("Value vs Quality Scatter")
    # ... (your value vs quality code here)

with tab_pe_ev:
    st.subheader("P/E vs EV/EBITDA Scatter")
    # ... (your P/E vs EV/EBITDA code here)

with tab_table:
    st.subheader("Underlying Data")
    # ... (your data table code here)


# --- BUFFETT SCREEN TAB ---
tab_buffett = st.tabs(["Buffett Screen"])[0]

with tab_buffett:
    st.subheader("What Would Warren Want? (to buy) Stock Screening Tool for SP500")

    # --- Metrics and Default Hurdles ---
    metrics = {
        'ROE_min': ('Return on Equity (%)', 0.15, 'higher', 'ROE_calc'),
        'Debt_to_Equity_max': ('Debt/Equity', 0.5, 'lower', 'Debt_to_Equity'),
        'Net_Margin_min': ('Net Margin (%)', 0.10, 'higher', 'Net_Margin'),
        'PE_max': ('P/E Ratio', 20, 'lower', 'PE'),
        'ROIC_min': ('ROIC (%)', 0.10, 'higher', 'ROIC'),
        'Current_Ratio_min': ('Current Ratio', 1.5, 'higher', 'Current_Ratio'),
        'Interest_Coverage_min': ('Interest Coverage', 3, 'higher', 'Interest_Coverage'),
        'Dividend_Yield_min': ('Dividend Yield (%)', 0.02, 'higher', 'Dividend_Yield'),
        'Payout_Ratio_max': ('Payout Ratio (%)', 0.60, 'lower', 'Payout_Ratio'),
        'Price_to_Book_max': ('Price/Book', 3, 'lower', 'Price_to_Book'),
        'Capex_to_OCF_max': ('Capex/OCF (%)', 0.20, 'lower', 'Capex_to_OCF'),
        'Positive_NI': ('Positive Net Income', True, 'higher', 'Positive_NI')
    }

    # --- Compute Derived Metrics ---
    df = df_filtered.copy()
    shares_outstanding = df['marketCapTTM'] / (df['priceToSalesRatioTTM'] * df['revenuePerShareTTM'])
    df['net_income'] = df['netIncomePerShareTTM'] * shares_outstanding
    df['revenue'] = df['revenuePerShareTTM'] * shares_outstanding
    df['equity'] = df['shareholdersEquityPerShareTTM'] * shares_outstanding
    df['debt'] = df['interestDebtPerShareTTM'] * shares_outstanding

    df['ROE_calc'] = df['net_income'] / df['equity']
    df['Debt_to_Equity'] = df['debt'] / df['equity']
    df['Net_Margin'] = df['net_income'] / df['revenue']
    df['PE'] = df['peRatioTTM']
    df['ROIC'] = df['roicTTM']
    df['Current_Ratio'] = df['currentRatioTTM']
    df['Interest_Coverage'] = df['interestCoverageTTM']
    df['Dividend_Yield'] = df['dividendYieldTTM']
    df['Payout_Ratio'] = df['payoutRatioTTM']
    df['Price_to_Book'] = df['pbRatioTTM']
    df['Capex_to_OCF'] = df['capexToOperatingCashFlowTTM']
    df['Positive_NI'] = df['net_income'] > 0

    # --- Controls ---
    st.markdown("#### Filter & Hurdle Controls")
    col_search, col_sector = st.columns([2, 2])
    with col_search:
        search_term = st.text_input("Search Symbol", "")
    with col_sector:
        sector_options = ['All'] + sorted(df['sector'].dropna().unique())
        sector_filter = st.selectbox("Sector", sector_options)

    # Metric selection
    metric_labels = [metrics[k][0] for k in metrics.keys()]
    metric_label_to_key = dict(zip(metric_labels, metrics.keys()))
    selected_labels = st.multiselect("Select metrics to display", options=metric_labels, default=metric_labels)
    selected_metrics = [metric_label_to_key[lbl] for lbl in selected_labels]

    exclude_non_pass = st.checkbox("Only show companies that meet all displayed metric hurdles", value=False)

    # --- Hurdle Inputs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        roe_min = st.number_input("ROE min (%)", value=15.0)
        debt_max = st.number_input("Debt/Equity max", value=0.5)
        margin_min = st.number_input("Net Margin min (%)", value=10.0)
        pe_max = st.number_input("PE max", value=20.0)
    with col2:
        roic_min = st.number_input("ROIC min (%)", value=10.0)
        current_min = st.number_input("Current Ratio min", value=1.5)
        interest_min = st.number_input("Interest Coverage min", value=3.0)
        dividend_min = st.number_input("Dividend Yield min (%)", value=2.0)
    with col3:
        payout_max = st.number_input("Payout Ratio max (%)", value=60.0)
        pb_max = st.number_input("Price/Book max", value=3.0)
        capex_max = st.number_input("Capex/OCF max (%)", value=20.0)

    criteria = {
        'ROE_min': roe_min / 100,
        'Debt_to_Equity_max': debt_max,
        'Net_Margin_min': margin_min / 100,
        'PE_max': pe_max,
        'ROIC_min': roic_min / 100,
        'Current_Ratio_min': current_min,
        'Interest_Coverage_min': interest_min,
        'Dividend_Yield_min': dividend_min / 100,
        'Payout_Ratio_max': payout_max / 100,
        'Price_to_Book_max': pb_max,
        'Capex_to_OCF_max': capex_max / 100,
        'Positive_NI': True
    }

    # Apply search and sector filter
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['symbol'].str.contains(search_term, case=False, na=False)]
    if sector_filter != 'All':
        filtered_df = filtered_df[filtered_df['sector'] == sector_filter]

    # --- Scoring ---
    scores = pd.DataFrame(index=filtered_df.index)
    for key in selected_metrics:
        col_name = metrics[key][3]
        direction = metrics[key][2]
        vals = filtered_df[col_name].astype(float) if key != 'Positive_NI' else filtered_df['Positive_NI'].astype(int)
        ascending = True if direction == 'lower' else False
        scores[key] = vals.rank(ascending=ascending, method='min')
    scores['Total_Score'] = scores.sum(axis=1)

    # Exclude non-pass if requested
    def passes_all(row):
        for key in selected_metrics:
            col_name = metrics[key][3]
            val = row[col_name]
            if key.endswith('_min') and val < criteria[key]: return False
            if key.endswith('_max') and val > criteria[key]: return False
            if key == 'Positive_NI' and not val: return False
        return True

    if exclude_non_pass:
        filtered_df = filtered_df[filtered_df.apply(passes_all, axis=1)]
        scores = scores.loc[filtered_df.index]

    # --- Build DataFrame for Streamlit ---
    display_df = pd.DataFrame(index=filtered_df.index)
    display_df['Symbol'] = filtered_df['symbol']
    for key in selected_metrics:
        col_name = metrics[key][3]
        vals = filtered_df[col_name]
        display_df[metrics[key][0]] = vals.apply(
            lambda v: f"✅ {v:.2f}" if (v >= 0 and ((key.endswith('_min') and v >= criteria[key]) or (key.endswith('_max') and v <= criteria[key]) or (key == 'Positive_NI' and v))) else f"❌ {v:.2f}" if isinstance(v, (int, float)) else f"❌ {v}"
        )
    display_df['Total Score'] = scores['Total_Score']

    # --- Show sortable table ---
    st.dataframe(display_df.sort_values('Total Score'), use_container_width=True)



# -------------------------
# DATA TABLE TAB
# -------------------------
with tab_table:
    st.subheader("Underlying Data")

    st.write("Filtered universe (after sidebar filters):")
    st.dataframe(
        df_filtered.head(500).style.format(precision=2),
        height=500,
    )

    st.caption("Showing first 500 rows for performance. Export from the original CSVs if you need the full dataset.")













