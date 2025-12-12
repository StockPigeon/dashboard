import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
import requests
import re

# NEW: import your dynamic global dataset
from update_data import load_global_data

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Equity Valuation Dashboard",
    layout="wide",
)

# -------------------------
# LOAD GLOBAL DATA
# -------------------------
@st.cache_data(show_spinner=True)
def load_universe():
    global_data = load_global_data()  # dict: index → df
    dfs = []

    for index_name, df in global_data.items():
        if df is None or df.empty:
            continue

        df = df.copy()

        # Ensure symbol upper-case
        df["symbol"] = df["symbol"].astype(str).str.upper()

        # Ensure sector column exists
        if "sector" not in df.columns:
            df["sector"] = "Unknown"

        # Ensure name column exists (some ETFs do not return it)
        if "name" not in df.columns:
            df["name"] = df["symbol"]

        # Add index label
        df["index"] = index_name

        dfs.append(df)

    # Combine all indices into a single universe
    all_raw = pd.concat(dfs, ignore_index=True)
    all_raw.columns = [c.strip() for c in all_raw.columns]

    return global_data, all_raw



global_data, all_raw = load_universe()


# ---------------------------------------------
# SIDEBAR INDEX SELECTION (dynamic, global)
# ---------------------------------------------
all_indices = list(global_data.keys())

# Default selection = S&P 500 + ASX 200 only
default_selection = []
for ix in ["S&P 500", "ASX 200"]:
    if ix in all_indices:
        default_selection.append(ix)

index_filter = st.sidebar.multiselect(
    "Index",
    options=all_indices,
    default=default_selection
)


# The rest of your existing sidebar
sector_options = sorted(all_raw["sector"].dropna().unique())
sector_filter = st.sidebar.multiselect(
    "Sector (optional filter)",
    options=sector_options,
    default=[],
    key="sector_filter_main"
)

show_cap_100 = st.sidebar.checkbox(
    "Cap multiples at 100 (P/E & EV/EBITDA)", 
    value=False,
    key="cap_multiples_checkbox"
)


# ---------------------------------------------
# FILTER UNIVERSE
# ---------------------------------------------
df_filtered = all_raw.copy()

if index_filter:
    df_filtered = df_filtered[df_filtered["index"].isin(index_filter)]

if sector_filter:
    df_filtered = df_filtered[df_filtered["sector"].isin(sector_filter)]

print(df_filtered.head)

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
        c
        for c in df.columns
        if "peratiottm" in c.lower()
        or ("pe" in c.lower() and "ttm" in c.lower())
    ]
    return candidates[0] if candidates else None


def detect_ev_ebitda_column(df):
    candidates = [
        c
        for c in df.columns
        if any(
            k in c.lower()
            for k in [
                "evtoebitda",
                "enterprisevalueoverebitda",
                "ev_ebitda",
                "enterprisevaluemultiple",
            ]
        )
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
# PORTFOLIO OPTIMISER HELPERS
# -------------------------
# Expect API key in Streamlit secrets or environment
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))


def fetch_prices_fmp(tickers, start_date, end_date, api_key):
    price_data = {}
    for ticker in tickers:
        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/"
            f"{ticker}?from={start_date}&to={end_date}&apikey={api_key}"
        )
        r = requests.get(url)
        data = r.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            price_data[ticker] = df["adjClose"]
    if not price_data:
        return pd.DataFrame()
    return pd.DataFrame(price_data)


def calc_returns_cov(prices, freq="ME"):
    if prices.empty:
        return None, None
    if freq == "ME":
        prices = prices.resample("ME").last()
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        return None, None
    exp_returns = returns.mean() * (12 if freq == "ME" else 252)
    cov_matrix = returns.cov() * (12 if freq == "ME" else 252)
    return exp_returns, cov_matrix


def optimise_portfolio(returns, cov, risk_free, goal):
    n = len(returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)

    def portfolio_perf(weights):
        port_return = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = (port_return - risk_free) / port_vol if port_vol > 0 else 0
        return port_return, port_vol, sharpe

    x0 = np.array(n * [1.0 / n])

    if goal == "Sharpe":
        def neg_sharpe(w):
            return -portfolio_perf(w)[2]

        result = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)

    elif goal == "Return":
        def neg_return(w):
            return -portfolio_perf(w)[0]

        result = minimize(neg_return, x0, bounds=bounds, constraints=constraints)

    elif goal == "Volatility":
        def port_vol(w):
            return portfolio_perf(w)[1]

        result = minimize(port_vol, x0, bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Unknown goal")

    return result, portfolio_perf


# # -------------------------
# # SIDEBAR FILTERS
# # -------------------------
# st.sidebar.title("Controls")

# index_filter = st.sidebar.multiselect(
#     "Index",
#     options=["S&P 500", "NASDAQ 100"],
#     default=["S&P 500", "NASDAQ 100"],
# )

# sector_options = sorted(all_raw["sector"].dropna().unique())
# sector_filter = st.sidebar.multiselect(
#     "Sector (optional filter)",
#     options=sector_options,
#     default=[],
# )

# show_cap_100 = st.sidebar.checkbox(
#     "Cap multiples at 100 (P/E & EV/EBITDA)", value=False
# )

# st.sidebar.markdown("---")
# st.sidebar.caption("CSV files must be in the same folder as app.py")

# Apply filters
df_filtered = all_raw.copy()
if index_filter:
    df_filtered = df_filtered[df_filtered["index"].isin(index_filter)]
if sector_filter:
    df_filtered = df_filtered[df_filtered["sector"].isin(sector_filter)]


# ---------------------------------------------
# INVESTMENT IDEAS TOOL (helpers + function)
# ---------------------------------------------
def _safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _compute_cagr(start, end, years):
    try:
        if pd.isna(start) or pd.isna(end) or start <= 0 or end <= 0 or years <= 0:
            return np.nan
        return (end / start) ** (1 / years) - 1
    except Exception:
        return np.nan

def _detect_hist_columns(df, base_key):
    """
    Detect candidate historical columns by name pattern for the given base_key.
    Returns a list of (order_key, col_name) sorted oldest..newest.

    Matches patterns like:
      - freeCashFlowPerShare_FY2021, _FY2022
      - freeCashFlowPerShare_1Y, _2Y, _3Y
      - freeCashFlowPerShare2021
    """
    cols = []
    base_lower = base_key.lower().replace("ttm", "")
    for c in df.columns:
        cl = c.lower()
        if base_lower in cl and c != base_key:
            m_year = re.search(r'(?:fy)?(\d{4})', cl)
            m_yago = re.search(r'_(\d+)y', cl)  # e.g., _1Y, _2Y
            if m_year:
                order = int(m_year.group(1))
            elif m_yago:
                order = -int(m_yago.group(1))  # older → more negative so sorts first
            else:
                m_num = re.search(r'_(\d+)$', cl)
                order = int(m_num.group(1)) if m_num else cl
            cols.append((order, c))
    try:
        cols.sort(key=lambda x: x[0])
    except Exception:
        cols.sort(key=lambda x: str(x[0]))
    return cols

def _compute_series_cagr(row, hist_cols):
    """Given sorted historical columns (oldest..newest), compute CAGR."""
    if not hist_cols or len(hist_cols) < 2:
        return np.nan
    start_val = row.get(hist_cols[0][1])
    end_val   = row.get(hist_cols[-1][1])
    years = len(hist_cols) - 1  # assume annual spacing
    return _compute_cagr(start_val, end_val, years)

def investment_ideas_tool(df_filtered):
    st.subheader("Investment Ideas – Companies Passing All Hurdles")

    st.markdown("""
### Metrics, Formulas & Hurdles
**1. Positive Operating Profit (proxy)**  
*Formula:* `operatingIncome` history (preferred) or `operatingCashFlowPerShare` history  
*Hurdle:* Positive for **each** of the past **7 years** (N/A → excluded)

**2. Profit Margin**  
*Formula:* `netIncomePerShareTTM / revenuePerShareTTM`  
*Hurdle:* Margin > sector average  
*Growth hurdle:* Margin CAGR ≥ **10% p.a.** (from historical NI/REV; N/A → excluded)

**3. ROE**  
*Formula:* `roeTTM`  
*Hurdle:* ROE > **15%** for **last 4 quarters** and **each of last 3 FYs** (N/A → excluded)

**4. Debt-to-Equity**  
*Formula:* `debtToEquityTTM`  
*Hurdle:* < sector average **AND** < **1.0**

**5. Capex/OCF**  
*Formula:* `capexToOperatingCashFlowTTM` if present, else `capexPerShareTTM / operatingCashFlowPerShareTTM`  
*Hurdle:* < **20%**

**6. Net Income / Tangible Assets**  
*Formula:* `(netIncomePerShareTTM) / (tangibleBookValuePerShareTTM)`  
*Hurdle:* > **20%**

**7. FCF per Share**  
*Formula:* `freeCashFlowPerShareTTM`

**8. Price per Share**  
*Formula:* `marketCapTTM / shares_outstanding_proxy`  
*Shares proxy:* `marketCapTTM / (priceToSalesRatioTTM * revenuePerShareTTM)`

**9. FCF Growth Rate (CAGR)**  
*Formula:* CAGR using historical `freeCashFlowPerShare_*` columns (e.g., `_FY2021`, `_3Y`)  
(N/A → excluded from ranking)

**10. Ranking Metric**  
*Formula:* `(Price per Share / FCF per Share) / FCF Growth Rate`  
(applied only to companies that pass all hurdles and have positive FCF growth)

**11. P/E**  
*Formula:* `peRatioTTM`

**12. Price/FCF**  
*Formula:* `Price per Share / FCF per Share`

**Important:** We do not fabricate data. If a required metric cannot be calculated, it is **shown as N/A** and the company is **excluded** (because all hurdles must be met).
    """)

    # Make sure numeric fields are numeric
    num_cols = [
        "operatingCashFlowPerShareTTM",
        "netIncomePerShareTTM",
        "revenuePerShareTTM",
        "tangibleBookValuePerShareTTM",
        "debtToEquityTTM",
        "roeTTM",
        "freeCashFlowPerShareTTM",
        "marketCapTTM",
        "priceToSalesRatioTTM",
        "capexPerShareTTM",
        "capexToOperatingCashFlowTTM",
        "peRatioTTM",
        "retainedEarningsTTM",
        "operatingIncomeTTM",
    ]
    dfN = numericify(df_filtered, num_cols)

    # Sector averages
    sector_averages = dfN.groupby("sector").agg({
        "netIncomePerShareTTM": "mean",
        "revenuePerShareTTM": "mean",
        "debtToEquityTTM": "mean",
    }).rename(columns={
        "netIncomePerShareTTM": "sector_net_income",
        "revenuePerShareTTM": "sector_revenue",
        "debtToEquityTTM": "sector_dte",
    })
    dfN = dfN.merge(sector_averages, left_on="sector", right_index=True, how="left")

    # Detect history columns
    fcf_hist_cols = _detect_hist_columns(dfN, "freeCashFlowPerShareTTM")
    ni_hist_cols  = _detect_hist_columns(dfN, "netIncomePerShareTTM")
    rev_hist_cols = _detect_hist_columns(dfN, "revenuePerShareTTM")
    roe_hist_cols = _detect_hist_columns(dfN, "roeTTM")
    op_inc_hist   = _detect_hist_columns(dfN, "operatingIncomeTTM")
    op_cf_hist    = _detect_hist_columns(dfN, "operatingCashFlowPerShareTTM")
    ret_earn_hist = _detect_hist_columns(dfN, "retainedEarningsTTM")
    ps_hist       = _detect_hist_columns(dfN, "priceToSalesRatioTTM")
    revps_hist    = _detect_hist_columns(dfN, "revenuePerShareTTM")

    rows_out = []

    for _, row in dfN.iterrows():
        # 1) Positive operating profit for 7 years
        op_series = []
        if op_inc_hist and len(op_inc_hist) >= 7:
            op_series = [row.get(cname) for _, cname in op_inc_hist[-7:]]
        elif op_cf_hist and len(op_cf_hist) >= 7:
            op_series = [row.get(cname) for _, cname in op_cf_hist[-7:]]
        op7_check = (len(op_series) == 7 and all(pd.notna(v) and v > 0 for v in op_series))
        op7_val = "✅ 7/7 positive" if op7_check else "N/A"

        # 2) Profit margin vs sector and margin CAGR ≥ 10%
        margin = _safe_div(row.get("netIncomePerShareTTM"), row.get("revenuePerShareTTM"))
        sector_margin = _safe_div(row.get("sector_net_income"), row.get("sector_revenue"))
        margin_above_sector = (pd.notna(margin) and pd.notna(sector_margin) and margin > sector_margin)

        margin_cagr = np.nan
        margin_cagr_check = False
        if ni_hist_cols and rev_hist_cols and len(ni_hist_cols) >= 2 and len(rev_hist_cols) >= 2:
            start_margin = _safe_div(row.get(ni_hist_cols[0][1]), row.get(rev_hist_cols[0][1]))
            end_margin   = _safe_div(row.get(ni_hist_cols[-1][1]), row.get(rev_hist_cols[-1][1]))
            years = max(1, len(ni_hist_cols) - 1)
            margin_cagr = _compute_cagr(start_margin, end_margin, years)
            margin_cagr_check = pd.notna(margin_cagr) and margin_cagr >= 0.10

        # 3) ROE > 15% last 4Q and each of last 3 FYs + TTM > 15%
        roe_ttm = row.get("roeTTM")
        roe_ttm_check = (pd.notna(roe_ttm) and roe_ttm > 0.15)
        roe_q_check = False
        roe_y_check = False
        if roe_hist_cols and len(roe_hist_cols) >= 4:
            last4 = [row.get(cname) for _, cname in roe_hist_cols[-4:]]
            roe_q_check = all(pd.notna(v) and v > 0.15 for v in last4)
        if roe_hist_cols and len(roe_hist_cols) >= 3:
            last3_years = [row.get(cname) for _, cname in roe_hist_cols[-3:]]
            roe_y_check = all(pd.notna(v) and v > 0.15 for v in last3_years)

        # 4) Debt-to-equity < sector avg and < 1
        dte = row.get("debtToEquityTTM")
        sector_dte = row.get("sector_dte")
        dte_check = (pd.notna(dte) and pd.notna(sector_dte) and dte < sector_dte and dte < 1.0)

        # 5) Capex/OCF < 20%
        if "capexToOperatingCashFlowTTM" in dfN.columns and pd.notna(row.get("capexToOperatingCashFlowTTM")):
            capex_ratio = row.get("capexToOperatingCashFlowTTM")
        else:
            capex_ratio = _safe_div(row.get("capexPerShareTTM"), row.get("operatingCashFlowPerShareTTM"))
        capex_check = (pd.notna(capex_ratio) and capex_ratio < 0.20)

        # 6) Net income / tangible assets > 20%
        ni_ta = _safe_div(row.get("netIncomePerShareTTM"), row.get("tangibleBookValuePerShareTTM"))
        ni_ta_check = (pd.notna(ni_ta) and ni_ta > 0.20)

        # 7) FCF per share
        fcf_ps = row.get("freeCashFlowPerShareTTM")

        # 8) Price per share via shares proxy
        shares_outstanding = _safe_div(row.get("marketCapTTM"), row.get("priceToSalesRatioTTM") * row.get("revenuePerShareTTM"))
        price_ps = _safe_div(row.get("marketCapTTM"), shares_outstanding)

        # 9) FCF growth rate (CAGR)
        fcf_growth_rate = np.nan
        if fcf_hist_cols and len(fcf_hist_cols) >= 2:
            fcf_growth_rate = _compute_series_cagr(row, fcf_hist_cols)

        # 10) Ranking metric
        ranking_metric = np.nan
        if pd.notna(price_ps) and pd.notna(fcf_ps) and pd.notna(fcf_growth_rate) and fcf_growth_rate > 0:
            ranking_metric = (price_ps / fcf_ps) / fcf_growth_rate

        # 11) P/E
        pe_ratio = row.get("peRatioTTM")

        # 12) Price/FCF
        price_fcf = _safe_div(price_ps, fcf_ps)

        # Proportionate increase in share price vs retained earnings over 5Y (>1 required)
        price5_ratio = "N/A"
        prop_check = False
        if ps_hist and revps_hist and len(ps_hist) >= 2 and len(revps_hist) >= 2 and ret_earn_hist and len(ret_earn_hist) >= 2:
            # proxy price = (P/S) * (Revenue/Share)
            price_old = _safe_div(row.get(ps_hist[0][1]) * row.get(revps_hist[0][1]), 1.0)
            price_new = _safe_div(row.get(ps_hist[-1][1]) * row.get(revps_hist[-1][1]), 1.0)
            re_old = row.get(ret_earn_hist[0][1])
            re_new = row.get(ret_earn_hist[-1][1])

            price_change = _safe_div(price_new - price_old, abs(price_old)) if pd.notna(price_old) and price_old != 0 else np.nan
            re_change    = _safe_div(re_new - re_old, abs(re_old)) if pd.notna(re_old) and re_old != 0 else np.nan

            if pd.notna(price_change) and pd.notna(re_change) and re_change != 0:
                ratio = price_change / re_change
                price5_ratio = f"{ratio:.2f}"
                prop_check = ratio > 1

        # Strict pass: must meet ALL hurdles
        passes_all = all([
            op7_check,
            margin_above_sector,
            margin_cagr_check,
            roe_ttm_check,
            roe_q_check,
            roe_y_check,
            dte_check,
            capex_check,
            ni_ta_check,
            prop_check,
        ])

        if passes_all:
            rows_out.append({
                "Symbol": row.get("symbol"),
                "Name": row.get("name"),
                "Sector": row.get("sector"),
                "Positive Op Profit (7y)": op7_val,
                "Profit Margin (%)": f"{(margin * 100):.2f}%" if pd.notna(margin) else "N/A",
                "Sector Margin (%)": f"{(sector_margin * 100):.2f}%" if pd.notna(sector_margin) else "N/A",
                "Margin CAGR (%, ≥10%)": f"{(margin_cagr * 100):.2f}%" if pd.notna(margin_cagr) else "N/A",
                "ROE TTM (%)": f"{(roe_ttm * 100):.2f}%" if pd.notna(roe_ttm) else "N/A",
                "ROE last 4Q": "✅" if roe_q_check else "❌",
                "ROE last 3 FY": "✅" if roe_y_check else "❌",
                "Debt/Equity": f"{dte:.2f}" if pd.notna(dte) else "N/A",
                "Sector D/E": f"{sector_dte:.2f}" if pd.notna(sector_dte) else "N/A",
                "Capex/OCF (%)": f"{(capex_ratio * 100):.2f}%" if pd.notna(capex_ratio) else "N/A",
                "NI / Tangible Assets (%)": f"{(ni_ta * 100)::.2f}%" if pd.notna(ni_ta) else "N/A",
                "Price per Share": f"{price_ps:.2f}" if pd.notna(price_ps) else "N/A",
                "FCF per Share": f"{fcf_ps:.2f}" if pd.notna(fcf_ps) else "N/A",
                "FCF Growth Rate (CAGR)": f"{(fcf_growth_rate * 100):.2f}%" if pd.notna(fcf_growth_rate) else "N/A",
                "P/E": f"{pe_ratio:.2f}" if pd.notna(pe_ratio) else "N/A",
                "Price/FCF": f"{price_fcf:.2f}" if pd.notna(price_fcf) else "N/A",
                "5Y Price vs Ret. Earn. Ratio": price5_ratio,
                "Ranking Metric": f"{(((price_ps / fcf_ps) / fcf_growth_rate)):.2f}" if (pd.notna(price_ps) and pd.notna(fcf_ps) and pd.notna(fcf_growth_rate) and fcf_growth_rate > 0) else "N/A",
            })

    if not rows_out:
        st.warning("No companies passed all metric hurdles with available multi-year data.")
    else:
        df_out = pd.DataFrame(rows_out)
        # Sort by Ranking Metric (numeric only)
        with np.errstate(invalid='ignore'):
            df_out["_rank_sort"] = pd.to_numeric(df_out["Ranking Metric"], errors="coerce")
        df_out = df_out.sort_values(by="_rank_sort", ascending=True).drop(columns="_rank_sort")
        st.dataframe(df_out, use_container_width=True)


# -------------------------
# MAIN LAYOUT
# -------------------------
st.title("Equity Valuation Dashboard")
st.caption("S&P 500 & NASDAQ 100 — sector valuations, spreads, factor scores & screens")


(
    tab_overview,
    tab_heatmap,
    tab_vq,
    tab_pe_ev,
    tab_roic_ey,
    tab_portfolio,
    tab_buffett,
    tab_table,
    tab_invest,  # NEW
) = st.tabs(
    [
        "Overview",
        "Sector Heatmap",
        "Value vs Quality",
        "P/E vs EV/EBITDA",
               "ROIC vs Earnings Yield",
        "Portfolio Optimizer",
        "Buffett Screen",
        "Data Table",
        "Investment Ideas",  # NEW
    ]
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
            pe_median = (
                dft[pe_col_det].replace([np.inf, -np.inf], np.nan).median()
            )
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

        quad_filter = st.multiselect(
            "Quadrant filter",
            options=[
                "Cheap & Quality",
                "Cheap but Low Quality",
                "Expensive but Quality",
                "Expensive & Low Quality",
            ],
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

        # --- NEW: de-duplicate legend so it only shows quadrants ---
        seen_quadrants = set()
        for trace in fig_vq.data:
            # trace.name will look like "Cheap & Quality, S&P 500"
            quad_name = trace.name.split(",")[0].strip()

            trace.legendgroup = quad_name  # group by quadrant

            if quad_name in seen_quadrants:
                trace.showlegend = False   # hide duplicate
            else:
                trace.name = quad_name     # show just the quadrant name
                trace.showlegend = True
                seen_quadrants.add(quad_name)
        # -----------------------------------------------------------

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
            df_vq["quadrant"]
            .value_counts()
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
with tab_roic_ey:
    st.subheader("ROIC vs Earnings Yield (Outliers & negatives removed, % axes)")

    df_roic_ey = df_filtered.copy()
    df_roic_ey = numericify(df_roic_ey, ["roicTTM", "earningsYieldTTM"])
    df_roic_ey = df_roic_ey.dropna(subset=["roicTTM", "earningsYieldTTM"])
    df_roic_ey = df_roic_ey[
        (df_roic_ey["roicTTM"] >= 0) & (df_roic_ey["earningsYieldTTM"] >= 0)
    ]

    if df_roic_ey.empty:
        st.warning("No valid ROIC / Earnings Yield data after cleaning.")
    else:
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
            (df_roic_ey["roicTTM"] < lower_roic)
            | (df_roic_ey["roicTTM"] > upper_roic)
            | (df_roic_ey["earningsYieldTTM"] < lower_ey)
            | (df_roic_ey["earningsYieldTTM"] > upper_ey)
        ]
        df_roic_ey_clean = df_roic_ey[
            ~df_roic_ey.index.isin(outliers.index)
        ].copy()

        if df_roic_ey_clean.empty:
            st.warning("All points were treated as outliers.")
        else:
            df_roic_ey_clean["ROIC (%)"] = df_roic_ey_clean["roicTTM"] * 100
            df_roic_ey_clean["Earnings Yield (%)"] = (
                df_roic_ey_clean["earningsYieldTTM"] * 100
            )

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

        if not outliers.empty:
            outliers_disp = outliers.copy()
            outliers_disp["ROIC (%)"] = outliers_disp["roicTTM"] * 100
            outliers_disp["Earnings Yield (%)"] = (
                outliers_disp["earningsYieldTTM"] * 100
            )
            st.write("#### Outliers (excluded from chart)")
            st.dataframe(
                outliers_disp[
                    ["symbol", "name", "ROIC (%)", "Earnings Yield (%)"]
                ].sort_values("ROIC (%)", ascending=False)
            )


# -------------------------
# PORTFOLIO OPTIMISER TAB
# -------------------------
with tab_portfolio:
    st.subheader("Portfolio Optimizer (Modern Portfolio Theory)")

    all_tickers = sorted(all_raw["symbol"].dropna().unique())
    selected_tickers = st.multiselect(
        "Select Holdings (tickers):",
        options=all_tickers,
        default=["AAPL", "MSFT", "GOOGL"],
    )

    goal = st.selectbox(
        "Optimization Goal:",
        options=[
            ("Max Sharpe Ratio", "Sharpe"),
            ("Max Return", "Return"),
            ("Min Volatility", "Volatility"),
        ],
        format_func=lambda x: x[0],
    )

    risk_free = st.text_input(
        "Risk-free rate (annual, decimal, e.g. 0.02):", value="0.02"
    )
    years = st.slider(
        "Lookback period (years):", min_value=1, max_value=10, value=3, step=1
    )
    freq = st.selectbox(
        "Return Frequency:",
        options=[("Monthly (ME)", "ME"), ("Daily", "D")],
        format_func=lambda x: x[0],
    )

    run_opt = st.button("Run Portfolio Optimizer")

    if run_opt:
        if not FMP_API_KEY:
            st.error(
                "No FMP API key found. Please set FMP_API_KEY in Streamlit secrets or environment."
            )
        else:
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
                start_date = end_date - pd.Timedelta(days=365 * years)
                prices = fetch_prices_fmp(
                    selected_tickers,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    FMP_API_KEY,
                )
                if prices.empty or prices.isnull().all().all():
                    st.error("No price data found for selected tickers.")
                else:
                    exp_returns, cov_matrix = calc_returns_cov(
                        prices, freq=freq[1]
                    )
                    if (
                        exp_returns is None
                        or cov_matrix is None
                        or np.any(np.isnan(exp_returns))
                        or np.any(np.isnan(cov_matrix))
                        or cov_matrix.shape[0] != len(exp_returns)
                    ):
                        st.error(
                            "Insufficient or invalid price data for optimization. "
                            "Try different tickers or a longer lookback."
                        )
                    else:
                        result, perf = optimise_portfolio(
                            exp_returns.values,
                            cov_matrix.values,
                            risk_free_val,
                            goal[1],
                        )
                        if not result.success:
                            st.error(f"Optimization failed: {result.message}")
                        else:
                            weights = result.x
                            port_ret, port_vol, port_sharpe = perf(weights)

                            st.success("Optimal Portfolio Allocation:")
                            alloc_df = pd.DataFrame(
                                {
                                    "Ticker": exp_returns.index,
                                    "Weight": weights,
                                }
                            )
                            st.dataframe(
                                alloc_df.style.format({"Weight": "{:.2%}"})
                            )

                            st.markdown(
                                f"- **Expected return:** {port_ret:.2%}\n"
                                f"- **Volatility:** {port_vol:.2%}\n"
                                f"- **Sharpe ratio:** {port_sharpe:.2f}"
                            )

                            fig = px.pie(
                                alloc_df,
                                names="Ticker",
                                values="Weight",
                                title=f"Optimal Portfolio Allocation ({goal[0]})",
                                hole=0.3,
                            )
                            st.plotly_chart(fig, use_container_width=True)


# -------------------------
# BUFFETT SCREEN TAB
# -------------------------
with tab_buffett:
    st.subheader("What Would Warren Want? – Buffett-style Screening (S&P 500 / NASDAQ)")

    df_buff = df_filtered.copy()

    # Ensure we have required columns, map P/B if needed
    required_cols = [
        "marketCapTTM",
        "priceToSalesRatioTTM",
        "revenuePerShareTTM",
        "netIncomePerShareTTM",
        "shareholdersEquityPerShareTTM",
        "interestDebtPerShareTTM",
        "peRatioTTM",
        "roicTTM",
        "currentRatioTTM",
        "interestCoverageTTM",
        "dividendYieldTTM",
        "payoutRatioTTM",
        "capexToOperatingCashFlowTTM",
    ]

    if "pbRatioTTM" not in df_buff.columns and "priceToBookRatioTTM" in df_buff.columns:
        df_buff["pbRatioTTM"] = df_buff["priceToBookRatioTTM"]

    required_cols.append("pbRatioTTM")

    missing = [c for c in required_cols if c not in df_buff.columns]
    if missing:
        st.warning(
            f"Missing required columns for Buffett screen: {missing}. "
            "Check your CSV exports."
        )
    else:
        df_buff = numericify(df_buff, required_cols)
        df_buff = df_buff.replace([np.inf, -np.inf], np.nan)

        # Drop rows with zero / negative denominators to avoid divide-by-zero
        denom_mask = (
            (df_buff["priceToSalesRatioTTM"] > 0)
            & (df_buff["revenuePerShareTTM"] > 0)
            & (df_buff["shareholdersEquityPerShareTTM"] > 0)
        )
        df_buff = df_buff[denom_mask].copy()

        # Compute derived metrics
        shares_outstanding = df_buff["marketCapTTM"] / (
            df_buff["priceToSalesRatioTTM"] * df_buff["revenuePerShareTTM"]
        )

        df_buff["net_income"] = df_buff["netIncomePerShareTTM"] * shares_outstanding
        df_buff["revenue"] = df_buff["revenuePerShareTTM"] * shares_outstanding
        df_buff["equity"] = (
            df_buff["shareholdersEquityPerShareTTM"] * shares_outstanding
        )
        df_buff["debt"] = (
            df_buff["interestDebtPerShareTTM"] * shares_outstanding
        )

        df_buff["ROE_calc"] = df_buff["net_income"] / df_buff["equity"]
        df_buff["Debt_to_Equity"] = df_buff["debt"] / df_buff["equity"]
        df_buff["Net_Margin"] = df_buff["net_income"] / df_buff["revenue"]
        df_buff["PE"] = df_buff["peRatioTTM"]
        df_buff["ROIC"] = df_buff["roicTTM"]
        df_buff["Current_Ratio"] = df_buff["currentRatioTTM"]
        df_buff["Interest_Coverage"] = df_buff["interestCoverageTTM"]
        df_buff["Dividend_Yield"] = df_buff["dividendYieldTTM"]
        df_buff["Payout_Ratio"] = df_buff["payoutRatioTTM"]
        df_buff["Price_to_Book"] = df_buff["pbRatioTTM"]
        df_buff["Capex_to_OCF"] = df_buff["capexToOperatingCashFlowTTM"]
        df_buff["Positive_NI"] = df_buff["net_income"] > 0

        metrics = {
            "ROE_min": ("Return on Equity (%)", 0.15, "higher", "ROE_calc"),
            "Debt_to_Equity_max": ("Debt/Equity", 0.5, "lower", "Debt_to_Equity"),
            "Net_Margin_min": ("Net Margin (%)", 0.10, "higher", "Net_Margin"),
            "PE_max": ("P/E Ratio", 20, "lower", "PE"),
            "ROIC_min": ("ROIC (%)", 0.10, "higher", "ROIC"),
            "Current_Ratio_min": ("Current Ratio", 1.5, "higher", "Current_Ratio"),
            "Interest_Coverage_min": (
                "Interest Coverage",
                3,
                "higher",
                "Interest_Coverage",
            ),
            "Dividend_Yield_min": (
                "Dividend Yield (%)",
                0.02,
                "higher",
                "Dividend_Yield",
            ),
            "Payout_Ratio_max": (
                "Payout Ratio (%)",
                0.60,
                "lower",
                "Payout_Ratio",
            ),
            "Price_to_Book_max": ("Price/Book", 3, "lower", "Price_to_Book"),
            "Capex_to_OCF_max": ("Capex/OCF (%)", 0.20, "lower", "Capex_to_OCF"),
            "Positive_NI": ("Positive Net Income", True, "higher", "Positive_NI"),
        }

        st.markdown("#### Filter & Hurdle Controls")
        col_search, col_sector = st.columns([2, 2])

        with col_sector:
            sector_options_buff = ["All"] + sorted(
                df_buff["sector"].dropna().unique()
            )
            sector_filter_buff = st.selectbox("Sector", sector_options_buff)

        filtered_df = df_buff.copy()
        if sector_filter_buff != "All":
            filtered_df = filtered_df[
                filtered_df["sector"] == sector_filter_buff
            ]

        with col_search:
            ticker_options = sorted(filtered_df["symbol"].dropna().unique())
            manual_selection = st.multiselect(
                "Select companies (optional)", options=ticker_options, default=[]
            )

        if manual_selection:
            filtered_df = filtered_df[
                filtered_df["symbol"].isin(manual_selection)
            ]

        metric_labels = [metrics[k][0] for k in metrics.keys()]
        metric_label_to_key = dict(zip(metric_labels, metrics.keys()))
        selected_labels = st.multiselect(
            "Select metrics to display",
            options=metric_labels,
            default=metric_labels,
        )
        selected_metrics = [metric_label_to_key[lbl] for lbl in selected_labels]

        exclude_non_pass = st.checkbox(
            "Only show companies that meet all displayed metric hurdles",
            value=False,
        )

        # Hurdle inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            roe_min = st.number_input("ROE min (%)", value=15.0)
            debt_max = st.number_input("Debt/Equity max", value=0.5)
            margin_min = st.number_input("Net Margin min (%)", value=10.0)
            pe_max = st.number_input("PE max", value=20.0)
        with col2:
            roic_min = st.number_input("ROIC min (%)", value=10.0)
            current_min = st.number_input("Current Ratio min", value=1.5)
            interest_min = st.number_input(
                "Interest Coverage min", value=3.0
            )
            dividend_min = st.number_input(
                "Dividend Yield min (%)", value=2.0
            )
        with col3:
            payout_max = st.number_input(
                "Payout Ratio max (%)", value=60.0
            )
            pb_max = st.number_input("Price/Book max", value=3.0)
            capex_max = st.number_input(
                "Capex/OCF max (%)", value=20.0
            )

        criteria = {
            "ROE_min": roe_min / 100,
            "Debt_to_Equity_max": debt_max,
            "Net_Margin_min": margin_min / 100,
            "PE_max": pe_max,
            "ROIC_min": roic_min / 100,
            "Current_Ratio_min": current_min,
            "Interest_Coverage_min": interest_min,
            "Dividend_Yield_min": dividend_min / 100,
            "Payout_Ratio_max": payout_max / 100,
            "Price_to_Book_max": pb_max,
            "Capex_to_OCF_max": capex_max / 100,
            "Positive_NI": True,
        }

        if filtered_df.empty:
            st.warning("No companies available after filters.")
        else:
            scores = pd.DataFrame(index=filtered_df.index)
            for key in selected_metrics:
                col_name = metrics[key][3]
                direction = metrics[key][2]
                if key == "Positive_NI":
                    vals = filtered_df["Positive_NI"].astype(int)
                else:
                    vals = filtered_df[col_name].astype(float)
                ascending = True if direction == "lower" else False
                scores[key] = vals.rank(
                    ascending=ascending, method="min"
                )
            scores["Total_Score"] = scores.sum(axis=1)

            def passes_all(row):
                for key in selected_metrics:
                    col_name = metrics[key][3]
                    val = row[col_name]
                    if key.endswith("_min") and val < criteria[key]:
                        return False
                    if key.endswith("_max") and val > criteria[key]:
                        return False
                    if key == "Positive_NI" and not val:
                        return False
                return True

            if exclude_non_pass:
                filtered_df = filtered_df[
                    filtered_df.apply(passes_all, axis=1)
                ]
                scores = scores.loc[filtered_df.index]

            display_df = pd.DataFrame(index=filtered_df.index)
            display_df["Symbol"] = filtered_df["symbol"]
            display_df["Name"] = filtered_df["name"]
            display_df["Sector"] = filtered_df["sector"]

            for key in selected_metrics:
                col_name = metrics[key][3]
                label = metrics[key][0]
                vals = filtered_df[col_name]

                def fmt(v, k=key):
                    if k == "Positive_NI":
                        return "✅ Positive" if v else "❌ Negative"
                    try:
                        v_float = float(v)
                    except Exception:
                        return f"❌ {v}"
                    passed = (
                        (k.endswith("_min") and v_float >= criteria[k])
                        or (k.endswith("_max") and v_float <= criteria[k])
                    )
                    icon = "✅" if passed else "❌"
                    return f"{icon} {v_float:.2f}"

                display_df[label] = vals.apply(fmt)

            display_df["Total Score"] = scores["Total_Score"]

            st.dataframe(
                display_df.sort_values("Total Score"),
                use_container_width=True,
            )

# -------------------------
# INVESTMENT IDEAS TAB (NEW)
# -------------------------
with tab_invest:
    investment_ideas_tool(df_filtered)

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
    st.caption(
        "Showing first 500 rows for performance. "
        "Export from the original CSVs if you need the full dataset."

    )
