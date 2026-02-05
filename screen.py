import os
import math
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Buffett-Style Screening Tool",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# API KEY PLACEHOLDER
# -----------------------------
# Recommended: put this into .streamlit/secrets.toml as:
# FMP_API_KEY = "your_key_here"
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", "")).strip()

# If you use a different provider, swap fetch functions accordingly.
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


# -----------------------------
# Helpers
# -----------------------------
def numericify(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric safely."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_get(d: Dict, key: str, default=np.nan):
    return d.get(key, default) if isinstance(d, dict) else default


def chunked(lst: List[str], n: int) -> List[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def normalize_tickers(raw: str) -> List[str]:
    if not raw:
        return []
    # allow comma/space/newline separated
    tickers = (
        raw.replace("\n", ",")
        .replace(" ", ",")
        .replace(";", ",")
        .split(",")
    )
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# -----------------------------
# Fetch data (optional)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fmp_get_json(url: str, params: Dict) -> List[Dict]:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Some endpoints return dict w/ "symbol" etc; normalize to list[dict]
    if isinstance(data, dict):
        return [data]
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=True, ttl=60 * 30)
def fetch_fmp_ttm_bundle(symbols: List[str], api_key: str) -> pd.DataFrame:
    """
    Attempt to build a dataframe with your required columns using FMP-like endpoints.
    This is optional; if you already export CSV, use the Upload CSV option.

    Endpoints used (typical FMP):
      - /profile/{symbol}
      - /ratios-ttm/{symbol}
      - /key-metrics-ttm/{symbol}

    Note: Availability of exact keys varies by provider/plan. Missing keys are handled.
    """
    if not api_key:
        raise ValueError("No API key provided.")

    rows = []
    progress = st.progress(0, text="Fetching data from API...")
    total = max(1, len(symbols))

    for i, sym in enumerate(symbols, start=1):
        try:
            prof = fmp_get_json(
                f"{FMP_BASE_URL}/profile/{sym}",
                {"apikey": api_key},
            )
            ratios_ttm = fmp_get_json(
                f"{FMP_BASE_URL}/ratios-ttm/{sym}",
                {"apikey": api_key},
            )
            metrics_ttm = fmp_get_json(
                f"{FMP_BASE_URL}/key-metrics-ttm/{sym}",
                {"apikey": api_key},
            )

            prof0 = prof[0] if prof else {}
            r0 = ratios_ttm[0] if ratios_ttm else {}
            m0 = metrics_ttm[0] if metrics_ttm else {}

            # Build a row containing BOTH your identifier fields + expected numeric fields.
            row = {
                "symbol": sym,
                "name": safe_get(prof0, "companyName", safe_get(prof0, "name", sym)),
                "sector": safe_get(prof0, "sector", np.nan),

                # Core inputs (attempt to map from ratios/metrics/profile)
                "marketCapTTM": safe_get(m0, "marketCapTTM", safe_get(prof0, "mktCap", np.nan)),
                "priceToSalesRatioTTM": safe_get(m0, "priceToSalesRatioTTM", safe_get(r0, "priceToSalesRatioTTM", safe_get(r0, "priceToSalesRatio", np.nan))),
                "revenuePerShareTTM": safe_get(m0, "revenuePerShareTTM", safe_get(r0, "revenuePerShareTTM", safe_get(r0, "revenuePerShare", np.nan))),
                "netIncomePerShareTTM": safe_get(m0, "netIncomePerShareTTM", safe_get(r0, "netIncomePerShareTTM", safe_get(r0, "netIncomePerShare", np.nan))),
                "shareholdersEquityPerShareTTM": safe_get(m0, "shareholdersEquityPerShareTTM", safe_get(r0, "shareholdersEquityPerShareTTM", safe_get(r0, "shareholdersEquityPerShare", np.nan))),
                "interestDebtPerShareTTM": safe_get(m0, "interestDebtPerShareTTM", safe_get(r0, "interestDebtPerShareTTM", safe_get(r0, "interestDebtPerShare", np.nan))),

                "peRatioTTM": safe_get(m0, "peRatioTTM", safe_get(r0, "peRatioTTM", safe_get(r0, "priceEarningsRatio", np.nan))),
                "pbRatioTTM": safe_get(m0, "pbRatioTTM", safe_get(r0, "pbRatioTTM", safe_get(r0, "priceToBookRatio", np.nan))),
                "roicTTM": safe_get(m0, "roicTTM", safe_get(r0, "returnOnCapitalEmployedTTM", safe_get(r0, "returnOnCapitalEmployed", np.nan))),

                "currentRatioTTM": safe_get(m0, "currentRatioTTM", safe_get(r0, "currentRatioTTM", safe_get(r0, "currentRatio", np.nan))),
                "interestCoverageTTM": safe_get(m0, "interestCoverageTTM", safe_get(r0, "interestCoverageTTM", safe_get(r0, "interestCoverage", np.nan))),

                "dividendYieldTTM": safe_get(m0, "dividendYieldTTM", safe_get(r0, "dividendYieldTTM", safe_get(r0, "dividendYield", np.nan))),
                "payoutRatioTTM": safe_get(m0, "payoutRatioTTM", safe_get(r0, "payoutRatioTTM", safe_get(r0, "payoutRatio", np.nan))),

                "capexToOperatingCashFlowTTM": safe_get(m0, "capexToOperatingCashFlowTTM", safe_get(r0, "capexToOperatingCashFlowTTM", safe_get(r0, "capexToOperatingCashFlow", np.nan))),
            }

            rows.append(row)

        except Exception as e:
            rows.append({"symbol": sym, "name": sym, "sector": np.nan, "error": str(e)})

        progress.progress(i / total, text=f"Fetching {i}/{total}: {sym}")

    progress.empty()
    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Screening Engine
# -----------------------------
def build_buffett_screen(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Returns:
      - df_buff (with derived columns)
      - display_df (empty for now; built later after UI controls)
      - metrics dict
    """

    df_buff = df_raw.copy()

    # Map P/B if needed
    if "pbRatioTTM" not in df_buff.columns and "priceToBookRatioTTM" in df_buff.columns:
        df_buff["pbRatioTTM"] = df_buff["priceToBookRatioTTM"]

    required_cols = [
        # identifiers used for filtering & display
        "symbol", "name", "sector",

        # numeric inputs required for derived metrics + screen
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
        "pbRatioTTM",
    ]

    missing = [c for c in required_cols if c not in df_buff.columns]
    if missing:
        raise KeyError(
            "Missing required columns for Buffett screen: "
            f"{missing}. Check your CSV/API mapping."
        )

    # numeric conversion
    numeric_cols = [c for c in required_cols if c not in ("symbol", "name", "sector")]
    df_buff = numericify(df_buff, numeric_cols).replace([np.inf, -np.inf], np.nan)

    # Denominator checks to prevent divide-by-zero and invalid share estimation
    denom_mask = (
        (df_buff["marketCapTTM"] > 0) &
        (df_buff["priceToSalesRatioTTM"] > 0) &
        (df_buff["revenuePerShareTTM"] > 0) &
        (df_buff["shareholdersEquityPerShareTTM"] > 0)
    )
    df_buff = df_buff.loc[denom_mask].copy()

    # Derived metrics
    price_implied = df_buff["priceToSalesRatioTTM"] * df_buff["revenuePerShareTTM"]

    with np.errstate(divide="ignore", invalid="ignore"):
        shares_outstanding = df_buff["marketCapTTM"] / price_implied

        df_buff["net_income"] = df_buff["netIncomePerShareTTM"] * shares_outstanding
        df_buff["revenue"] = df_buff["revenuePerShareTTM"] * shares_outstanding
        df_buff["equity"] = df_buff["shareholdersEquityPerShareTTM"] * shares_outstanding
        df_buff["debt"] = df_buff["interestDebtPerShareTTM"] * shares_outstanding

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

    # Metrics definition: (Label, default_threshold, direction, derived_col)
    metrics = {
        "ROE_min": ("Return on Equity (%)", 0.15, "higher", "ROE_calc"),
        "Debt_to_Equity_max": ("Debt/Equity", 0.5, "lower", "Debt_to_Equity"),
        "Net_Margin_min": ("Net Margin (%)", 0.10, "higher", "Net_Margin"),
        "PE_max": ("P/E Ratio", 20, "lower", "PE"),
        "ROIC_min": ("ROIC (%)", 0.10, "higher", "ROIC"),
        "Current_Ratio_min": ("Current Ratio", 1.5, "higher", "Current_Ratio"),
        "Interest_Coverage_min": ("Interest Coverage", 3.0, "higher", "Interest_Coverage"),
        "Dividend_Yield_min": ("Dividend Yield (%)", 0.02, "higher", "Dividend_Yield"),
        "Payout_Ratio_max": ("Payout Ratio (%)", 0.60, "lower", "Payout_Ratio"),
        "Price_to_Book_max": ("Price/Book", 3.0, "lower", "Price_to_Book"),
        "Capex_to_OCF_max": ("Capex/OCF (%)", 0.20, "lower", "Capex_to_OCF"),
        "Positive_NI": ("Positive Net Income", True, "higher", "Positive_NI"),
    }

    return df_buff, metrics


def compute_scores(filtered_df: pd.DataFrame, metrics: Dict, selected_metrics: List[str]) -> pd.DataFrame:
    scores = pd.DataFrame(index=filtered_df.index)

    for key in selected_metrics:
        col_name = metrics[key][3]
        direction = metrics[key][2]

        if key == "Positive_NI":
            vals = filtered_df[col_name].astype(int)
        else:
            vals = pd.to_numeric(filtered_df[col_name], errors="coerce")

        ascending = True if direction == "lower" else False
        scores[key] = vals.rank(ascending=ascending, method="min")

    scores["Total_Score"] = scores.sum(axis=1)
    return scores


def build_criteria_from_inputs() -> Dict:
    """UI inputs -> criteria dict (in decimals where needed)."""
    st.markdown("#### Hurdle Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        roe_min = st.number_input("ROE min (%)", value=15.0, step=0.5)
        debt_max = st.number_input("Debt/Equity max", value=0.50, step=0.05, format="%.2f")
        margin_min = st.number_input("Net Margin min (%)", value=10.0, step=0.5)
        pe_max = st.number_input("PE max", value=20.0, step=1.0)

    with col2:
        roic_min = st.number_input("ROIC min (%)", value=10.0, step=0.5)
        current_min = st.number_input("Current Ratio min", value=1.5, step=0.1, format="%.2f")
        interest_min = st.number_input("Interest Coverage min", value=3.0, step=0.5, format="%.2f")
        dividend_min = st.number_input("Dividend Yield min (%)", value=2.0, step=0.25)

    with col3:
        payout_max = st.number_input("Payout Ratio max (%)", value=60.0, step=1.0)
        pb_max = st.number_input("Price/Book max", value=3.0, step=0.25, format="%.2f")
        capex_max = st.number_input("Capex/OCF max (%)", value=20.0, step=1.0)

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
    return criteria


def passes_all(row: pd.Series, selected_metrics: List[str], metrics: Dict, criteria: Dict) -> Tuple[bool, List[str]]:
    """
    Returns (pass_bool, fail_reasons)
    Fail NaNs explicitly.
    """
    fails = []
    for key in selected_metrics:
        col_name = metrics[key][3]
        val = row.get(col_name, np.nan)

        # Fail missing
        if pd.isna(val):
            fails.append(f"{metrics[key][0]} is NaN")
            continue

        # Special case
        if key == "Positive_NI":
            if not bool(val):
                fails.append("Net income is not positive")
            continue

        v = float(val)

        if key.endswith("_min"):
            if v < criteria[key]:
                fails.append(f"{metrics[key][0]} < {criteria[key]}")
        elif key.endswith("_max"):
            if v > criteria[key]:
                fails.append(f"{metrics[key][0]} > {criteria[key]}")

    return (len(fails) == 0), fails


def fmt_value(v, key: str, criteria: Dict, metrics: Dict) -> str:
    if key == "Positive_NI":
        return "✅ Positive" if bool(v) else "❌ Negative"
    if pd.isna(v):
        return "❌ NaN"
    try:
        v_float = float(v)
    except Exception:
        return f"❌ {v}"

    passed = (
        (key.endswith("_min") and v_float >= criteria[key]) or
        (key.endswith("_max") and v_float <= criteria[key])
    )
    icon = "✅" if passed else "❌"
    return f"{icon} {v_float:.2f}"


# -----------------------------
# UI: Header
# -----------------------------
st.title("📈 Buffett-Style Screening Tool")
st.caption(
    "Upload a fundamentals CSV or fetch from an API, then filter and rank companies against hurdle criteria."
)

with st.expander("What columns do I need (for CSV uploads)?", expanded=False):
    st.markdown(
        """
**Required identifiers:**
- `symbol`, `name`, `sector`

**Required numeric inputs (TTM):**
- `marketCapTTM`
- `priceToSalesRatioTTM`
- `revenuePerShareTTM`
- `netIncomePerShareTTM`
- `shareholdersEquityPerShareTTM`
- `interestDebtPerShareTTM`
- `peRatioTTM`
- `roicTTM`
- `currentRatioTTM`
- `interestCoverageTTM`
- `dividendYieldTTM`
- `payoutRatioTTM`
- `capexToOperatingCashFlowTTM`
- `pbRatioTTM` (or `priceToBookRatioTTM` which will be mapped automatically)
        """
    )


# -----------------------------
# UI: Data Source
# -----------------------------
st.sidebar.header("Data Source")
source = st.sidebar.radio(
    "Choose data source",
    ["Upload CSV", "Fetch via API (FMP-style)"],
    index=0,
)

df_input = None

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded CSV: {df_input.shape[0]} rows, {df_input.shape[1]} columns")
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

else:
    st.sidebar.markdown("### API Settings")
    if not FMP_API_KEY:
        st.sidebar.warning(
            "No API key found. Add `FMP_API_KEY` to `.streamlit/secrets.toml` or set env var."
        )

    tickers_raw = st.sidebar.text_area(
        "Tickers (comma/space/newline separated)",
        value="AAPL, MSFT, BRK-B",
        height=100,
    )
    tickers = normalize_tickers(tickers_raw)

    if st.sidebar.button("Fetch Data", type="primary", use_container_width=True):
        if not tickers:
            st.sidebar.error("Please enter at least one ticker.")
        elif not FMP_API_KEY:
            st.sidebar.error("API key missing. Add FMP_API_KEY to secrets or env var.")
        else:
            try:
                df_input = fetch_fmp_ttm_bundle(tickers, FMP_API_KEY)
                st.sidebar.success(f"Fetched: {df_input.shape[0]} rows")
            except Exception as e:
                st.sidebar.error(f"API fetch failed: {e}")


if df_input is None:
    st.info("Load data using the sidebar to begin.")
    st.stop()


# -----------------------------
# Build Screen Base
# -----------------------------
try:
    df_buff, metrics = build_buffett_screen(df_input)
except Exception as e:
    st.error(str(e))
    st.stop()

if df_buff.empty:
    st.warning("No valid companies after denominator checks (market cap / P/S / revenue per share / equity per share).")
    st.stop()


# -----------------------------
# Controls: Filters & Metrics
# -----------------------------
st.markdown("### Filters & Metrics")

col_search, col_sector, col_opts = st.columns([2, 2, 2])

with col_sector:
    sector_options = ["All"] + sorted(df_buff["sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sector_options, index=0)

filtered_df = df_buff.copy()
if sector_filter != "All":
    filtered_df = filtered_df[filtered_df["sector"] == sector_filter]

with col_search:
    ticker_options = sorted(filtered_df["symbol"].dropna().unique().tolist())
    manual_selection = st.multiselect(
        "Select companies (optional)",
        options=ticker_options,
        default=[],
    )
if manual_selection:
    filtered_df = filtered_df[filtered_df["symbol"].isin(manual_selection)]

with col_opts:
    exclude_non_pass = st.checkbox(
        "Only show companies that meet all displayed metric hurdles",
        value=False,
    )
    show_fail_reasons = st.checkbox(
        "Show fail reasons (when not excluding non-pass)",
        value=True,
    )

metric_labels = [metrics[k][0] for k in metrics.keys()]
metric_label_to_key = dict(zip(metric_labels, metrics.keys()))

selected_labels = st.multiselect(
    "Select metrics to display (and to score)",
    options=metric_labels,
    default=metric_labels,
)

selected_metrics = [metric_label_to_key[lbl] for lbl in selected_labels]
criteria = build_criteria_from_inputs()


# -----------------------------
# Apply Pass/Fail & Score
# -----------------------------
if filtered_df.empty:
    st.warning("No companies available after filters.")
    st.stop()

# compute pass/fail and reasons
pass_mask = []
fail_reasons_list = []

for idx, row in filtered_df.iterrows():
    ok, fails = passes_all(row, selected_metrics, metrics, criteria)
    pass_mask.append(ok)
    fail_reasons_list.append("; ".join(fails))

pass_mask = pd.Series(pass_mask, index=filtered_df.index)

if exclude_non_pass:
    filtered_df = filtered_df.loc[pass_mask].copy()
    if filtered_df.empty:
        st.warning("No companies pass all selected metric hurdles with current thresholds.")
        st.stop()

scores = compute_scores(filtered_df, metrics, selected_metrics)

# -----------------------------
# Build Display Table
# -----------------------------
display_df = pd.DataFrame(index=filtered_df.index)
display_df["Symbol"] = filtered_df["symbol"]
display_df["Name"] = filtered_df["name"]
display_df["Sector"] = filtered_df["sector"]

for key in selected_metrics:
    col_name = metrics[key][3]
    label = metrics[key][0]
    display_df[label] = filtered_df[col_name].apply(lambda v: fmt_value(v, key, criteria, metrics))

display_df["Total Score"] = scores["Total_Score"]

if not exclude_non_pass and show_fail_reasons:
    # align fail reasons to current filtered_df index
    # (fail_reasons_list was built from pre-exclude filter set; re-map)
    tmp = pd.Series(fail_reasons_list, index=pass_mask.index)
    display_df["Fail Reasons"] = tmp.loc[display_df.index].replace("", "✅ Pass")

# sort: lower total score is better
display_df_sorted = display_df.sort_values("Total Score", ascending=True)

st.markdown("### Results")
st.dataframe(display_df_sorted, use_container_width=True)

# Downloads
csv_bytes = display_df_sorted.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download results as CSV",
    data=csv_bytes,
    file_name="buffett_screen_results.csv",
    mime="text/csv",
    use_container_width=True,
)

# Optional: show underlying numeric dataframe (debug)
with st.expander("Debug: show derived numeric columns", expanded=False):
    cols_show = [
        "symbol", "name", "sector",
        "ROE_calc", "Debt_to_Equity", "Net_Margin",
        "PE", "ROIC", "Current_Ratio", "Interest_Coverage",
        "Dividend_Yield", "Payout_Ratio", "Price_to_Book", "Capex_to_OCF",
        "Positive_NI",
    ]
    cols_show = [c for c in cols_show if c in df_buff.columns]
    st.dataframe(df_buff[cols_show].copy(), use_container_width=True)
