# update_data.py
import requests
import pandas as pd
from datetime import datetime
import streamlit as st

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
FMP_API_KEY = st.secrets["FMP_API_KEY"]
ETF_BASE = "https://financialmodelingprep.com/stable/etf/holdings"
BULK_METRICS_URL = "https://financialmodelingprep.com/stable/key-metrics-ttm-bulk"


# --------------------------------------------------------------
# LOAD BULK KEY METRICS (CSV)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bulk_key_metrics():
    url = f"{BULK_METRICS_URL}?apikey={FMP_API_KEY}"
    df = pd.read_csv(url)

    df.columns = [c.strip() for c in df.columns]
    df["symbol"] = df["symbol"].astype(str).str.upper()

    return df

@st.cache_data(show_spinner=True)
def load_bulk_profiles():
    """
    Loads ALL company profiles (sector, companyName) from profile-bulk CSV parts 0..N.
    FMP returns CSV, not JSON, so we parse each part with pandas.read_csv().
    """
    all_parts = []
    part = 0

    while True:
        url = f"https://financialmodelingprep.com/stable/profile-bulk?part={part}&apikey={FMP_API_KEY}"

        try:
            # Try reading CSV directly
            df = pd.read_csv(url)

            # If empty → stop
            if df.empty:
                break

            # Standardise symbol
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.upper()
            else:
                # If no symbol column is present, skip this part
                print(f"Part {part} missing 'symbol' column. Columns: {df.columns.tolist()}")
                break

            all_parts.append(df)
            part += 1

        except pd.errors.EmptyDataError:
            # Happens if CSV endpoint returns nothing (end of parts)
            break

        except Exception as e:
            print(f"Error loading profile-bulk part {part}: {e}")
            break

    if not all_parts:
        print("No valid profile-bulk CSV data returned.")
        return pd.DataFrame(columns=["symbol", "sector", "companyName"])

    profiles = pd.concat(all_parts, ignore_index=True)

    # Ensure required columns exist
    for col in ["symbol", "sector", "companyName"]:
        if col not in profiles.columns:
            profiles[col] = None

    # Only keep necessary columns
    profiles = profiles[["symbol", "sector", "companyName"]]

    return profiles




# --------------------------------------------------------------
# ETF HOLDINGS
# --------------------------------------------------------------
def get_etf_holdings(etf_symbol="A200.AX"):
    params = {"symbol": etf_symbol, "apikey": FMP_API_KEY}
    r = requests.get(ETF_BASE, params=params)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Unexpected ETF response for {etf_symbol}: {data}")

    df = pd.DataFrame(data)

    # Intelligent symbol extraction
    if "asset" in df.columns:
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        df = df.rename(columns={"asset": "symbol"})
    elif "symbol" not in df.columns:
        raise ValueError(
            f"No 'asset' or 'symbol' columns returned for {etf_symbol}. "
            f"Columns: {list(df.columns)}"
        )

    def extract_symbol_from_cell(x):
        if isinstance(x, dict):
            return x.get("ticker", x.get("symbol", str(x)))
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if not x.empty:
                return str(x.iloc[0]) if isinstance(x, pd.Series) else str(x.iloc[0,0])
            return None
        if isinstance(x, list) and len(x):
            return extract_symbol_from_cell(x[0])
        return str(x)

    df["symbol"] = df["symbol"].apply(extract_symbol_from_cell).astype(str).str.upper()
    df = df.dropna(subset=["symbol"])

    keep_cols = ["symbol"]
    for c in ["name", "weightPercentage", "marketValue", "sharesNumber"]:
        if c in df.columns:
            keep_cols.append(c)

    return df[keep_cols]


# --------------------------------------------------------------
# MERGE ETF HOLDINGS WITH BULK METRICS
# --------------------------------------------------------------
def merge_etf_with_metrics(holdings_df, metrics_df):
    metrics_df = metrics_df.copy()
    metrics_df["symbol"] = metrics_df["symbol"].astype(str).str.upper()

    merged = holdings_df.merge(metrics_df, on="symbol", how="left")
    return merged


# --------------------------------------------------------------
# INDEX ETF MAP
# --------------------------------------------------------------
INDEX_ETF_MAP = {
    "S&P 500": "SPY",
    "NASDAQ 100": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM",
    "S&P 400 Midcap": "IJH",
    "S&P 600 Smallcap": "IJR",

    "FTSE 100": "EWU",
    "Euro Stoxx 50": "FEZ",
    "DAX (Germany)": "EWG",
    "CAC 40": "EWQ",

    "Nikkei 225": "EWJ",
    "Hang Seng": "EWH",
    "China A50": "AFTY",
    "CSI 300": "ASHR",

    "ASX 200": "A200.AX",
    "TSX 60": "XIU.TO",
    "MSCI World": "URTH",
    "MSCI Emerging Markets": "EEM",
}


# --------------------------------------------------------------
# GLOBAL INDEX METRICS PIPELINE
# (Runs automatically on first import)
# --------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_global_data():
    """Runs once per session or when the cache invalidates (daily/hourly/etc.)"""

    bulk = load_bulk_key_metrics()
    profiles = load_bulk_profiles()

    # Build lookup dictionaries
    sector_map = profiles.set_index("symbol")["sector"].to_dict()
    name_map = profiles.set_index("symbol")["companyName"].to_dict()

    # Standardize metric names coming from bulk endpoint
    COLUMN_MAP = {
        "returnOnEquityTTM": "roeTTM",
        "returnOnAssetsTTM": "roaTTM",
        "returnOnInvestedCapitalTTM": "roicTTM",
        "evToEBITDATTM": "evToEbitdaTTM",
    }

    # Apply renaming to bulk metrics
    bulk = bulk.rename(columns=COLUMN_MAP)

    results = {}

    for index_name, etf_symbol in INDEX_ETF_MAP.items():
        try:
            holdings = get_etf_holdings(etf_symbol)
            merged = merge_etf_with_metrics(holdings, bulk)

            # Apply same renaming to merged dataset
            merged = merged.rename(columns=COLUMN_MAP)

            # Add sector and name enrichment
            merged["sector"] = merged["symbol"].map(sector_map)
            merged["name"] = merged["symbol"].map(name_map).fillna(merged["name"])
            merged["sector"] = merged["sector"].astype("string").fillna("Unknown")


            results[index_name] = merged
            
            # missing_symbols = merged.loc[merged["sector"].isna(), "symbol"].unique().tolist()
            # print("Symbols missing sector:", missing_symbols[:50])


        except Exception as e:
            results[index_name] = pd.DataFrame({"error": [str(e)]})

    return results
