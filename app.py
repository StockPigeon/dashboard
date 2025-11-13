import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

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
