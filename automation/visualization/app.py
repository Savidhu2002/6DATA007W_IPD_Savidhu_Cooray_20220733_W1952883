# ============================
# ONE NOTEBOOK CELL SOLUTION
# This cell will:
# 1) Write the Streamlit app to app_8505.py in the SAME folder as your notebook
# 2) Run Streamlit on port 8505
# 3) Open: http://localhost:8505
# ============================

import textwrap, os, sys, subprocess

APP_FILE = "app_8505.py"
PORT = 8505

app_code = r"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Cut to Ship Dashboard", layout="wide")

EXCEL_PATH = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report.xlsx"
SHEET_NAME = "Final"

# -----------------------------
# HELPERS
# -----------------------------
def safe_div(n, d):
    if d is None or pd.isna(d) or d == 0:
        return np.nan
    return n / d

def percent_fmt(x):
    return "NA" if pd.isna(x) else f"{x*100:,.2f}%"

def num_fmt(x):
    return "NA" if pd.isna(x) else f"{x:,.0f}"

def clean_week(series):
    w = series.astype(str).str.strip()
    w = w.str.replace("Week", "", regex=False).str.replace("W", "", regex=False).str.strip()
    return pd.to_numeric(w, errors="coerce")

def weekly_totals(df):
    agg = {
        "OrderQty": ("OrderQty", "sum"),
        "CutQty": ("CutQty", "sum"),
        "ShipQty": ("ShipQty", "sum"),
    }
    if "CutShipDiff" in df.columns:
        agg["CutShipDiff"] = ("CutShipDiff", "sum")

    wk = df.groupby(["Year", "Week_Num"], dropna=False, as_index=False).agg(**agg)
    wk["Cut/Ship"] = wk["ShipQty"] / wk["CutQty"]
    wk["Order/Ship"] = wk["ShipQty"] / wk["OrderQty"]
    wk["Order/Cut"] = wk["CutQty"] / wk["OrderQty"]
    wk = wk.replace([np.inf, -np.inf], np.nan).sort_values(["Year", "Week_Num"])
    return wk

def top_n_by_ratio(df, group_col, ratio_name, n=10):
    g = df.groupby(group_col, as_index=False).agg(
        OrderQty=("OrderQty", "sum"),
        CutQty=("CutQty", "sum"),
        ShipQty=("ShipQty", "sum"),
    )

    if ratio_name == "Cut/Ship":
        g[ratio_name] = g["ShipQty"] / g["CutQty"]
    elif ratio_name == "Order/Ship":
        g[ratio_name] = g["ShipQty"] / g["OrderQty"]
    elif ratio_name == "Order/Cut":
        g[ratio_name] = g["CutQty"] / g["OrderQty"]
    else:
        raise ValueError("Invalid ratio_name")

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=[ratio_name])
    return g.sort_values(ratio_name, ascending=False).head(n)

def diff_breakdown_cols(df):
    possible = [
        "Fabric Damage", "Colour Shading", "Finishing Damage", "Shade Band", "Pilot",
        "Wash Reference Sample", "Cut panel rejection qty", "Sewing Reject Qty",
        "EMB / Printing Damages", "Washing Damages", "Sample qty", "Shortage qty",
        "Unreconciled qty - panel form", "Unreconciled qty - GMT form",
        "Second Quality", "Good garments", "PO Mix",
        "Transfer to other SOD", "Transfer from other SOD"
    ]
    return [c for c in possible if c in df.columns]

def sum_diff_breakdown(df):
    cols = diff_breakdown_cols(df)
    if not cols:
        return pd.Series(dtype=float)
    tmp = df[cols].copy()
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    return tmp.sum(numeric_only=True).sort_values(ascending=False)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df.columns = [c.replace("  ", " ").strip() for c in df.columns]

    # Drop 'Customers' if present. We only use Calling Name.
    if "Customers" in df.columns:
        df = df.drop(columns=["Customers"])

    # Rename business dimensions
    df = df.rename(columns={
        "Unit": "Factory",
        "Calling Name": "Customer",
        "Garment item type": "Product"
    })

    # Standardize qty columns (your file uses spaces)
    df = df.rename(columns={
        "Order Qty": "OrderQty",
        "Cut Qty": "CutQty",
        "Ship Qty": "ShipQty",
        "Cutship Difference": "CutShipDiff"
    })

    # Remove exact duplicate columns after renaming
    df = df.loc[:, ~df.columns.duplicated()]

    # Required columns
    required = ["Year", "Week", "Factory", "Customer", "Product", "OrderQty", "CutQty", "ShipQty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Numeric
    for c in ["OrderQty", "CutQty", "ShipQty", "CutShipDiff"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Year & Week parsing
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Week_Num"] = clean_week(df["Week"])

    # Drop rows where all qty are blank
    df = df.dropna(subset=["OrderQty", "CutQty", "ShipQty"], how="all")

    return df

df = load_data(EXCEL_PATH, SHEET_NAME)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

years_all = sorted([int(x) for x in df["Year"].dropna().unique().tolist()])
weeks_all = sorted([int(x) for x in df["Week_Num"].dropna().unique().tolist()])

factories_all = sorted(df["Factory"].dropna().unique().tolist())
customers_all = sorted(df["Customer"].dropna().unique().tolist())
products_all = sorted(df["Product"].dropna().unique().tolist())

f_year = st.sidebar.multiselect("Year", years_all, default=years_all)
f_week = st.sidebar.multiselect("Week", weeks_all, default=[])
f_factory = st.sidebar.multiselect("Factory", factories_all, default=[])
f_customer = st.sidebar.multiselect("Customer", customers_all, default=[])
f_product = st.sidebar.multiselect("Product", products_all, default=[])

fdf = df.copy()
if f_year:
    fdf = fdf[fdf["Year"].isin(f_year)]
if f_week:
    fdf = fdf[fdf["Week_Num"].isin(f_week)]
if f_factory:
    fdf = fdf[fdf["Factory"].isin(f_factory)]
if f_customer:
    fdf = fdf[fdf["Customer"].isin(f_customer)]
if f_product:
    fdf = fdf[fdf["Product"].isin(f_product)]

st.sidebar.caption(f"Rows after filters: {len(fdf):,}")

# -----------------------------
# PAGE SELECTOR
# -----------------------------
page = st.selectbox(
    "Select Page",
    [
        "1 Overall",
        "2 Cut/Ship",
        "3 Order/Ship",
        "4 Order/Cut",
        "5 Cut Ship Difference",
        "6 Latest Week Deep Dive"
    ]
)

# -----------------------------
# PAGE 1
# -----------------------------
if page == "1 Overall":
    st.title("Overall Performance")

    total_order = fdf["OrderQty"].sum()
    total_cut = fdf["CutQty"].sum()
    total_ship = fdf["ShipQty"].sum()

    cut_ship = safe_div(total_ship, total_cut)
    order_ship = safe_div(total_ship, total_order)
    order_cut = safe_div(total_cut, total_order)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cut/Ship", percent_fmt(cut_ship))
    c2.metric("Order/Ship", percent_fmt(order_ship))
    c3.metric("Order/Cut", percent_fmt(order_cut))

    c4, c5, c6 = st.columns(3)
    c4.metric("Total Order Qty", num_fmt(total_order))
    c5.metric("Total Cut Qty", num_fmt(total_cut))
    c6.metric("Total Ship Qty", num_fmt(total_ship))

    st.divider()
    st.subheader("Week wise movement (totals, not averages)")

    wk = weekly_totals(fdf)
    if wk.empty:
        st.warning("No weekly data available after filters.")
    else:
        wk_long = wk.melt(
            id_vars=["Year", "Week_Num", "OrderQty", "CutQty", "ShipQty"],
            value_vars=["Cut/Ship", "Order/Ship", "Order/Cut"],
            var_name="Metric",
            value_name="Value"
        )
        fig1 = px.line(wk_long, x="Week_Num", y="Value", color="Metric", facet_col="Year", markers=True)
        fig1.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)

        wk_qty = wk.melt(
            id_vars=["Year", "Week_Num"],
            value_vars=["OrderQty", "CutQty", "ShipQty"],
            var_name="QtyType",
            value_name="Qty"
        )
        fig2 = px.bar(wk_qty, x="Week_Num", y="Qty", color="QtyType", facet_col="Year", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Weekly totals table"):
            st.dataframe(wk)

# -----------------------------
# PAGE 2
# -----------------------------
elif page == "2 Cut/Ship":
    st.title("Cut/Ship | Top 10 by Customer, Product, Factory")
    tabs = st.tabs(["Customer (Top 10)", "Product (Top 10)", "Factory (Top 10)"])
    for tab, col in zip(tabs, ["Customer", "Product", "Factory"]):
        with tab:
            top = top_n_by_ratio(fdf, col, "Cut/Ship", n=10)
            fig = px.bar(top, x="Cut/Ship", y=col, orientation="h")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top)

# -----------------------------
# PAGE 3
# -----------------------------
elif page == "3 Order/Ship":
    st.title("Order/Ship | Top 10 by Customer, Product, Factory")
    tabs = st.tabs(["Customer (Top 10)", "Product (Top 10)", "Factory (Top 10)"])
    for tab, col in zip(tabs, ["Customer", "Product", "Factory"]):
        with tab:
            top = top_n_by_ratio(fdf, col, "Order/Ship", n=10)
            fig = px.bar(top, x="Order/Ship", y=col, orientation="h")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top)

# -----------------------------
# PAGE 4
# -----------------------------
elif page == "4 Order/Cut":
    st.title("Order/Cut | Top 10 by Customer, Product, Factory")
    tabs = st.tabs(["Customer (Top 10)", "Product (Top 10)", "Factory (Top 10)"])
    for tab, col in zip(tabs, ["Customer", "Product", "Factory"]):
        with tab:
            top = top_n_by_ratio(fdf, col, "Order/Cut", n=10)
            fig = px.bar(top, x="Order/Cut", y=col, orientation="h")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top)

# -----------------------------
# PAGE 5
# -----------------------------
elif page == "5 Cut Ship Difference":
    st.title("Cut Ship Difference Analysis")

    if "CutShipDiff" not in fdf.columns:
        st.error("Column 'Cutship Difference' not found in dataset (expected as CutShipDiff after rename).")
    else:
        g_factory = fdf.groupby("Factory", as_index=False)["CutShipDiff"].sum().sort_values("CutShipDiff", ascending=False)
        g_customer = fdf.groupby("Customer", as_index=False)["CutShipDiff"].sum().sort_values("CutShipDiff", ascending=False)
        g_product = fdf.groupby("Product", as_index=False)["CutShipDiff"].sum().sort_values("CutShipDiff", ascending=False)

        t1, t2, t3 = st.tabs(["Factory", "Customer", "Product"])
        with t1:
            st.plotly_chart(px.bar(g_factory.head(20), x="CutShipDiff", y="Factory", orientation="h"), use_container_width=True)
            st.dataframe(g_factory)
        with t2:
            st.plotly_chart(px.bar(g_customer.head(20), x="CutShipDiff", y="Customer", orientation="h"), use_container_width=True)
            st.dataframe(g_customer)
        with t3:
            st.plotly_chart(px.bar(g_product.head(20), x="CutShipDiff", y="Product", orientation="h"), use_container_width=True)
            st.dataframe(g_product)

        st.divider()
        st.subheader("How Cut Ship Difference is made (Reason breakdown)")

        breakdown = sum_diff_breakdown(fdf)
        if breakdown.empty:
            st.warning("No breakdown columns found (Fabric Damage to Transfer from other SOD).")
        else:
            bdf = breakdown.reset_index()
            bdf.columns = ["Reason", "Qty"]
            st.plotly_chart(px.bar(bdf, x="Qty", y="Reason", orientation="h"), use_container_width=True)
            st.dataframe(bdf)

# -----------------------------
# PAGE 6
# -----------------------------
elif page == "6 Latest Week Deep Dive":
    st.title("Latest Week Deep Dive (Latest Year + Latest Week)")

    latest_year = int(df["Year"].dropna().max())
    latest_week = int(df[df["Year"] == latest_year]["Week_Num"].dropna().max())
    st.info(f"Latest Year: {latest_year} | Latest Week: {latest_week}")

    base = df[(df["Year"] == latest_year) & (df["Week_Num"] == latest_week)].copy()

    if f_factory:
        base = base[base["Factory"].isin(f_factory)]
    if f_customer:
        base = base[base["Customer"].isin(f_customer)]
    if f_product:
        base = base[base["Product"].isin(f_product)]

    if base.empty:
        st.warning("No data available for latest week after applying filters.")
    else:
        st.subheader("Factory wise summary (latest week)")
        fac = base.groupby("Factory", as_index=False).agg(
            OrderQty=("OrderQty", "sum"),
            CutQty=("CutQty", "sum"),
            ShipQty=("ShipQty", "sum"),
            CutShipDiff=("CutShipDiff", "sum") if "CutShipDiff" in base.columns else ("ShipQty", "size")
        )
        fac["Cut/Ship"] = fac["ShipQty"] / fac["CutQty"]
        fac["Order/Ship"] = fac["ShipQty"] / fac["OrderQty"]
        fac["Diff/ShipQty"] = fac["CutShipDiff"] / fac["ShipQty"]
        fac = fac.replace([np.inf, -np.inf], np.nan)

        st.dataframe(fac.sort_values("Cut/Ship"))

        st.divider()
        st.subheader("Select a Factory to diagnose")

        sel_factory = st.selectbox("Factory", sorted(base["Factory"].dropna().unique().tolist()))
        b1 = base[base["Factory"] == sel_factory].copy()

        cust = b1.groupby("Customer", as_index=False).agg(
            OrderQty=("OrderQty", "sum"),
            CutQty=("CutQty", "sum"),
            ShipQty=("ShipQty", "sum"),
            CutShipDiff=("CutShipDiff", "sum") if "CutShipDiff" in b1.columns else ("ShipQty", "size")
        )
        cust["Cut/Ship"] = cust["ShipQty"] / cust["CutQty"]
        cust["Order/Ship"] = cust["ShipQty"] / cust["OrderQty"]
        cust["Diff/ShipQty"] = cust["CutShipDiff"] / cust["ShipQty"]
        cust = cust.replace([np.inf, -np.inf], np.nan)

        prod = b1.groupby("Product", as_index=False).agg(
            OrderQty=("OrderQty", "sum"),
            CutQty=("CutQty", "sum"),
            ShipQty=("ShipQty", "sum"),
            CutShipDiff=("CutShipDiff", "sum") if "CutShipDiff" in b1.columns else ("ShipQty", "size")
        )
        prod["Cut/Ship"] = prod["ShipQty"] / prod["CutQty"]
        prod["Order/Ship"] = prod["ShipQty"] / prod["OrderQty"]
        prod["Diff/ShipQty"] = prod["CutShipDiff"] / prod["ShipQty"]
        prod = prod.replace([np.inf, -np.inf], np.nan)

        t1, t2 = st.tabs(["Customer drivers", "Product drivers"])
        with t1:
            st.plotly_chart(px.bar(cust.sort_values("Cut/Ship").head(25), x="Cut/Ship", y="Customer", orientation="h"), use_container_width=True)
            st.plotly_chart(px.bar(cust.sort_values("Order/Ship").head(25), x="Order/Ship", y="Customer", orientation="h"), use_container_width=True)
            st.plotly_chart(px.bar(cust.sort_values("Diff/ShipQty", ascending=False).head(25), x="Diff/ShipQty", y="Customer", orientation="h"), use_container_width=True)
            st.dataframe(cust)

        with t2:
            st.plotly_chart(px.bar(prod.sort_values("Cut/Ship").head(25), x="Cut/Ship", y="Product", orientation="h"), use_container_width=True)
            st.plotly_chart(px.bar(prod.sort_values("Order/Ship").head(25), x="Order/Ship", y="Product", orientation="h"), use_container_width=True)
            st.plotly_chart(px.bar(prod.sort_values("Diff/ShipQty", ascending=False).head(25), x="Diff/ShipQty", y="Product", orientation="h"), use_container_width=True)
            st.dataframe(prod)

        st.divider()
        st.subheader("Reason breakdown for selected Factory (latest week)")

        bcols = diff_breakdown_cols(b1)
        if bcols:
            tmp = b1[bcols].copy()
            for c in bcols:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            bsum = tmp.sum(numeric_only=True).sort_values(ascending=False).reset_index()
            bsum.columns = ["Reason", "Qty"]
            st.plotly_chart(px.bar(bsum, x="Qty", y="Reason", orientation="h"), use_container_width=True)
            st.dataframe(bsum)
        else:
            st.caption("No reason breakdown columns found in this dataset view.")
"""

# Write file in current working directory (no new folders)
with open(APP_FILE, "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(app_code).lstrip())

print(f"Wrote Streamlit app to: {os.path.abspath(APP_FILE)}")
print(f"Starting Streamlit on port {PORT} ...")
print(f"Open this in Chrome: http://localhost:{PORT}")

# Run Streamlit (this will keep running until you stop the cell/kernel)
cmd = [sys.executable, "-m", "streamlit", "run", APP_FILE, "--server.port", str(PORT)]
subprocess.run(cmd)
