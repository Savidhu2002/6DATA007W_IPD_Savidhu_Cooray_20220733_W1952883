# ------------------------------------------------------------
# Cut to Ship Report Automation
# Robust to column name variations
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import re

# -----------------------------
# Helpers
# -----------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: remove weird whitespace, trim, keep as strings."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)  # collapse whitespace/newlines/tabs
        .str.strip()
    )
    return df

def find_col(df: pd.DataFrame, text: str, prefer_exact: bool = True) -> str:
    """
    Find a column that matches text.
    - If prefer_exact, first try exact match (case-insensitive).
    - Otherwise find first column containing the text (case-insensitive).
    """
    cols = list(df.columns)

    if prefer_exact:
        for c in cols:
            if c.strip().lower() == text.strip().lower():
                return c

    matches = [c for c in cols if text.lower() in c.lower()]
    if not matches:
        raise KeyError(
            f"Could not find a column containing '{text}'. Available columns:\n{cols}"
        )
    return matches[0]

def ensure_datetime(s: pd.Series) -> pd.Series:
    """Convert series to datetime safely."""
    return pd.to_datetime(s, errors="coerce")

def safe_str(s: pd.Series) -> pd.Series:
    """Convert series to string, preserving NaN as empty string for concatenations."""
    return s.fillna("").astype(str)

# -----------------------------
# Defining the paths for the datasets
# -----------------------------
D1_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Style Closure -Week 43.xlsx"
D2_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Garment Sales Order-Week 43.xlsx"
D3_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Order Book -Week 43.xlsx"
D4_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Transaction Summary -Week 43.xlsx"
D5_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Export Summary -Week 43.xlsx"
D6_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Last Shipment- Week 43.xlsx"
D7_path = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report Automation\Master sheet - customer name.xlsx"

# -----------------------------
# Loading the datasets
# -----------------------------
D1 = pd.read_excel(D1_path, sheet_name="Sheet1")
D2 = pd.read_excel(D2_path, sheet_name="Sheet1")
D3 = pd.read_excel(D3_path, sheet_name="Sheet1")
D4 = pd.read_excel(D4_path, sheet_name="Sheet1")
D5 = pd.read_excel(D5_path, sheet_name="Sheet1")
D6 = pd.read_excel(D6_path, sheet_name="Sheet1")
D7 = pd.read_excel(D7_path, sheet_name="Sheet1")

# Clean column names
D1 = clean_cols(D1)
D2 = clean_cols(D2)
D3 = clean_cols(D3)
D4 = clean_cols(D4)
D5 = clean_cols(D5)
D6 = clean_cols(D6)
D7 = clean_cols(D7)

# -----------------------------
# Resolving column names
# -----------------------------
# D2 filter columns
D2_gen = find_col(D2, "General sales order")
D2_sample = find_col(D2, "Sample sales orders")
D2_salesman = find_col(D2, "Salesman order")
D2_sales_order = find_col(D2, "Sales order")  # could be Sales order, Sales order.1, etc.

# D1 columns
D1_sales_order = find_col(D1, "Sales order")
D1_customer = find_col(D1, "Customer account")
D1_style_closed = find_col(D1, "Style closed date")

# D7 columns
D7_customer = find_col(D7, "Customer")
D7_calling = find_col(D7, "Calling Name")

# D3 columns
D3_sales_order = find_col(D3, "Sales order")
D3_division = find_col(D3, "Division")
D3_season = find_col(D3, "Season")
D3_style_no = find_col(D3, "Style number")
D3_item_type = find_col(D3, "Garment item type")
D3_site = find_col(D3, "Site")
D3_set_garment = find_col(D3, "Set garment")
D3_qty = find_col(D3, "Quantity")

# D4 columns
D4_sales_order = find_col(D4, "Sales order")
D4_unit = find_col(D4, "Unit")
D4_qty = find_col(D4, "Qty")

# D5 columns
D5_sales_order = find_col(D5, "Sales order")
D5_site = find_col(D5, "Site")
D5_invoice = find_col(D5, "Customer invoice")
D5_date = find_col(D5, "Date")
D5_invoice_qty = find_col(D5, "Invoice qty")
D5_fob = find_col(D5, "FOB")

# D6 columns
D6_sales_order = find_col(D6, "Sales order")
D6_approved_date = find_col(D6, "Approved date")

# Ensure datetime fields
D1[D1_style_closed] = ensure_datetime(D1[D1_style_closed])
D6[D6_approved_date] = ensure_datetime(D6[D6_approved_date])
D5[D5_date] = ensure_datetime(D5[D5_date])

# -----------------------------
# Filtering D2 for Sales orders where all three columns are "No"
# -----------------------------
filtered_D2 = D2[
    (D2[D2_gen] == "No") &
    (D2[D2_sample] == "No") &
    (D2[D2_salesman] == "No")
][[D2_sales_order]].copy()

# -----------------------------
# Filtering D1 to keep only Sales orders matching filtered D2 (semi_join)
# -----------------------------
filtered_D1 = D1[D1[D1_sales_order].isin(filtered_D2[D2_sales_order])].copy()

# -----------------------------
# Initialize result with filtered Sales orders
# -----------------------------
result = filtered_D1[[D1_sales_order]].copy()
result = result.rename(columns={D1_sales_order: "Sales_order"})

# -----------------------------
# Add Customer from D1
# -----------------------------
result = result.merge(
    D1[[D1_sales_order, D1_customer]].rename(columns={D1_sales_order: "Sales_order", D1_customer: "Customer"}),
    on="Sales_order",
    how="left"
)

# -----------------------------
# Add Calling Name from D7 based on Customer
# -----------------------------
result = result.merge(
    D7[[D7_customer, D7_calling]].rename(columns={D7_customer: "Customer", D7_calling: "Calling_Name"}),
    on="Customer",
    how="left"
)

# -----------------------------
# Add other columns from D3 by matching Sales order
# -----------------------------
result = result.merge(
    D3[[D3_sales_order, D3_division, D3_season, D3_style_no, D3_item_type, D3_site, D3_set_garment]].rename(
        columns={
            D3_sales_order: "Sales_order",
            D3_division: "Division",
            D3_season: "Season",
            D3_style_no: "Style_number",
            D3_item_type: "Garment_item_type",
            D3_site: "Unit",
            D3_set_garment: "Set_garment",
        }
    ),
    on="Sales_order",
    how="left"
)

# -----------------------------
# Add "Last Shipped" from D6 (max Approved date per Sales order)
# -----------------------------
last_shipped = (
    D6.groupby(D6_sales_order, as_index=False)[D6_approved_date]
    .max()
    .rename(columns={D6_sales_order: "Sales_order", D6_approved_date: "Last_Shipped"})
)

result = result.merge(last_shipped, on="Sales_order", how="left")

# -----------------------------
# Add "Style closed date" from D1
# -----------------------------
result = result.merge(
    D1[[D1_sales_order, D1_style_closed]].rename(columns={D1_sales_order: "Sales_order", D1_style_closed: "Style_closed_date"}),
    on="Sales_order",
    how="left"
)

# -----------------------------
# Add Month and Week
# -----------------------------
# Month abbreviations like Jan, Feb, ...
result["Month"] = result["Style_closed_date"].dt.month_name().str[:3]

# ISO week number, formatted Week 02, Week 43, ...
iso_week = result["Style_closed_date"].dt.isocalendar().week.astype("Int64")
result["Week"] = iso_week.apply(lambda x: f"Week {int(x):02d}" if pd.notna(x) else np.nan)

# -----------------------------
# Add Operation based on Sales order prefix
# -----------------------------
so_str = safe_str(result["Sales_order"])
result["Operation"] = np.where(
    so_str.str.startswith("N"), "Knit Operation",
    np.where(so_str.str.startswith("W"), "Woven Operation", "Other Operation")
)

# -----------------------------
# Remove Sales orders with "Oritapparels" and "Southasiatextiles" in Customer column (case-insensitive)
# -----------------------------
result = result[~result["Customer"].astype(str).str.lower().str.contains("oritapparels|southasiatextiles", regex=True, na=False)].copy()

# -----------------------------
# Code column combining Sales order and Unit
# -----------------------------
result["Code"] = safe_str(result["Sales_order"]) + safe_str(result["Unit"])

# Create Code columns in D3, D4, D5
D3["Code"] = safe_str(D3[D3_sales_order]) + safe_str(D3[D3_site])
D4["Code"] = (safe_str(D4[D4_sales_order]) + safe_str(D4[D4_unit])).str.upper()
D5["Code"] = safe_str(D5[D5_sales_order]) + safe_str(D5[D5_site])

# -----------------------------
# Summarize quantities by Code
# -----------------------------
# Order Qty: exclude Site containing HIKH-SAMP|HKSAM
order_qty = (
    D3[~D3[D3_site].astype(str).str.contains("HIKH-SAMP|HKSAM", regex=True, na=False)]
    .groupby("Code", as_index=False)[D3_qty]
    .sum()
    .rename(columns={D3_qty: "Order_Qty"})
)

# Cut Qty: sum Qty by Code (Code already upper in D4)
cut_qty = (
    D4.groupby("Code", as_index=False)[D4_qty]
    .sum()
    .rename(columns={D4_qty: "Cut_Qty"})
)

# Ship Qty: exclude invoices starting with scl|rtn|dummy|sms|ss (case-insensitive),
# then distinct by (Customer invoice, Date, Invoice qty) and sum Invoice qty by Code
ship_qty = (
    D5[
        ~D5[D5_invoice].astype(str).str.lower().str.match(r"^(scl|rtn|dummy|sms|ss)", na=False)
    ]
    .drop_duplicates(subset=[D5_invoice, D5_date, D5_invoice_qty])
    .groupby("Code", as_index=False)[D5_invoice_qty]
    .sum()
    .rename(columns={D5_invoice_qty: "Ship_Qty"})
)

# Merge quantities into result
result = result.merge(order_qty, on="Code", how="left")
result = result.merge(cut_qty, on="Code", how="left")
result = result.merge(ship_qty, on="Code", how="left")

# -----------------------------
# SO Type
# -----------------------------
result["SO_Type"] = "Bulk"

# -----------------------------
# Pcs based on Set_garment and Style_number
# R logic:
# if Single -> 1
# else if contains Pack -> extract digits from Style_number before PK or P
# limit Pcs > 10 to NA
# adjust Cut_Qty = Cut_Qty / Pcs when Pcs not NA
# -----------------------------
set_g = result["Set_garment"].astype(str)
style_num = result["Style_number"].astype(str)

pcs = np.where(
    set_g.eq("Single"), 1,
    np.where(
        set_g.str.contains("Pack", na=False),
        pd.to_numeric(style_num.str.extract(r"(\d+)(?=PK|P)")[0], errors="coerce"),
        np.nan
    )
)

result["Pcs"] = pcs.astype(float)
result.loc[result["Pcs"] > 10, "Pcs"] = np.nan

# Adjust Cut Qty based on Pcs
result["Cut_Qty"] = pd.to_numeric(result["Cut_Qty"], errors="coerce")
result.loc[result["Pcs"].notna(), "Cut_Qty"] = result.loc[result["Pcs"].notna(), "Cut_Qty"] / result.loc[result["Pcs"].notna(), "Pcs"]

# -----------------------------
# Remove duplicates in Code (keep first)
# -----------------------------
result = result.drop_duplicates(subset=["Code"], keep="first").copy()

# -----------------------------
# Ratios (no rounding)
# -----------------------------
result["Cut/Ship"] = result["Ship_Qty"] / result["Cut_Qty"]
result["Order/Ship"] = result["Ship_Qty"] / result["Order_Qty"]
result["Order/Cut"] = result["Cut_Qty"] / result["Order_Qty"]

# -----------------------------
# FOB calculation from D5: sum(FOB * Invoice qty) by Code
# -----------------------------
D5_fob_total = pd.to_numeric(D5[D5_fob], errors="coerce") * pd.to_numeric(D5[D5_invoice_qty], errors="coerce")
fob_data = (
    D5.assign(FOB_Total=D5_fob_total)
    .groupby("Code", as_index=False)["FOB_Total"]
    .sum()
    .rename(columns={"FOB_Total": "FOB"})
)

result = result.merge(fob_data, on="Code", how="left")

# -----------------------------
# Final column selection and ordering (matching your R output)
# -----------------------------
result_out = result.rename(columns={
    "Sales_order": "Sales order",
    "Calling_Name": "Calling Name",
    "Division": "Div",
    "Style_number": "Style number",
    "Garment_item_type": "Garment item type",
    "Last_Shipped": "Last Shipped",
    "Style_closed_date": "Style closed date",
    "Order_Qty": "Order Qty",
    "Cut_Qty": "Cut Qty",
    "Ship_Qty": "Ship Qty",
    "SO_Type": "SO Type",
    "Set_garment": "Set garment",
})

# Filter for Style closed date in 2024
result_out["Style closed date"] = ensure_datetime(result_out["Style closed date"])
result_out = result_out[result_out["Style closed date"].dt.year == 2024].copy()

# Replace missing Cut Qty with Ship Qty if available
result_out["Cut Qty"] = pd.to_numeric(result_out["Cut Qty"], errors="coerce")
result_out["Ship Qty"] = pd.to_numeric(result_out["Ship Qty"], errors="coerce")
result_out.loc[result_out["Cut Qty"].isna() & result_out["Ship Qty"].notna(), "Cut Qty"] = result_out.loc[
    result_out["Cut Qty"].isna() & result_out["Ship Qty"].notna(), "Ship Qty"
]

# Filter out Unit contains Corporate or Sample
result_out["Unit"] = result_out["Unit"].astype(str)
result_out = result_out[~result_out["Unit"].str.contains("Corporate|Sample", regex=True, na=False)].copy()

# Keep your exact output column order (including FOB after Code in your R, but you selected Code then FOB at end)
# Your R final select order ends with: Code, FOB
final_cols = [
    "Sales order",
    "Customer",
    "Calling Name",
    "Div",
    "Season",
    "Style number",
    "Garment item type",
    "Unit",
    "Last Shipped",
    "Style closed date",
    "Month",
    "Order Qty",
    "Cut Qty",
    "Ship Qty",
    "Cut/Ship",
    "Order/Ship",
    "Order/Cut",
    "SO Type",
    "Week",
    "Operation",
    "Set garment",
    "Pcs",
    "Code",
    "FOB",
]

# Some columns might not exist if source files miss them, so select safely
final_cols_existing = [c for c in final_cols if c in result_out.columns]
result_out = result_out[final_cols_existing].copy()

# -----------------------------
# Write output
# -----------------------------
output_path = (
    r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects"
    r"\Cut to Ship Prediction Model\Cut to Ship Report Automation"
    r"\Cut_to_Ship_Week_43_R2.xlsx"
)

result_out.to_excel(output_path, index=False)
print("Saved:", output_path)
