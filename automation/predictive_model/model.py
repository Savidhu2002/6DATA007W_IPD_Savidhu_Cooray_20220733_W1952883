import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
EXCEL_PATH = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report.xlsx"
SHEET_NAME = "Final"

# Final UI inputs
INPUT_COLS = [
    "Year",
    "Calling Name",
    "Div",
    "Season",
    "Garment item type",
    "Unit",
    "Operation",
    "Month",
    "Type",
    "Operation 2",
    "Pcs",
    "Order Qty",
]

TARGET_CUT = "Cut Qty"
TARGET_SHIP = "Ship Qty"

# Rare category handling
RARE_COLS = ["Div", "Season", "Calling Name", "Garment item type"]
MIN_COUNT = 10
OTHER_LABEL = "OTHER"

# Reason columns (damage + adjustments)
# This is based on your file structure description. If any are missing, code will ignore them safely.
REASON_COLS = [
    "Fabric Damage",
    "Colour Shading",
    "Finishing Damage",
    "Shade Band",
    "Pilot",
    "Wash Reference Sample",
    "Cut panel rejection qty",
    "Sewing Reject Qty",
    "EMB / Printing  Damages",
    "Washing Damages",
    "Sample qty",
    "Shortage qty",
    "Unreconciled qty -panel form",
    "Unreconciled qty -GMT form",
    "Second Quality",
    "Good garments",
    "PO Mix",
    "Transfer to other SOD",
    "Transfer from other SOD",
]

OUTPUT_DIR = os.path.dirname(EXCEL_PATH)
MODEL_CUT_PATH = os.path.join(OUTPUT_DIR, "model_cut_ratio.pkl")
MODEL_LOSS_PATH = os.path.join(OUTPUT_DIR, "model_loss_ratio.pkl")
META_PATH = os.path.join(OUTPUT_DIR, "model_meta.pkl")
APP_PATH = os.path.join(OUTPUT_DIR, "streamlit_app.py")

# =========================
# HELPERS
# =========================
def to_num(series):
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")

def group_rare(df, col, min_count=10, other_label="OTHER"):
    vc = df[col].value_counts(dropna=False)
    rare_vals = vc[vc < min_count].index
    df[col] = df[col].replace(rare_vals, other_label)
    return df

def safe_sum(df, cols):
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index)
    return df[present].sum(axis=1)

# Build lookup tables for historical behavior features with fallback levels
def build_lookup_tables(df, key_cols, feat_cols):
    # Full key
    full = df.groupby(key_cols, dropna=False)[feat_cols].mean().reset_index()

    # Backoff 1: remove Calling Name (more general)
    backoff1_keys = [c for c in key_cols if c != "Calling Name"]
    backoff1 = df.groupby(backoff1_keys, dropna=False)[feat_cols].mean().reset_index()

    # Backoff 2: remove Calling Name + Season (even more general)
    backoff2_keys = [c for c in backoff1_keys if c != "Season"]
    backoff2 = df.groupby(backoff2_keys, dropna=False)[feat_cols].mean().reset_index()

    # Global mean
    global_mean = df[feat_cols].mean().to_dict()

    return {
        "key_cols": key_cols,
        "backoff1_keys": backoff1_keys,
        "backoff2_keys": backoff2_keys,
        "full": full,
        "backoff1": backoff1,
        "backoff2": backoff2,
        "global_mean": global_mean,
        "feat_cols": feat_cols
    }

def lookup_behavior(row_dict, lookups):
    feat_cols = lookups["feat_cols"]
    global_mean = lookups["global_mean"]

    # Try full
    full_keys = lookups["key_cols"]
    full_df = lookups["full"]
    mask = np.ones(len(full_df), dtype=bool)
    for k in full_keys:
        mask &= (full_df[k].astype(str).values == str(row_dict[k]))
    match = full_df.loc[mask]
    if len(match) > 0:
        return {f: float(match.iloc[0][f]) for f in feat_cols}

    # Try backoff1
    b1_keys = lookups["backoff1_keys"]
    b1_df = lookups["backoff1"]
    mask = np.ones(len(b1_df), dtype=bool)
    for k in b1_keys:
        mask &= (b1_df[k].astype(str).values == str(row_dict[k]))
    match = b1_df.loc[mask]
    if len(match) > 0:
        return {f: float(match.iloc[0][f]) for f in feat_cols}

    # Try backoff2
    b2_keys = lookups["backoff2_keys"]
    b2_df = lookups["backoff2"]
    mask = np.ones(len(b2_df), dtype=bool)
    for k in b2_keys:
        mask &= (b2_df[k].astype(str).values == str(row_dict[k]))
    match = b2_df.loc[mask]
    if len(match) > 0:
        return {f: float(match.iloc[0][f]) for f in feat_cols}

    # Fall back to global
    return {f: float(global_mean.get(f, 0.0)) for f in feat_cols}

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]

required = INPUT_COLS + [TARGET_CUT, TARGET_SHIP]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in sheet '{SHEET_NAME}': {missing}")

# Keep relevant columns plus any reason cols that exist
present_reason_cols = [c for c in REASON_COLS if c in df.columns]
df = df[INPUT_COLS + [TARGET_CUT, TARGET_SHIP] + present_reason_cols].copy()

# Numeric conversions
df["Order Qty"] = to_num(df["Order Qty"])
df["Pcs"] = to_num(df["Pcs"])
df[TARGET_CUT] = to_num(df[TARGET_CUT])
df[TARGET_SHIP] = to_num(df[TARGET_SHIP])

for c in present_reason_cols:
    df[c] = to_num(df[c]).fillna(0)

# Basic cleaning
df = df.dropna(subset=["Order Qty", "Pcs", TARGET_CUT, TARGET_SHIP])
df = df[(df["Order Qty"] > 0) & (df["Order Qty"] < 1_000_000)]
df = df[(df["Pcs"] > 0) & (df["Pcs"] < 1_000_000)]
df = df[(df[TARGET_CUT] >= 0) & (df[TARGET_CUT] < 1_000_000)]
df = df[(df[TARGET_SHIP] >= 0) & (df[TARGET_SHIP] < 1_000_000)]

# Force categorical strings
for c in INPUT_COLS:
    if c not in ["Order Qty", "Pcs"]:
        df[c] = df[c].astype(str).fillna("Unknown").str.strip()

# Rare bucketing for key categorical columns
for c in RARE_COLS:
    df = group_rare(df, c, min_count=MIN_COUNT, other_label=OTHER_LABEL)

# =========================
# ENGINEER TARGETS (2-stage)
# =========================
eps = 1e-9
df["Cut_Ratio"] = df[TARGET_CUT] / (df["Order Qty"] + eps)
df["Loss_Ratio"] = (df[TARGET_CUT] - df[TARGET_SHIP]) / (df[TARGET_CUT] + eps)

# Clip to sane ranges (helps stability)
df = df[(df["Cut_Ratio"] > 0) & (df["Cut_Ratio"] < 2)]
df = df[(df["Loss_Ratio"] >= 0) & (df["Loss_Ratio"] < 1)]

# =========================
# ENGINEER HISTORICAL BEHAVIOR FEATURES FROM REASONS
# =========================
# Build aggregated damage features as ratios to cut
transfer_to = df["Transfer to other SOD"] if "Transfer to other SOD" in df.columns else 0.0
transfer_from = df["Transfer from other SOD"] if "Transfer from other SOD" in df.columns else 0.0

# "Quality loss" reasons (excluding transfers and "Good garments" etc. because those can be confusing)
quality_reason_candidates = [
    "Fabric Damage",
    "Colour Shading",
    "Finishing Damage",
    "Shade Band",
    "Pilot",
    "Wash Reference Sample",
    "Cut panel rejection qty",
    "Sewing Reject Qty",
    "EMB / Printing  Damages",
    "Washing Damages",
    "Sample qty",
    "Shortage qty",
    "Unreconciled qty -panel form",
    "Unreconciled qty -GMT form",
    "Second Quality",
    "PO Mix",
]
quality_cols = [c for c in quality_reason_candidates if c in df.columns]

df["Quality_Issue_Qty"] = safe_sum(df, quality_cols)
df["Net_Transfer_Qty"] = (transfer_from - transfer_to) if isinstance(transfer_from, pd.Series) else 0.0

df["Quality_Issue_Ratio"] = df["Quality_Issue_Qty"] / (df[TARGET_CUT] + eps)
df["Net_Transfer_Ratio"] = df["Net_Transfer_Qty"] / (df[TARGET_CUT] + eps)

# Clip ratios
df["Quality_Issue_Ratio"] = df["Quality_Issue_Ratio"].clip(0, 1)
df["Net_Transfer_Ratio"] = df["Net_Transfer_Ratio"].clip(-1, 1)

# =========================
# BUILD LOOKUPS (historical behavior features, no user inputs)
# =========================
KEY_COLS = [
    "Year",
    "Calling Name",
    "Div",
    "Season",
    "Garment item type",
    "Unit",
    "Operation",
    "Month",
    "Type",
    "Operation 2",
]

BEHAVIOR_FEATURES = ["Quality_Issue_Ratio", "Net_Transfer_Ratio"]

lookups = build_lookup_tables(df, KEY_COLS, BEHAVIOR_FEATURES)

# Apply lookups to each row to create "expected behavior" features used by the model
# (This makes the model learn loss tendencies for similar orders)
def add_behavior_features(df, lookups):
    out_quality = []
    out_transfer = []
    for _, r in df.iterrows():
        row_dict = {k: r[k] for k in lookups["key_cols"]}
        feats = lookup_behavior(row_dict, lookups)
        out_quality.append(feats["Quality_Issue_Ratio"])
        out_transfer.append(feats["Net_Transfer_Ratio"])
    df["Hist_Quality_Issue_Ratio"] = out_quality
    df["Hist_Net_Transfer_Ratio"] = out_transfer
    return df

df = add_behavior_features(df, lookups)

print("Rows ready for training:", len(df))

# =========================
# STAGE A MODEL: predict Cut_Ratio
# =========================
X_base = df[INPUT_COLS].copy()
X_base["Hist_Quality_Issue_Ratio"] = df["Hist_Quality_Issue_Ratio"].astype(float)
X_base["Hist_Net_Transfer_Ratio"] = df["Hist_Net_Transfer_Ratio"].astype(float)

y_cut = df["Cut_Ratio"].astype(float)

cat_cols = [c for c in INPUT_COLS if c not in ["Order Qty", "Pcs"]]
num_cols = ["Order Qty", "Pcs", "Hist_Quality_Issue_Ratio", "Hist_Net_Transfer_Ratio"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

cut_model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=600,
        max_depth=22,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X_base, y_cut, test_size=0.2, random_state=42)
cut_model.fit(X_train, y_train)

pred_cut = cut_model.predict(X_test)
print("\nSTAGE A (Cut_Ratio) evaluation")
print("MAE:", mean_absolute_error(y_test, pred_cut))
print("R2 :", r2_score(y_test, pred_cut))

# =========================
# STAGE B MODEL: predict Loss_Ratio
# Include actual Cut_Ratio as an input feature (represents planned cutting strategy)
# At runtime we will use predicted Cut_Ratio.
# =========================
X_loss = X_base.copy()
X_loss["Cut_Ratio"] = df["Cut_Ratio"].astype(float)
y_loss = df["Loss_Ratio"].astype(float)

num_cols_loss = num_cols + ["Cut_Ratio"]

preprocess_loss = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols_loss),
    ]
)

loss_model = Pipeline(steps=[
    ("prep", preprocess_loss),
    ("model", RandomForestRegressor(
        n_estimators=600,
        max_depth=22,
        random_state=42,
        n_jobs=-1
    ))
])

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_loss, y_loss, test_size=0.2, random_state=42)
loss_model.fit(X_train2, y_train2)

pred_loss = loss_model.predict(X_test2)
print("\nSTAGE B (Loss_Ratio) evaluation")
print("MAE:", mean_absolute_error(y_test2, pred_loss))
print("R2 :", r2_score(y_test2, pred_loss))

# =========================
# SAVE MODELS + META (including lookups + allowed values)
# =========================
joblib.dump(cut_model, MODEL_CUT_PATH)
joblib.dump(loss_model, MODEL_LOSS_PATH)

allowed_values = {}
for c in INPUT_COLS:
    if c not in ["Order Qty", "Pcs"]:
        allowed_values[c] = sorted(df[c].astype(str).unique().tolist())

meta = {
    "excel_path": EXCEL_PATH,
    "sheet_name": SHEET_NAME,
    "input_cols": INPUT_COLS,
    "key_cols": KEY_COLS,
    "rare_cols": RARE_COLS,
    "min_count": MIN_COUNT,
    "other_label": OTHER_LABEL,
    "allowed_values": allowed_values,
    "behavior_features": BEHAVIOR_FEATURES,
    "lookups": {
        "key_cols": lookups["key_cols"],
        "backoff1_keys": lookups["backoff1_keys"],
        "backoff2_keys": lookups["backoff2_keys"],
        "full": lookups["full"],
        "backoff1": lookups["backoff1"],
        "backoff2": lookups["backoff2"],
        "global_mean": lookups["global_mean"],
        "feat_cols": lookups["feat_cols"]
    }
}
joblib.dump(meta, META_PATH)

print("\nSaved Stage A model:", MODEL_CUT_PATH)
print("Saved Stage B model:", MODEL_LOSS_PATH)
print("Saved meta:", META_PATH)

# =========================
# WRITE STREAMLIT APP (2-stage, no reason inputs)
# =========================
app_code = f"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_CUT_PATH = r"{MODEL_CUT_PATH}"
MODEL_LOSS_PATH = r"{MODEL_LOSS_PATH}"
META_PATH = r"{META_PATH}"

cut_model = joblib.load(MODEL_CUT_PATH)
loss_model = joblib.load(MODEL_LOSS_PATH)
meta = joblib.load(META_PATH)

INPUT_COLS = meta["input_cols"]
ALLOWED = meta["allowed_values"]
RARE_COLS = set(meta["rare_cols"])
OTHER_LABEL = meta["other_label"]

lookups = meta["lookups"]
feat_cols = lookups["feat_cols"]

full_df = pd.DataFrame(lookups["full"])
b1_df = pd.DataFrame(lookups["backoff1"])
b2_df = pd.DataFrame(lookups["backoff2"])
global_mean = lookups["global_mean"]

def lookup_behavior(row_dict):
    # Try full
    mask = np.ones(len(full_df), dtype=bool)
    for k in lookups["key_cols"]:
        mask &= (full_df[k].astype(str).values == str(row_dict[k]))
    m = full_df.loc[mask]
    if len(m) > 0:
        return {{f: float(m.iloc[0][f]) for f in feat_cols}}

    # Backoff1
    mask = np.ones(len(b1_df), dtype=bool)
    for k in lookups["backoff1_keys"]:
        mask &= (b1_df[k].astype(str).values == str(row_dict[k]))
    m = b1_df.loc[mask]
    if len(m) > 0:
        return {{f: float(m.iloc[0][f]) for f in feat_cols}}

    # Backoff2
    mask = np.ones(len(b2_df), dtype=bool)
    for k in lookups["backoff2_keys"]:
        mask &= (b2_df[k].astype(str).values == str(row_dict[k]))
    m = b2_df.loc[mask]
    if len(m) > 0:
        return {{f: float(m.iloc[0][f]) for f in feat_cols}}

    # Global
    return {{f: float(global_mean.get(f, 0.0)) for f in feat_cols}}

st.set_page_config(page_title="Cut to Ship Prediction", layout="wide")
st.title("Cut to Ship Prediction (2-Stage Model)")

st.write(
    "Stage A predicts the cutting plan (Cut Qty based on Cut Ratio). "
    "Stage B predicts execution loss (Ship Qty based on Loss Ratio). "
    "The reasons data is used as historical behavior patterns, so users do not need to enter it."
)

st.subheader("Enter Order Details")
cols = st.columns(3)
inputs = {{}}

for i, col in enumerate(INPUT_COLS):
    with cols[i % 3]:
        if col in ["Order Qty", "Pcs"]:
            inputs[col] = st.number_input(col, min_value=1, step=1)
        else:
            options = ALLOWED.get(col, [])
            if options:
                inputs[col] = st.selectbox(col, options)
            else:
                inputs[col] = st.text_input(col, value="Unknown")

# Force strings for categoricals
for c in INPUT_COLS:
    if c not in ["Order Qty", "Pcs"]:
        inputs[c] = str(inputs[c]).strip()

# Rare bucketing at input time
for c in RARE_COLS:
    if inputs.get(c, OTHER_LABEL) not in ALLOWED.get(c, []):
        inputs[c] = OTHER_LABEL

# Lookup historical behavior features
row_dict = {{k: inputs[k] for k in lookups["key_cols"]}}
beh = lookup_behavior(row_dict)

X = pd.DataFrame([[inputs[c] for c in INPUT_COLS]], columns=INPUT_COLS)
X["Hist_Quality_Issue_Ratio"] = beh.get("Quality_Issue_Ratio", 0.0)
X["Hist_Net_Transfer_Ratio"] = beh.get("Net_Transfer_Ratio", 0.0)

if st.button("Predict"):
    order_qty = float(inputs["Order Qty"])
    eps = 1e-9

    # Stage A
    cut_ratio_pred = float(cut_model.predict(X)[0])
    cut_ratio_pred = max(0.0, min(2.0, cut_ratio_pred))
    cut_qty_pred = order_qty * cut_ratio_pred

    # Stage B (add cut ratio as a feature)
    X2 = X.copy()
    X2["Cut_Ratio"] = cut_ratio_pred
    loss_ratio_pred = float(loss_model.predict(X2)[0])
    loss_ratio_pred = max(0.0, min(1.0, loss_ratio_pred))

    ship_qty_pred = cut_qty_pred * (1 - loss_ratio_pred)

    # Ratios
    cut_ship = ship_qty_pred / (cut_qty_pred + eps)
    order_ship = ship_qty_pred / (order_qty + eps)
    order_cut = cut_qty_pred / (order_qty + eps)

    # Simple risk logic
    if cut_ship < 0.95 or order_ship < 0.95:
        risk = "HIGH RISK"
        color = "red"
    elif cut_ship < 0.98 or order_ship < 0.98:
        risk = "MEDIUM RISK"
        color = "orange"
    else:
        risk = "LOW RISK"
        color = "green"

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Cut Qty", f"{{int(round(cut_qty_pred)):,}}")
    c2.metric("Predicted Ship Qty", f"{{int(round(ship_qty_pred)):,}}")
    c3.metric("Order Qty", f"{{int(order_qty):,}}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Cut / Ship", round(float(cut_ship), 3))
    c5.metric("Order / Ship", round(float(order_ship), 3))
    c6.metric("Order / Cut", round(float(order_cut), 3))

    st.markdown(f"<h3 style='color:{{color}}'>Overall Risk: {{risk}}</h3>", unsafe_allow_html=True)

    with st.expander("Model details used (historical behavior features)"):
        st.write({{
            "Hist_Quality_Issue_Ratio": float(X["Hist_Quality_Issue_Ratio"].iloc[0]),
            "Hist_Net_Transfer_Ratio": float(X["Hist_Net_Transfer_Ratio"].iloc[0]),
            "Pred_Cut_Ratio": float(cut_ratio_pred),
            "Pred_Loss_Ratio": float(loss_ratio_pred),
        }})
"""

with open(APP_PATH, "w", encoding="utf-8") as f:
    f.write(app_code)

print("\nStreamlit app written to:", APP_PATH)

# =========================
# LAUNCH STREAMLIT FROM JUPYTER
# =========================
time.sleep(1)
subprocess.Popen([sys.executable, "-m", "streamlit", "run", APP_PATH], cwd=OUTPUT_DIR)
print("If it does not open automatically: http://localhost:8501")



import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
EXCEL_PATH = r"D:\Savidhu_OneDrive\OneDrive - Hirdaramani Group\Projects\Cut to Ship Prediction Model\Cut to Ship Report.xlsx"
SHEET_NAME = "2026"

OUTPUT_DIR = os.path.dirname(EXCEL_PATH)
MODEL_CUT_PATH = os.path.join(OUTPUT_DIR, "model_cut_ratio.pkl")
MODEL_LOSS_PATH = os.path.join(OUTPUT_DIR, "model_loss_ratio.pkl")
META_PATH = os.path.join(OUTPUT_DIR, "model_meta.pkl")

TARGET_CUT = "Cut Qty"
TARGET_SHIP = "Ship Qty"

# =========================
# HELPERS
# =========================
def to_num(series):
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("()", "0")
    )
    return pd.to_numeric(s, errors="coerce")

def within_pct(y_true, y_pred, pct=5):
    eps = 1e-9
    return np.mean(
        np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps) <= pct / 100
    ) * 100

# =========================
# LOAD MODELS + META
# =========================
cut_model = joblib.load(MODEL_CUT_PATH)
loss_model = joblib.load(MODEL_LOSS_PATH)
meta = joblib.load(META_PATH)

INPUT_COLS = meta["input_cols"]
RARE_COLS = meta["rare_cols"]
MIN_COUNT = meta["min_count"]
OTHER_LABEL = meta["other_label"]
lookups = meta["lookups"]

full_df = pd.DataFrame(lookups["full"])
b1_df = pd.DataFrame(lookups["backoff1"])
b2_df = pd.DataFrame(lookups["backoff2"])
global_mean = lookups["global_mean"]
feat_cols = lookups["feat_cols"]

def lookup_behavior(row_dict):
    for df_, keys in [
        (full_df, lookups["key_cols"]),
        (b1_df, lookups["backoff1_keys"]),
        (b2_df, lookups["backoff2_keys"]),
    ]:
        mask = np.ones(len(df_), dtype=bool)
        for k in keys:
            mask &= df_[k].astype(str).values == str(row_dict[k])
        m = df_.loc[mask]
        if len(m) > 0:
            return {f: float(m.iloc[0][f]) for f in feat_cols}
    return {f: float(global_mean.get(f, 0.0)) for f in feat_cols}

# =========================
# LOAD 2026 DATA
# =========================
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]

required = INPUT_COLS + [TARGET_CUT, TARGET_SHIP]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in 2026 sheet: {missing}")

df = df[INPUT_COLS + [TARGET_CUT, TARGET_SHIP]].copy()

# Numeric cleanup
df["Order Qty"] = to_num(df["Order Qty"])
df["Pcs"] = to_num(df["Pcs"])
df[TARGET_CUT] = to_num(df[TARGET_CUT])
df[TARGET_SHIP] = to_num(df[TARGET_SHIP])

df = df.dropna(subset=["Order Qty", "Pcs", TARGET_CUT, TARGET_SHIP])
df = df[(df["Order Qty"] >= 500) & (df[TARGET_CUT] >= 500)]

# Categoricals
for c in INPUT_COLS:
    if c not in ["Order Qty", "Pcs"]:
        df[c] = df[c].astype(str).fillna("Unknown").str.strip()

# Rare bucket handling
for c in RARE_COLS:
    df[c] = df[c].where(df[c].isin(meta["allowed_values"][c]), OTHER_LABEL)

# =========================
# BUILD FEATURES
# =========================
X = df[INPUT_COLS].copy()

hist_q, hist_t = [], []
for _, r in df.iterrows():
    row_dict = {k: r[k] for k in lookups["key_cols"]}
    beh = lookup_behavior(row_dict)
    hist_q.append(beh["Quality_Issue_Ratio"])
    hist_t.append(beh["Net_Transfer_Ratio"])

X["Hist_Quality_Issue_Ratio"] = hist_q
X["Hist_Net_Transfer_Ratio"] = hist_t

# =========================
# PREDICT
# =========================
order_qty = df["Order Qty"].values
cut_ratio_pred = np.clip(cut_model.predict(X), 0, 2)
cut_qty_pred = order_qty * cut_ratio_pred

X2 = X.copy()
X2["Cut_Ratio"] = cut_ratio_pred
loss_ratio_pred = np.clip(loss_model.predict(X2), 0, 1)
ship_qty_pred = cut_qty_pred * (1 - loss_ratio_pred)

# =========================
# EVALUATION
# =========================
y_cut = df[TARGET_CUT].values
y_ship = df[TARGET_SHIP].values

print("UNSEEN DATA EVALUATION (2026)")
print("Rows:", len(df))

print("\nCUT QTY")
print("MAE:", mean_absolute_error(y_cut, cut_qty_pred))
print("Within 5% :", within_pct(y_cut, cut_qty_pred, 5), "%")
print("Within 10%:", within_pct(y_cut, cut_qty_pred, 10), "%")

print("\nSHIP QTY")
print("MAE:", mean_absolute_error(y_ship, ship_qty_pred))
print("Within 5% :", within_pct(y_ship, ship_qty_pred, 5), "%")
print("Within 10%:", within_pct(y_ship, ship_qty_pred, 10), "%")

# Preview
out = df.copy()
out["Pred_CutQty"] = cut_qty_pred.round(0)
out["Pred_ShipQty"] = ship_qty_pred.round(0)
out["Ship_Error_%"] = ((out["Pred_ShipQty"] - out[TARGET_SHIP]).abs() / out[TARGET_SHIP]) * 100

print("\nSample comparison:")
print(out[[
    "Order Qty", TARGET_CUT, TARGET_SHIP,
    "Pred_CutQty", "Pred_ShipQty", "Ship_Error_%"
]].head(10))
