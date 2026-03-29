import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Telangana PDS Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 0.9rem;
    padding-bottom: 1rem;
    max-width: 1600px;
}
.main-box {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.main-title {
    color: white;
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    color: #cbd5e1;
    font-size: 0.98rem;
}
.kpi-card {
    background: linear-gradient(135deg, #e0f2fe, #f8fafc);
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    padding: 12px 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.kpi-title {
    font-size: 0.88rem;
    color: #475569;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0f172a;
}
.section-note {
    font-size: 0.90rem;
    color: #64748b;
    margin-bottom: 8px;
}
.small-muted {
    color: #64748b;
    font-size: 0.84rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
SHOP_FILE = BASE_DIR / "shop_clustered.csv"
ANOMALY_FILE = BASE_DIR / "anomaly_shops.csv"
CLUSTER_FILE = BASE_DIR / "cluster_profile.csv"

# =========================================================
# HELPERS
# =========================================================
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def kpi_card(col, title, value):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def normalize_series(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    if s.nunique() <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_text(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def get_cluster_name(cid, persona=None):
    cluster_map = {
        0: "High Utilization Shops",
        1: "Portability Hub Shops",
        2: "Balanced Commodity Shops",
        3: "Low Volume Shops"
    }
    if pd.notna(cid) and cid in cluster_map:
        return cluster_map[cid]
    if persona is not None and str(persona).strip():
        return str(persona).strip()
    return "Unknown Cluster"

def anomaly_label(row):
    peer = row.get("peer_anomaly_flag", 0)
    dbs = row.get("is_anomaly", 0)
    gap = row.get("utilization_gap", np.nan)
    if pd.notna(peer) and int(peer) == 1:
        return "Anomaly"
    if pd.notna(dbs) and int(dbs) == 1:
        return "Anomaly"
    if pd.notna(gap) and abs(gap) >= 0.40:
        return "Anomaly"
    return "Normal"

def to_display(df):
    rename_map = {
        "shopNo": "Shop No",
        "district_name": "District",
        "office_name": "Office",
        "fpsStatus": "FPS Status",
        "fpsType": "FPS Type",
        "cluster_id": "Cluster ID",
        "cluster_name": "Cluster Name",
        "noOfTrans": "Transactions",
        "totalRcs": "Total Cards",
        "utilization_ratio": "Utilization Ratio",
        "portability_ratio": "Portability Ratio",
        "total_rice": "Rice",
        "wheat": "Wheat",
        "sugar": "Sugar",
        "kerosene": "Kerosene",
        "salt": "Salt",
        "commodity_total": "Commodity Total",
        "otherShopTransCnt": "Other Shop Trans",
        "utilization_gap": "Utilization Gap",
        "anomaly_status": "Anomaly Status",
        "anomaly_score": "Anomaly Score",
        "risk_level": "Risk Level",
        "shop_score": "Shop Score",
        "rank": "Rank",
        "rf_pred_transactions": "RF Predicted Transactions",
        "mlp_pred_transactions": "DL Predicted Transactions"
    }
    return df.rename(columns=rename_map)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    missing = []
    for f in [SHOP_FILE, ANOMALY_FILE, CLUSTER_FILE]:
        if not f.exists():
            missing.append(f.name)
    if missing:
        return None, None, None, missing

    shop = safe_read_csv(SHOP_FILE)
    anomaly = safe_read_csv(ANOMALY_FILE)
    cluster = safe_read_csv(CLUSTER_FILE)

    shop.columns = [str(c).strip() for c in shop.columns]
    anomaly.columns = [str(c).strip() for c in anomaly.columns]
    cluster.columns = [str(c).strip() for c in cluster.columns]

    return shop, anomaly, cluster, []

shop_df, anomaly_df, cluster_df, missing_files = load_data()

if missing_files:
    st.error("Missing files: " + ", ".join(missing_files))
    st.stop()

# =========================================================
# CLEAN DATA
# =========================================================
num_cols = [
    "shopNo", "distCode", "officeCode_txn", "noOfRcs", "noOfTrans", "total_rice",
    "wheat", "sugar", "rgdal", "kerosene", "salt", "totalAmount", "otherShopTransCnt",
    "totalRcs", "totalUnits", "utilization_ratio", "rice_wheat_ratio", "portability_ratio",
    "commodity_total", "longitude", "latitude", "cluster_id", "pca1", "pca2",
    "dbscan_label", "is_anomaly", "cluster_avg_utilization", "utilization_gap",
    "utilization_gap_abs", "peer_anomaly_flag"
]

shop_df = safe_numeric(shop_df, num_cols)
anomaly_df = safe_numeric(anomaly_df, num_cols)
cluster_df = safe_numeric(cluster_df, [c for c in cluster_df.columns if c != "cluster_persona"])

text_cols = ["district_name", "office_name", "fpsStatus", "fpsType", "cluster_persona"]
shop_df = clean_text(shop_df, text_cols)
anomaly_df = clean_text(anomaly_df, text_cols)
if "cluster_persona" in cluster_df.columns:
    cluster_df["cluster_persona"] = cluster_df["cluster_persona"].fillna("").astype(str).str.strip()

shop_df["cluster_name"] = [
    get_cluster_name(cid, persona)
    for cid, persona in zip(shop_df["cluster_id"], shop_df.get("cluster_persona", ""))
]
anomaly_df["cluster_name"] = [
    get_cluster_name(cid, persona)
    for cid, persona in zip(anomaly_df["cluster_id"], anomaly_df.get("cluster_persona", ""))
]
cluster_df["cluster_name"] = [
    get_cluster_name(cid, persona if "cluster_persona" in cluster_df.columns else "")
    for cid, persona in zip(cluster_df["cluster_id"], cluster_df.get("cluster_persona", pd.Series([""] * len(cluster_df))))
]

shop_df["anomaly_status"] = shop_df.apply(anomaly_label, axis=1)
anomaly_df["anomaly_status"] = "Anomaly"

# valid geo only if actual coordinates > 0
if "latitude" in shop_df.columns and "longitude" in shop_df.columns:
    shop_df["has_valid_geo"] = (
        shop_df["latitude"].notna() &
        shop_df["longitude"].notna() &
        (shop_df["latitude"] > 0) &
        (shop_df["longitude"] > 0)
    )
else:
    shop_df["has_valid_geo"] = False

# safe columns for plots
shop_df["plot_size_cards"] = pd.to_numeric(shop_df.get("totalRcs", 0), errors="coerce").fillna(1).clip(lower=1)
shop_df["plot_size_commodity"] = pd.to_numeric(shop_df.get("commodity_total", 0), errors="coerce").fillna(1).clip(lower=1)
shop_df["plot_size_trans"] = pd.to_numeric(shop_df.get("noOfTrans", 0), errors="coerce").fillna(1).clip(lower=1)

# =========================================================
# AI / SCORE / RISK
# =========================================================
feature_cols = [c for c in ["totalRcs", "utilization_ratio", "portability_ratio", "commodity_total", "otherShopTransCnt"] if c in shop_df.columns]
if feature_cols:
    X = shop_df[feature_cols].fillna(0)
    y = shop_df["noOfTrans"].fillna(0)

    rf = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=10)
    rf.fit(X, y)
    shop_df["rf_pred_transactions"] = rf.predict(X)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    mlp.fit(X, y)
    shop_df["mlp_pred_transactions"] = mlp.predict(X)
else:
    shop_df["rf_pred_transactions"] = np.nan
    shop_df["mlp_pred_transactions"] = np.nan

shop_df["anomaly_score"] = (
    normalize_series(shop_df.get("utilization_gap_abs", 0)) * 35 +
    normalize_series(abs(shop_df["noOfTrans"] - shop_df["rf_pred_transactions"])) * 45 +
    normalize_series(shop_df.get("portability_ratio", 0)) * 20
) * 100
shop_df["anomaly_score"] = shop_df["anomaly_score"].round(2)

shop_df["risk_level"] = pd.cut(
    shop_df["anomaly_score"],
    bins=[-0.01, 33.33, 66.66, 100.00],
    labels=["Low", "Medium", "High"]
)

shop_df["shop_score"] = (
    normalize_series(shop_df["noOfTrans"]) * 40 +
    normalize_series(shop_df.get("utilization_ratio", 0)) * 30 +
    normalize_series(shop_df.get("commodity_total", 0)) * 20 +
    normalize_series(shop_df.get("otherShopTransCnt", 0)) * 10
) * 100
shop_df["shop_score"] = shop_df["shop_score"].round(2)
shop_df["rank"] = shop_df["shop_score"].rank(ascending=False, method="dense").astype(int)

# =========================================================
# CLUSTER SUMMARY FROM SHOP DATA
# =========================================================
cluster_summary = (
    shop_df.groupby(["cluster_id", "cluster_name"], as_index=False)[
        [c for c in ["noOfTrans", "totalRcs", "utilization_ratio", "portability_ratio",
                     "total_rice", "wheat", "sugar", "kerosene", "salt",
                     "commodity_total", "otherShopTransCnt"] if c in shop_df.columns]
    ]
    .mean()
    .sort_values("cluster_id")
    .reset_index(drop=True)
)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.title("📊 Dashboard Filters")

auto_refresh = st.sidebar.checkbox("Auto Refresh (Live Monitor Style)", value=False)
if auto_refresh:
    wait_s = st.sidebar.slider("Refresh every seconds", 5, 60, 15)
    time.sleep(wait_s)
    st.rerun()

district_options = ["All"] + sorted([x for x in shop_df["district_name"].dropna().unique().tolist() if x != ""])
cluster_options = ["All"] + sorted([x for x in shop_df["cluster_name"].dropna().unique().tolist() if x != ""])
fps_status_options = ["All"] + sorted([x for x in shop_df["fpsStatus"].dropna().unique().tolist() if x != ""])
fps_type_options = ["All"] + sorted([x for x in shop_df["fpsType"].dropna().unique().tolist() if x != ""])
risk_options = ["All"] + sorted([str(x) for x in shop_df["risk_level"].dropna().unique().tolist()])

selected_district = st.sidebar.selectbox("District", district_options)
selected_cluster = st.sidebar.selectbox("Cluster", cluster_options)
selected_fps_status = st.sidebar.selectbox("FPS Status", fps_status_options)
selected_fps_type = st.sidebar.selectbox("FPS Type", fps_type_options)
selected_risk = st.sidebar.selectbox("Risk Level", risk_options)
selected_anomaly = st.sidebar.selectbox("Anomaly Status", ["All", "Normal", "Anomaly"])
map_only = st.sidebar.checkbox("Show only map-ready rows", value=False)

trans_min = int(shop_df["noOfTrans"].min())
trans_max = int(shop_df["noOfTrans"].max())
selected_trans = st.sidebar.slider("Transactions Range", trans_min, trans_max, (trans_min, trans_max))

# avoid huge 418 default confusion by clipping slider max to 99th percentile view
util_upper = float(np.nanpercentile(shop_df["utilization_ratio"], 99))
util_max = float(shop_df["utilization_ratio"].max())
util_slider_max = max(util_upper, 1.0)
selected_util = st.sidebar.slider(
    "Utilization Range",
    0.0,
    float(util_max),
    (0.0, float(util_slider_max))
)

comm_min = float(shop_df["commodity_total"].min())
comm_max = float(shop_df["commodity_total"].max())
selected_comm = st.sidebar.slider("Commodity Total Range", comm_min, comm_max, (comm_min, comm_max))

top_n = st.sidebar.slider("Top N rows/charts", 5, 30, 10)
search_text = st.sidebar.text_input("Search Shop No / Office")

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = shop_df.copy()

if selected_district != "All":
    filtered_df = filtered_df[filtered_df["district_name"] == selected_district]

if selected_cluster != "All":
    filtered_df = filtered_df[filtered_df["cluster_name"] == selected_cluster]

if selected_fps_status != "All":
    filtered_df = filtered_df[filtered_df["fpsStatus"] == selected_fps_status]

if selected_fps_type != "All":
    filtered_df = filtered_df[filtered_df["fpsType"] == selected_fps_type]

if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["risk_level"].astype(str) == selected_risk]

if selected_anomaly != "All":
    filtered_df = filtered_df[filtered_df["anomaly_status"] == selected_anomaly]

filtered_df = filtered_df[filtered_df["noOfTrans"].between(selected_trans[0], selected_trans[1])]
filtered_df = filtered_df[filtered_df["utilization_ratio"].between(selected_util[0], selected_util[1])]
filtered_df = filtered_df[filtered_df["commodity_total"].between(selected_comm[0], selected_comm[1])]

if map_only:
    filtered_df = filtered_df[filtered_df["has_valid_geo"]]

if search_text:
    search_mask = (
        filtered_df["shopNo"].astype(str).str.contains(search_text, case=False, na=False) |
        filtered_df["office_name"].astype(str).str.contains(search_text, case=False, na=False)
    )
    filtered_df = filtered_df[search_mask]

if filtered_df.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

filtered_normal = filtered_df[filtered_df["anomaly_status"] == "Normal"].copy()
filtered_anomaly = filtered_df[filtered_df["anomaly_status"] == "Anomaly"].copy()

# cluster summary based on filtered data
filtered_cluster_summary = (
    filtered_df.groupby(["cluster_id", "cluster_name"], as_index=False)[
        [c for c in ["noOfTrans", "totalRcs", "utilization_ratio", "portability_ratio",
                     "total_rice", "wheat", "sugar", "kerosene", "salt",
                     "commodity_total", "otherShopTransCnt"] if c in filtered_df.columns]
    ]
    .mean()
    .sort_values("cluster_id")
    .reset_index(drop=True)
)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-box">
    <div class="main-title">Telangana PDS Analytics Dashboard</div>
    <div class="main-subtitle">
        Clear cluster profiling, anomaly monitoring, AI-based demand estimation, ranking, and map intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI
# =========================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_card(k1, "Total Shops", f"{len(filtered_df):,}")
kpi_card(k2, "Clusters", f"{filtered_df['cluster_id'].nunique()}")
kpi_card(k3, "Anomaly Shops", f"{len(filtered_anomaly):,}")
kpi_card(k4, "Avg Transactions", f"{filtered_df['noOfTrans'].mean():,.1f}")
kpi_card(k5, "Avg Utilization", f"{filtered_df['utilization_ratio'].mean():.2f}")
kpi_card(k6, "Map-ready Shops", f"{filtered_df['has_valid_geo'].sum():,}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview",
    "🧩 Cluster Analysis",
    "🌾 Commodity Analysis",
    "🚨 Anomaly Analysis",
    "🤖 AI & Ranking",
    "🗺️ Map",
    "🔎 Shop Explorer / Data"
])

# =========================================================
# TAB 1 OVERVIEW
# =========================================================
with tab1:
    st.subheader("1. Overall Dashboard Summary")
    st.markdown('<div class="section-note">This section gives a quick understanding of shop distribution, district presence, transaction levels, and anomaly share.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cluster_counts = (
            filtered_df.groupby("cluster_name", as_index=False)
            .size()
            .rename(columns={"size": "Shop Count"})
            .sort_values("Shop Count", ascending=False)
        )
        fig = px.bar(
            cluster_counts,
            x="cluster_name",
            y="Shop Count",
            color="cluster_name",
            text="Shop Count",
            title="Cluster-wise Shop Count"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Shop Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        district_counts = (
            filtered_df.groupby("district_name", as_index=False)
            .size()
            .rename(columns={"size": "Shop Count"})
            .sort_values("Shop Count", ascending=False)
            .head(top_n)
        )
        fig = px.bar(
            district_counts,
            x="district_name",
            y="Shop Count",
            color="district_name",
            text="Shop Count",
            title=f"Top {top_n} Districts by Shop Count"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Shop Count")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        anomaly_share = pd.DataFrame({
            "Type": ["Normal", "Anomaly"],
            "Count": [len(filtered_normal), len(filtered_anomaly)]
        })
        fig = px.pie(
            anomaly_share,
            names="Type",
            values="Count",
            hole=0.50,
            title="Normal vs Anomaly Share"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(
            filtered_df,
            x="noOfTrans",
            color="cluster_name",
            nbins=35,
            title="Transactions Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2. Top Shops by Transactions")
    top_trans = filtered_df.sort_values("noOfTrans", ascending=False).head(top_n)
    fig = px.bar(
        top_trans,
        x="shopNo",
        y="noOfTrans",
        color="cluster_name",
        hover_data=["district_name", "office_name"],
        title=f"Top {top_n} Shops by Transactions"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3. Overview Summary Table")
    overview_cols = [
        "shopNo", "district_name", "office_name", "cluster_name", "fpsStatus",
        "fpsType", "noOfTrans", "totalRcs", "utilization_ratio",
        "portability_ratio", "commodity_total", "anomaly_status"
    ]
    st.dataframe(
        to_display(filtered_df[overview_cols].sort_values("noOfTrans", ascending=False).reset_index(drop=True)),
        use_container_width=True,
        hide_index=True,
        height=320
    )

# =========================================================
# TAB 2 CLUSTER ANALYSIS
# =========================================================
with tab2:
    st.subheader("1. Cluster Summary Table")
    st.markdown('<div class="section-note">Instead of radar chart, this section uses clearer charts: grouped bars, normalized heatmap, commodity stack, and PCA view.</div>', unsafe_allow_html=True)

    cluster_cols_show = [
        "cluster_id", "cluster_name", "noOfTrans", "totalRcs", "utilization_ratio",
        "portability_ratio", "commodity_total", "otherShopTransCnt",
        "total_rice", "wheat", "sugar", "kerosene", "salt"
    ]
    cluster_cols_show = [c for c in cluster_cols_show if c in filtered_cluster_summary.columns]
    st.dataframe(
        to_display(filtered_cluster_summary[cluster_cols_show].reset_index(drop=True)),
        use_container_width=True,
        hide_index=True,
        height=220
    )

    st.markdown("### 2. Main Cluster Identity Chart")
    cluster_identity = filtered_cluster_summary.melt(
        id_vars=["cluster_name"],
        value_vars=[c for c in ["noOfTrans", "totalRcs", "utilization_ratio", "portability_ratio", "commodity_total"] if c in filtered_cluster_summary.columns],
        var_name="Metric",
        value_name="Value"
    )
    fig = px.bar(
        cluster_identity,
        x="cluster_name",
        y="Value",
        color="Metric",
        barmode="group",
        title="Transactions, Cards, Utilization, Portability, Commodity Total by Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3. Normalized Cluster Heatmap")
    heat_cols = [c for c in ["noOfTrans", "totalRcs", "utilization_ratio", "portability_ratio", "commodity_total", "otherShopTransCnt"] if c in filtered_cluster_summary.columns]
    heat_df = filtered_cluster_summary[["cluster_name"] + heat_cols].copy()
    for c in heat_cols:
        heat_df[c] = normalize_series(heat_df[c])
    heat_matrix = heat_df.set_index("cluster_name")
    fig = px.imshow(
        heat_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Normalized Cluster Comparison Heatmap (Easy to Understand)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 4. Commodity by Cluster")
    commodity_cols = [c for c in ["total_rice", "wheat", "sugar", "kerosene", "salt"] if c in filtered_cluster_summary.columns]
    commodity_long = filtered_cluster_summary.melt(
        id_vars=["cluster_name"],
        value_vars=commodity_cols,
        var_name="Commodity",
        value_name="Value"
    )
    fig = px.bar(
        commodity_long,
        x="cluster_name",
        y="Value",
        color="Commodity",
        barmode="stack",
        title="Commodity Composition by Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

    if "pca1" in filtered_df.columns and "pca2" in filtered_df.columns:
        st.markdown("### 5. PCA Cluster View")
        fig = px.scatter(
            filtered_df,
            x="pca1",
            y="pca2",
            color="cluster_name",
            hover_data=["shopNo", "district_name", "office_name"],
            title="PCA Projection of Shops"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 COMMODITY ANALYSIS
# =========================================================
with tab3:
    st.subheader("1. Commodity Summary")
    st.markdown('<div class="section-note">This section helps understand which commodities dominate overall, by cluster, and by district.</div>', unsafe_allow_html=True)

    commodity_totals = {
        "Rice": filtered_df["total_rice"].sum() if "total_rice" in filtered_df.columns else 0,
        "Wheat": filtered_df["wheat"].sum() if "wheat" in filtered_df.columns else 0,
        "Sugar": filtered_df["sugar"].sum() if "sugar" in filtered_df.columns else 0,
        "Kerosene": filtered_df["kerosene"].sum() if "kerosene" in filtered_df.columns else 0,
        "Salt": filtered_df["salt"].sum() if "salt" in filtered_df.columns else 0
    }
    commodity_df = pd.DataFrame(list(commodity_totals.items()), columns=["Commodity", "Value"])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            commodity_df,
            names="Commodity",
            values="Value",
            title="Overall Commodity Share"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.treemap(
            commodity_df,
            path=["Commodity"],
            values="Value",
            title="Commodity Treemap"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2. Commodity vs Transactions")
    # FIXED: safe bubble size column used
    fig = px.scatter(
        filtered_df,
        x="commodity_total",
        y="noOfTrans",
        color="cluster_name",
        size="plot_size_cards",
        size_max=35,
        hover_data=["shopNo", "district_name", "office_name"],
        title="How Commodity Total Relates to Transactions"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3. District-wise Commodity Total")
    district_commodity = (
        filtered_df.groupby("district_name", as_index=False)[
            [c for c in ["total_rice", "wheat", "sugar", "kerosene", "salt", "commodity_total"] if c in filtered_df.columns]
        ]
        .sum()
        .sort_values("commodity_total", ascending=False)
        .head(top_n)
    )
    st.dataframe(
        to_display(district_commodity.reset_index(drop=True)),
        use_container_width=True,
        hide_index=True,
        height=280
    )

# =========================================================
# TAB 4 ANOMALY ANALYSIS
# =========================================================
with tab4:
    st.subheader("1. Anomaly Monitoring")
    st.markdown('<div class="section-note">This section shows how anomaly shops differ from normal shops and where anomalies are concentrated.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        anomaly_by_cluster = (
            filtered_df.groupby(["cluster_name", "anomaly_status"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig = px.bar(
            anomaly_by_cluster,
            x="cluster_name",
            y="Count",
            color="anomaly_status",
            barmode="group",
            text="Count",
            title="Normal vs Anomaly Shops by Cluster"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered_df,
            x="anomaly_status",
            y="utilization_gap",
            color="anomaly_status",
            title="Utilization Gap Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            filtered_df,
            x="utilization_ratio",
            y="portability_ratio",
            color="anomaly_status",
            size="plot_size_trans",
            size_max=35,
            hover_data=["shopNo", "district_name", "office_name", "cluster_name"],
            title="Utilization vs Portability Pattern"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        if not filtered_anomaly.empty:
            top_anom_dist = (
                filtered_anomaly.groupby("district_name", as_index=False)
                .size()
                .rename(columns={"size": "Anomaly Count"})
                .sort_values("Anomaly Count", ascending=False)
                .head(top_n)
            )
            fig = px.bar(
                top_anom_dist,
                x="district_name",
                y="Anomaly Count",
                color="district_name",
                text="Anomaly Count",
                title=f"Top {top_n} Districts with Anomalies"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomaly rows in current filter.")

    st.markdown("### 2. High Risk / Anomaly Table")
    anom_cols = [
        "shopNo", "district_name", "office_name", "cluster_name", "fpsStatus", "fpsType",
        "noOfTrans", "totalRcs", "utilization_ratio", "portability_ratio",
        "commodity_total", "utilization_gap", "anomaly_score", "risk_level", "anomaly_status"
    ]
    anom_cols = [c for c in anom_cols if c in filtered_df.columns]
    st.dataframe(
        to_display(filtered_df[anom_cols].sort_values(["anomaly_score", "noOfTrans"], ascending=[False, False]).reset_index(drop=True)),
        use_container_width=True,
        hide_index=True,
        height=320
    )

# =========================================================
# TAB 5 AI & RANKING
# =========================================================
with tab5:
    st.subheader("1. AI Demand Estimation and Ranking")
    st.markdown('<div class="section-note">This section estimates expected transactions, compares RF and DL models, and ranks shops using a score-based approach.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            filtered_df,
            x="noOfTrans",
            y="rf_pred_transactions",
            color="cluster_name",
            hover_data=["shopNo", "district_name", "office_name"],
            title="Actual vs Random Forest Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            filtered_df,
            x="rf_pred_transactions",
            y="mlp_pred_transactions",
            color="risk_level",
            hover_data=["shopNo", "cluster_name"],
            title="RF vs Deep Learning Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        top_ranked = filtered_df.sort_values(["shop_score", "noOfTrans"], ascending=[False, False]).head(top_n)
        top_cols = [
            "shopNo", "district_name", "office_name", "cluster_name",
            "noOfTrans", "utilization_ratio", "commodity_total", "shop_score", "rank"
        ]
        st.markdown(f"### 2. Top {top_n} Ranked Shops")
        st.dataframe(
            to_display(top_ranked[top_cols].reset_index(drop=True)),
            use_container_width=True,
            hide_index=True,
            height=280
        )

    with c4:
        high_risk = filtered_df[filtered_df["risk_level"].astype(str) == "High"].sort_values("anomaly_score", ascending=False).head(top_n)
        risk_cols = [
            "shopNo", "district_name", "office_name", "cluster_name",
            "noOfTrans", "anomaly_score", "risk_level", "anomaly_status"
        ]
        st.markdown(f"### 3. Top {top_n} High Risk Shops")
        st.dataframe(
            to_display(high_risk[risk_cols].reset_index(drop=True)),
            use_container_width=True,
            hide_index=True,
            height=280
        )

    st.markdown("### 4. Risk Distribution")
    risk_dist = (
        filtered_df.groupby("risk_level", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    fig = px.bar(
        risk_dist,
        x="risk_level",
        y="Count",
        color="risk_level",
        text="Count",
        title="Low / Medium / High Risk Distribution"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 6 MAP
# =========================================================
with tab6:
    st.subheader("1. Geo Intelligence")
    st.markdown('<div class="section-note">Only valid coordinates are shown on the map. Zero or empty latitude/longitude rows are automatically removed.</div>', unsafe_allow_html=True)

    geo_df = filtered_df[filtered_df["has_valid_geo"]].copy()

    if geo_df.empty:
        st.warning("No valid geo coordinates available for the current filter.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            try:
                fig = px.scatter_mapbox(
                    geo_df,
                    lat="latitude",
                    lon="longitude",
                    color="cluster_name",
                    size="plot_size_trans",
                    size_max=22,
                    hover_name="shopNo",
                    hover_data=["district_name", "office_name", "anomaly_status", "risk_level"],
                    zoom=5,
                    mapbox_style="carto-positron",
                    title="Cluster-based Shop Map"
                )
                fig.update_layout(margin=dict(l=0, r=0, t=45, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Map chart error: {e}")

        with c2:
            try:
                fig = px.density_mapbox(
                    geo_df,
                    lat="latitude",
                    lon="longitude",
                    z="noOfTrans",
                    radius=9,
                    zoom=5,
                    mapbox_style="carto-positron",
                    title="Transaction Density Heatmap"
                )
                fig.update_layout(margin=dict(l=0, r=0, t=45, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Heatmap error: {e}")

        st.markdown("### 2. Geo Table")
        geo_cols = [
            "shopNo", "district_name", "office_name", "cluster_name",
            "latitude", "longitude", "noOfTrans", "anomaly_status", "risk_level"
        ]
        st.dataframe(
            to_display(geo_df[geo_cols].reset_index(drop=True)),
            use_container_width=True,
            hide_index=True,
            height=300
        )

# =========================================================
# TAB 7 SHOP EXPLORER / DATA
# =========================================================
with tab7:
    st.subheader("1. Search Single Shop")
    search_shop = st.text_input("Enter exact or partial Shop No")
    temp = filtered_df.copy()

    if search_shop:
        temp = temp[temp["shopNo"].astype(str).str.contains(search_shop, case=False, na=False)]

    if temp.empty:
        st.warning("No shop found.")
    else:
        row = temp.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Shop No", str(int(row["shopNo"])) if pd.notna(row["shopNo"]) else "-")
        m2.metric("Cluster", str(row["cluster_name"]))
        m3.metric("Transactions", f"{row['noOfTrans']:.0f}")
        m4.metric("Risk", str(row["risk_level"]))

        st.markdown("### 2. Selected Shop vs Cluster Average")
        compare_cols = [c for c in ["noOfTrans", "totalRcs", "utilization_ratio", "portability_ratio", "commodity_total", "otherShopTransCnt"] if c in filtered_df.columns]

        cluster_avg = (
            filtered_df[filtered_df["cluster_name"] == row["cluster_name"]][compare_cols]
            .mean()
            .reset_index()
        )
        cluster_avg.columns = ["Metric", "Cluster Average"]

        shop_vals = pd.DataFrame({
            "Metric": compare_cols,
            "Shop Value": [row[c] for c in compare_cols]
        })

        compare_df = cluster_avg.merge(shop_vals, on="Metric", how="left")
        fig = px.bar(
            compare_df,
            x="Metric",
            y=["Cluster Average", "Shop Value"],
            barmode="group",
            title="Selected Shop Compared with Cluster Average"
        )
        st.plotly_chart(fig, use_container_width=True)

        show_cols = [
            "shopNo", "district_name", "office_name", "fpsStatus", "fpsType",
            "cluster_name", "noOfTrans", "totalRcs", "utilization_ratio",
            "portability_ratio", "total_rice", "wheat", "sugar", "kerosene",
            "salt", "commodity_total", "anomaly_status", "anomaly_score",
            "risk_level", "shop_score", "rank", "rf_pred_transactions", "mlp_pred_transactions"
        ]
        show_cols = [c for c in show_cols if c in temp.columns]
        st.dataframe(
            to_display(temp[show_cols].reset_index(drop=True)),
            use_container_width=True,
            hide_index=True,
            height=250
        )

    st.markdown("### 3. Full Data Table")
    explorer_cols = [
        "shopNo", "district_name", "office_name", "fpsStatus", "fpsType",
        "cluster_name", "noOfTrans", "totalRcs", "utilization_ratio",
        "portability_ratio", "commodity_total", "otherShopTransCnt",
        "anomaly_status", "anomaly_score", "risk_level",
        "shop_score", "rank", "rf_pred_transactions", "mlp_pred_transactions"
    ]
    st.dataframe(
        to_display(filtered_df[explorer_cols].sort_values("rank").reset_index(drop=True)),
        use_container_width=True,
        hide_index=True,
        height=350
    )

    st.markdown("### 4. Download Filtered Data")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="telangana_pds_filtered_dashboard_data.csv",
        mime="text/csv"
    )

    st.markdown("### 5. Simple Report Download")
    report_lines = [
        "TELANGANA PDS DASHBOARD REPORT",
        f"Total Shops: {len(filtered_df)}",
        f"Total Clusters: {filtered_df['cluster_id'].nunique()}",
        f"Anomaly Shops: {len(filtered_anomaly)}",
        f"Average Transactions: {filtered_df['noOfTrans'].mean():.2f}",
        f"Average Utilization Ratio: {filtered_df['utilization_ratio'].mean():.4f}",
        f"Average Commodity Total: {filtered_df['commodity_total'].mean():.2f}",
        "",
        "Top Ranked Shops:"
    ]
    for _, r in filtered_df.sort_values("shop_score", ascending=False).head(5).iterrows():
        report_lines.append(
            f"- Shop {int(r['shopNo'])} | District: {r['district_name']} | Cluster: {r['cluster_name']} | Score: {r['shop_score']:.2f}"
        )
    report_text = "\n".join(report_lines)

    st.download_button(
        "Download Text Report",
        data=report_text.encode("utf-8"),
        file_name="telangana_pds_dashboard_report.txt",
        mime="text/plain"
    )
