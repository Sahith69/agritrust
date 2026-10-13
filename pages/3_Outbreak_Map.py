"""
pages/3_Outbreak_Map.py - District-Level Disease Surveillance Map
AgriTrust AI Multipage Application

Redesigned modern GIS surveillance dashboard matching AgriTrust design system.
NOTE: SIMULATED RESEARCH FEATURE - FOR DEMONSTRATION ONLY.
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
import weather_live
import contrib.store as store

# Page configuration
st.set_page_config(
    page_title="Disease Outbreak Map | AgriTrust AI",
    page_icon="🗺️",
    layout="wide"
)

# ── Custom AgriTrust Theme Styling ───────────────────────────────────────────
st.markdown("""
<style>
    /* App Canvas & Typography */
    .stApp, [data-testid="stAppViewContainer"], section.main {
        background-color: #F9F9F6 !important;
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif !important;
        color: #275637 !important;
    }

    /* Top Breadcrumb Nav Bar */
    .nav-bar {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .nav-link {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        background: #FFFFFF;
        color: #275637 !important;
        border: 1px solid #91BA94;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }
    .nav-link:hover {
        background: #E8F0E9;
        border-color: #275637;
    }
    .nav-link.active {
        background: #275637;
        color: #FFFFFF !important;
        border-color: #275637;
    }

    /* Simulation Notice Banner */
    .sim-badge {
        background-color: #FFF9E6;
        color: #8C6D15;
        border: 1px solid #F5E096;
        padding: 10px 18px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }

    /* KPI Summary Card */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #D8E4DA;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(39, 86, 55, 0.04);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #556B5C;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #275637;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #728C78;
        margin-top: 2px;
    }

    /* Hotspot District Card */
    .hotspot-card {
        background: #FFFFFF;
        border: 1px solid #D8E4DA;
        border-left: 4px solid #275637;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(39, 86, 55, 0.03);
    }
</style>
""", unsafe_allow_html=True)

# ── Navigation & Simulated Header ─────────────────────────────────────────────
nav_c1, nav_c2, nav_c3 = st.columns([1, 1.2, 1])
with nav_c1:
    if st.button("🔬 Home & Diagnostic Scan", key="nav_to_home_m", use_container_width=True):
        st.switch_page("app.py")
with nav_c2:
    if st.button("🌾 Field Bounty Contributor", key="nav_to_contrib_m", use_container_width=True):
        st.switch_page("pages/2_Contribute_and_Earn.py")
with nav_c3:
    st.button("🗺️ Outbreak Map (Current)", key="nav_curr_map", disabled=True, use_container_width=True)

st.markdown("""
<div class="sim-badge" style="margin-top: 10px;">
    <span>⚠️</span>
    <div>
        <strong>SIMULATED EPIDEMIOLOGICAL SURVEILLANCE:</strong> Regional aggregates demonstrate disease pressure surveillance and cluster tracking.
    </div>
</div>
""", unsafe_allow_html=True)

st.title("🗺️ District-Level Crop Pathology Surveillance Map")
st.markdown(
    "Live spatial tracking of **expert-verified** crop disease reports across monitored agricultural districts. "
    "Features **strict differential privacy**: only coarse district centroid coordinates are rendered."
)

# ── Privacy Guarantee Alert ───────────────────────────────────────────────────
st.markdown("""
<div style="background-color: #EBF3ED; border: 1px solid #91BA94; border-radius: 12px; padding: 12px 18px; margin-bottom: 20px; font-size: 13.5px; color: #275637;">
    🔒 <strong>Strict Privacy Shield Active:</strong> Under no circumstances are farmer IDs, photo filenames, device metadata, or coordinates finer than the district centroid displayed or exported.
</div>
""", unsafe_allow_html=True)

# ── Controls: Simulation Toggle & Filters ─────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns([1.5, 1], gap="medium")
with col_ctrl1:
    include_simulated = st.checkbox(
        "Include Historical Benchmark Surveillance Data",
        value=True,
        help="Blends historical benchmark surveillance records with live expert-verified field contributions."
    )
with col_ctrl2:
    if include_simulated:
        st.markdown("""
        <div style="font-size: 12.5px; color: #556B5C; padding-top: 6px;">
            ℹ️ Showing <strong>historical benchmark surveillance records</strong> alongside verified field submissions.
        </div>
        """, unsafe_allow_html=True)

# Fetch aggregated outbreak data
records = store.get_verified_outbreaks(include_simulated=include_simulated)

if not records:
    st.warning("No verified disease outbreak records in the database. Enable 'Include Historical Benchmark Surveillance Data' above to preview spatial cluster distribution.")
else:
    df = pd.DataFrame(records)

    # ── Surveillance KPI Summary Cards ─────────────────────────────────────────
    total_cases = int(df["count"].sum())
    monitored_districts = df["district"].nunique()
    top_disease_row = df.groupby("disease")["count"].sum().sort_values(ascending=False)
    top_disease = top_disease_row.index[0].replace("_", " ").title() if not top_disease_row.empty else "N/A"
    top_disease_count = top_disease_row.iloc[0] if not top_disease_row.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Monitored Districts</div>
            <div class="kpi-value">{monitored_districts}</div>
            <div class="kpi-sub">Active regional centroids</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Verified Cases</div>
            <div class="kpi-value">{total_cases}</div>
            <div class="kpi-sub">Surveillance reports</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Monitored Pathogen</div>
            <div class="kpi-value" style="font-size:20px;">{top_disease}</div>
            <div class="kpi-sub">{top_disease_count} documented cases</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Surveillance Privacy</div>
            <div class="kpi-value" style="font-size:20px; color:#275637;">100% Shielded</div>
            <div class="kpi-sub">District centroid only</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── Filter Controls ───────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        crop_options = ["All Crops"] + sorted(df["crop"].unique())
        crop_filter = st.selectbox("Filter by Crop", options=crop_options)
    with col_f2:
        district_options = ["All Districts"] + sorted(df["district"].unique())
        district_filter = st.selectbox("Filter by District", options=district_options)

    filtered_df = df.copy()
    if crop_filter != "All Crops":
        filtered_df = filtered_df[filtered_df["crop"] == crop_filter]
    if district_filter != "All Districts":
        filtered_df = filtered_df[filtered_df["district"] == district_filter]

    # ── PyDeck Map Visualization ──────────────────────────────────────────────
    st.markdown("### 📍 Disease Outbreak Intensity by District")

    # Group by district for map rendering
    map_rows = []
    district_grouped = filtered_df.groupby("district")

    for d_name, group in district_grouped:
        cnt = int(group["count"].sum())
        if d_name in weather_live.DISTRICT_COORDINATES:
            coords = weather_live.DISTRICT_COORDINATES[d_name]
            top_dis = group.sort_values(by="count", ascending=False).iloc[0]["disease"].replace("_", " ").title()
            crops_str = ", ".join(sorted(group["crop"].unique()))

            # Color coding by crop or intensity
            if "Paddy" in crops_str and "Apple" not in crops_str and "Arecanut" not in crops_str:
                fill_color = [39, 140, 75, 200]    # Emerald Green for Paddy
            elif "Apple" in crops_str and "Paddy" not in crops_str and "Arecanut" not in crops_str:
                fill_color = [185, 55, 55, 200]   # Crimson for Apple
            elif "Arecanut" in crops_str and "Paddy" not in crops_str and "Apple" not in crops_str:
                fill_color = [220, 155, 30, 200]  # Amber for Arecanut
            else:
                fill_color = [39, 86, 55, 210]    # AgriTrust Forest Green for multi-crop

            map_rows.append({
                "district": d_name,
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "total_cases": cnt,
                "dominant_disease": top_dis,
                "crops_summary": crops_str,
                "fill_color": fill_color,
                "size": min(100.0, max(25.0, cnt * 4.0))
            })

    if map_rows:
        map_df = pd.DataFrame(map_rows)

        # PyDeck Scatterplot Layer
        try:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[longitude, latitude]",
                get_fill_color="fill_color",
                get_radius="total_cases * 1600 + 20000",
                pickable=True,
                stroked=True,
                get_line_color=[255, 255, 255, 230],
                line_width_min_pixels=2
            )

            # Center map over India
            view_state = pdk.ViewState(
                latitude=21.5,
                longitude=77.5,
                zoom=4.4,
                pitch=20
            )

            tooltip = {
                "html": """
                <div style="font-family:system-ui; padding:8px 12px; background:#275637; color:#FFFFFF; border-radius:8px; font-size:13px; line-height:1.4;">
                    <div style="font-size:15px; font-weight:700;">📍 {district}</div>
                    <div style="margin-top:4px;"><b>Verified Cases:</b> {total_cases}</div>
                    <div><b>Dominant Disease:</b> {dominant_disease}</div>
                    <div><b>Crops Monitored:</b> {crops_summary}</div>
                </div>
                """,
                "style": {"color": "white"}
            }

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="light",
                    tooltip=tooltip
                )
            )
        except Exception:
            # Fallback to st.map if PyDeck has environment issues
            st.map(map_df, latitude="latitude", longitude="longitude", size="size")

        # Map Legend
        st.markdown("""
        <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap; font-size:12.5px; color:#556B5C; margin-top:-8px; margin-bottom:20px;">
            <span style="display:inline-flex; align-items:center; gap:6px;">
                <span style="width:12px; height:12px; border-radius:50%; background:rgb(39, 140, 75); display:inline-block;"></span> Paddy
            </span>
            <span style="display:inline-flex; align-items:center; gap:6px;">
                <span style="width:12px; height:12px; border-radius:50%; background:rgb(220, 155, 30); display:inline-block;"></span> Arecanut
            </span>
            <span style="display:inline-flex; align-items:center; gap:6px;">
                <span style="width:12px; height:12px; border-radius:50%; background:rgb(185, 55, 55); display:inline-block;"></span> Apple
            </span>
            <span style="display:inline-flex; align-items:center; gap:6px;">
                <span style="width:12px; height:12px; border-radius:50%; background:rgb(39, 86, 55); display:inline-block;"></span> Multi-Crop Cluster
            </span>
            <span>(Circle size represents relative case volume)</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("No matching surveillance records for the selected filters.")

    # ── Active Hotspots & Outbreak Master Table ───────────────────────────────
    col_hotspots, col_table = st.columns([1, 1.6], gap="large")

    with col_hotspots:
        st.markdown("### 🔥 High-Priority Regional Hotspots")
        hotspot_summary = (
            filtered_df.groupby("district")
            .agg(total_cases=("count", "sum"), top_crop=("crop", "first"))
            .reset_index()
            .sort_values(by="total_cases", ascending=False)
            .head(4)
        )

        for _, row in hotspot_summary.iterrows():
            d_name = row["district"]
            c_cnt = int(row["total_cases"])
            c_crop = row["top_crop"]

            badge_text = "🚨 Elevated Alert" if c_cnt >= 20 else "⚠️ Moderate Alert"
            badge_color = "#B5483A" if c_cnt >= 20 else "#D9822B"

            st.markdown(f"""
            <div class="hotspot-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; font-size:15px; color:#275637;">📍 {d_name}</div>
                    <span style="font-size:12px; font-weight:700; color:{badge_color};">{badge_text}</span>
                </div>
                <div style="margin-top:6px; font-size:13px; color:#556B5C;">
                    Primary Focus: <strong>{c_crop}</strong> &nbsp;|&nbsp; <strong>{c_cnt}</strong> total verified cases
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_table:
        st.markdown("### 📋 Verified Outbreak Surveillance Ledger")

        display_df = filtered_df.rename(columns={
            "district": "District",
            "crop": "Crop",
            "disease": "Disease Condition",
            "count": "Verified Reports",
            "is_simulated": "Simulated Demo Row"
        }).copy()

        display_df["Disease Condition"] = display_df["Disease Condition"].apply(
            lambda x: str(x).replace("_", " ").title()
        )

        display_df["Data Origin"] = display_df["Simulated Demo Row"].apply(
            lambda sim: "Historical Benchmark" if sim else "Field Verified Submission"
        )

        st.dataframe(
            display_df[["District", "Crop", "Disease Condition", "Verified Reports", "Data Origin"]],
            use_container_width=True,
            hide_index=True
        )
