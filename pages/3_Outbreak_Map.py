"""
pages/3_Outbreak_Map.py - District-Level Disease Surveillance Map
AgriTrust AI Multipage Application

NOTE: SIMULATED RESEARCH FEATURE - FOR DEMONSTRATION ONLY.
"""

import streamlit as st
import pandas as pd
import weather_live
import contrib.store as store

# Page configuration
st.set_page_config(
    page_title="Outbreak Map | AgriTrust AI",
    page_icon="🗺️",
    layout="wide"
)

# Mandatory SIMULATED Banner
st.markdown("""
<div style="background-color: #FFF3CD; color: #856404; padding: 14px 20px; border-radius: 8px; border: 1.5px solid #FFEEBA; margin-bottom: 20px; font-weight: 600; text-align: center; font-size: 16px;">
    ⚠️ SIMULATED FEATURE — FOR RESEARCH & PROTOTYPING DEMONSTRATION ONLY.
</div>
""", unsafe_allow_html=True)

st.title("🗺️ District-Level Outbreak Surveillance Map")
st.markdown(
    "Live spatial aggregation of **expert-verified** crop pathology reports across monitored agricultural districts. "
    "Maintains strict farmer privacy: only coarse district-level disease aggregates are exposed."
)

# Privacy disclosure alert
st.info("🔒 **Privacy Guarantee:** Under no circumstances are farmer IDs, photo filenames, device metadata, or coordinates finer than the district centroid displayed or exported.")

# Simulated Demo Toggle (Default: OFF)
include_simulated = st.checkbox("Include simulated demo data", value=False)

if include_simulated:
    st.markdown("""
    <div style="background-color: #E3F2FD; color: #0D47A1; padding: 12px 18px; border-radius: 6px; border: 1px solid #BBDEFB; margin-bottom: 16px; font-weight: 500;">
        ℹ️ <strong>SIMULATED DEMO DATA ACTIVE:</strong> The map and surveillance tables currently display historical benchmark surveillance records alongside verified submissions.
    </div>
    """, unsafe_allow_html=True)

# Fetch aggregated outbreak data
records = store.get_verified_outbreaks(include_simulated=include_simulated)

if not records:
    st.warning("No verified disease outbreak records in the database. Enable 'Include simulated demo data' above to preview spatial cluster distribution.")
else:
    df = pd.DataFrame(records)

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        crop_filter = st.selectbox("Filter by Crop", options=["All Crops"] + sorted(df["crop"].unique()))
    with col_f2:
        district_filter = st.selectbox("Filter by District", options=["All Districts"] + sorted(df["district"].unique()))

    filtered_df = df.copy()
    if crop_filter != "All Crops":
        filtered_df = filtered_df[filtered_df["crop"] == crop_filter]
    if district_filter != "All Districts":
        filtered_df = filtered_df[filtered_df["district"] == district_filter]

    # Map Aggregation (One marker per district)
    map_rows = []
    district_totals = filtered_df.groupby("district")["count"].sum().reset_index()

    for _, row in district_totals.iterrows():
        d_name = row["district"]
        cnt = int(row["count"])
        if d_name in weather_live.DISTRICT_COORDINATES:
            coords = weather_live.DISTRICT_COORDINATES[d_name]
            map_rows.append({
                "district": d_name,
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "total_cases": cnt,
                # Scaled marker size for map rendering
                "size": min(100.0, max(20.0, cnt * 4.0))
            })

    if map_rows:
        map_df = pd.DataFrame(map_rows)
        st.subheader("📍 Disease Outbreak Intensity by District")
        st.map(map_df, latitude="latitude", longitude="longitude", size="size")
    else:
        st.info("No matching records for the selected filters.")

    # Aggregated Summary Table
    st.subheader("📋 Verified Outbreak Summary Table")

    display_df = filtered_df.rename(columns={
        "district": "District",
        "crop": "Crop",
        "disease": "Disease",
        "count": "Verified Reports",
        "is_simulated": "Simulated Demo Row"
    })

    display_df["Data Origin"] = display_df["Simulated Demo Row"].apply(
        lambda sim: "Simulated Historical Benchmark" if sim else "Verified Field Submission"
    )

    st.dataframe(
        display_df[["District", "Crop", "Disease", "Verified Reports", "Data Origin"]],
        use_container_width=True,
        hide_index=True
    )
