"""
pages/4_Admin_Portal.py - Dedicated Agronomist & Admin Verification Portal
AgriTrust AI Multipage Application

Provides expert review, label correction, and bounty disbarment for community field submissions.
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image
import contrib.store as store
from ui_theme import CSS_STYLES, render_sidebar

# Page setup
st.set_page_config(
    page_title="Agronomist Portal | AgriTrust AI",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)

# ─── ACCESS GATE: REQUIRE AUTHENTICATION ─────────────────────────────────────
from auth import require_auth
require_auth(current_page="admin")

render_sidebar(current_page="admin")

st.markdown("""
<style>
    .admin-header-card {
        background: linear-gradient(135deg, #275637 0%, #153320 100%);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(39, 86, 55, 0.12);
    }
    .queue-card {
        background: #FFFFFF;
        border: 1.5px solid #91BA94;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(39, 86, 55, 0.04);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="admin-header-card">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <div style="font-size:12px; font-weight:700; color:#B1D27D; text-transform:uppercase; letter-spacing:0.8px;">
                ICAR / KVK Verification Desk
            </div>
            <h2 style="margin:4px 0 0 0; color:#FFFFFF; font-size:26px;">
                🔐 Agronomist Verification Portal
            </h2>
            <div style="margin-top:6px; font-size:13.5px; color:#E3ECE5;">
                Expert review, disease reclassification, and cryptographic verification of field pathology photos.
            </div>
        </div>
        <div style="font-size:42px;">🔬</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ACCESS GATE ───────────────────────────────────────────────────────────────
admin_pass_env = os.environ.get("ADMIN_PASS", "agritrust")
if "ADMIN_PASS" not in os.environ:
    st.info("ℹ️ **Demo Key Mode:** `ADMIN_PASS` environment variable is not explicitly set. Using demo key: `agritrust`")

col_pass, _ = st.columns([1.5, 1])
with col_pass:
    admin_input = st.text_input("Enter Agronomist / Administrator Access Key", type="password")

if admin_input == admin_pass_env:
    admin_auth = True
    st.success("✅ Agronomist session authenticated.")
elif admin_input:
    admin_auth = False
    st.error("❌ Authentication failed: Invalid access key.")
else:
    admin_auth = False
    st.caption("🔒 Please enter the administrator access key to review pending field contributions.")

if admin_auth:
    pending_items = store.get_pending_contributions()

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Pending In Queue", f"{len(pending_items)} photos")
    with col_stat2:
        st.metric("Active Bounty Rate", "₹1.00 / verified")
    with col_stat3:
        st.metric("Reviewer Session", "admin_expert (KVK)")

    st.markdown("---")

    if not pending_items:
        st.markdown("""
        <div style="text-align: center; padding: 48px 24px; background: #FFFFFF; border-radius: 16px; border: 1.5px solid #D8E4DA; margin-top: 12px;">
            <div style="font-size: 40px; margin-bottom: 10px;">🎉</div>
            <div style="font-weight: 700; font-size: 18px; color: #275637;">All Submissions Verified</div>
            <p style="margin: 6px 0 0 0; font-size: 14px; color: #556B5C;">
                There are no pending submissions awaiting review in the queue.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"### 📋 {len(pending_items)} Submission(s) Awaiting Certification")

        for item in pending_items:
            with st.container():
                st.markdown(f"""
                <div class="queue-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                        <span style="font-weight:800; color:#275637; font-size:17px;">
                            Submission #{item['id']} &nbsp;•&nbsp; {item['crop']}
                        </span>
                        <span style="background:#E8F0E9; color:#275637; padding:4px 12px; border-radius:12px; font-size:12px; font-weight:700;">
                            Farmer: {item['farmer_id']} &nbsp;|&nbsp; 📍 {item['district']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_img, col_detail = st.columns([1, 1.4], gap="medium")

                with col_img:
                    if os.path.exists(item["filename"]):
                        st.image(item["filename"], caption=f"Perceptual Cluster: {item['cluster_id']}", use_container_width=True)
                    else:
                        st.warning("Image file missing from pool.")

                with col_detail:
                    st.markdown(f"""
                    - **Crop:** `{item['crop']}`
                    - **Farmer Claimed Condition:** `{item['claimed_label']}`
                    - **Submitted At:** `{item['timestamp']}`
                    - **Bounty:** **₹{item['reward']:.2f}**
                    - **Cluster ID:** `{item['cluster_id']}`
                    """)

                    all_crop_diseases = store.CROP_CLASSES.get(item["crop"], [item["claimed_label"]])
                    corrected_label = st.selectbox(
                        f"Confirm / Correct Diagnosis for #{item['id']}",
                        options=all_crop_diseases,
                        index=all_crop_diseases.index(item["claimed_label"]) if item["claimed_label"] in all_crop_diseases else 0,
                        key=f"label_{item['id']}"
                    )

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button(f"✅ Approve & Disburse ₹{item['reward']:.0f}", key=f"app_{item['id']}", type="primary", use_container_width=True):
                            store.verify_contribution(
                                contrib_id=item["id"],
                                verified_by="admin_expert",
                                corrected_label=corrected_label
                            )
                            st.success(f"Submission #{item['id']} verified & ₹{item['reward']:.0f} credited to {item['farmer_id']}!")
                            st.rerun()

                    with col_btn2:
                        if st.button(f"❌ Reject Submission", key=f"rej_{item['id']}", use_container_width=True):
                            store.reject_contribution(
                                contrib_id=item["id"],
                                verified_by="admin_expert"
                            )
                            st.warning(f"Submission #{item['id']} rejected.")
                            st.rerun()

                st.markdown("---")
