"""
pages/2_Contribute_and_Earn.py - Community Image Contribution & Bounty Wallet
AgriTrust AI Multipage Application

NOTE: SIMULATED RESEARCH FEATURE - NO REAL PAYMENTS PROCESSED.
"""

import os
import streamlit as st
from PIL import Image
import contrib.store as store

# Page setup
st.set_page_config(
    page_title="Contribute & Earn | AgriTrust AI",
    page_icon="🌾",
    layout="wide"
)

# Mandatory SIMULATED Banner
st.markdown("""
<div style="background-color: #FFF3CD; color: #856404; padding: 14px 20px; border-radius: 8px; border: 1.5px solid #FFEEBA; margin-bottom: 20px; font-weight: 600; text-align: center; font-size: 16px;">
    ⚠️ SIMULATED FEATURE — FOR RESEARCH & PROTOTYPING DEMONSTRATION ONLY. NO REAL PAYMENTS OR STATUTORY CERTIFICATIONS.
</div>
""", unsafe_allow_html=True)

st.title("🌾 Field Contribution & Farmer Bounty Wallet")
st.markdown(
    "Contribute geotagged crop pathology photos to expand public training distributions. "
    "Earn simulated bounties (₹1 for standard classes, ₹10 bonus for rare diseases) credited upon expert verification."
)

# Tabs
tab_contribute, tab_wallet, tab_admin = st.tabs([
    "📤 Contribute Photo",
    "👛 Bounty Wallet",
    "🔐 Expert Review (Admin)"
])

# ── TAB 1: Contribute Photo
with tab_contribute:
    st.subheader("Submit a Field Pathology Photo")
    st.caption("All submissions undergo automated quality gating, deduplication, and benchmark leakage prevention.")

    col_form, col_preview = st.columns([1.2, 1])

    with col_form:
        farmer_id = st.text_input("Farmer ID / Phone Identifier", value="farmer_KA_101", help="Used to credit your bounty wallet.")
        crop = st.selectbox("Crop Type", options=["Paddy", "Arecanut", "Apple"])
        district = st.selectbox("District (Location)", options=store.DISTRICTS)

        diseases = store.CROP_CLASSES[crop]
        claimed_label = st.selectbox(
            "Observed / Claimed Disease",
            options=diseases,
            format_func=lambda x: f"{x} (⭐ ₹10.00 Rare Bounty)" if x in store.RARE_CLASSES else f"{x} (₹1.00 Standard)"
        )

        consent_given = st.checkbox(
            "I consent to sharing this photo and district-level location data for disease tracking and research.",
            value=False
        )

        uploaded_file = st.file_uploader(
            "Upload Leaf / Tree Image",
            type=["jpg", "jpeg", "png"],
            help="Take a clear, well-focused close-up photo of the affected plant tissue."
        )

        submit_btn = st.button("Submit Contribution", type="primary")

    with col_preview:
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            st.image(uploaded_bytes, caption="Uploaded Image Preview", use_container_width=True)
            # Live blur check preview
            lap_var = store.compute_laplacian_variance(uploaded_bytes)
            sharpness_color = "green" if lap_var >= store.BLUR_THRESHOLD else "red"
            st.markdown(f"**Focus Quality (Laplacian Variance):** <span style='color:{sharpness_color};font-weight:bold;'>{lap_var:.1f}</span> (Threshold: {store.BLUR_THRESHOLD:.1f})", unsafe_allow_html=True)
        else:
            st.info("Upload an image on the left to preview sharpness and submission details.")

    if submit_btn:
        if uploaded_file is None:
            st.error("Please upload an image file before submitting.")
        else:
            img_bytes = uploaded_file.getvalue()
            res = store.add_contribution(
                farmer_id=farmer_id.strip(),
                crop=crop,
                district=district,
                claimed_label=claimed_label,
                consent_given=consent_given,
                img_bytes=img_bytes,
                filename=uploaded_file.name
            )

            if not res["valid"]:
                st.error(f"❌ **Submission Blocked:** {res['error']}")
            else:
                st.success("✅ **Photo Accepted! Status: Pending Expert Verification**")
                st.markdown(f"""
                <div style="background-color: #E8F5E9; border-left: 5px solid #2E7D32; padding: 14px 18px; border-radius: 6px; margin-top: 10px;">
                    <h4 style="margin: 0; color: #1B5E20;">Quality & Integrity Gating Passed:</h4>
                    <ul style="margin: 8px 0 0 0; color: #2E7D32;">
                        <li><strong>Location Consent:</strong> Verified</li>
                        <li><strong>Sharpness Test:</strong> Passed (Laplacian Variance: {res['laplacian_var']:.1f} ≥ {store.BLUR_THRESHOLD:.1f})</li>
                        <li><strong>Evaluation Split Gating:</strong> Clean (0 matches in benchmark datasets)</li>
                        <li><strong>Duplicate & Near-Duplicate Check:</strong> Passed</li>
                        <li><strong>Cluster ID:</strong> <code>{res['cluster_id']}</code></li>
                        <li><strong>Potential Bounty:</strong> ₹{res['reward']:.2f} ({'Rare Disease' if claimed_label in store.RARE_CLASSES else 'Standard'})</li>
                        <li><strong>Storage Pool:</strong> <code>contributed_pool/{crop.lower()}/</code></li>
                    </ul>
                    <p style="margin: 10px 0 0 0; font-size: 14px; color: #388E3C;">
                        Bounties are credited to your simulated wallet once verified by an agronomist in the Admin portal.
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 2: Bounty Wallet
with tab_wallet:
    st.subheader("Farmer Bounty Wallet & Payout Status")
    wallet_farmer_id = st.text_input("Enter Farmer ID to check balance", value=farmer_id if 'farmer_id' in locals() else "farmer_KA_101")

    wallet_data = store.get_farmer_wallet(wallet_farmer_id.strip())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Verified Balance", f"₹{wallet_data['verified_balance']:.2f}")
    with c2:
        st.metric("Pending Verification", f"₹{wallet_data['pending_balance']:.2f}")
    with c3:
        st.metric("Approved Photos", f"{wallet_data['count_verified']}")
    with c4:
        st.metric("Pending Photos", f"{wallet_data['count_pending']}")

    st.markdown("---")
    st.markdown("### Simulated Payout Threshold: ₹50.00")
    progress_frac = min(1.0, wallet_data["verified_balance"] / store.PAYOUT_THRESHOLD)
    st.progress(progress_frac)

    if wallet_data["payout_eligible"]:
        st.success(f"🎉 Threshold reached! You have ₹{wallet_data['verified_balance']:.2f} eligible for simulated payout.")
        if st.button("Request Simulated Payout"):
            st.info("Simulated payout request generated. In a production deployment, this transfers via UPI/DBT.")
    else:
        remaining = store.PAYOUT_THRESHOLD - wallet_data["verified_balance"]
        st.caption(f"₹{remaining:.2f} more in verified bounties needed to reach the ₹50.00 simulated payout threshold.")

# ── TAB 3: Admin Review
with tab_admin:
    st.subheader("🔐 Agronomist Verification Portal")
    st.caption("Review pending crowd-sourced submissions. Protected by administrator credentials.")

    admin_pass_env = os.environ.get("ADMIN_PASS")
    if not admin_pass_env:
        st.warning("⚠️ **ADMIN_PASS environment variable is not configured.** Set `ADMIN_PASS` in your host shell to enable admin actions.")
        admin_auth = False
    else:
        admin_input = st.text_input("Enter Administrator Password", type="password")
        if admin_input == admin_pass_env:
            st.success("✅ Administrator authenticated.")
            admin_auth = True
        elif admin_input:
            st.error("Incorrect administrator password.")
            admin_auth = False
        else:
            admin_auth = False

    if admin_auth:
        pending_items = store.get_pending_contributions()
        if not pending_items:
            st.info("No contributions currently pending verification. All submissions have been processed!")
        else:
            st.write(f"**{len(pending_items)} submission(s) awaiting review:**")
            for item in pending_items:
                with st.container():
                    st.markdown(f"#### Submission #{item['id']} — Farmer: `{item['farmer_id']}` ({item['district']})")
                    col_img, col_detail = st.columns([1, 1.5])

                    with col_img:
                        if os.path.exists(item["filename"]):
                            st.image(item["filename"], caption=f"Cluster: {item['cluster_id']}", width=320)
                        else:
                            st.warning("Image file missing from pool.")

                    with col_detail:
                        st.write(f"**Crop:** {item['crop']}")
                        st.write(f"**Farmer Claimed Disease:** `{item['claimed_label']}`")
                        st.write(f"**Submitted At:** {item['timestamp']}")
                        st.write(f"**Assigned Bounty:** ₹{item['reward']:.2f}")

                        all_crop_diseases = store.CROP_CLASSES.get(item["crop"], [item["claimed_label"]])
                        corrected_label = st.selectbox(
                            f"Confirm / Correct Disease Label (ID #{item['id']})",
                            options=all_crop_diseases,
                            index=all_crop_diseases.index(item["claimed_label"]) if item["claimed_label"] in all_crop_diseases else 0,
                            key=f"label_{item['id']}"
                        )

                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button(f"✅ Approve & Credit ₹{item['reward']:.0f}", key=f"app_{item['id']}", type="primary"):
                                store.verify_contribution(
                                    contrib_id=item["id"],
                                    verified_by="admin_expert",
                                    corrected_label=corrected_label
                                )
                                st.success(f"Submission #{item['id']} verified!")
                                st.rerun()

                        with col_btn2:
                            if st.button(f"❌ Reject Submission", key=f"rej_{item['id']}"):
                                store.reject_contribution(
                                    contrib_id=item["id"],
                                    verified_by="admin_expert"
                                )
                                st.warning(f"Submission #{item['id']} rejected (bounty forfeited).")
                                st.rerun()

                    st.markdown("---")
