"""
pages/2_Contribute_and_Earn.py - Community Image Contribution & Bounty Wallet
AgriTrust AI Multipage Application

Redesigned modern UI matching AgriTrust design system.
NOTE: SIMULATED RESEARCH FEATURE - NO REAL PAYMENTS PROCESSED.
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image
import contrib.store as store

# Page setup
st.set_page_config(
    page_title="Contribute & Earn | AgriTrust AI",
    page_icon="🌾",
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

    /* Styled Container Card */
    .ui-card {
        background: #FFFFFF;
        border: 1px solid #D8E4DA;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(39, 86, 55, 0.03);
    }

    /* Bounty Tier Badges */
    .tier-badge-std {
        background: #E8F2EA;
        color: #275637;
        border: 1px solid #91BA94;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .tier-badge-rare {
        background: #FFF3D6;
        color: #8C5E08;
        border: 1px solid #F6D05F;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Digital Farmer Pass Wallet Card */
    .wallet-card {
        background: linear-gradient(135deg, #1A472A 0%, #275637 100%);
        color: #FFFFFF;
        border-radius: 18px;
        padding: 24px 28px;
        box-shadow: 0 8px 24px rgba(39, 86, 55, 0.16);
        position: relative;
        overflow: hidden;
        margin-bottom: 22px;
    }

    /* Button Styling */
    button, .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Navigation & Simulated Header ─────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <a href="/" target="_self" class="nav-link">🔬 Home & Diagnostic Scan</a>
    <a href="/Contribute_and_Earn" target="_self" class="nav-link active">🌾 Field Bounty Contributor</a>
    <a href="/Outbreak_Map" target="_self" class="nav-link">🗺️ Disease Outbreak Map</a>
</div>
<div class="sim-badge">
    <span>⚠️</span>
    <div>
        <strong>SIMULATED RESEARCH FEATURE:</strong> For algorithmic integrity and crowd-sourcing evaluation only. No statutory payments or real INR currency processed.
    </div>
</div>
""", unsafe_allow_html=True)

st.title("🌾 Field Pathology Contribution & Farmer Bounty Wallet")
st.markdown(
    "Contribute verified field pathology images to close distribution shifts. "
    "Every contribution undergoes **automated focus gating**, **leakage prevention**, and **expert agronomist verification**."
)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_contribute, tab_wallet, tab_admin = st.tabs([
    "📤 Submit Field Photo",
    "👛 Farmer Bounty Wallet",
    "🔐 Agronomist Verification Desk"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SUBMIT FIELD PHOTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_contribute:
    st.markdown("""
    <div class="ui-card" style="margin-top: 10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <h4 style="margin:0; color:#275637;">Bounty Incentive Structure</h4>
                <p style="margin:4px 0 0 0; color:#556B5C; font-size:14px;">
                    Submissions are credited upon expert confirmation by agricultural officers.
                </p>
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="tier-badge-std">🌾 Standard Class: ₹1.00</span>
                <span class="tier-badge-rare">⭐ Rare Pathology Bonus: ₹10.00</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_preview = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown("#### 1. Submission Details")
        farmer_id = st.text_input(
            "Farmer Identifier / Phone ID",
            value="farmer_KA_101",
            help="Your identifier used to track submissions and credit your bounty wallet."
        )

        col_crop, col_dist = st.columns(2)
        with col_crop:
            crop = st.selectbox("Crop Type", options=["Paddy", "Arecanut", "Apple"])
        with col_dist:
            district = st.selectbox("District (Location Centroid)", options=store.DISTRICTS)

        diseases = store.CROP_CLASSES[crop]
        claimed_label = st.selectbox(
            "Observed Disease Condition",
            options=diseases,
            format_func=lambda x: f"{x} — ⭐ ₹10.00 (Rare)" if x in store.RARE_CLASSES else f"{x} — ₹1.00 (Standard)"
        )

        # Bounty Tier preview chip
        is_rare = claimed_label in store.RARE_CLASSES
        reward_preview = store.REWARD_RARE if is_rare else store.REWARD_STANDARD
        tier_label = "⭐ Rare Disease Bounty" if is_rare else "🌾 Standard Disease Bounty"
        badge_class = "tier-badge-rare" if is_rare else "tier-badge-std"

        st.markdown(f"""
        <div style="margin: 8px 0 16px 0;">
            <span class="{badge_class}">
                {tier_label}: <strong>₹{reward_preview:.2f} credited upon verification</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 2. Research & Location Consent")
        consent_given = st.checkbox(
            "I grant consent to share this leaf photograph and district centroid for regional epidemiological tracking and open-source agricultural research.",
            value=False
        )

        uploaded_file = st.file_uploader(
            "Upload Field Photo (Leaf / Nut / Trunk)",
            type=["jpg", "jpeg", "png"],
            help="Ensure good ambient daylight and focused close-up framing."
        )

        submit_btn = st.button("🚀 Submit Contribution for Verification", type="primary", use_container_width=True)

    with col_preview:
        st.markdown("#### 3. Real-time Quality Inspection")
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            st.image(uploaded_bytes, caption=f"Preview: {uploaded_file.name}", use_container_width=True)

            # Live blur evaluation
            lap_var = store.compute_laplacian_variance(uploaded_bytes)
            is_sharp = (lap_var >= store.BLUR_THRESHOLD)

            status_color = "#275637" if is_sharp else "#B5483A"
            status_bg = "#E8F5E9" if is_sharp else "#FDECEA"
            status_text = "🟢 Crystal Sharp (Passed)" if is_sharp else "🔴 Blurry / Out of Focus (Fails Threshold)"

            st.markdown(f"""
            <div style="background-color: {status_bg}; border: 1.5px solid {status_color}; border-radius: 12px; padding: 14px 18px; margin-top: 12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:{status_color}; font-size:14px;">Optical Focus Quality</span>
                    <span style="font-weight:700; color:{status_color}; font-size:14px;">{status_text}</span>
                </div>
                <div style="margin-top:8px; font-size:13px; color:#556B5C;">
                    Variance of Laplacian: <strong>{lap_var:.1f}</strong> &nbsp;|&nbsp; Minimum Required: <strong>{store.BLUR_THRESHOLD:.1f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="border: 2px dashed #91BA94; border-radius: 14px; padding: 48px 24px; text-align: center; background: #FFFFFF; color: #556B5C;">
                <div style="font-size: 36px; margin-bottom: 8px;">📷</div>
                <div style="font-weight: 700; font-size: 16px; color: #275637;">Live Focus & Quality Inspector</div>
                <p style="margin: 6px 0 0 0; font-size: 13px;">
                    Upload an image to evaluate edge clarity, perceptual sharpness, and evaluation-set integrity.
                </p>
            </div>
            """, unsafe_allow_html=True)

    if submit_btn:
        if uploaded_file is None:
            st.error("⚠️ Please select an image file to upload before submitting.")
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
                st.error(f"❌ **Submission Rejected:** {res['error']}")
            else:
                st.success("✅ **Quality & Integrity Checks Passed! Submission Enqueued for Review.**")
                st.markdown(f"""
                <div style="background-color: #E8F5E9; border-left: 5px solid #275637; border-radius: 12px; padding: 18px 22px; margin-top: 14px;">
                    <h4 style="margin: 0; color: #275637;">🛡️ Gating Verification Summary</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 12px; font-size: 13.5px; color: #275637;">
                        <div>• <strong>Submission ID:</strong> #{res['id']}</div>
                        <div>• <strong>Reserved Bounty:</strong> ₹{res['reward']:.2f} ({'Rare Pathology' if is_rare else 'Standard'})</div>
                        <div>• <strong>Focus Score:</strong> {res['laplacian_var']:.1f} (≥ {store.BLUR_THRESHOLD:.1f})</div>
                        <div>• <strong>Cluster ID:</strong> <code>{res['cluster_id']}</code></div>
                        <div>• <strong>Evaluation Split Shield:</strong> 0 matches (Leakage Clean)</div>
                        <div>• <strong>Status:</strong> ⏳ Pending Expert Review</div>
                    </div>
                    <p style="margin: 12px 0 0 0; font-size: 13px; color: #4B6E52;">
                        Your bounty will be credited to your simulated wallet once verified by an agronomist in the Admin Desk.
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FARMER BOUNTY WALLET
# ══════════════════════════════════════════════════════════════════════════════
with tab_wallet:
    st.markdown("#### Check Farmer Bounty Balance & Payout Status")
    wallet_farmer_id = st.text_input(
        "Enter Farmer Identifier",
        value=farmer_id if 'farmer_id' in locals() else "farmer_KA_101",
        key="wallet_search_input"
    ).strip()

    wallet_data = store.get_farmer_wallet(wallet_farmer_id)
    history_items = store.get_farmer_history(wallet_farmer_id)

    # Digital Farmer Pass Card
    v_bal = wallet_data["verified_balance"]
    p_bal = wallet_data["pending_balance"]
    threshold = store.PAYOUT_THRESHOLD
    progress_frac = min(1.0, v_bal / threshold)

    st.markdown(f"""
    <div class="wallet-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
            <div>
                <span style="font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: #A8D5B5; font-weight: 700;">AgriTrust Farmer Pass</span>
                <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{wallet_farmer_id}</div>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">
                🌱 Active Contributor
            </div>
        </div>
        <div style="display: flex; gap: 36px; align-items: baseline; flex-wrap: wrap;">
            <div>
                <div style="font-size: 12px; color: #C2E2CC; font-weight: 600;">Verified Available Balance</div>
                <div style="font-size: 34px; font-weight: 800; color: #FFFFFF; line-height: 1.2;">₹{v_bal:.2f}</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #C2E2CC; font-weight: 600;">In-Review Escrow</div>
                <div style="font-size: 22px; font-weight: 700; color: #F6D05F; line-height: 1.2;">₹{p_bal:.2f}</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #C2E2CC; font-weight: 600;">Approved Photos</div>
                <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; line-height: 1.2;">{wallet_data['count_verified']}</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #C2E2CC; font-weight: 600;">Pending Verification</div>
                <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; line-height: 1.2;">{wallet_data['count_pending']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Payout Milestone Section
    st.markdown(f"#### 🎯 Simulated Payout Goal: ₹{threshold:.2f}")
    st.progress(progress_frac)

    if wallet_data["payout_eligible"]:
        st.markdown(f"""
        <div style="background-color: #E8F5E9; border: 1.5px solid #275637; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="margin:0; color:#275637;">🎉 Payout Milestone Reached!</h4>
                    <p style="margin:4px 0 0 0; color:#406E4A; font-size:14px;">
                        You have <strong>₹{v_bal:.2f}</strong> eligible for simulated Direct Benefit Transfer (DBT).
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("💳 Request Simulated DBT / UPI Payout", type="primary"):
            st.success("✅ Simulated DBT transfer request initiated! In production, funds transfer directly to farmer bank account via DBT/UPI.")
    else:
        remaining = threshold - v_bal
        st.caption(f"💡 Earn **₹{remaining:.2f}** more in verified photo bounties to reach the ₹{threshold:.2f} simulated payout threshold.")

    # Activity Ledger
    st.markdown("---")
    st.markdown("#### 📜 Field Submission Activity Ledger")
    if not history_items:
        st.info("No contributions recorded yet for this Farmer ID. Submit a photo in the first tab to start earning!")
    else:
        ledger_df = pd.DataFrame(history_items)
        display_ledger = ledger_df[["id", "crop", "claimed_label", "district", "timestamp", "reward", "status"]].copy()
        display_ledger.rename(columns={
            "id": "ID",
            "crop": "Crop",
            "claimed_label": "Claimed Disease",
            "district": "District",
            "timestamp": "Submitted At",
            "reward": "Bounty (₹)",
            "status": "Status"
        }, inplace=True)

        st.dataframe(display_ledger, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AGRONOMIST VERIFICATION DESK
# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
        <div>
            <h3 style="margin:0; color:#275637;">🔐 Agronomist Verification Portal</h3>
            <p style="margin:4px 0 0 0; color:#556B5C; font-size:14px;">
                Review, correct, and certify community field pathology submissions prior to wallet disbursement.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    admin_pass_env = os.environ.get("ADMIN_PASS", "agritrust")
    if "ADMIN_PASS" not in os.environ:
        st.info("ℹ️ **Demo Access Mode:** `ADMIN_PASS` environment variable is not set. Using demo key: `agritrust`")

    admin_input = st.text_input("Enter Administrator Access Key", type="password")
    if admin_input == admin_pass_env:
        st.success("✅ Agronomist session authenticated.")
        admin_auth = True
    elif admin_input:
        st.error("❌ Authentication failed: Invalid access key.")
        admin_auth = False
    else:
        admin_auth = False

    if admin_auth:
        pending_items = store.get_pending_contributions()
        if not pending_items:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; background: #FFFFFF; border-radius: 14px; border: 1px solid #D8E4DA;">
                <div style="font-size: 36px; margin-bottom: 8px;">🎉</div>
                <div style="font-weight: 700; font-size: 16px; color: #275637;">All Submissions Verified</div>
                <p style="margin: 6px 0 0 0; font-size: 13.5px; color: #556B5C;">
                    There are no pending submissions awaiting review in the queue.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(pending_items)} submission(s) awaiting expert verification:**")

            for item in pending_items:
                with st.container():
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #91BA94; border-radius:14px; padding:18px 22px; margin-bottom:16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <span style="font-weight:700; color:#275637; font-size:16px;">Submission #{item['id']}</span>
                            <span style="background:#E8F0E9; color:#275637; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:600;">
                                Farmer: {item['farmer_id']} • {item['district']}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_img, col_detail = st.columns([1, 1.4], gap="medium")

                    with col_img:
                        if os.path.exists(item["filename"]):
                            st.image(item["filename"], caption=f"Cluster: {item['cluster_id']}", use_container_width=True)
                        else:
                            st.warning("Image file missing from pool.")

                    with col_detail:
                        st.markdown(f"""
                        - **Crop:** `{item['crop']}`
                        - **Farmer Claimed Condition:** `{item['claimed_label']}`
                        - **Submitted At:** `{item['timestamp']}`
                        - **Bounty:** **₹{item['reward']:.2f}**
                        - **Perceptual Cluster ID:** `{item['cluster_id']}`
                        """)

                        all_crop_diseases = store.CROP_CLASSES.get(item["crop"], [item["claimed_label"]])
                        corrected_label = st.selectbox(
                            f"Confirm / Correct Disease Diagnosis (ID #{item['id']})",
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
                                st.success(f"Submission #{item['id']} verified & credited!")
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
