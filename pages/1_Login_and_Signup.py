"""
pages/1_Login_and_Signup.py - Farmer & Specialist Account Portal
AgriTrust AI Multipage Application

Provides secure authentication with duplicate username prevention and session state persistence.
"""

import streamlit as st
import auth
from ui_theme import CSS_STYLES, render_sidebar

# Page config
st.set_page_config(
    page_title="Account & Login | AgriTrust AI",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)
render_sidebar(current_page="account")

# Custom page card styling
st.markdown("""
<style>
    .auth-card {
        background: #FFFFFF;
        border: 1px solid #D8E4DA;
        border-radius: 16px;
        padding: 28px 32px;
        box-shadow: 0 4px 16px rgba(39, 86, 55, 0.04);
        max-width: 600px;
        margin: 0 auto;
    }
    .user-profile-badge {
        background: linear-gradient(135deg, #275637 0%, #1A472A 100%);
        color: #FFFFFF !important;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(39, 86, 55, 0.15);
    }
    .user-profile-badge h1,
    .user-profile-badge h2,
    .user-profile-badge h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔑 Farmer & Specialist Account Portal")
st.markdown(
    "Sign in or register your account to track field pathology submissions, receive regional disease advisories, "
    "and access your digital bounty wallet."
)

is_logged_in = st.session_state.get("is_logged_in", False)
current_user = st.session_state.get("user", None)

if is_logged_in and current_user:
    # ── ALREADY LOGGED IN VIEW ────────────────────────────────────────────────
    role = st.session_state.get("user_role", "Farmer").title()
    st.markdown(f"""
    <div class="user-profile-badge">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;">
            <div>
                <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#B1D27D !important;">
                    Active Farmer Account
                </div>
                <h2 style="margin:4px 0 0 0; color:#FFFFFF !important; font-size:26px; font-weight:800;">
                    Welcome back, {current_user}!
                </h2>
                <div style="margin-top:6px; font-size:13px; color:#E3ECE5;">
                    Role: <strong>{role}</strong> &nbsp;|&nbsp; Status: <span style="color:#B1D27D; font-weight:700;">Verified Active</span>
                </div>
            </div>
            <div style="font-size:44px;">🌾</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🌾 Plant Pathology Advisory", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        if st.button("👛 Field Bounty Wallet", use_container_width=True):
            st.switch_page("pages/2_Contribute_and_Earn.py")
    with col3:
        if st.button("🗺️ Surveillance Map", use_container_width=True):
            st.switch_page("pages/3_Outbreak_Map.py")
    with col4:
        if st.button("🚪 Sign Out", type="secondary", use_container_width=True):
            st.session_state["is_logged_in"] = False
            st.session_state["user"] = None
            st.session_state["user_role"] = None
            st.success("You have been signed out.")
            st.rerun()

else:
    # ── SIGN IN / SIGN UP TABS ────────────────────────────────────────────────
    tab_signin, tab_signup = st.tabs(["🔐 Sign In to Account", "📝 Create New Account"])

    with tab_signin:
        st.markdown("#### Enter your credentials to sign in:")
        col_form, _ = st.columns([1.5, 1])
        with col_form:
            with st.form("signin_form"):
                in_username = st.text_input("Username / Farmer ID", placeholder="e.g. farmer_KA_101").strip()
                in_password = st.text_input("Password", type="password", placeholder="Enter your password")
                btn_signin = st.form_submit_button("🚀 Sign In", type="primary", use_container_width=True)

            if btn_signin:
                if not in_username or not in_password:
                    st.error("⚠️ Please enter both username and password.")
                else:
                    success, user_data, msg = auth.verify_user(in_username, in_password)
                    if success:
                        st.session_state["is_logged_in"] = True
                        st.session_state["user"] = user_data["username"]
                        st.session_state["user_role"] = user_data["role"]
                        st.success(f"✅ Welcome back, **{user_data['username']}**!")
                        try:
                            st.switch_page("app.py")
                        except Exception:
                            st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    with tab_signup:
        st.markdown("#### Create a new unique account:")
        col_reg, _ = st.columns([1.5, 1])
        with col_reg:
            with st.form("signup_form"):
                new_username = st.text_input(
                    "Choose a Unique Username",
                    placeholder="e.g. farmer_ka_105",
                    help="Usernames are unique across the platform and cannot be duplicated."
                ).strip()

                new_password = st.text_input(
                    "Create Password",
                    type="password",
                    placeholder="At least 4 characters"
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-type password"
                )

                new_role = st.selectbox(
                    "Account Type",
                    options=["Farmer", "Agronomist / Extension Expert", "Agricultural Researcher"],
                    index=0
                )

                btn_signup = st.form_submit_button("🌱 Register Account", type="primary", use_container_width=True)

            if btn_signup:
                if not new_username:
                    st.error("⚠️ Username cannot be empty.")
                elif len(new_username) < 3:
                    st.error("⚠️ Username must be at least 3 characters.")
                elif not new_password:
                    st.error("⚠️ Password cannot be empty.")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match. Please verify your entries.")
                else:
                    ok, reg_msg = auth.register_user(
                        username=new_username,
                        password=new_password,
                        role="agronomist" if "Agronomist" in new_role else "farmer"
                    )
                    if ok:
                        st.session_state["is_logged_in"] = True
                        st.session_state["user"] = new_username
                        st.session_state["user_role"] = "agronomist" if "Agronomist" in new_role else "farmer"
                        st.success(f"🎉 Account created successfully! Logged in as **{new_username}**.")
                        try:
                            st.switch_page("app.py")
                        except Exception:
                            st.rerun()
                    else:
                        st.error(f"❌ {reg_msg}")
