"""
auth.py - Simple SQLite User Authentication Engine for AgriTrust AI
Provides secure local username & password authentication with unique username enforcement.
"""

import os
import sqlite3
import hashlib
import datetime
from typing import Tuple, Dict, Any, Optional

DB_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "contrib", "contrib.db")

# Fixed salt for demo hashing (SHA-256 + salt)
PASSWORD_SALT = "agritrust_ai_secure_salt_2026"


def hash_password(password: str) -> str:
    """Hashes a password with a static project salt."""
    salted = f"{PASSWORD_SALT}:{password.strip()}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def init_user_db(db_path: str = DB_PATH_DEFAULT) -> None:
    """Ensures the users table exists in SQLite database."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT DEFAULT 'farmer'
            )
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username COLLATE NOCASE);")
    conn.close()


def register_user(
    username: str,
    password: str,
    role: str = "farmer",
    db_path: str = DB_PATH_DEFAULT
) -> Tuple[bool, str]:
    """
    Registers a new user with username and password.
    Enforces uniqueness (case-insensitive) and basic password validation.
    """
    init_user_db(db_path)
    u = username.strip()
    p = password.strip()

    if not u:
        return False, "Username cannot be empty."
    if len(u) < 3:
        return False, "Username must be at least 3 characters long."
    if not p:
        return False, "Password cannot be empty."
    if len(p) < 4:
        return False, "Password must be at least 4 characters long."

    pw_hash = hash_password(p)
    now_ts = datetime.datetime.now().isoformat()

    conn = sqlite3.connect(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, created_at, role)
                VALUES (?, ?, ?, ?)
            """, (u, pw_hash, now_ts, role))
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, f"Username '{u}' is already taken. Please choose another username."
    except Exception as e:
        return False, f"Database error during registration: {str(e)}"
    finally:
        conn.close()


def verify_user(
    username: str,
    password: str,
    db_path: str = DB_PATH_DEFAULT
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates a user against SQLite.
    Returns (success, user_dict, error_message).
    """
    init_user_db(db_path)
    u = username.strip()
    p = password.strip()

    if not u or not p:
        return False, None, "Please provide both username and password."

    pw_hash = hash_password(p)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, password_hash, created_at, role
            FROM users
            WHERE username = ? COLLATE NOCASE
        """, (u,))
        row = cursor.fetchone()

        if not row:
            return False, None, "Username not found. Please sign up first."

        if row["password_hash"] != pw_hash:
            return False, None, "Incorrect password. Please try again."

        user_dict = {
            "id": row["id"],
            "username": row["username"],
            "created_at": row["created_at"],
            "role": row["role"]
        }
        return True, user_dict, "Login successful!"
    except Exception as e:
        return False, None, f"Database error during login: {str(e)}"
    finally:
        conn.close()


def get_user(username: str, db_path: str = DB_PATH_DEFAULT) -> Optional[Dict[str, Any]]:
    """Retrieves user profile by username."""
    init_user_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, created_at, role FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "username": row["username"],
                "created_at": row["created_at"],
                "role": row["role"]
            }
        return None
    finally:
        conn.close()


def require_auth(current_page: str = "app") -> bool:
    """
    Enforces authentication across all app pages.
    If the user is not authenticated:
    1. Attempts to switch to pages/1_Login_and_Signup.py.
    2. Renders an authentication wall card and calls st.stop() to halt any page logic or data leakage.
    Returns True if user is logged in.
    """
    import streamlit as st

    if st.session_state.get("is_logged_in", False) and st.session_state.get("user"):
        return True

    # Not logged in: Redirect or display access restriction
    try:
        st.switch_page("pages/1_Login_and_Signup.py")
    except Exception:
        pass

    # Fallback gate rendering in case switch_page is not executed
    st.markdown("""
    <div style="background: #FFFFFF; border: 2px solid #B5483A; border-radius: 16px; padding: 28px; max-width: 600px; margin: 2rem auto; text-align: center; box-shadow: 0 4px 20px rgba(181, 72, 58, 0.08);">
        <div style="font-size: 42px; margin-bottom: 8px;">🔒</div>
        <h2 style="color: #B5483A; margin: 0 0 8px 0; font-size: 22px;">Authentication Required</h2>
        <p style="color: #556B5C; font-size: 14.5px; line-height: 1.5; margin: 0 0 20px 0;">
            You must <strong>Sign In</strong> or <strong>Create an Account</strong> before accessing AgriTrust AI clinical crop pathology tools, bounty wallets, and regional disease surveillance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔑 Proceed to Login / Sign Up Page", type="primary", use_container_width=True):
        try:
            st.switch_page("pages/1_Login_and_Signup.py")
        except Exception:
            pass
    st.stop()
    return False

