"""
contrib/store.py - Local Contribution Pool, SQLite Storage & Validation Engine
AgriTrust AI Multipage Extension

Features:
- SQLite table 'contributions' in contrib/contrib.db
- Isolated storage: images saved only to contributed_pool/<crop>/
- Zero data leakage: assert_not_in_eval_splits() blocks any match (MD5 or pHash <= 8 across orig/hflip/vflip)
  against the Paddy, Arecanut, and Apple benchmark evaluation splits.
- Automated quality gating: blur detection (variance of Laplacian >= 30.0), deduplication, daily quota (10/day),
  mandatory location consent.
- Near-duplicate clustering: perceptual hash distance <= 16 clusters related submissions.
- Simulated bounty wallet with rare disease bonuses (Rs 10 vs Rs 1) and Rs 50 simulated payout threshold.
"""

import os
import io
import time
import uuid
import json
import sqlite3
import hashlib
import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from PIL import Image
import imagehash

# ── Configuration & Domain Constants
DB_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "contrib.db")
POOL_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "contributed_pool")
SIGNATURES_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "eval_split_signatures.npz")
SEED_DATA_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "simulated_seed_data.json")

RARE_CLASSES = {
    "bud_borer",
    "stem_cracking",
    "cedar_apple_rust",
    "downy_mildew",
    "bacterial_leaf_streak"
}

DISTRICTS = [
    "Mangaluru",
    "Udupi",
    "Shivamogga",
    "Mysuru",
    "Srinagar",
    "Shimla",
    "Kullu",
    "Dehradun"
]

CROP_CLASSES = {
    "Paddy": [
        "bacterial_leaf_blight",
        "bacterial_leaf_streak",
        "bacterial_panicle_blight",
        "blast",
        "brown_spot",
        "dead_heart",
        "downy_mildew",
        "hispa",
        "normal",
        "tungro"
    ],
    "Arecanut": [
        "bud_borer",
        "healthy_foot",
        "healthy_leaf",
        "healthy_nut",
        "healthy_trunk",
        "mahali_koleroga",
        "stem_bleeding",
        "stem_cracking",
        "yellow_leaf_disease"
    ],
    "Apple": [
        "apple_scab",
        "black_rot",
        "cedar_apple_rust",
        "healthy"
    ]
}

BLUR_THRESHOLD = 30.0
DAILY_UPLOAD_CAP = 10
PAYOUT_THRESHOLD = 50.0
REWARD_STANDARD = 1.0
REWARD_RARE = 10.0


# ── Database Initialization
def init_db(db_path: str = DB_PATH_DEFAULT) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id TEXT NOT NULL,
                crop TEXT NOT NULL,
                claimed_label TEXT NOT NULL,
                district TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                md5 TEXT NOT NULL,
                phash TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                status TEXT NOT NULL,
                reward REAL NOT NULL,
                filename TEXT NOT NULL,
                verified_by TEXT,
                verified_at TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contrib_farmer ON contributions(farmer_id, timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contrib_md5 ON contributions(md5);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contrib_status ON contributions(status);")
    conn.close()


def get_db(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Quality & Hash Utilities
def compute_laplacian_variance(img_bytes_or_pil) -> float:
    """Compute variance of the Laplacian to evaluate image sharpness."""
    if isinstance(img_bytes_or_pil, (bytes, bytearray)):
        nparr = np.frombuffer(img_bytes_or_pil, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img_bytes_or_pil, Image.Image):
        img = cv2.cvtColor(np.array(img_bytes_or_pil), cv2.COLOR_RGB2GRAY)
    elif isinstance(img_bytes_or_pil, str) and os.path.isfile(img_bytes_or_pil):
        img = cv2.imread(img_bytes_or_pil, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported image input type for blur calculation")

    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_image_signatures(img_bytes: bytes) -> Dict[str, Any]:
    """Compute MD5 and multi-orientation perceptual hashes (orig, hflip, vflip)."""
    md5_hash = hashlib.md5(img_bytes).hexdigest()
    with Image.open(io.BytesIO(img_bytes)) as pil_img:
        pil_rgb = pil_img.convert("RGB")
        h_orig = imagehash.phash(pil_rgb)
        h_hflip = imagehash.phash(pil_rgb.transpose(Image.FLIP_LEFT_RIGHT))
        h_vflip = imagehash.phash(pil_rgb.transpose(Image.FLIP_TOP_BOTTOM))

    u_orig = np.uint64(int(str(h_orig), 16))
    u_hflip = np.uint64(int(str(h_hflip), 16))
    u_vflip = np.uint64(int(str(h_vflip), 16))

    return {
        "md5": md5_hash,
        "phash_hex": str(h_orig),
        "phash_u64": u_orig,
        "hflip_u64": u_hflip,
        "vflip_u64": u_vflip,
    }


# ── Split Leakage Prevention Engine
_SPLIT_SIGNATURES = None

def load_split_signatures(signatures_path: str = SIGNATURES_PATH_DEFAULT):
    global _SPLIT_SIGNATURES
    if _SPLIT_SIGNATURES is not None:
        return _SPLIT_SIGNATURES

    if not os.path.exists(signatures_path):
        raise FileNotFoundError(f"Signatures cache not found at {signatures_path}. Run build script.")

    data = np.load(signatures_path)
    _SPLIT_SIGNATURES = {
        "md5_set": set(data["md5s"]),
        "orig": data["orig"],
        "hflip": data["hflip"],
        "vflip": data["vflip"],
    }
    return _SPLIT_SIGNATURES


def assert_not_in_eval_splits(
    img_bytes: bytes,
    signatures_path: str = SIGNATURES_PATH_DEFAULT
) -> None:
    """
    Compare image's MD5 and pHash (orig, hflip, vflip) with distance <= 8 against all
    images listed in paddy, arecanut, and apple split CSVs.
    Raises ValueError on any match.
    """
    sigs = load_split_signatures(signatures_path)
    img_sig = compute_image_signatures(img_bytes)

    # 1. Exact MD5 check
    if img_sig["md5"] in sigs["md5_set"]:
        raise ValueError(f"Contributed image matches an evaluation split image by MD5 hash ({img_sig['md5']}). Submission rejected.")

    # 2. Multi-orientation Hamming distance check (<= 8)
    cand_hashes = [img_sig["phash_u64"], img_sig["hflip_u64"], img_sig["vflip_u64"]]
    split_arrays = [sigs["orig"], sigs["hflip"], sigs["vflip"]]

    for cand_h in cand_hashes:
        for split_arr in split_arrays:
            diffs = np.bitwise_count(split_arr ^ cand_h)
            min_dist = int(np.min(diffs))
            if min_dist <= 8:
                raise ValueError(
                    f"Contributed image matches an evaluation split image by perceptual hash "
                    f"(Hamming distance {min_dist} <= 8). Submission rejected to protect benchmark integrity."
                )


# ── Automated Validation Pipeline
def validate_contribution(
    farmer_id: str,
    crop: str,
    district: str,
    claimed_label: str,
    consent_given: bool,
    img_bytes: bytes,
    db_path: str = DB_PATH_DEFAULT,
    signatures_path: str = SIGNATURES_PATH_DEFAULT
) -> Dict[str, Any]:
    """
    Executes automated gating:
    1. Consent verification
    2. Daily rate limit (<= 10 uploads in last 24h)
    3. Image blur check (Laplacian variance >= 30.0)
    4. Benchmark evaluation split contamination check
    5. Contribution pool duplicate check (MD5 or pHash <= 8)
    6. Cluster assignment (shares cluster_id if pHash <= 16)
    """
    # 1. Mandatory consent
    if not consent_given:
        return {
            "valid": False,
            "error": "Consent required: You must consent to sharing district-level location data for outbreak mapping."
        }

    # 2. Daily rate limiting
    conn = get_db(db_path)
    now_dt = datetime.datetime.now()
    since_dt = (now_dt - datetime.timedelta(hours=24)).isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM contributions WHERE farmer_id = ? AND timestamp >= ?",
        (farmer_id, since_dt)
    )
    daily_count = cursor.fetchone()[0]
    if daily_count >= DAILY_UPLOAD_CAP:
        conn.close()
        return {
            "valid": False,
            "error": f"Daily upload quota reached: Maximum {DAILY_UPLOAD_CAP} uploads per 24 hours allowed."
        }

    # 3. Blur test
    lap_var = compute_laplacian_variance(img_bytes)
    if lap_var < BLUR_THRESHOLD:
        conn.close()
        return {
            "valid": False,
            "error": f"Image rejected for blur: Laplacian variance is {lap_var:.2f} (minimum required: {BLUR_THRESHOLD:.1f}). Please capture a well-focused photo."
        }

    # 4. Evaluation split integrity
    try:
        assert_not_in_eval_splits(img_bytes, signatures_path)
    except ValueError as e:
        conn.close()
        return {"valid": False, "error": str(e)}

    # 5. Duplicate check against existing contributions
    img_sig = compute_image_signatures(img_bytes)
    cursor.execute("SELECT id, md5, phash, cluster_id FROM contributions")
    existing_rows = cursor.fetchall()
    conn.close()

    matched_cluster = None
    for row in existing_rows:
        # Exact MD5 match
        if row["md5"] == img_sig["md5"]:
            return {
                "valid": False,
                "error": f"Duplicate photo: Matches existing contribution #{row['id']} by exact MD5 hash."
            }

        # Perceptual hash distance
        try:
            ex_u64 = np.uint64(int(row["phash"], 16))
            dist = int(np.bitwise_count(ex_u64 ^ img_sig["phash_u64"]))
            if dist <= 8:
                return {
                    "valid": False,
                    "error": f"Duplicate photo: Perceptual hash matches existing contribution #{row['id']} (distance {dist} <= 8)."
                }
            elif dist <= 16 and matched_cluster is None:
                matched_cluster = row["cluster_id"]
        except Exception:
            pass

    cluster_id = matched_cluster if matched_cluster else f"cluster_{uuid.uuid4().hex[:8]}"
    reward = REWARD_RARE if claimed_label in RARE_CLASSES else REWARD_STANDARD

    return {
        "valid": True,
        "error": None,
        "signatures": img_sig,
        "laplacian_var": lap_var,
        "cluster_id": cluster_id,
        "reward": reward
    }


# ── Record Ingestion & Admin Actions
def add_contribution(
    farmer_id: str,
    crop: str,
    district: str,
    claimed_label: str,
    consent_given: bool,
    img_bytes: bytes,
    filename: str,
    db_path: str = DB_PATH_DEFAULT,
    pool_dir: str = POOL_DIR_DEFAULT,
    signatures_path: str = SIGNATURES_PATH_DEFAULT
) -> Dict[str, Any]:
    """
    Validates, saves the image to contributed_pool/<crop>/, and logs record into SQLite.
    Images are never saved under data/, outputs/, or any split folder.
    """
    init_db(db_path)
    check_res = validate_contribution(
        farmer_id=farmer_id,
        crop=crop,
        district=district,
        claimed_label=claimed_label,
        consent_given=consent_given,
        img_bytes=img_bytes,
        db_path=db_path,
        signatures_path=signatures_path
    )

    if not check_res["valid"]:
        return check_res

    # Save image strictly inside contributed_pool/<crop>/
    crop_dir = os.path.join(pool_dir, crop.lower())
    os.makedirs(crop_dir, exist_ok=True)
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{os.path.basename(filename)}"
    save_path = os.path.join(crop_dir, safe_name)
    with open(save_path, "wb") as f:
        f.write(img_bytes)

    # Insert into database
    conn = get_db(db_path)
    ts = datetime.datetime.now().isoformat()
    sigs = check_res["signatures"]
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contributions (
                farmer_id, crop, claimed_label, district, timestamp,
                md5, phash, cluster_id, status, reward, filename,
                verified_by, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            farmer_id,
            crop,
            claimed_label,
            district,
            ts,
            sigs["md5"],
            sigs["phash_hex"],
            check_res["cluster_id"],
            "pending",
            check_res["reward"],
            save_path,
            None,
            None
        ))
        record_id = cursor.lastrowid
    conn.close()

    return {
        "valid": True,
        "id": record_id,
        "farmer_id": farmer_id,
        "crop": crop,
        "claimed_label": claimed_label,
        "district": district,
        "timestamp": ts,
        "status": "pending",
        "reward": check_res["reward"],
        "cluster_id": check_res["cluster_id"],
        "laplacian_var": check_res["laplacian_var"],
        "saved_path": save_path
    }


def verify_contribution(
    contrib_id: int,
    verified_by: str,
    corrected_label: Optional[str] = None,
    db_path: str = DB_PATH_DEFAULT
) -> bool:
    """Verifies a contribution, optionally updating the label, and credits reward."""
    conn = get_db(db_path)
    now_ts = datetime.datetime.now().isoformat()
    with conn:
        cursor = conn.cursor()
        if corrected_label:
            reward = REWARD_RARE if corrected_label in RARE_CLASSES else REWARD_STANDARD
            cursor.execute("""
                UPDATE contributions
                SET status = 'verified',
                    claimed_label = ?,
                    reward = ?,
                    verified_by = ?,
                    verified_at = ?
                WHERE id = ?
            """, (corrected_label, reward, verified_by, now_ts, contrib_id))
        else:
            cursor.execute("""
                UPDATE contributions
                SET status = 'verified',
                    verified_by = ?,
                    verified_at = ?
                WHERE id = ?
            """, (verified_by, now_ts, contrib_id))
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def reject_contribution(
    contrib_id: int,
    verified_by: str,
    db_path: str = DB_PATH_DEFAULT
) -> bool:
    """Rejects a contribution; payout reward becomes 0 upon rejection."""
    conn = get_db(db_path)
    now_ts = datetime.datetime.now().isoformat()
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE contributions
            SET status = 'rejected',
                reward = 0.0,
                verified_by = ?,
                verified_at = ?
            WHERE id = ?
        """, (verified_by, now_ts, contrib_id))
        updated = cursor.rowcount > 0
    conn.close()
    return updated


# ── Query & Wallet APIs
def get_farmer_wallet(farmer_id: str, db_path: str = DB_PATH_DEFAULT) -> Dict[str, Any]:
    """Retrieves farmer's verified wallet earnings, pending amounts, and payout threshold status."""
    conn = get_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN status = 'verified' THEN reward ELSE 0.0 END), 0.0) as verified_balance,
            COALESCE(SUM(CASE WHEN status = 'pending' THEN reward ELSE 0.0 END), 0.0) as pending_balance,
            COUNT(CASE WHEN status = 'verified' THEN 1 END) as count_verified,
            COUNT(CASE WHEN status = 'pending' THEN 1 END) as count_pending,
            COUNT(CASE WHEN status = 'rejected' THEN 1 END) as count_rejected,
            COUNT(*) as count_total
        FROM contributions
        WHERE farmer_id = ?
    """, (farmer_id,))
    row = cursor.fetchone()
    conn.close()

    v_bal = float(row["verified_balance"])
    p_bal = float(row["pending_balance"])

    return {
        "farmer_id": farmer_id,
        "verified_balance": v_bal,
        "pending_balance": p_bal,
        "count_verified": int(row["count_verified"]),
        "count_pending": int(row["count_pending"]),
        "count_rejected": int(row["count_rejected"]),
        "count_total": int(row["count_total"]),
        "payout_threshold": PAYOUT_THRESHOLD,
        "payout_eligible": (v_bal >= PAYOUT_THRESHOLD)
    }


def get_farmer_history(farmer_id: str, db_path: str = DB_PATH_DEFAULT) -> List[Dict[str, Any]]:
    """Retrieves itemized contribution history for a given farmer."""
    conn = get_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, crop, claimed_label, district, timestamp, cluster_id, status, reward, filename
        FROM contributions
        WHERE farmer_id = ?
        ORDER BY id DESC
    """, (farmer_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_pending_contributions(db_path: str = DB_PATH_DEFAULT) -> List[Dict[str, Any]]:
    conn = get_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, farmer_id, crop, claimed_label, district, timestamp, cluster_id, reward, filename
        FROM contributions
        WHERE status = 'pending'
        ORDER BY id ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_verified_outbreaks(
    include_simulated: bool = False,
    db_path: str = DB_PATH_DEFAULT,
    seed_path: str = SEED_DATA_PATH_DEFAULT
) -> List[Dict[str, Any]]:
    """
    Aggregates verified contributions by district, crop, and disease.
    Strict Privacy Rule: NEVER returns farmer_id, filenames, timestamps, or sub-district locations.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT district, crop, claimed_label as disease, COUNT(*) as count
        FROM contributions
        WHERE status = 'verified'
        GROUP BY district, crop, claimed_label
        ORDER BY count DESC
    """)
    records = []
    for r in cursor.fetchall():
        records.append({
            "district": r["district"],
            "crop": r["crop"],
            "disease": r["disease"],
            "count": int(r["count"]),
            "is_simulated": False
        })
    conn.close()

    if include_simulated and os.path.exists(seed_path):
        with open(seed_path, "r") as f:
            seeds = json.load(f)
            records.extend(seeds)

    return records


# ── Threshold Justification Utility
def justify_blur_threshold(samples: Optional[List[Tuple[str, str]]] = None) -> str:
    """Evaluates 20 sample images to justify the Laplacian variance threshold (30.0)."""
    if samples is None:
        samples = [("tests/samples/blurry_leaf.jpg", "Field Blurry Sample (Out of Focus)")]
        # Sample split images: 6 paddy, 6 arecanut, 5 apple
        for crop_file, crop_name, n_take in [
            ("outputs/paddy_splits.csv", "Paddy", 6),
            ("outputs/arecanut_splits.csv", "Arecanut", 6),
            ("outputs/apple_splits.csv", "Apple", 5)
        ]:
            if os.path.exists(crop_file):
                import pandas as pd
                df = pd.read_csv(crop_file)
                for _, r in df.head(n_take).iterrows():
                    samples.append((r["image_path"], f"{crop_name}: {r['disease_class']}"))

        # Synthetic blur
        if len(samples) > 1 and os.path.exists(samples[1][0]):
            gray = cv2.imread(samples[1][0], cv2.IMREAD_GRAYSCALE)
            k15_path = "/tmp/sample_synthetic_blur15.jpg"
            k31_path = "/tmp/sample_synthetic_blur31.jpg"
            cv2.imwrite(k15_path, cv2.GaussianBlur(gray, (15, 15), 0))
            cv2.imwrite(k31_path, cv2.GaussianBlur(gray, (31, 31), 0))
            samples.append((k15_path, "Synthetic Gaussian Blur (k=15)"))
            samples.append((k31_path, "Synthetic Gaussian Blur (k=31)"))

    samples = samples[:20]
    lines = [
        "| Index | Sample Description | Variance of Laplacian | Gating Verdict |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for idx, (path, desc) in enumerate(samples, 1):
        if os.path.exists(path):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
        else:
            var = 0.0
        verdict = "**PASS (Sharp)**" if var >= BLUR_THRESHOLD else "❌ **REJECT (Blurry)**"
        lines.append(f"| {idx} | {desc} | {var:.2f} | {verdict} |")

    table_md = "\n".join(lines)
    return table_md
