#!/usr/bin/env python3
"""
download_models.py — Downloads deployed model checkpoints for AgriTrust AI.

Downloads the trained model weights and OOD guard parameters required by app.py
and ood_guard.py from a remote repository into outputs/models/ and outputs/,
with strict SHA-256 checksum verification, download progress, resume support,
and 3 retry attempts on failure.
"""

import os
import sys
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from typing import Dict, Any

# Default base URL pointing to HuggingFace repository (overridable with --url or MODEL_URL env var)
DEFAULT_MODEL_URL = "https://huggingface.co/sahith77/agritrust/tree/main"
MODEL_BASE_URL = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)

# Deployed checkpoints and parameter artifacts required by app.py & ood_guard.py
# (Computed via sha256sum and stat -c %s from pristine trained checkpoints)
DEPLOYED_MODELS: Dict[str, Dict[str, Any]] = {
    "paddy_disease_classifier.pth": {
        "dest": "outputs/models/paddy_disease_classifier.pth",
        "size_bytes": 44804363,
        "sha256": "0e83d174f115565b3690109d5156c80450c3ea4354aa39892fd5a0b8c2639dcf",
        "description": "Paddy ResNet-18 disease classifier (10 classes)",
    },
    "paddy_ordinal.pth": {
        "dest": "outputs/models/paddy_ordinal.pth",
        "size_bytes": 44792203,
        "sha256": "a80b6291724a12ddf34cc508f748360ec57d7ae742850bec0e427f8f8b133b42",
        "description": "Paddy ResNet-18 severity model (Ordinal CORN)",
    },
    "paddy_baseline.pth": {
        "dest": "outputs/models/paddy_baseline.pth",
        "size_bytes": 44794379,
        "sha256": "9288ec099c54e50a3dd01dc379d61ae509ebd0af628b2b94c676e35ae432617c",
        "description": "Paddy ResNet-18 baseline severity model (Standard Softmax)",
    },
    "arecanut_disease_classifier.pth": {
        "dest": "outputs/models/arecanut_disease_classifier.pth",
        "size_bytes": 44802827,
        "sha256": "caf2267364aa1750465d2dcf4e528afc1fb6ffeb202cb2be5a884e0c40c68da8",
        "description": "Arecanut ResNet-18 disease classifier (9 classes)",
    },
    "arecanut_ordinal.pth": {
        "dest": "outputs/models/arecanut_ordinal.pth",
        "size_bytes": 44792587,
        "sha256": "35d9c2508b85bdf738949a23a8c7ffcea468531ea6ddb1f735805a1999e0b1e6",
        "description": "Arecanut ResNet-18 severity model (Ordinal CORN)",
    },
    "arecanut_baseline.pth": {
        "dest": "outputs/models/arecanut_baseline.pth",
        "size_bytes": 44794763,
        "sha256": "8ca9aa3defe19d7fa27c6b9c9268c8b272bc35462a20626e8b64835debc51644",
        "description": "Arecanut ResNet-18 baseline severity model (Standard Softmax)",
    },
    "apple_disease_classifier.pth": {
        "dest": "outputs/models/apple_disease_classifier.pth",
        "size_bytes": 44794251,
        "sha256": "047575112bc1d1a4887dbcd875c19dcfc5f5e7236363920638622491eee54a7a",
        "description": "Apple ResNet-18 disease classifier (4 classes)",
    },
    "apple_ordinal.pth": {
        "dest": "outputs/models/apple_ordinal.pth",
        "size_bytes": 44792267,
        "sha256": "8615a23a73e0fea5b5abb13cfde4f5ff2cb156ebb00c53859edfacff018d462a",
        "description": "Apple ResNet-18 severity model (Ordinal CORN)",
    },
    "ood_params.pt": {
        "dest": "outputs/ood_params.pt",
        "size_bytes": 3226681,
        "sha256": "83170bbc2e67729786291eb12f71632e57f50b759df3666cc90a4163db22593c",
        "description": "Mahalanobis OOD guard calibrated class means and shrinkage precision",
    },
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_file(filepath: str, expected_sha: str, expected_size: int) -> bool:
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) != expected_size:
        return False
    return compute_sha256(filepath).lower() == expected_sha.lower()


def normalize_download_url(base_url: str, filename: str) -> str:
    cleaned = base_url.rstrip("/")
    if "huggingface.co" in cleaned and "/tree/" in cleaned:
        cleaned = cleaned.replace("/tree/", "/resolve/")
    return f"{cleaned}/{filename}"


def download_file_with_resume(url: str, dest_path: str, expected_size: int, max_retries: int = 3) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    tmp_path = dest_path + ".tmp"

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = 0
            mode = "wb"
            headers = {"User-Agent": "AgriTrust-Downloader/1.0"}

            if os.path.exists(tmp_path):
                existing_bytes = os.path.getsize(tmp_path)
                if 0 < existing_bytes < expected_size:
                    headers["Range"] = f"bytes={existing_bytes}-"
                    downloaded = existing_bytes
                    mode = "ab"
                elif existing_bytes >= expected_size:
                    os.remove(tmp_path)
                    downloaded = 0

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, mode) as f:
                if resp.status == 200 and mode == "ab":
                    f.seek(0)
                    f.truncate(0)
                    downloaded = 0

                block_size = 1024 * 128
                last_update = 0

                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_update > 0.25 or downloaded >= expected_size:
                        pct = (downloaded / expected_size) * 100 if expected_size else 0.0
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = expected_size / (1024 * 1024)
                        bar_len = 25
                        filled = int(bar_len * downloaded / expected_size) if expected_size else 0
                        bar = "=" * filled + ("-" * (bar_len - filled))
                        sys.stdout.write(f"\r  Progress: [{bar}] {pct:5.1f}% ({mb_done:5.1f} / {mb_total:5.1f} MB)")
                        sys.stdout.flush()
                        last_update = now

            sys.stdout.write("\n")
            sys.stdout.flush()

            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.replace(tmp_path, dest_path)
            return True

        except Exception as e:
            sys.stdout.write("\n")
            sys.stdout.flush()
            print(f"  [WARN] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
            else:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Download and verify deployed AgriTrust AI model checkpoints.")
    parser.add_argument("--url", default=MODEL_BASE_URL, help=f"Base URL for downloads (default: {MODEL_BASE_URL})")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files, do not download")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print("=" * 80)
    print("AGRITRUST AI — MODEL CHECKPOINT DOWNLOADER & VERIFIER")
    print(f"Base URL: {base_url}")
    print(f"Mode: {'Verify Only' if args.verify_only else 'Download & Verify'}")
    print("=" * 80)

    total_bytes = sum(m["size_bytes"] for m in DEPLOYED_MODELS.values())
    print(f"Manifest: {len(DEPLOYED_MODELS)} deployed checkpoints ({total_bytes / (1024*1024):.2f} MB total)")

    all_passed = True
    for filename, meta in DEPLOYED_MODELS.items():
        dest = meta["dest"]
        expected_size = meta["size_bytes"]
        expected_sha = meta["sha256"]
        desc = meta["description"]
        size_mb = expected_size / (1024 * 1024)

        print(f"\n[{filename}] ({size_mb:.2f} MB)")
        print(f"  Description: {desc}")
        print(f"  Destination: {dest}")

        if os.path.exists(dest):
            if verify_file(dest, expected_sha, expected_size):
                print(f"  [OK] File exists and matches expected SHA-256 ({expected_sha[:16]}...).")
                continue
            else:
                print(f"  [WARN] File exists but checksum/size mismatch! Re-downloading.")
                if args.verify_only:
                    all_passed = False
                    continue

        if args.verify_only:
            print(f"  [MISSING] File not found.")
            all_passed = False
            continue

        download_url = normalize_download_url(base_url, filename)
        print(f"  Downloading from: {download_url}")
        success = download_file_with_resume(download_url, dest, expected_size, max_retries=3)

        if success:
            if verify_file(dest, expected_sha, expected_size):
                print(f"  [SUCCESS] Downloaded and verified SHA-256 ({expected_sha[:16]}...).")
            else:
                print(f"  [ERROR] Verification failed for {dest}! Checksum mismatch. Deleting file.")
                print(f"  [FAILING URL] {download_url}")
                if os.path.exists(dest):
                    os.remove(dest)
                all_passed = False
        else:
            print(f"  [ERROR] Failed to download {filename} after 3 attempts.")
            print(f"  [FAILING URL] {download_url}")
            if os.path.exists(dest):
                os.remove(dest)
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL DEPLOYED CHECKPOINTS VERIFIED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("VERIFICATION INCOMPLETE OR FAILED: One or more checkpoints are missing or invalid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
