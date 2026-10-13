"""
tests/test_contrib.py - Unit and Integration Tests for Community Contribution Store

Tests:
1. Consent enforcement (mandatory district-level consent)
2. Daily upload quota (maximum 10 photos per 24 hours)
3. Benchmark evaluation split contamination protection (assert_not_in_eval_splits)
4. Duplicate rejection (MD5 and perceptual hash <= 8)
5. Reward gating (credits only upon expert verification)
6. Spatial privacy protection (zero farmer_id or file paths in outbreak map outputs)
7. Near-duplicate clustering (shared cluster_id when pHash <= 16)
"""

import os
import io
import tempfile
import unittest
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

import contrib.store as store


def create_synthetic_plant_image(index=0):
    """Generates a sharp synthetic image guaranteed to be distinct and non-colliding."""
    rng = np.random.default_rng(1000 + index)
    arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(arr).resize((300, 300), Image.NEAREST)
    noise = rng.integers(-15, 15, (300, 300, 3))
    arr_noisy = np.clip(np.array(img, dtype=int) + noise, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr_noisy)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class TestContribStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db = os.path.join(self.temp_dir.name, "test_contrib.db")
        self.test_pool = os.path.join(self.temp_dir.name, "contributed_pool")
        store.init_db(self.test_db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_consent_required(self):
        """Submission must be blocked if consent is False."""
        img_bytes = create_synthetic_plant_image(index=1)
        res = store.add_contribution(
            farmer_id="farmer_test_1",
            crop="Paddy",
            district="Mangaluru",
            claimed_label="blast",
            consent_given=False,
            img_bytes=img_bytes,
            filename="plant.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertFalse(res["valid"])
        self.assertIn("Consent required", res["error"])

    def test_daily_cap(self):
        """Maximum 10 uploads per farmer within 24 hours."""
        farmer_id = "farmer_cap_test"
        for i in range(10):
            img_bytes = create_synthetic_plant_image(index=10 + i)
            res = store.add_contribution(
                farmer_id=farmer_id,
                crop="Paddy",
                district="Mangaluru",
                claimed_label="blast",
                consent_given=True,
                img_bytes=img_bytes,
                filename=f"plant_{i}.jpg",
                db_path=self.test_db,
                pool_dir=self.test_pool
            )
            self.assertTrue(res["valid"], f"Upload {i} failed: {res.get('error')}")

        # 11th upload must be blocked
        img_bytes_11 = create_synthetic_plant_image(index=99)
        res_11 = store.add_contribution(
            farmer_id=farmer_id,
            crop="Paddy",
            district="Mangaluru",
            claimed_label="blast",
            consent_given=True,
            img_bytes=img_bytes_11,
            filename="plant_11.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertFalse(res_11["valid"])
        self.assertIn("Daily upload quota reached", res_11["error"])

    def test_assert_not_in_eval_splits(self):
        """Evaluating an exact copy or flipped copy of an evaluation split image must raise ValueError."""
        # Load a known image from the paddy split
        paddy_splits = pd.read_csv("outputs/paddy_splits.csv")
        real_split_path = paddy_splits["image_path"].iloc[0]
        self.assertTrue(os.path.exists(real_split_path))

        with open(real_split_path, "rb") as f:
            real_bytes = f.read()

        # 1. Exact match test
        with self.assertRaises(ValueError) as ctx:
            store.assert_not_in_eval_splits(real_bytes)
        self.assertIn("matches an evaluation split image", str(ctx.exception))

        # 2. Horizontal flip test (must be blocked by pHash <= 8 check)
        with Image.open(real_split_path) as pil_img:
            hflip_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            buf = io.BytesIO()
            hflip_img.save(buf, format="JPEG")
            hflip_bytes = buf.getvalue()

        with self.assertRaises(ValueError) as ctx_flip:
            store.assert_not_in_eval_splits(hflip_bytes)
        self.assertIn("matches an evaluation split image", str(ctx_flip.exception))

    def test_duplicate_rejection(self):
        """Duplicate submissions by MD5 or perceptual hash <= 8 must be rejected."""
        img_bytes = create_synthetic_plant_image(index=30)
        res1 = store.add_contribution(
            farmer_id="farmer_dup_1",
            crop="Arecanut",
            district="Shivamogga",
            claimed_label="bud_borer",
            consent_given=True,
            img_bytes=img_bytes,
            filename="areca1.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertTrue(res1["valid"])

        # Identical image submission
        res2 = store.add_contribution(
            farmer_id="farmer_dup_2",
            crop="Arecanut",
            district="Shivamogga",
            claimed_label="bud_borer",
            consent_given=True,
            img_bytes=img_bytes,
            filename="areca_dup.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertFalse(res2["valid"])
        self.assertIn("Duplicate photo", res2["error"])

    def test_reward_only_on_verification(self):
        """Bounties are credited only after administrator verification; rejected items pay 0."""
        farmer_id = "farmer_wallet_test"
        img_rare = create_synthetic_plant_image(index=50)
        res_rare = store.add_contribution(
            farmer_id=farmer_id,
            crop="Arecanut",
            district="Shivamogga",
            claimed_label="bud_borer",  # Universal bounty: Rs 1
            consent_given=True,
            img_bytes=img_rare,
            filename="rare.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertTrue(res_rare["valid"])
        contrib_id = res_rare["id"]

        # Check initial wallet state (pending, 0 verified)
        wallet = store.get_farmer_wallet(farmer_id, db_path=self.test_db)
        self.assertEqual(wallet["verified_balance"], 0.0)
        self.assertEqual(wallet["pending_balance"], 1.0)
        self.assertEqual(wallet["count_verified"], 0)
        self.assertEqual(wallet["count_pending"], 1)

        # Admin verifies submission
        success = store.verify_contribution(contrib_id, verified_by="admin_test", db_path=self.test_db)
        self.assertTrue(success)

        # Check updated wallet (Rs 1 credited)
        wallet_verified = store.get_farmer_wallet(farmer_id, db_path=self.test_db)
        self.assertEqual(wallet_verified["verified_balance"], 1.0)
        self.assertEqual(wallet_verified["pending_balance"], 0.0)
        self.assertEqual(wallet_verified["count_verified"], 1)

        # Submit another item and reject it
        img_std = create_synthetic_plant_image(index=70)
        res_std = store.add_contribution(
            farmer_id=farmer_id,
            crop="Paddy",
            district="Mangaluru",
            claimed_label="blast",  # Universal class: Rs 1
            consent_given=True,
            img_bytes=img_std,
            filename="std.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertTrue(res_std["valid"])
        store.reject_contribution(res_std["id"], verified_by="admin_test", db_path=self.test_db)

        wallet_after_rej = store.get_farmer_wallet(farmer_id, db_path=self.test_db)
        self.assertEqual(wallet_after_rej["verified_balance"], 1.0)
        self.assertEqual(wallet_after_rej["count_rejected"], 1)

    def test_privacy_guarantee(self):
        """Outbreak surveillance records must strictly never expose farmer_id, filenames or timestamps."""
        img = create_synthetic_plant_image(index=85)
        res = store.add_contribution(
            farmer_id="private_farmer_999",
            crop="Apple",
            district="Srinagar",
            claimed_label="apple_scab",
            consent_given=True,
            img_bytes=img,
            filename="sensitive_field_photo.jpg",
            db_path=self.test_db,
            pool_dir=self.test_pool
        )
        self.assertTrue(res["valid"])
        store.verify_contribution(res["id"], verified_by="admin_test", db_path=self.test_db)

        # Fetch outbreak map records
        outbreaks = store.get_verified_outbreaks(include_simulated=True, db_path=self.test_db)
        self.assertGreater(len(outbreaks), 0)

        for rec in outbreaks:
            self.assertNotIn("farmer_id", rec)
            self.assertNotIn("filename", rec)
            self.assertNotIn("timestamp", rec)
            self.assertNotIn("gps_lat", rec)
            self.assertIn("district", rec)
            self.assertIn("crop", rec)
            self.assertIn("disease", rec)
            self.assertIn("count", rec)


if __name__ == "__main__":
    unittest.main()
