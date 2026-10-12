#!/usr/bin/env python3
"""
test_alert_gating.py — Unit test for 48-Hour Disease Weather Alert Gating.
Loops over all 23 classes across Paddy (10), Arecanut (9), and Apple (4),
and asserts that compute_48h_disease_risk produces an alert for EXACTLY 3 classes:
  1. Apple -> apple_scab
  2. Paddy -> blast
  3. Arecanut -> mahali_koleroga
For all other 20 classes (including hispa, tungro, black_rot, cedar_apple_rust,
bud_borer, yellow_leaf_disease, healthy foliage classes), it asserts:
  - compute_48h_disease_risk returns None (no weather alert)
  - No weather sentence is added to voice advisory
"""

import os
import sys
import unittest
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from weather_live import compute_48h_disease_risk
from app import PADDY_DISPLAY, ARECA_DISPLAY, APPLE_DISPLAY

class TestAlertGating(unittest.TestCase):
    def setUp(self):
        # Synthetic 48h weather dataset designed to trigger all conditions:
        # T=22°C (20-30°C for blast, 6-24°C for scab), RH=95% (>85% scab, >90% blast/koleroga), Precip=1.5 mm/h (>0.1 scab, >0.5 koleroga)
        self.mock_forecast = {
            "hourly": {
                "temperature_2m": [22.0] * 48,
                "relative_humidity_2m": [95.0] * 48,
                "precipitation": [1.5] * 48,
                "soil_moisture_0_to_1cm": [0.35] * 48,
            }
        }
        self.crop_class_map = {
            "Paddy": list(PADDY_DISPLAY.keys()),
            "Arecanut": list(ARECA_DISPLAY.keys()),
            "Apple": list(APPLE_DISPLAY.keys()),
        }

    def test_alert_gating_all_23_classes(self):
        expected_allowed = {
            ("Apple", "apple_scab"),
            ("Paddy", "blast"),
            ("Arecanut", "mahali_koleroga"),
        }

        total_classes_checked = 0
        actual_allowed = set()

        print("\n" + "=" * 80)
        print("RUNNING 48-HOUR ALERT GATING UNIT TEST ACROSS ALL 23 CLASSES")
        print("=" * 80)

        for crop, class_list in self.crop_class_map.items():
            print(f"\n--- Checking Crop: {crop} ({len(class_list)} classes) ---")
            for raw_cls in class_list:
                total_classes_checked += 1
                disp_name = (
                    PADDY_DISPLAY.get(raw_cls) if crop == "Paddy"
                    else ARECA_DISPLAY.get(raw_cls) if crop == "Arecanut"
                    else APPLE_DISPLAY.get(raw_cls)
                )

                alert_res = compute_48h_disease_risk(
                    forecast_data=self.mock_forecast,
                    crop=crop,
                    disease_name=disp_name,
                    pred_cls=raw_cls
                )

                if (crop, raw_cls) in expected_allowed:
                    self.assertIsNotNone(alert_res, f"Expected alert for ({crop}, {raw_cls}) but got None!")
                    self.assertGreater(alert_res["matched_count"], 0)
                    self.assertIn("voice_alerts", alert_res)
                    actual_allowed.add((crop, raw_cls))
                    print(f"  [PASS - ALERT ACTIVE]  {crop:<8} -> {raw_cls:<25} ({alert_res['rule_name']}): Matched {alert_res['matched_count']}h | Risk: {alert_res['risk_cat']}")
                else:
                    self.assertIsNone(alert_res, f"Expected NO alert for ({crop}, {raw_cls}) but got: {alert_res}")
                    print(f"  [PASS - NO ALERT]      {crop:<8} -> {raw_cls:<25} : Successfully blocked (None)")

        self.assertEqual(total_classes_checked, 23, f"Expected 23 classes, checked {total_classes_checked}")
        self.assertEqual(actual_allowed, expected_allowed, "Alert did not trigger for exactly the 3 expected classes")
        print("\n" + "=" * 80)
        print(f"UNIT TEST RESULT: Successfully verified {total_classes_checked}/23 classes.")
        print(f"Exactly {len(actual_allowed)} classes triggered weather alerts: {sorted(list(actual_allowed))}")
        print("All remaining 20 classes (including hispa, tungro, black_rot, cedar_apple_rust, bud_borer, YLD, healthy) strictly blocked.")
        print("=" * 80 + "\n")

    def test_blast_phrasing_wording(self):
        # Specifically verify paddy blast uses "before conditions worsen"
        alert_res = compute_48h_disease_risk(
            forecast_data=self.mock_forecast,
            crop="Paddy",
            disease_name="Rice Blast",
            pred_cls="blast"
        )
        self.assertIsNotNone(alert_res)
        self.assertIn("before conditions worsen", alert_res["alert_text"])
        self.assertIn("before conditions worsen", alert_res["voice_alerts"]["en"])
        self.assertIn("ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ", alert_res["voice_alerts"]["kn"])
        self.assertIn("स्थिति बिगड़ने से पहले", alert_res["voice_alerts"]["hi"])
        self.assertNotIn("before the wet spell", alert_res["alert_text"])
        print("[PASS] Verified Paddy Blast uses 'before conditions worsen' in EN, KN, HI.")


if __name__ == "__main__":
    unittest.main()
