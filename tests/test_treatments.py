#!/usr/bin/env python3
"""
test_treatments.py — Unit test verifying completeness of treatment plans in app.py's TREATMENT_DATABASE.
Verifies:
  - 92 severity plans (23 classes x 4 severity grades: 0, 1, 2, 3)
  - 4 Apple disease-level advisories ('disease_advice')
  - Total entries: exactly 96 (92 severity plans + 4 disease advisories)
  - All titles and descriptions are non-empty across all 23 classes.
"""

import sys
import unittest
sys.path.insert(0, '/home/sahith/Projects/Hackathon')

import app

class TestTreatmentDatabase(unittest.TestCase):
    def test_treatment_database_completeness(self):
        db = app.TREATMENT_DATABASE
        severity_plans_count = 0
        disease_advisories_count = 0
        total_entries = 0

        crop_counts = {}

        for crop, diseases in db.items():
            crop_counts[crop] = {"severity_plans": 0, "disease_advice": 0}
            for disease, plans in diseases.items():
                for key, val in plans.items():
                    total_entries += 1
                    self.assertIsInstance(val, tuple, f"Plan value must be a tuple for {crop} -> {disease} -> {key}")
                    self.assertEqual(len(val), 2, f"Plan must be a 2-tuple (title, desc) for {crop} -> {disease} -> {key}")
                    title, desc = val
                    self.assertTrue(len(title) > 0, f"Empty title for {crop} -> {disease} -> {key}")
                    self.assertTrue(len(desc) > 0, f"Empty description for {crop} -> {disease} -> {key}")

                    if isinstance(key, int):
                        self.assertIn(key, [0, 1, 2, 3], f"Invalid grade {key} for {crop} -> {disease}")
                        severity_plans_count += 1
                        crop_counts[crop]["severity_plans"] += 1
                    elif key == "disease_advice":
                        disease_advisories_count += 1
                        crop_counts[crop]["disease_advice"] += 1
                    else:
                        self.fail(f"Unexpected plan key {key} for {crop} -> {disease}")

        # Assertions
        self.assertEqual(severity_plans_count, 92, f"Expected exactly 92 severity plans, found {severity_plans_count}")
        self.assertEqual(disease_advisories_count, 4, f"Expected exactly 4 disease advisories, found {disease_advisories_count}")
        self.assertEqual(total_entries, 96, f"Expected exactly 96 total entries, found {total_entries}")

        # Crop-specific breakdown assertions
        self.assertEqual(crop_counts["Paddy"]["severity_plans"], 40)   # 10 classes x 4 grades
        self.assertEqual(crop_counts["Arecanut"]["severity_plans"], 36) # 9 classes x 4 grades
        self.assertEqual(crop_counts["Apple"]["severity_plans"], 16)    # 4 classes x 4 grades
        self.assertEqual(crop_counts["Apple"]["disease_advice"], 4)     # 4 Apple disease advisories

        print("\n" + "=" * 80)
        print("TREATMENT DATABASE COMPLETENESS TEST")
        print("=" * 80)
        print(f"Verified completeness for {severity_plans_count}/92 severity plans (23 classes x 4 grades 0..3):")
        print(f"  • Paddy:    {crop_counts['Paddy']['severity_plans']}/40 severity plans (10 classes)")
        print(f"  • Arecanut: {crop_counts['Arecanut']['severity_plans']}/36 severity plans (9 classes)")
        print(f"  • Apple:    {crop_counts['Apple']['severity_plans']}/16 severity plans (4 classes)")
        print(f"Verified completeness for {disease_advisories_count}/4 Apple general disease advisories.")
        print(f"Total verified database entries: {total_entries}/96 entries (100% complete, non-empty).")
        print("=" * 80)

if __name__ == "__main__":
    unittest.main()
