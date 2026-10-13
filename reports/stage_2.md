# Stage 2 Report: Community Upload Pool, Bounty Wallet & Outbreak Map

## 1. Overview and Architecture
Stage 2 implements a crowdsourced data contribution pool, simulated bounty wallet, and spatial outbreak surveillance system entirely in dedicated modular files without modifying `app.py`:
- **`contrib/store.py`**: Local SQLite database (`contrib/contrib.db`) and validation engine.
- **`pages/2_Contribute_and_Earn.py`**: Streamlit contribution portal with automated gating, bounty wallet, and administrator review dashboard.
- **`pages/3_Outbreak_Map.py`**: District-level spatial disease mapping and aggregated outbreak tables.
- **`tests/test_contrib.py`**: Unit test suite covering privacy, split leakage prevention, rate limiting, and deduplication.

Both Streamlit pages display a prominent banner:
`⚠️ SIMULATED FEATURE — FOR RESEARCH & PROTOTYPING DEMONSTRATION ONLY. NO REAL PAYMENTS OR STATUTORY CERTIFICATIONS.`

---

## 2. Benchmark Leakage Prevention & Quality Gating

### A. Zero Evaluation Split Contamination (`assert_not_in_eval_splits`)
Every submitted image undergoes automated verification against all 15,885 benchmark images across Paddy (`outputs/paddy_splits.csv`), Arecanut (`outputs/arecanut_splits.csv`), and Apple (`outputs/apple_splits.csv`):
1. **Exact MD5 hash match:** Rejects any image whose exact MD5 matches an evaluation split image.
2. **Perceptual Hash Multi-Orientation Matching:** Computes pHash across original, horizontally-flipped, and vertically-flipped orientations. If Hamming distance to any split image is $\le 8$, submission is blocked with an informative error.
3. **Execution Time:** Vectorized `uint64` evaluation executes across all 15,885 split signatures in under 35 ms.

### B. Image Sharpness Justification (Variance of Laplacian)
Quality gating rejects unfocused, blurry photos where pathology features cannot be discerned. A threshold of $\text{Var}(\text{Laplacian}) \ge 30.0$ was established and validated across 20 sample images:

| Index | Sample Description | Variance of Laplacian | Gating Verdict |
| :--- | :--- | :--- | :--- |
| 1 | Field Blurry Sample (`tests/samples/blurry_leaf.jpg`) | 1.80 | ❌ **REJECT (Blurry)** |
| 2 | Paddy: bacterial_leaf_blight (`104605.jpg`) | 536.88 | **PASS (Sharp)** |
| 3 | Paddy: bacterial_leaf_blight (`103276.jpg`) | 2730.22 | **PASS (Sharp)** |
| 4 | Paddy: bacterial_leaf_blight (`101918.jpg`) | 818.66 | **PASS (Sharp)** |
| 5 | Paddy: bacterial_leaf_blight (`100998.jpg`) | 2804.65 | **PASS (Sharp)** |
| 6 | Paddy: bacterial_leaf_blight (`108526.jpg`) | 2819.33 | **PASS (Sharp)** |
| 7 | Paddy: bacterial_leaf_blight (`109067.jpg`) | 4919.00 | **PASS (Sharp)** |
| 8 | Arecanut: healthy_nut (`Healthy_Nut_original_45.jpg`) | 43.68 | **PASS (Sharp)** |
| 9 | Arecanut: healthy_nut (`Healthy_Nut_original_40.jpg`) | 255.39 | **PASS (Sharp)** |
| 10 | Arecanut: healthy_nut (`Healthy_Nut_original_49.jpg`) | 39.90 | **PASS (Sharp)** |
| 11 | Arecanut: healthy_nut (`Healthy_Nut_original_43.jpg`) | 129.92 | **PASS (Sharp)** |
| 12 | Arecanut: healthy_nut (`Healthy_Nut_original_40.jpg`) | 163.88 | **PASS (Sharp)** |
| 13 | Arecanut: healthy_nut (`Healthy_Nut_original_47.jpg`) | 45.29 | **PASS (Sharp)** |
| 14 | Apple: apple_scab (`892da302...JPG`) | 1228.11 | **PASS (Sharp)** |
| 15 | Apple: apple_scab (`51530f6a...JPG`) | 59.61 | **PASS (Sharp)** |
| 16 | Apple: apple_scab (`c7a9601c...JPG`) | 5501.20 | **PASS (Sharp)** |
| 17 | Apple: apple_scab (`f887b059...JPG`) | 6660.46 | **PASS (Sharp)** |
| 18 | Apple: apple_scab (`61373045...JPG`) | 671.86 | **PASS (Sharp)** |
| 19 | Synthetic Gaussian Blur ($k=15$) | 7.37 | ❌ **REJECT (Blurry)** |
| 20 | Synthetic Gaussian Blur ($k=31$) | 2.67 | ❌ **REJECT (Blurry)** |

*Justification*: Genuine field leaves maintain high spatial frequency details (variances between 39.9 and 6660.5). Out-of-focus optics and heavy blur yield variances $\le 7.4$. A threshold of $30.0$ strictly excludes blurred photos while safely admitting focused leaf imagery.

### C. Automated Policy Enforcement
- **Consent:** Submissions without explicit consent for sharing district-level location are blocked immediately.
- **Daily Upload Cap:** Farmers are restricted to 10 uploads per 24-hour rolling window.
- **Duplicate & Near-Duplicate Handling:** Identical or near-identical images (pHash distance $\le 8$) are rejected. Images with $8 < \text{pHash distance} \le 16$ are accepted but assigned to the same `cluster_id` as the precursor submission.
- **Storage Isolation:** Contributed photos are stored exclusively in `contributed_pool/<crop>/` and never placed in `data/` or `outputs/`.

---

## 3. Bounty Wallet & Outbreak Surveillance

- **Simulated Bounty Schedule:**
  - Standard disease classes: ₹1.00
  - Rare disease bonus (`bud_borer`, `stem_cracking`, `cedar_apple_rust`, `downy_mildew`, `bacterial_leaf_streak`): ₹10.00
  - Bounties are credited to verified balance only upon agronomist approval in the Admin portal; rejected submissions pay ₹0.00.
  - Payout threshold: ₹50.00 (simulated).
- **Admin Verification Portal:**
  - Protected behind the `ADMIN_PASS` environment variable (no default credentials in code; documented in `README.md`).
  - Allows reviewing submitted photos, claimed disease label, and optional relabeling. Logs `verified_by` and UTC timestamp.
- **Spatial Outbreak Map:**
  - Aggregates verified reports by district and disease class using district centroid coordinates from `weather_live.DISTRICT_COORDINATES`.
  - Toggle for synthetic demo data (`contrib/simulated_seed_data.json`) defaults to OFF with an explicit banner when activated.
  - **Strict Privacy Guarantee:** Never exposes `farmer_id`, filenames, timestamps, or sub-district coordinates.

---

## 4. Verification and Test Logs

### A. Unit Tests (`tests/test_contrib.py`)
```
test_assert_not_in_eval_splits (tests.test_contrib.TestContribStore.test_assert_not_in_eval_splits)
Evaluating an exact copy or flipped copy of an evaluation split image must raise ValueError. ... ok
test_consent_required (tests.test_contrib.TestContribStore.test_consent_required)
Submission must be blocked if consent is False. ... ok
test_daily_cap (tests.test_contrib.TestContribStore.test_daily_cap)
Maximum 10 uploads per farmer within 24 hours. ... ok
test_duplicate_rejection (tests.test_contrib.TestContribStore.test_duplicate_rejection)
Duplicate submissions by MD5 or perceptual hash <= 8 must be rejected. ... ok
test_privacy_guarantee (tests.test_contrib.TestContribStore.test_privacy_guarantee)
Outbreak surveillance records must strictly never expose farmer_id, filenames or timestamps. ... ok
test_reward_only_on_verification (tests.test_contrib.TestContribStore.test_reward_only_on_verification)
Bounties are credited only after administrator verification; rejected items pay 0. ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.474s

OK
```

### B. Headless AppTest Verification
```
Testing AppTest on pages/2_Contribute_and_Earn.py...
PASS: pages/2_Contribute_and_Earn.py loaded cleanly!
Testing AppTest on pages/3_Outbreak_Map.py...
PASS: pages/3_Outbreak_Map.py loaded cleanly!
```
- Git status confirms `contributed_pool/` and `contrib/contrib.db` are strictly ignored by `.gitignore`.
- Zero modifications were made to `app.py`.
