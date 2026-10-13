# Stage 1 Report: Multi-Crop Cross-Crop OOD Guard

## 1. Overview and Implementation
In Stage 1, a multi-crop cross-crop guard was implemented in `app.py` and `ui_theme.py`.
For every uploaded image, the OOD Mahalanobis distance score divided by the crop-specific threshold ($r = \text{score} / \text{threshold}$) is evaluated across all three supported crops (`Paddy`, `Arecanut`, `Apple`).

### Guard Logic:
- Let $r_{\text{sel}}$ be the ratio for the user's selected crop mode.
- Let $r_{\text{min\_other}} = \min_{c \neq \text{sel}} r_c$, and $c_{\text{suggested}} = \arg\min_{c \neq \text{sel}} r_c$.
- If all three crops have $r > 1.0$, the image is strictly non-crop OOD; the existing standard OOD refusal card is displayed.
- Otherwise, the cross-crop refusal card is triggered if:
  1. $r_{\text{sel}} > 1.0$ AND $r_{\text{min\_other}} < 1.0$, OR
  2. $r_{\text{sel}} - r_{\text{min\_other}} \ge 0.25$.
- When triggered:
  - Diagnosis, severity grading, advisory plan, weather evaluation, soil health card, audio voice generation, and chat context are suppressed.
  - A calm grey refusal card is rendered:
    - Header: `"Crop Type Discrepancy" / "ಬೆಳೆ ಆಯ್ಕೆ ವ್ಯತ್ಯಾಸ" / "फसल चयन में विसंगति"`
    - Message: `"This looks more like a {other_crop} photo. Check your crop selection." / "ಇದು {other_crop} ಚಿತ್ರದಂತೆ ಕಾಣುತ್ತಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಬೆಳೆ ಆಯ್ಕೆಯನ್ನು ಪರಿಶೀಲಿಸಿ." / "यह {other_crop} की तस्वीर जैसी लग रही है। कृपया अपनी फसल के चयन की जाँच करें।"`
  - Flagged for native-speaker audit in `native_review_todo.md` (Section 12).

---

## 2. Evaluation Results (Unrounded)

### A. Wrong-Crop Mode Evaluation (200 test images per crop across 2 wrong modes)

| True Crop | Selected Mode | N | Old OOD Only Block Rate | Cross-Crop Refusal Rate | Total Blocked Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Paddy** | Arecanut | 200 | 13.5000% (27/200) | 23.5000% (47/200) | **23.5000%** (47/200) |
| **Paddy** | Apple | 200 | 100.0000% (200/200) | 100.0000% (200/200) | **100.0000%** (200/200) |
| **Arecanut** | Paddy | 200 | 87.0000% (174/200) | 82.5000% (165/200) | **91.0000%** (182/200) |
| **Arecanut** | Apple | 200 | 100.0000% (200/200) | 91.5000% (183/200) | **100.0000%** (200/200) |
| **Apple** | Paddy | 200 | 96.0000% (192/200) | 94.0000% (188/200) | **97.0000%** (194/200) |
| **Apple** | Arecanut | 200 | 69.5000% (139/200) | 73.5000% (147/200) | **76.5000%** (153/200) |

*Observations*:
- Total block rate for Paddy tested in Arecanut mode increased from 13.5000% to 23.5000% (+10.0000%).
- Total block rate for Arecanut tested in Paddy mode increased from 87.0000% to 91.0000% (+4.0000%).
- Total block rate for Apple tested in Arecanut mode increased from 69.5000% to 76.5000% (+7.0000%).

---

### B. Non-Crop Disk Images (60 images tested across all 3 modes)

| Selected Mode | N | Old OOD Only Block Rate | Cross-Crop Refusal Rate | Total Blocked Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Paddy** | 60 | 100.0000% (60/60) | 60.0000% (36/60) | **100.0000%** (60/60) |
| **Arecanut** | 60 | 40.0000% (24/60) | 0.0000% (0/60) | **40.0000%** (24/60) |
| **Apple** | 60 | 100.0000% (60/60) | 60.0000% (36/60) | **100.0000%** (60/60) |

*Note*: Non-crop images continue to be blocked at 100% in Paddy and Apple modes, and 40% in Arecanut mode (identical to the underlying OOD guard baseline).

---

### C. Correct-Crop Test Images (False Block Rate per crop)

| Crop | Total Test N | Old OOD Block Rate | Cross-Crop False Block Rate | Total Blocked Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Paddy** | 1,507 | 5.2422% (79/1,507) | **4.6450%** (70/1,507) | 5.5740% (84/1,507) |
| **Arecanut** | 486 | 6.9959% (34/486) | **0.2058%** (1/486) | 6.9959% (34/486) |
| **Apple** | 476 | 3.5714% (17/476) | **0.6303%** (3/476) | 3.5714% (17/476) |

*Compliance with Stop Condition*:
- Paddy Cross-Crop False Block Rate: **$4.6450\% \le 5.0\%$** (PASS)
- Arecanut Cross-Crop False Block Rate: **$0.2058\% \le 5.0\%$** (PASS)
- Apple Cross-Crop False Block Rate: **$0.6303\% \le 5.0\%$** (PASS)
- Since no crop's cross-crop false block rate exceeds $5.0\%$, the stop condition did not trigger.
- *Margin Sensitivity Analysis*: Testing with margin = 0.30 yielded Paddy false block = 2.4552%, Arecanut = 0.2058%, Apple = 0.4202%. Margin = 0.25 was retained as prescribed by the specification since it satisfies the $\le 5.0\%$ threshold.

---

## 3. Verification Suite Results

1. **Python Syntax Compilation**:
   - `python3 -m py_compile app.py ui_theme.py`: **0 errors** (Exit code 0).
2. **Tab Check**:
   - `grep -P '\t' app.py | wc -l`: **0** (Strictly space indentation).
3. **Treatment Database Integrity**:
   - `tests/test_treatments.py`: **96/96** lookups verified (100% coverage, 0 fallbacks).
4. **48-Hour Alert Gating**:
   - `tests/test_alert_gating.py`: **23/23** classes verified. Exactly 3 active (`blast`, `mahali_koleroga`, `apple_scab`), 20 strictly gated.
5. **Real App Execution Flows (`scripts/dev/run_real_flows.py`)**:
   - Paddy Blast (Mangaluru): `⏱️ 48-Hour Alert (Moderate Risk): Blast-favourable conditions expected in about 3 hours (11 hours of favourable weather in the next 48 h). Apply a protective measure before conditions worsen, as advised by your KVK.`
   - Apple Scab (Srinagar): `⏱️ 48-Hour Alert (High Risk): Scab-favourable conditions expected in about 16 hours (23 hours of favourable weather in the next 48 h). Apply a protective measure before the wet spell, as advised by your KVK.`
   - Diagnosis, alert strings, and voice texts match the baseline execution runs.
6. **Streamlit AppTest Smoke Tests**:
   - Tested headless across `en`, `kn`, and `hi` modes: All runs completed with `exception == None`.
