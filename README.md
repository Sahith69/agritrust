# AgriTrust AI — Multimodal Crop Disease & Severity Engine

AgriTrust AI is a Streamlit prototype (needs internet for voice and weather) designed for crop pathology diagnosis and severity assessment across **Paddy**, **Arecanut**, and **Apple**. It combines ResNet-18 vision models, ordinal disease severity grading (CORN), user-entered field/weather condition synthesis, safety guardrails, and multilingual voice advisories (English, Kannada, Hindi).

---

## 🔍 Data Leakage Audit Findings

Cross-split duplicate and near-duplicate leakage audits were performed across train, validation, and test splits (recorded in `audit_report.json` and `leakage_adjusted_report.json`):

1. **Paddy (Paddy Doctor):**
   - **Setting 2 Twin Audit (Test vs. Train+Val, Original/H-Flip/V-Flip, Hamming $\le 8$):** **51.4%** (`775 / 1507` test images flagged as twins; twin-free $N = 732$; lesion-only twin-free $N = 343$).
   - **Deployed Single-Model Evaluation (No Ensembles, No TTA, Argmax Decode):**
     - *Full Test ($N = 1507$):* Disease Accuracy `0.9655` [0.9555, 0.9735], Macro-F1 `0.9609` [0.9490, 0.9709]; CORN Severity QWK `0.9390` [0.9232, 0.9525], Exact Acc `0.9098` [0.8945, 0.9230], Within-1 `0.9881` [0.9821, 0.9934], MAE `0.1048` [0.0876, 0.1241] (Class-only baseline: QWK `0.9114`, MAE `0.1865`).
     - *Lesion-Only Full ($N = 757$):* Disease Accuracy `0.9696` [0.9564, 0.9815], Macro-F1 `0.5880` [0.5818, 0.7296]; CORN Severity QWK `0.7925` [0.7342, 0.8496], Exact Acc `0.8864` [0.8653, 0.9089], Within-1 `0.9855` [0.9762, 0.9934], MAE `0.1334` [0.1044, 0.1625] (Class-only baseline: QWK `0.6018`, MAE `0.2616`).
     - *Twin-Free Full ($N = 732$):* Disease Accuracy `0.9686` [0.9563, 0.9809], Macro-F1 `0.5879` [0.5821, 0.7262]; CORN Severity QWK `0.7869` [0.7198, 0.8441], Exact Acc `0.8839` [0.8579, 0.9057], Within-1 `0.9850` [0.9754, 0.9932], MAE `0.1366` [0.1079, 0.1694] (Class-only baseline: QWK `0.5828`, MAE `0.2705`).
     - *Lesion-Only Twin-Free ($N = 343$):* Disease Accuracy `0.9592` [0.9388, 0.9796], Macro-F1 `0.3941` [0.3912, 0.5684]; CORN Severity QWK `0.6736` [0.5602, 0.7815], Exact Acc `0.8309` [0.7901, 0.8688], Within-1 `0.9796` [0.9621, 0.9942], MAE `0.1983` [0.1487, 0.2479] (Class-only baseline: QWK `0.0000`, MAE `0.4373`).

2. **Arecanut (Field Dataset):**
   - *Arecanut checkpoint and split are under reconciliation; severity figures are provisional.*
   - **Disease Classifier Provenance:** `arecanut_disease_classifier_v2.pth` is explicitly excluded from deployment because it was trained on external leaky images (`kaggle_merged` without cross-split leakage deduplication). The deployed model is `arecanut_disease_classifier.pth` (trained strictly on clean local capture data).
   - **Timestamped Camera Burst Overlap ($\le 120$ s):** 880 total images contain camera timestamps in their filenames (528 train, 181 val, 171 test). Among timestamped test images, **70.8%** (`121 / 171`) occurred within a 120-second camera burst of a training image (**24.9%** across all 486 test images).
   - Near-duplicates at pHash $\le 8$ were 0.0%, but soft perceptual similarity (pHash $\le 16$) reached 32.1%.
   - When twin images were excluded ($N = 335$), disease classification macro-F1 was 77.0%. Severity assessment remains provisional pending complete physical cluster reconciliation.

3. **Apple (PlantVillage):**
   - **pHash Near-Duplicate Rate ($\le 8$):** **6.9%** (`33 / 476` test images).
   - **DINOv2 Feature Cosine ($\ge 0.95$):** **12.4%** (`59 / 476` test images).
   - **Combined Twin Rate:** **18.5%** (`88 / 476` test images flagged as twins).
   - On the twin-free test subset ($N = 388$), disease accuracy remained 99.7%, reflecting the homogeneous lab backdrop of the dataset.

---

## 🛡️ Out-of-Distribution (OOD) Guard Performance

An OOD guard was constructed using class-conditional Mahalanobis distance with Ledoit-Wolf shrinkage covariance fitted on penultimate ResNet-18 features of **TRAIN images only**. Thresholds were calibrated to retain 95% of in-domain **VAL images**:
- **Paddy:** Threshold = `28.265936` (In-domain test retention: 94.8%)
- **Arecanut:** Threshold = `36.752533` (In-domain test retention: 93.0%)
- **Apple:** Threshold = `31.297428` (In-domain test retention: 96.4%)

### Evaluation Summary
- **Paddy & Apple:** Strong OOD detection. Paddy rejects 100% of non-crop test images (desk, wall, person, wallpapers) and 91.3% of cross-crop images. Apple rejects 100% of non-crop test images and 100% of cross-crop images.
- **Arecanut:** Weak OOD detection. The wide feature variance of heterogeneous palm views (trunk, foot, nut, leaf) results in a broad covariance ellipsoid that rejects only **38% of non-crop images** (23 / 60) and **13.5% of paddy images** (27 / 200) in arecanut mode.

---

## ⚠️ Known Limitations

- **The app only supports close-up photos of paddy, arecanut and apple, and unsupported crops are not reliably detected.**
- **Apple Data Provenance:** The Apple disease model was trained on public laboratory dataset imagery (PlantVillage) and has not been field-validated in commercial orchards. Severity grading is suppressed accordingly.
- **Cedar Apple Rust:** Class from US public data; confirm locally with your KVK.
- **Arecanut Bud Borer / YLD Confusion:** High visual overlap exists between Arecanut Bud Borer and Yellow Leaf Disease (YLD); physical confirmation by local Krishi Vigyan Kendra (KVK) or CPCRI extension officers is advised.
- **Arecanut Reconciliation Status:** Arecanut checkpoint and split are under reconciliation; figures are provisional.
- **Explainability Limitations:** Grad-CAM saliency heatmaps indicate feature activation regions but do not guarantee precise pixel-level lesion boundary localization.
