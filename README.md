# AgriTrust AI — Multimodal Crop Disease & Severity Engine

> **Research Prototype:** AgriTrust AI is an experimental multimodal decision support system for crop pathology. It is intended for research and extension demonstration, not as a replacement for certified agronomist diagnosis or statutory agricultural extension services.

AgriTrust AI is designed for crop pathology diagnosis and severity assessment across **Paddy**, **Arecanut**, and **Apple**. It combines ResNet-18 vision models, ordinal disease severity grading (CORN), user-entered field/weather condition synthesis, safety guardrails, and multilingual voice advisories (English, Kannada, Hindi).

---

## 🚀 Quick Start (4 Steps)

```bash
# 1. Clone repository
git clone https://github.com/sahith77/agritrust.git && cd agritrust

# 2. Create and activate virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Download and verify deployed model checkpoints (~345 MB)
python download_models.py

# Launch application
streamlit run app.py
```

### 🌐 System Requirements & Data Notes
- **Model Checkpoints Download Size:** ~345 MB across 9 deployed checkpoints (`outputs/models/*.pth` and `outputs/ood_params.pt`).
- **Internet Connectivity Required:** The app requires an internet connection for:
  - **Live Weather Forecasts:** 48-hour hourly bioclimatic risk projections from Open-Meteo API.
  - **Multilingual Voice Advisories:** Text-to-speech audio rendering via `gTTS`.
  - **First-Run Downloads:** Downloading deployed checkpoints (`python download_models.py`) and the DINOv2 vision transformer backbone (`dinov2_vits14`, ~85 MB) from PyTorch Hub.
- **Training Data Directory (`data/`):** The `data/` directory (21 GB) is strictly for offline model training, dataset curation, and split reproduction audits. It is **not** required for deploying, serving, or running the Streamlit app.
- **Data Source Notes & Provenance:**
  - **Paddy:** Paddy Doctor dataset (10 classes: 9 disease types + healthy).
  - **Arecanut:** 9 classes: 5 disease (`bud_borer`, `mahali_koleroga`, `stem_bleeding`, `stem_cracking`, `yellow_leaf_disease`) + 4 healthy views (`healthy_foot`, `healthy_leaf`, `healthy_nut`, `healthy_trunk`). About 71% of training images come from a merged public Kaggle collection; deployed checkpoint local-photo accuracy 81.9% ($N=155$), `bud_borer` recall 9/25.
  - **Apple:** Fine-tuned on laboratory PlantVillage benchmark and public orchard photos (FGVC / Cornell dataset, 4 classes: apple scab, black rot, cedar apple rust, healthy). Black rot was trained on lab photos only.


## 🔍 Leakage & Model Audit Summary

| Metric / Audit Slice | Value | Description & Findings |
| :--- | :--- | :--- |
| **Paddy Test Twin Rate** | **51.4% (775 / 1507)** | Near-duplicate twins (pHash $\le 8$, H/V flip) between test set and train/val sets |
| **Paddy Deployed Severity QWK (Full Test)** | **0.94** (0.9390) | Headline ordinal severity metric across entire test set ($N=1507$) |
| **Paddy Deployed Severity QWK (Twin-free Lesion Classes)** | **0.67** (0.6736) | Realistic field test performance on unique lesion-only cases ($N=343$) |
| **Class-Only Baseline QWK (Full vs. Twin-free Lesion)** | **0.91 vs. 0.00** | Predicting mode grade per disease achieves 0.91 on full test but collapses to 0.00 on twin-free lesion cases |
| **Abstain Confidence Threshold** | **0.70** | App declines to diagnose when top disease prediction confidence is below 70% |
| **Arecanut Training Kaggle Origin** | **~71%** | About 71% of Arecanut training images come from a merged public Kaggle collection |
| **Arecanut Local-Photo Accuracy** | **81.9% ($N=155$)** | Deployed checkpoint evaluation on clean local plantation photographs |
| **Arecanut Bud Borer Recall** | **9 / 25 (36.0%)** | Deployed checkpoint recall on Bud Borer local test images (frequent confusion with YLD) |
| **Apple Orchard Generalization ($N=346$)** | **Raw: 96.0% [93.6%, 98.0%]<br>Guarded: 89.9% [86.7%, 92.8%]** | Held-out orchard split of one public dataset, N=346: raw accuracy 96.0% (95% CI: [93.6%, 98.0%]), guarded accuracy 89.9% (coverage 92.5%, diagnosed accuracy 97.2%); on images farthest from training: 95.7% ($N=304$). Not northern-India validated. |


## 🛡️ Out-of-Distribution (OOD) Guard Performance

An OOD guard was constructed using class-conditional Mahalanobis distance with Ledoit-Wolf shrinkage covariance fitted on penultimate ResNet-18 features of **TRAIN images only**. Thresholds were calibrated to retain 95% of in-domain **VAL images**:
- **Paddy:** Threshold = `28.265936` (In-domain test retention: 94.8%)
- **Arecanut:** Threshold = `36.752533` (In-domain test retention: 93.0%)
- **Apple (v2):** Threshold = `29.0189` (Combined validation calibration; in-domain test retention: 95.4% orchard, 97.5% lab)

### Evaluation Summary
- **Paddy & Apple:** Strong OOD detection. Paddy rejects 100% of non-crop test images (desk, wall, person, wallpapers) and 91.3% of cross-crop images. Apple rejects 100% of non-crop test images and 96.0% of cross-crop images.
- **Arecanut:** Weak OOD detection. The wide feature variance of heterogeneous palm views (trunk, foot, nut, leaf) results in a broad covariance ellipsoid that rejects only **38%–40% of non-crop images** (24 / 60) and **13.5% of paddy images** (27 / 200) in arecanut mode.


## ⚠️ Known Limitations

- **Research Prototype:** AgriTrust AI is an experimental decision support system intended for research and extension demonstration, not as a replacement for certified agronomist diagnosis or statutory agricultural extension services.
- **Crop Scope:** The app only supports close-up photos of paddy, arecanut and apple, and unsupported crops are not reliably detected.
- **Apple Data Provenance & Geography:** Apple model was fine-tuned on public orchard photographs (Cornell FGVC dataset, not northern India). Held-out orchard split of one public dataset, N=346: raw accuracy 96.0% [93.6%, 98.0%], guarded accuracy 89.9% (coverage 92.5%, diagnosed accuracy 97.2%); on images farthest from training: 95.7%. Not validated in commercial orchards in northern India (Kashmir/Himachal Pradesh). Black rot was trained on lab photos only; severity grading is withheld accordingly.
- **Cedar Apple Rust:** Class from US public data; confirm locally with your KVK.
- **Arecanut Bud Borer / YLD Confusion:** About 71% of Arecanut training images come from a merged public Kaggle collection. Deployed checkpoint local-photo accuracy is 81.9% ($N=155$), but `bud_borer` recall is 9/25 (36.0%) due to frequent confusion between Bud Borer and Yellow Leaf Disease (YLD). Confirm with KVK / CPCRI before treatment.
- **Arecanut Severity Provisional:** Arecanut severity assessment is provisional and has not been independently validated.
- **Explainability Limitations:** Grad-CAM saliency heatmaps indicate feature activation regions but do not guarantee precise pixel-level lesion boundary localization.

## 👨‍💼 Expert Verification & Admin Access

The expert verification dashboard on the **Contribute & Earn** page (`pages/2_Contribute_and_Earn.py`) is protected by administrator authentication:
- **Environment Variable:** Set `ADMIN_PASS` in your environment (e.g. `export ADMIN_PASS="your_secure_password"`) before launching the app.
- For security, no default administrator password exists in the codebase.
