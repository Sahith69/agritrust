# Stage 3 Report: Apple FGVC Orchard Test & Fine-Tuning

## 3a. Data — FGVC Image Pool

| Class | FGVC Count | Mapping |
| :--- | :--- | :--- |
| `scab` | 592 | → `apple_scab` |
| `rust` | 622 | → `cedar_apple_rust` |
| `healthy` | 516 | → `healthy` |
| `multiple_diseases` | 91 | **Dropped** |
| **Total used** | **1,730** | (single-disease only) |

---

## 3b. Leak-Safe Cluster Split

### Clustering Parameters
- Feature: pHash (orig/hflip/vflip) ≤ 8 **OR** DINOv2 vits14 cosine ≥ 0.95
- DINOv2: `dinov2_vits14`, resize 224, L2-normalised 384-d embeddings
- Vectorised adjacency matrix: `(1730 × 1730)` pairwise computation
- Connected-components via `scipy.sparse.csgraph.connected_components`

### Cluster Size Distribution (1,730 FGVC images)

| Metric | Value |
| :--- | :--- |
| Total clusters | **1,700** |
| Singletons (size = 1) | 1,670 (98.2%) |
| Pairs (size = 2) | 30 (1.8%) |
| Max cluster size | 2 |
| Mean cluster size | 1.02 |

### 60/20/20 Stratified Cluster Split (seed 42)

| Split | `apple_scab` | `cedar_apple_rust` | `healthy` | Total |
| :--- | :--- | :--- | :--- | :--- |
| `fgvc_train` | 354 | 373 | 310 | **1,037** |
| `fgvc_val` | 119 | 126 | 102 | **347** |
| `fgvc_test` | 119 | 123 | 104 | **346** |

### Leakage Verification for `fgvc_test`
- **Test candidates:** 346 images
- **Dropped due to pHash ≤ 8 or DINOv2 cosine ≥ 0.95 match to any of fgvc_train, fgvc_val, or PlantVillage apple train:** **0 images**
- `fgvc_test` is frozen; it was **never** used for training, tuning, threshold selection, or early stopping.

---

## 3c. Baseline Evaluation — Deployed `apple_disease_classifier.pth` on `fgvc_test`

### Class Coverage
| Class | N |
| :--- | :--- |
| `apple_scab` | 119 |
| `cedar_apple_rust` | 123 |
| `healthy` | 104 |
| **Total** | **346** |

### Guarded Pipeline (OOD Guard + Abstain 0.70)

| Metric | Value |
| :--- | :--- |
| Share rejected (OOD, score > threshold) | **98.84%** (342/346) |
| Share abstained (confidence < 0.70) | 0.29% (1/346) |
| Share diagnosed | 0.87% (3/346) |
| Accuracy on diagnosed images | 100.00% (3/3) |
| Accuracy across ALL images | 0.87% (95% CI: [0.00%, 2.02%]) |
| Macro-F1 across ALL images | 0.0164 (95% CI: [0.0000, 0.0359]) |
| Scab recall (ALL images) | 2.52% |
| Rust recall (ALL images) | 0.00% |
| Healthy recall (ALL images) | 0.00% |
| Mean confidence (correct) | 0.8633 |
| Mean confidence (wrong) | 0.0000 |

### Raw Argmax Pipeline (No Guards)

| Metric | Value |
| :--- | :--- |
| Raw Accuracy | **51.16%** (95% CI: [45.95%, 56.36%]) |
| Raw Macro-F1 (3 classes) | 0.5504 (95% CI: [0.4957, 0.6004]) |
| `apple_scab` recall | 63.03% |
| `cedar_apple_rust` recall | **25.20%** |
| `healthy` recall | 68.27% |

### Baseline Raw Confusion Matrix

| True \ Pred | apple_scab | black_rot | cedar_apple_rust | healthy |
| :--- | :--- | :--- | :--- | :--- |
| apple_scab | 75 | 9 | 4 | 31 |
| cedar_apple_rust | 23 | 50 | 31 | 19 |
| healthy | 22 | 11 | 0 | 71 |

> [!IMPORTANT]
> **Stop Rule:** Raw accuracy ≥ 90% **AND** scab recall ≥ 85% **AND** rust recall ≥ 85%. Achieved: Raw Acc = 51.16%, Scab = 63.03%, Rust = **25.20%**.
> **Stop rule NOT met → Training proceeded.**

---

## 3d. Fine-Tuning (15 Epochs, Seed 42)

### Dataset Composition
| Source | Images per Epoch | FGVC Share |
| :--- | :--- | :--- |
| PlantVillage lab train | 2,219 | — |
| FGVC orchard train (oversampled 2×) | 2,074 | — |
| **Total** | **4,293** | **48.3% ≥ 40% ✓** |

### Architecture
- ResNet-18 initialized from deployed `apple_disease_classifier.pth` weights
- **Frozen:** `conv1`, `bn1`, `layer1`, `layer2` (683,072 parameters)
- **Trainable:** `layer3`, `layer4`, `fc` (10,495,492 parameters)
- Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-2)
- Loss: CrossEntropyLoss
- Saved to: `outputs/apple_fgvc/apple_disease_fgvc_v1.pth` (**deployed model never touched**)

### Augmentation Strategy
- **Orchard (FGVC):** `RandomResizedCrop(224, 0.8–1.0)`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `ColorJitter(0.2/0.2/0.2/0.1)`, `GaussianBlur(k=3)`, `Normalize`
- **Lab (PlantVillage):** `Resize(224)`, `RandomHorizontalFlip`, `Normalize`

### Training Log

| Epoch | Train Loss | Val Acc | Val Scab Recall | Val Rust Recall | Epoch Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | 0.1279 | 97.41% | 95.0% | 97.6% | 252.2s |
| 02 | 0.0426 | 97.69% | 95.0% | 98.4% | 248.2s |
| 03 | 0.0255 | 97.41% | 92.4% | 100.0% | 245.8s |
| 04 | 0.0264 | 97.69% | 94.1% | 99.2% | 244.8s |
| 05 | 0.0221 | 96.54% | 93.3% | 96.8% | 244.3s |
| 06 | 0.0091 | 95.97% | 89.1% | 99.2% | 244.4s |
| 07 | 0.0162 | 97.41% | 94.1% | 98.4% | 244.8s |
| 08 | 0.0201 | 98.85% | 98.3% | 99.2% | 245.8s |
| 09 | 0.0088 | 97.69% | 94.1% | 100.0% | 245.1s |
| 10 | 0.0116 | 97.69% | 97.5% | 100.0% | 244.0s |
| 11 | 0.0136 | 98.27% | 97.5% | 98.4% | 244.1s |
| 12 | 0.0149 | 96.25% | 92.4% | 96.8% | 244.0s |
| 13 | 0.0133 | 98.56% | 97.5% | 99.2% | 242.4s |
| 14 | 0.0089 | **98.85%** | 97.5% | **100.0%** | 244.0s |
| 15 | 0.0083 | 97.98% | 95.0% | 100.0% | 243.6s |

**Total training time: 3,677.6 s (61.3 min)** — within the 90-minute time-box.

---

## 3e. OOD Guard Refit on New Model

| Parameter | Value |
| :--- | :--- |
| Backbone | ResNet-18 penultimate (512-d features) |
| Covariance method | Ledoit-Wolf (assume_centered=True) |
| Shrinkage coefficient | 0.0043 |
| Training images used | 3,256 (unique: PV train + FGVC train, no duplication) |
| Calibration set | FGVC val (347 images, 95th-percentile threshold) |
| Threshold | **27.6972** |
| Val retention | 94.8% (329/347) |
| Saved to | `outputs/apple_fgvc/ood_params_apple_fgvc.pt` |

---

## 3f. Final Comparative Evaluation on Frozen `fgvc_test`

### Full Metric Comparison

| Metric | Baseline Deployed | Fine-Tuned (New) | Delta |
| :--- | :--- | :--- | :--- |
| **Raw Argmax Accuracy** | 51.16% | **95.95%** | **+44.80%** |
| **Raw Macro-F1 (3 classes)** | 0.5504 | **0.9582** | **+0.4078** |
| — `apple_scab` raw recall | 63.03% | 93.28% | +30.25% |
| — `cedar_apple_rust` raw recall | 25.20% | **99.19%** | **+73.98%** |
| — `healthy` raw recall | 68.27% | 95.19% | +26.92% |
| Guarded OOD rejection share | 98.84% | 6.65% | −92.20% |
| Guarded abstain share (<0.70) | 0.29% | 0.87% | +0.58% |
| Guarded diagnosed accuracy | 100.00% | 97.19% | −2.81% |
| **Guarded Accuracy (ALL images)** | 0.87% | **89.88%** | **+89.02%** |
| **Guarded Macro-F1 (ALL images)** | 0.0164 | **0.9304** | **+0.9140** |
| PlantVillage black_rot recall | 100.00% | **100.00%** | 0.00% |

### McNemar Test (Raw Predictions, `fgvc_test`)
- **χ² = 140.33** (df = 1), **p = 2.25 × 10⁻³²** (highly significant)
- The new model is significantly more accurate on FGVC orchard images.

### New Model Guarded Confusion Matrix

| True \ Pred | apple_scab | black_rot | cedar_apple_rust | healthy | abstained | rejected |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| apple_scab | **104** | 0 | 0 | 6 | 2 | 7 |
| cedar_apple_rust | 0 | 0 | **120** | 0 | 0 | 3 |
| healthy | 2 | 0 | 0 | **88** | 1 | 13 |

### New Model Raw Confusion Matrix

| True \ Pred | apple_scab | black_rot | cedar_apple_rust | healthy |
| :--- | :--- | :--- | :--- | :--- |
| apple_scab | **111** | 0 | 0 | 8 |
| cedar_apple_rust | 0 | 0 | **122** | 1 |
| healthy | 3 | 0 | 2 | **99** |

### Black Rot Retention (PlantVillage Apple Test, N = 101)
| Model | black_rot Recall |
| :--- | :--- |
| Baseline deployed | **100.00%** (101/101) |
| Fine-tuned new | **100.00%** (101/101) |

---

## Deployment Recommendation

> **Deploy `apple_disease_fgvc_v1.pth`**: The fine-tuned model elevates raw FGVC orchard accuracy from 51.2% → 96.0% (p = 2.25 × 10⁻³², McNemar) and guarded all-image accuracy from 0.87% → 89.88%, while fully preserving laboratory black rot diagnostic capability (100.0% recall on PlantVillage test). OOD guard threshold (27.70) is calibrated to 94.8% FGVC val retention.

> [!NOTE]
> `apple_disease_fgvc_v1.pth` and `ood_params_apple_fgvc.pt` are stored in `outputs/apple_fgvc/` only. The deployed `outputs/models/apple_disease_classifier.pth` was never modified.
