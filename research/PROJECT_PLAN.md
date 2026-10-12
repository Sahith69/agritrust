# Explainable ML Framework for Ordinal Disease Severity Grading in Arecanut and Paddy

## Project Goal
Build a multi-crop framework that predicts **disease severity** (ordinal grade 0-3: Healthy/Mild/Moderate/Severe) for arecanut and paddy leaf/plant images, with explainable AI validating that predictions are based on actual disease symptoms rather than background artifacts.

---

## Datasets

**Arecanut**
- Primary: `arecanut_annotations.json` (895 images, 9 classes, already has `severity_grade` 0-3)
  - Schema: `{image_path, filename, original_label, severity_grade}`
  - Classes: bud_borer(0), healthy_foot(1), healthy_leaf(2), healthy_nut(3), healthy_trunk(4), mahali_koleroga(5), stem_cracking(6), stem_bleeding(7), yellow_leaf_disease(8)
  - KNOWN ISSUE: only ~239 unique parent images behind 895 files (avg 3.7x augmentation). Classes 1 (7 parents), 4 (15), 6 (14) are thin. MUST split by parent image group, never by individual file.
- Supplement (optional, for thin classes): Karnataka field-image dataset, IEEE DataPort, DOI 10.21227/q6wr-ca62 (Beena K & Sangeetha V, VTU)

**Paddy**
- Primary: HuggingFace `Project-AgML/paddy_disease_classification` (10,407 images, 10 classes, parquet)
  - Load: `from datasets import load_dataset; ds = load_dataset("Project-AgML/paddy_disease_classification")`
  - Classes: bacterial_leaf_blight(479), bacterial_leaf_streak(380), bacterial_panicle_blight(337), blast(1738), brown_spot(965), dead_heart(1442), downy_mildew(620), hispa(1594), normal(1764), tungro(1088)
  - No severity labels exist — must be generated (see below)

---

## Phase 1: Severity Label Generation (Paddy — Arecanut already has labels, needs resplit only)

Disease classes split into two groups; do NOT apply one formula to all:

### Group A — lesion/spot diseases (segmentation-based)
blast, brown_spot, tungro, bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, downy_mildew
+ arecanut: mahali_koleroga, stem_bleeding (leaf/surface lesion-like), yellow_leaf_disease

Pipeline:
1. Leaf/plant-part segmentation: **use Otsu's method or K-means clustering computed per-image** (NOT fixed HSV thresholds — proven to fail on real field images due to shadows/water/glare in earlier testing). Follow with largest-contour selection + morphological cleanup, same as prototype.
2. Erode leaf mask ~9px before lesion detection to avoid boundary anti-aliasing false positives (confirmed issue in earlier testing).
3. Lesion detection: color/texture thresholding for brown/necrotic/discolored regions, restricted to inside eroded leaf mask, connected-component filtering to drop small noise blobs (<15px).
4. Compute ratio = lesion_px / leaf_px.
5. Map to grade using **Horsfall-Barratt / EPPO standard thresholds**: 0% = Grade 0, 0-5% = Grade 1, 5-15% = Grade 2, >15% = Grade 3. (NOT the earlier arbitrary 2%/10% — these are the real plant-pathology-literature-cited boundaries.)

### Group B — structurally different diseases (custom proxies)
- **dead_heart**: central shoot death, not a leaf lesion. Use discoloration ratio (brown/dead tissue vs green) on the visible plant structure — same segmentation mechanic as Group A, different color target.
- **hispa**: insect scraping damage (pale/white linear streaks on green, not a solid-color blob). Try texture-based detection (local variance / edge density) as a proxy ratio. If this proves unreliable during testing, FALL BACK to a fixed class-level severity grade (document this explicitly as a limitation, don't force a false per-image signal).
- **normal**: Grade 0, no processing needed.

### Confidence scoring (for curation — NEW REQUIREMENT)
Segmentation is NOT a trained classifier, so there's no native softmax confidence. Construct a composite confidence score (0-100%) from:
1. **Leaf mask plausibility**: leaf area as % of total image should fall in a sane range (e.g. 5%-75%); contour solidity (area / convex hull area) should be reasonably high (a real leaf blob is fairly convex; fragmented/jagged masks indicate background contamination — score these low).
2. **Threshold-boundary distance**: ratio close to a grade boundary (5% or 15%) is inherently less certain than a ratio solidly in the middle of a band — score boundary-adjacent cases lower.
3. **Lesion mask fragmentation**: extremely fragmented lesion masks (many tiny blobs vs one/few coherent regions) may indicate noise rather than real symptoms — factor into confidence.

Combine into one score. **Only auto-accept images with composite confidence >= 80%.** Route everything below that threshold to a `needs_review/` folder + separate CSV — do not silently include low-confidence labels in the training set.

### Validation (do regardless of automation)
- Self-grade ~80-100 images blind (assign your own severity judgment BEFORE looking at script output), then compare against automated grades. Report as an informal validation metric in the writeup, explicitly labeled as non-expert.
- Document as a limitation: "severity is a symptom-area proxy, not clinical/expert staging."

---

## Phase 2: Data Integrity (non-negotiable, applies to BOTH crops)
- **Group by parent/source image before any train/val/test split.** For arecanut, parent ID = the `original_NN` token in the filename. For paddy, check if Paddy Doctor filenames encode any augmentation lineage; if images are all originals (no augmentation applied yet), a normal stratified split is fine — but VERIFY this assumption first (don't assume).
- Stratify splits by disease class AND severity grade jointly where possible.
- Recommended split: 70/15/15 or 80/10/10 train/val/test.

---

## Phase 3: Model Training (2 models — one per crop)

- Backbone: ResNet-18 or ResNet-34, ImageNet-pretrained, mostly frozen (unfreeze only last conv block + head — needed given small arecanut set and CPU-only training budget).
- **Ordinal head**: CORN loss (Conditional Ordinal Regression for Neural networks) preferred over plain softmax — for 4 grades this means 3 binary "severity > k" outputs. If CORN proves too complex to implement cleanly, fall back to weighted ordinal cross-entropy (penalize by |predicted_grade - true_grade|).
- Baseline model: same backbone, plain 4-way softmax — trained for comparison, to demonstrate the ordinal approach's benefit.
- Augmentation: rotation, flip, mild brightness/contrast jitter. AVOID heavy color jitter — color is meaningful disease signal here, not noise.
- Track **QWK (quadratic weighted kappa)** per epoch on validation set, not just accuracy/loss — use for early stopping.
- Batch size 16-32, AdamW, lr ~1e-4, 15-25 epochs with early stopping.

---

## Phase 4: Explainability
- Grad-CAM (or Grad-CAM++) on trained classifier for both correct and incorrect predictions.
- **Quantify agreement between Grad-CAM heatmap and the Phase-1 segmentation lesion mask** (e.g., IoU or normalized overlap score). This is the core XAI validation metric — answers "does the model actually look at the disease symptoms, or cheat on background/lighting cues?"
- Optional: SHAP on ordinal probability outputs if tabular features (lesion ratio, crop age/variety metadata from Paddy Doctor) are fused into the model.

---

## Phase 5: Evaluation
- Primary metric: **QWK** (not just accuracy).
- Per-class recall/precision, confusion matrix (visualize magnitude of errors — grade 1→2 miss is very different from grade 0→3 miss).
- Ablations: ordinal loss vs. softmax baseline; with/without Grad-CAM-segmentation consistency check.
- Cross-crop comparison + discussion of why one crop might outperform the other (data quality, class balance, unique-parent-image counts, lesion visual distinctiveness).
- Error analysis: pull worst misclassifications, look for patterns (specific disease type, specific image conditions like clutter/glare).

---

## Phase 6: Usage / Deliverables (build in this order)
1. **Inference script**: single image in → severity grade + per-grade confidence + Grad-CAM overlay out.
2. **Batch evaluation script**: runs full test set → CSV + confusion matrix + QWK report.
3. **Streamlit demo**: upload image → predicted grade, confidence bar chart, Grad-CAM overlay + segmentation mask overlay side-by-side. This is the primary presentation/demo artifact.
4. (Optional, low priority) Flask/FastAPI wrapper if a "deployable system" claim is wanted.

---

## Writeup Structure
Abstract → Related Work → Data & Labeling Methodology (be explicit about the proxy-based severity approach and Horsfall-Barratt grounding) → Model Architecture → Results (QWK, confusion matrices, baseline comparison) → Explainability Analysis (Grad-CAM/IoU findings) → Limitations (proxy severity not clinical staging; no expert/CPCRI validation — descoped; thin arecanut classes named explicitly; Group B disease-mechanism mismatch) → Conclusion.

Realistic target venue: regional/national conference, workshop, or Q2-Q3 journal — matches the scope of available resources (no expert panel, modest compute, no large annotated benchmark).

---

## Explicitly Descoped (decisions already made, don't revisit without reason)
- CPCRI / expert validation outreach — considered, deliberately dropped due to time constraints. Documented as a limitation instead.
- RiceSEG dataset — evaluated, ruled out (organ/phenotype segmentation, not disease lesion segmentation — wrong task).
- Isaac Ritharson's "Severity-Based Rice Disease Classification" Kaggle dataset — evaluated as a potential "teacher model" source, ruled out (confirmed 88.89% train/val leakage via MD5 hash duplicates, and only ~15 unique parent leaves per class — high risk of the trained model memorizing dataset-specific artifacts rather than learning generalizable severity features). Can optionally be used as a small independent external sanity-check, never as training data.
- Full CNN-based active-learning pipeline (200-image manual seed set → train → confidence-gated propagation) — considered as an alternative to segmentation-based auto-labeling; deprioritized in favor of the segmentation + confidence-gating approach above to save time, given repeated concern that large-scale manual grading is not a good use of time. Can be revisited for the `needs_review/` low-confidence bucket specifically, at smaller scale, if time allows.

---

## Immediate Next Steps (in order)
1. Verify assumption: are Paddy Doctor images pre-augmented (need parent-grouping) or all-original (safe to split normally)? Check filename patterns / metadata.
2. Build the Otsu/K-means segmentation + confidence-scoring script (Phase 1, Group A + confidence section above). Test on a small sample with a debug gallery before running on the full dataset.
3. Build the parent-grouped split script for arecanut (known augmentation issue, needed regardless).
4. Run Phase 1 across both datasets, review `needs_review/` bucket manually, finalize severity-labeled dataset.
5. Build and run Phase 3 training (both crops).
6. Build Phase 4/5 (Grad-CAM, IoU validation, evaluation metrics).
7. Build Phase 6 deliverables (inference script → batch eval → Streamlit demo).
