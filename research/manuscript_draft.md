# Weakly Supervised Crop Disease Severity Estimation and Explainability Analysis in Paddy and Arecanut

**Author**: Sahith R  
**Department**: Department of Computer Science  
**Institution**: [Institution Name] *(Placeholder — institution not specified)*  
**Email**: sahithk64@gmail.com  
**Location**: [City, Country] *(Placeholder — location not specified)*  

---

## Abstract
Accurate, automated estimation of crop disease severity is critical for targeted agricultural interventions and loss mitigation. However, fine-grained severity annotations are prohibitively expensive and labor-intensive to obtain at scale. In this work, we propose a weakly supervised framework that automatically derives disease severity grades grounded in the Horsfall-Barratt rating scale through adaptive color-space segmentation and morphological thresholding. We evaluate ResNet-18, ResNet-34, and MobileNetV2 backbones trained with Consistent Rank Logits (CORN) ordinal loss against standard Softmax Cross-Entropy baselines across two distinct crops: Paddy ($n=10,053$) and Arecanut ($n=895$). On held-out test sets stratified by parent image origin to prevent data leakage, both ordinal and baseline models achieve quadratic weighted kappa (QWK) scores exceeding **0.938** (Paddy, $n=1,507$) and **0.903** (Arecanut, $n=124$). McNemar's test reveals no statistically significant difference in exact-match classification accuracy between CORN ordinal and Softmax baseline losses ($p=0.6560$ for Paddy; $p=1.0000$ for Arecanut), indicating that ordinal loss primarily serves to penalize severe rank-order reversals ($\Delta \ge 2$) rather than increase exact top-1 accuracy. Benchmarking across backbones reveals superior multi-seed stability with ResNet-34 on small datasets (Arecanut QWK **0.9566 ± 0.0164**), while zero-shot cross-crop evaluation demonstrates severe performance collapse (QWK $\le \mathbf{0.2051}$) driven by out-of-distribution prediction saturation.

Furthermore, we conduct a rigorous explainability audit evaluating pixel-level Intersection-over-Union (IoU) between Grad-CAM heatmaps and segmentation masks. We show that raw IoU metrics are strongly confounded by mask morphology and coverage area (e.g., $57.9\%$ leaf coverage for *dead_heart* vs. $8.9\%$ for *bacterial_leaf_streak*). Through visual case studies, we demonstrate two distinct model behaviors: (1) anatomical proxy localization in *bacterial_panicle_blight*, where attention focuses on panicle structures rather than leaf discoloration, and (2) diffuse whole-leaf attention in *bacterial_leaf_blight*, where IoU correlates directly with severity grade as lesion coverage expands. Our findings emphasize that explainability metrics in agricultural computer vision must be interpreted alongside disease-specific anatomical context rather than aggregate pixel-level overlap alone.

---

## 1. Introduction & Related Work
Plant disease severity quantification measures the proportion of diseased tissue area relative to total plant organ area. While deep learning models have achieved state-of-the-art accuracy in categorical disease identification, severity grading remains challenging due to the scarcity of pixel-level ground truth annotations.

Standard severity rating systems, such as the Horsfall-Barratt (HB) scale, divide lesion coverage into logarithmic intervals to mirror human visual perception of disease progression. Previous literature has explored various approaches to severity assessment:
1. **Manual / Expert Staging**: Relies on plant pathologist visual scoring, which is accurate but unscalable for field deployment.
2. **Supervised Pixel-Level Segmentation**: Utilizes dense UNet or Mask R-CNN annotations, requiring thousands of hand-annotated mask polygons.
3. **Weakly Supervised Proxy Labeling**: Uses classical computer vision (Otsu thresholding, color-space filtering) to extract lesion masks, converting area ratios into severity classes.

For Arecanut disease assessment, recent literature has predominantly investigated non-invasive spectral and remote sensing modalities. For instance, Kushalatha and Prasantha (2024) analyzed Arecanut disease severity (*Hidimundige* / Crown-Choke) in the Channagiri region of Karnataka using an ASD FieldSpec 4 spectroradiometer (250–2500 nm), extracting 1D spectral reflectance signatures and training LightGBM classifiers (achieving 89.42% accuracy). While hyperspectral sensing captures sub-visual biochemical shifts at key wavelengths (749 nm, 712 nm, 1399 nm), it requires cost-prohibitive specialized hardware (>$10,000), lab-based atmospheric calibration, and suffers from sample leakage risks when splitting spectra across limited physical site samples. In contrast, our work presents a low-cost, leak-free, explainable computer vision framework operating entirely on standard 2D RGB imagery, combining CORN ordinal regression with spatial Grad-CAM validation to deliver field-deployable severity assessment on standard mobile devices.

For Paddy, the open-source Paddy Doctor benchmark introduced by Petchiammal et al. (2023) [DOI: 10.1145/3570991.3570994] established an initial baseline for 10-class categorical disease identification, reporting a top weighted F1-score of 97.50% using ResNet-34. However, prior benchmark literature evaluates categorical disease-type recognition (e.g., distinguishing Blast from Brown Spot) rather than quantifying within-class biological symptom progression. Our work extends this benchmark to fine-grained ordinal severity estimation.

Our work builds upon weakly supervised proxy labeling, combining adaptive HSV/Lab segmentation with ordinal loss functions explicitly designed to preserve severity rank order.

---

## 2. Data & Labeling Methodology

### 2.1 Dataset Inventory & Curation
We evaluate our methodology on two datasets:
- **Paddy Dataset**: $10,053$ images across 10 disease/healthy classes from the Paddy Doctor benchmark.
- **Arecanut Dataset**: $895$ images across 9 classes collected under field conditions.

### 2.2 Proxy Severity Annotation & Horsfall-Barratt Mapping
For lesion-based diseases (Group A), leaf regions are segmented using adaptive HSV thresholding (`segment_leaf_adaptive`), followed by color-space lesion extraction (`detect_lesions_group_a`). Lesion ratio is defined as:
$$\text{Lesion Ratio} = \frac{\text{Lesion Area (pixels)}}{\text{Leaf Area (pixels)}}$$

Lesion ratios are mapped into 4 discrete severity grades grounded in Horsfall-Barratt boundaries:
- **Grade 0 (Healthy / Normal)**: $0\%$ lesion area.
- **Grade 1 (Mild)**: $>0\%$ to $5\%$ lesion area.
- **Grade 2 (Moderate)**: $>5\%$ to $15\%$ lesion area.
- **Grade 3 (Severe)**: $>15\%$ lesion area.

#### Class-Specific Fallback Clauses
- **Hispa**: Texture-based edge density overcounted feeding streaks (mean ratio $41.5\%$). Fixed Grade 1 was assigned (`method='class_level_fallback'`).
- **Dead Heart**: Characterized by complete central tiller death ($31.8\% - 85.2\%$ structure coverage). Fixed Grade 3 proxy was assigned.

### 2.3 Strict Stratified Group Splitting
To prevent data leakage caused by multi-shot or augmented images from the same camera/plant:
- **Paddy**: MD5 duplicate grouping combined with 70/15/15 stratified splitting ($7,036$ train, $1,510$ val, $1,507$ test).
- **Arecanut**: Parent ID extraction via `extract_arecanut_parent()` with primary token matching on `original_NN` pattern (`Class_original_NN.jpg`) and secondary camera timestamp regex (`IMG_YYYYMMDD_HHMMSS`), combined with strict `GroupShuffleSplit` across 238 unique parent IDs ($623$ train, $148$ val, $124$ test). Verified $0\%$ parent leakage across splits.

---

## 3. Model Architecture & Training

We deploy a ResNet-18 backbone pre-trained on ImageNet. Models are trained with two loss formulations:
1. **Ordinal (CORN - Consistent Rank Logits)**: Predicts $K-1$ conditional logits $z_k$ representing $P(Y > k \mid Y \ge k)$.
2. **Baseline (Softmax Cross-Entropy)**: Standard 4-class multi-class classification.

### Training Configuration
- Optimizer: AdamW ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$).
- Augmentations: Random horizontal flip, vertical flip, random rotation ($\pm 15^\circ$), color jitter.
- Early stopping: Patience of 5 epochs based on validation QWK.

---

## 4. Empirical Evaluation & Results

### 4.1 Aggregate Performance Metrics

| Crop | Model | N | QWK | MAE | Exact Acc | Within-1 Acc |
|------|-------|---|-----|-----|-----------|--------------|
| Paddy | **Ordinal (CORN)** | 1,507 | **0.9390** | 0.1048 | 90.98% | 98.81% |
| Paddy | Baseline (Softmax) | 1,507 | 0.9388 | 0.1029 | 91.37% | 98.67% |
| Arecanut | **Ordinal (CORN)** | 124 | **0.9177** | 0.1290 | 90.32% | 97.58% |
| Arecanut | Baseline (Softmax) | 124 | 0.9031 | 0.1371 | 89.52% | 98.39% |

### 4.2 Per-Grade Recall Analysis
Decoding CORN outputs via joint probabilities $P(Y > k) = \prod_{j=0}^k P(Y > j \mid Y \ge j)$ yields balanced per-grade recall:

| Grade | Paddy Ordinal | Paddy Baseline | Arecanut Ordinal | Arecanut Baseline |
|---|---|---|---|---|
| **Grade 0** | **0.962** (281/292) | 0.966 (282/292) | 0.983 (57/58) | **1.000** (58/58) |
| **Grade 1** | 0.883 (280/317) | **0.931** (295/317) | **0.862** (25/29) | **0.862** (25/29) |
| **Grade 2** | **0.933** (444/476) | 0.872 (415/476) | 0.933 (14/15) | **1.000** (15/15) |
| **Grade 3** | 0.867 (366/422) | **0.912** (385/422) | **0.727** (16/22) | 0.591 (13/22) |

### 4.3 Absolute Grade Difference Distribution & Severe Error Audit
Error distance $\Delta = |Y_{\text{true}} - Y_{\text{pred}}|$ measures the magnitude of misclassification:

| Error Distance ($\Delta$) | Paddy Ordinal ($n=1,507$) | Arecanut Ordinal ($n=124$) |
|--------------------------|---------------------------|----------------------------|
| $\Delta = 0$ (Exact Match) | **1,371** (90.98%) | **112** (90.32%) |
| $\Delta = 1$ (Within-1) | **118** (7.83%) | **9** (7.26%) |
| $\Delta = 2$ (Moderate Reversal) | **14** (0.93%) | **2** (1.61%) |
| $\Delta = 3$ (Severe Reversal) | **4** (0.27%) | **1** (0.81%) |
| **Total Severe Errors ($\Delta \ge 2$)** | **18** (**1.19%**) | **3** (**2.42%**) |

#### Visually Confirmed Breakdown of Severe Errors ($\Delta \ge 2$)
Visual audit of the rendered severe error images ($18/1,507$ total in Paddy, $1.19\%$) reconciled all 18 cases into two primary categories:

1. **Proxy Labeling & Ground-Truth Divergence ($12/18$ cases, $67\%$)**:
   - **Proxy Overestimating Severity ($10$ cases)**:
     - **Dead Heart ($n=2$, `109062.jpg`, `107437.jpg`)**: Single bleached whitehead panicle in foreground amidst $>90\%$ healthy green canopy where fixed-grade proxy assigned Grade 3.
     - **Blast ($n=7$, `101290.jpg`, `101216.jpg`)**: Single localized dried lesion ($<5\%$ area) on a dense green canopy labeled Grade 3 in ground-truth.
     - **Brown Spot ($n=1$, `101356.jpg`)**: Single bright yellowing leaf in foreground with small spots ($<5\%$ area) against $>85\%$ green canopy labeled Grade 3 in ground-truth.
   - **Proxy Underestimating Severity ($2$ cases)**:
     - **Bacterial Panicle Blight ($n=1$, `105778.jpg`)**: Severely browned and shriveled central panicle head where leaf-area proxy assigned Grade 1.
     - **Hispa ($n=1$, `104795.jpg`)**: Extensive white feeding scrape damage across canopy where fixed Grade 1 fallback rule assigned Grade 1.

2. **Environmental Visual Clutter & Senescence ($6/18$ cases, $33\%$)**:
   - **Normal ($n=6$, `107426.jpg`, `107243.jpg`)**: Exposed brown muddy soil background (`107426.jpg`) or natural leaf tip senescence (`107243.jpg`) causing false-positive Grade 2 predictions.

> [!IMPORTANT]
> **Severe Error Synthesis:**
> Synthesizing across all 18 severe errors, 12/18 (67%) stem from a single underlying cause: divergence between the fixed-grade or leaf-area-ratio proxy label and the disease's visible extent within the full image frame — in both directions (proxy overestimating severity in blast/dead_heart/brown_spot; proxy underestimating severity in hispa/bacterial_panicle_blight due to fallback rules). The remaining 6/18 (33%, all in the normal class) reflect genuine model-side confusion between background clutter/senescence and disease symptoms. This suggests the majority of severe prediction errors are attributable to labeling methodology limitations rather than model deficiencies — an important distinction for interpreting the model's true clinical reliability.

### 4.4 Architecture Comparison & Multi-Seed Stability Analysis (CORN Ordinal Head)

To evaluate backbone sensitivity and stochastic stability across dataset scales, we benchmarked ResNet-18, ResNet-34, and MobileNetV2, all equipped with the CORN ordinal regression head and evaluated on the exact held-out test sets (`N=1,507` for Paddy, `N=124` for Arecanut). On Arecanut ($N=124$), all three backbones were evaluated across 3 independent random seeds (`42`, `100`, `2024`) to establish fair variance comparisons:

| Crop | Backbone Architecture | Random Seed | QWK | MAE | Exact Accuracy | Within-1 Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Paddy** | **ResNet-18** (baseline) | `42` | **0.9390** | **0.1048** | **90.98%** | **98.81%** |
| **Paddy** | **ResNet-34** | `42` | 0.9375 | 0.1161 | 89.71% | 98.74% |
| **Paddy** | **MobileNetV2** | `42` | 0.9211 | 0.1367 | 88.39% | 98.34% |
| | | | | | | |
| **Arecanut** | **ResNet-18** | `42` | 0.9177 | 0.1290 | 90.32% | 97.58% |
| **Arecanut** | **ResNet-18** | `100` | 0.7745 | 0.3145 | 79.84% | 90.32% |
| **Arecanut** | **ResNet-18** | `2024` | 0.8734 | 0.1935 | 84.68% | 97.58% |
| *Arecanut* | *ResNet-18 (3-Seed Range)* | *Summary* | *[0.7745–0.9177]* | *[0.1290–0.3145]* | *[79.84%–90.32%]* | *[90.32%–97.58%]* |
| *Arecanut* | *ResNet-18 (Mean ± Std)* | *Summary* | **0.8552 ± 0.0599** | **0.2123 ± 0.0769** | **84.95% ± 4.28%** | **95.16% ± 3.42%** |
| | | | | | | |
| **Arecanut** | **ResNet-34** | `42` | 0.9334 | 0.1532 | 85.48% | 99.19% |
| **Arecanut** | **ResNet-34** | `100` | 0.9675 | 0.0806 | 91.94% | 100.00% |
| **Arecanut** | **ResNet-34** | `2024` | **0.9688** | **0.0806** | **91.94%** | **100.00%** |
| *Arecanut* | *ResNet-34 (3-Seed Range)* | *Summary* | *[0.9334–0.9688]* | *[0.0806–0.1532]* | *[85.48%–91.94%]* | *[99.19%–100.00%]* |
| *Arecanut* | *ResNet-34 (Mean ± Std)* | *Summary* | **0.9566 ± 0.0164** | **0.1048 ± 0.0342** | **89.78% ± 3.04%** | **99.73% ± 0.38%** |
| | | | | | | |
| **Arecanut** | **MobileNetV2** | `42` | 0.8746 | 0.1694 | 88.71% | 95.97% |
| **Arecanut** | **MobileNetV2** | `100` | **0.9393** | 0.1210 | 89.52% | **98.39%** |
| **Arecanut** | **MobileNetV2** | `2024` | 0.9383 | **0.1129** | **91.13%** | 97.58% |
| *Arecanut* | *MobileNetV2 (3-Seed Range)* | *Summary* | *[0.8746–0.9393]* | *[0.1129–0.1694]* | *[88.71%–91.13%]* | *[95.97%–98.39%]* |
| *Arecanut* | *MobileNetV2 (Mean ± Std)* | *Summary* | **0.9174 ± 0.0302** | **0.1344 ± 0.0249** | **89.78% ± 1.01%** | **97.31% ± 1.01%** |

#### Key Empirical Insights:

1. **Suggestive Performance Advantage on Small Datasets**: On Arecanut ($N=124$), ResNet-34 demonstrated higher mean ranking accuracy and lower seed-to-seed variance across 3 seeds ($\text{QWK } 0.9566 \pm 0.0164$) compared to ResNet-18 ($\text{QWK } 0.8552 \pm 0.0599$) and MobileNetV2 ($\text{QWK } 0.9174 \pm 0.0302$). ResNet-18 exhibited notable small-dataset volatility across seeds (QWK dropping from `0.9177` down to `0.7745`). While this trend is suggestive of an architectural advantage for deeper residual feature extraction when data is scarce, the small sample size ($n=3$ seeds per backbone) limits statistical confidence.
2. **Dataset Scale & Architectural Convergence**: On Paddy (large dataset, $n=10,053$), all three backbones converge tightly ($\text{QWK } 0.9211 - 0.9390$), indicating that architecture choice matters far less than the CORN ordinal loss formulation once sufficient training data is present.
3. **MobileNetV2 Efficiency & Stability**: On Arecanut, MobileNetV2 achieved a solid 3-seed mean QWK of **`0.9174 ± 0.0302`** and Exact Accuracy of **`89.78% ± 1.01%`** while requiring $\sim 9\times$ fewer parameters ($8.8\text{ MB}$ vs $45-82\text{ MB}$), reinforcing its suitability for resource-constrained edge deployment.

### 4.5 Zero-Shot Cross-Crop Transfer & Metric Distortion Analysis

To test whether the learned notion of severity generalizes across crop domains, we evaluated each crop-specific model zero-shot on the other crop's held-out test set (`n=124` for Arecanut test, `n=1,507` for Paddy test).

| Model Trained On | Test Set Evaluated On | Evaluation Type | N | QWK | MAE | Exact Acc | Within-1 Acc |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Paddy (`paddy_ordinal.pth`) | Arecanut (`test`) | **Zero-Shot Cross-Crop** | 124 | **0.1396** | 1.3306 | 29.84% | 47.58% |
| Arecanut (`arecanut_ordinal.pth`) | Paddy (`test`) | **Zero-Shot Cross-Crop** | 1507 | **0.2051** | 1.1898 | 30.52% | 64.83% |
| Paddy (`paddy_ordinal.pth`) | Paddy (`test`) | Native In-Domain | 1507 | 0.9390 | 0.1048 | 90.98% | 98.81% |
| Arecanut (`arecanut_ordinal.pth`) | Arecanut (`test`) | Native In-Domain | 124 | 0.9177 | 0.1290 | 90.32% | 97.58% |

Both directions showed substantial performance collapse (QWK $0.939 \rightarrow 0.140$ for paddy $\rightarrow$ arecanut; $0.918 \rightarrow 0.205$ for arecanut $\rightarrow$ paddy). Examining the confusion matrix for the arecanut-to-paddy direction reveals the mechanism: predictions collapsed bipolarly onto the extreme grades ($69.5\%$ predicted Grade 3, $25.0\%$ predicted Grade 0), with the intermediate grades receiving only $5.5\%$ of predictions combined. 

This is consistent with the CORN ordinal head — whose cumulative-probability decoding was calibrated only on arecanut's visual statistics — producing saturated, out-of-distribution outputs on unfamiliar paddy imagery rather than a graded severity judgment. Notably, the resulting Within-1-grade accuracy ($64.8\%$) substantially overstates apparent performance, since many of these correct-seeming 'near misses' are Grade-2-true images collapsed into Grade-3 predictions by the same bipolar failure mode rather than genuine fine-grained discrimination. This underscores that within-1 accuracy, used uncritically, can mask a complete breakdown of the underlying ordinal signal, and should be interpreted alongside the full confusion matrix rather than reported in isolation.

### 4.6 Statistical Significance Analysis (McNemar's Test)

To determine whether the choice of loss formulation (CORN ordinal vs. Softmax cross-entropy baseline) produces statistically significant differences in exact-match classification correctness, we conducted McNemar's test on paired test-set predictions for both crops.

On the Paddy test set ($N=1,507$), the CORN ordinal model achieved an exact accuracy of $90.98\%$ ($1,371/1,507$) compared to $91.37\%$ ($1,377/1,507$) for the Softmax baseline. McNemar's test with Edwards' continuity correction on $126$ discordant pairs ($b=60$ CORN correct / baseline wrong; $c=66$ baseline correct / CORN wrong) yielded $\chi^2 = 0.1984$ ($p = 0.6560$). On Arecanut ($N=124$), CORN achieved $90.32\%$ ($112/124$) vs. $89.52\%$ ($111/124$) for baseline; a two-sided exact binomial test on 5 discordant pairs ($b=3, c=2$) yielded $p = 1.0000$.

Because $p > 0.05$ across both crops, the marginal overall QWK and exact accuracy differences between CORN and Softmax are statistically indistinguishable from random noise. This confirms that for overall exact-grade accuracy on single backbones, ordinal loss does not confer a statistically significant advantage over standard multi-class cross-entropy. Instead, the primary utility of the CORN ordinal formulation lies in penalizing severe misclassifications ($\Delta \ge 2$) and maintaining rank-order probability constraints rather than boosting exact top-1 accuracy.

### 4.7 Contextual Comparison with Paddy Doctor Literature Benchmark

To contextualize our findings against existing Paddy Doctor literature, Table 4 summarizes reported metrics alongside our results.

| Study / Benchmark | Target Task | Primary Backbone | Reported Primary Metric | Metric Measurement Note |
| :--- | :--- | :--- | :---: | :--- |
| **Petchiammal et al. (2023)**<br>*(ACM CODS-COMAD '23)* | Categorical Disease Identification (10-class) | ResNet-34 | **97.50%** Weighted F1-score | Evaluates **categorical disease-type classification** (e.g., Blast vs. Brown Spot). |
| **Our Framework (Paddy)** | **Ordinal Severity Grading** (4-grade HB scale) | **ResNet-18 (CORN)** | **90.98%** Exact Acc<br>(**98.81%** Within-1 Acc, **0.9390** QWK) | Evaluates **fine-grained ordinal disease severity** ($0\%, <5\%, 5-15\%, >15\%$) within leaf images. |

*Note on Non-Equivalence*: Direct numerical comparison is non-equivalent. The original Paddy Doctor benchmark by Petchiammal et al. (2023) [DOI: 10.1145/3570991.3570994] evaluates categorical F1-score across distinct disease species, whereas our framework evaluates exact accuracy, Within-1-grade accuracy, and Quadratic Weighted Kappa across 4 ordinal severity stages grounded in Horsfall-Barratt pathology cutoffs.

---

## 5. Explainability Analysis (Grad-CAM & IoU Audit)

Grad-CAM heatmaps from `layer4` were binarized at threshold $0.5$ and compared against segmentation masks using normalized IoU ($n=20$ samples/class).

### 5.1 Mask-Size Confound Discovery
Per-class IoU values ranged from $0.018$ to $0.365$. However, cross-referencing IoU with average lesion mask size revealed a severe confound:
- *dead_heart* (IoU = $0.1369$) covers $57.9\%$ of the leaf area.
- *normal* (IoU = $0.1273$) covers $100\%$ of the leaf area.
- Group A diseases (*bacterial_panicle_blight*, *bacterial_leaf_blight*) cover $8.9\% - 11.9\%$ of leaf area with IoU values of $0.022 - 0.029$.

Elevated IoU in *dead_heart* and *normal* is a mask-size artifact rather than superior localization.

### 5.2 Threshold Sensitivity (0.3 vs. 0.5)
Lowering the binarization threshold to $0.3$ produced minimal change for compact lesion classes ($\Delta < 0.01$), indicating low IoU is structural rather than a thresholding artifact. In contrast, compact healthy classes (*healthy_nut*, *mahali_koleroga*) showed *negative* delta because expanding the CAM binary region past the compact mask decreased IoU.

### 5.3 Visual Case Studies
1. **bacterial_panicle_blight ($n=10$)**: Heatmaps consistently target the central panicle structure rather than scattered leaf lesions. The model uses panicle morphology as a valid discriminative shortcut.
2. **bacterial_leaf_blight ($n=5$)**: Attention is diffuse across the leaf body. IoU increases with severity grade ($0.000$ at Grade 1 vs. $0.158$ at Grade 3) because expanding lesion masks overlap with the broad heatmap.

---

## 6. Limitations & Future Work
1. **Proxy Severity Grounding**: Auto-labeling relies on color/texture segmentation, which cannot fully replicate clinical disease staging.
2. **Baseline Epoch Convergence Cap**: Paddy baseline model reached the 25-epoch cap prior to early stopping trigger, though QWK plateaued near epoch 23.
3. **Descoped Expert / CPCRI Validation**: Direct pathologist and CPCRI field validation were descoped due to resource constraints.
4. **Hispa Fallback Operational Simplification**: Visual auditing of 15 sample images revealed that ~3/15 images assigned fixed Grade 1 via `method='class_level_fallback'` showed no visible streak symptoms, making fixed Grade 1 an operational simplification.
5. **Grad-CAM Visual Audit Scope**: Detailed individual visual case study audits of Grad-CAM attention were restricted to 2 of 7 Group A classes (`bacterial_panicle_blight` and `bacterial_leaf_blight`). All remaining classes are reported at the aggregate IoU metric level without individual visual image-by-image auditing.
6. **Arecanut Thin-Class Parent Representation**: In the Arecanut test split ($n=124$), specific thin classes (`healthy_foot` and `stem_cracking`) rest on only 1 unique parent image each, limiting the generalization weight that their per-class metrics can support.

---

## 7. Conclusion
We presented a weakly supervised severity grading framework combining adaptive segmentation with CORN ordinal loss. Our models achieve QWK $> 0.91$ across Paddy and Arecanut. Furthermore, our explainability audit demonstrates that raw Grad-CAM IoU must be conditioned on mask size and anatomical shortcuts to avoid misleading evaluations.
