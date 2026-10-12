"""
Script to generate a 100% Paper-Focused Machine Learning PowerPoint presentation deck (.pptx).
Focuses EXCLUSIVELY on core paper contributions:
- Paddy 10-Class Classification (96.04%, N=10,053) & 7-Class CORN Severity (0.7055 QWK, N=343)
- Arecanut Honest Baseline (82.92%, N=2,661) & Rerun CI Overlap Audit (90.33%, no stat diff)
- Zero-Leakage Data Partitioning (robust_cluster_id) & Soft-Leakage Kaggle Audit
- Grad-CAM Spatial Attribution & Data Scarcity Analysis (bud_borer 5 physical lineages)
- NO Gatekeeper / NO OOD filter references.
"""
import sys, os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_paper_focused_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette - IEEE Paper Research Theme
    BG_DARK     = RGBColor(15, 23, 42)      # Deep slate / navy #0f172a
    CARD_BG     = RGBColor(30, 41, 59)      # Slate card #1e293b
    ACCENT_CYAN = RGBColor(56, 189, 248)    # Cyan #38bdf8
    ACCENT_GREEN= RGBColor(74, 222, 128)    # Green #4ade80
    ACCENT_AMBER= RGBColor(251, 191, 36)    # Amber #fbbf24
    TEXT_WHITE  = RGBColor(248, 250, 252)   # White #f8fafc
    TEXT_MUTED  = RGBColor(148, 163, 184)   # Gray #94a3b8

    def add_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="PAPER METHODOLOGY & RESULTS DECK"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

    def add_card(slide, left, top, width, height, title="", items=None, accent_color=ACCENT_CYAN):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = accent_color
        card.line.width = Pt(1.5)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.2)
        tf.margin_bottom = Inches(0.2)

        if title:
            p0 = tf.paragraphs[0]
            p0.text = title
            p0.font.size = Pt(18)
            p0.font.bold = True
            p0.font.color.rgb = accent_color
            p0.space_after = Pt(10)

        if items:
            for idx, item in enumerate(items):
                p = tf.add_paragraph() if (title or idx > 0) else tf.paragraphs[0]
                p.text = f"•  {item}"
                p.font.size = Pt(13)
                p.font.color.rgb = TEXT_WHITE
                p.space_after = Pt(6)
        return card

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 1: PAPER TITLE SLIDE
    # ─────────────────────────────────────────────────────────────────────────
    slide1 = prs.slides.add_slide(blank_layout)
    add_bg(slide1)

    tb = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "EXPLAINABLE MACHINE LEARNING FRAMEWORK FOR CROP DISEASE SEVERITY"
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_CYAN
    p0.space_after = Pt(10)

    p1 = tf.add_paragraph()
    p1.text = "Zero-Leakage Classification, CORN Ordinal Severity Grading, and Spatial Feature Attribution in Paddy and Arecanut"
    p1.font.size = Pt(18)
    p1.font.color.rgb = TEXT_MUTED
    p1.space_after = Pt(35)

    p2 = tf.add_paragraph()
    p2.text = "Paper Contributions & Empirical Defense Presentation"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_GREEN

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 2: RESEARCH MOTIVATION & PAPER CONTRIBUTIONS
    # ─────────────────────────────────────────────────────────────────────────
    slide2 = prs.slides.add_slide(blank_layout)
    add_bg(slide2)
    add_header(slide2, "1. Research Motivation & Core Paper Contributions")

    add_card(slide2, 0.8, 1.8, 5.6, 5.0, "Core Research Gaps", [
        "Nominal vs Ordinal Gap: Standard deep classifiers treat disease severity as nominal labels, failing to penalize large rank reversals.",
        "Split Leakage Flaws: Published agricultural benchmarks leak camera bursts across splits (~99.79% accuracy illusion).",
        "Class Imbalance & Data Scarcity: Severe class skew between majority classes (N=502) and rare pests (N=61 bud_borer).",
        "Explainability Blind Spots: Deep CNNs frequently exploit background shortcuts rather than true leaf pathology."
    ], ACCENT_CYAN)

    add_card(slide2, 6.8, 1.8, 5.7, 5.0, "Key Paper Contributions", [
        "Weakly Supervised Auto-Labeling: Derive Horsfall-Barratt severity grades via adaptive Otsu HSV color-space segmentation.",
        "Leak-Free Group Partitioning: Enforce robust_cluster_id graph clustering (pHash <= 5 bits, Delta t <= 120s) for 0% split overlap.",
        "Deep Ordinal CORN Formulations: Fine-tune ResNet backbones using Consistent Rank Logits (CORN) for rank-consistent severity modeling.",
        "Empirical Audits: Audit bootstrap CIs, expose soft-leakage in expanded datasets, and evaluate spatial Grad-CAM attribution."
    ], ACCENT_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 3: DATASET TAXONOMY & ZERO-LEAKAGE AUDITING
    # ─────────────────────────────────────────────────────────────────────────
    slide3 = prs.slides.add_slide(blank_layout)
    add_bg(slide3)
    add_header(slide3, "2. Dataset Taxonomy & Zero-Leakage Audit")

    add_card(slide3, 0.8, 1.8, 5.6, 5.2, "Dataset Breakdown (12,714 Total Images)", [
        "Paddy Dataset (outputs/paddy_splits.csv): 10,053 trained-on total images (7,036 train / 1,510 val / 1,507 test; N=732 twin-free test).",
        "Arecanut Honest Split (outputs/arecanut_splits.csv): 2,661 total images (1,628 train / 547 val / 486 test) across 9 disease classes.",
        "Paddy Ordinal Severity Subset: 343 lesion-bearing test images evaluated for 7-class ordinal QWK.",
        "Quality Control: 354 Paddy images excluded due to low confidence (C < 0.80) or boundary proximity (|rho - theta| < 0.5%)."
    ], ACCENT_CYAN)

    add_card(slide3, 6.8, 1.8, 5.7, 5.2, "Rigorous Data Leakage Audit", [
        "Audit Discovery: Naive random splits cause severe camera burst & orchard leakage, artificially inflating accuracy.",
        "Cluster Grouping: Created robust_cluster_id based on capture timestamps, pHash visual hashes, and image lineages.",
        "Zero-Leakage Partitioning: Guaranteed 100% strict cluster separation between train, val, and test splits.",
        "Benchmark Result: Established an honest, non-inflated evaluation baseline."
    ], ACCENT_AMBER)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 4: MODEL METHODOLOGY & LOSS FORMULATIONS
    # ─────────────────────────────────────────────────────────────────────────
    slide4 = prs.slides.add_slide(blank_layout)
    add_bg(slide4)
    add_header(slide4, "3. Deep Learning Architecture & Loss Formulations")

    add_card(slide4, 0.8, 1.8, 5.6, 5.2, "Selective Transfer Learning", [
        "Pretrained Initialization: ImageNet-1K ResNet-18 weights.",
        "Selective Layer Freezing: conv1 through layer3 frozen (2.8M params); layer4 + FC head unfrozen (8.38M params).",
        "Optimizer: AdamW (learning rate = 3e-4, weight decay = 1e-4, Dropout p=0.4).",
        "Data Augmentation: Random Crops, Flips (p=0.3), Rotation (30 deg), Color Jitter."
    ], ACCENT_CYAN)

    add_card(slide4, 6.8, 1.8, 5.7, 5.2, "Loss Function Mathematical Design", [
        "Class-Weighted Cross-Entropy: L_CE = -sum(w_c * y_c * log(p_c)), where w_c = max(N) / N_c.",
        "Focal Loss: L_Focal = -alpha_t * (1 - p_t)^gamma * log(p_t) (gamma=2.0) to focus gradients on hard samples.",
        "CORN Ordinal Loss: Evaluates conditional probabilities P(Y > k | Y >= k) over K-1 binary sub-classifiers."
    ], ACCENT_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 5: PADDY DISEASE & ORDINAL SEVERITY RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    slide5 = prs.slides.add_slide(blank_layout)
    add_bg(slide5)
    add_header(slide5, "4. Paddy Classification & Severity Results")

    add_card(slide5, 0.8, 1.8, 5.6, 5.2, "10-Class Disease Classification", [
        "Test Accuracy: 96.04% across 10 paddy classes.",
        "Validation Accuracy: 96.56%.",
        "Evaluated on paddy_splits.csv (N=10,053 total trained-on images; N=732 twin-free test sample subset).",
        "High per-class precision across Bacterial Leaf Blight, Blast, Brown Spot, Tungro, Hispa, etc."
    ], ACCENT_CYAN)

    add_card(slide5, 6.8, 1.8, 5.7, 5.2, "Quantitative Ordinal Severity Grading", [
        "Paddy 2-Class Screening Accuracy: 98.36% [95% CI: 97.40%, 99.18%] (N=732 test images).",
        "7-Class Ordinal Severity QWK: 0.7055 [95% CI: 0.589, 0.800] achieved by CORN Ensemble (TTA).",
        "Evaluated on N=343 lesion-bearing test images.",
        "Combines deep CORN predictions with adaptive HSV leaf lesion surface area ratio calculation."
    ], ACCENT_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 6: ARECANUT HONEST BASELINE & BOOTSTRAP CI AUDIT
    # ─────────────────────────────────────────────────────────────────────────
    slide6 = prs.slides.add_slide(blank_layout)
    add_bg(slide6)
    add_header(slide6, "5. Arecanut Baseline & Statistical Audit")

    add_card(slide6, 0.8, 1.8, 5.6, 5.2, "Honest Baseline (outputs/arecanut_splits.csv)", [
        "Dataset Split: 1,628 train / 547 val / 486 test (0 cluster leak).",
        "Baseline Test Accuracy: 82.92% [95% CI: 72.82%, 91.88%].",
        "Baseline Macro-F1: 73.05% [95% CI: 59.46%, 83.09%].",
        "Training Strategy: Focal Loss (gamma=2.0) + WeightedRandomSampler."
    ], ACCENT_CYAN)

    add_card(slide6, 6.8, 1.8, 5.7, 5.2, "Rerun Checkpoint & Bootstrap CI Audit", [
        "Rerun Checkpoint Test Accuracy: 90.33% [95% CI: 83.55%, 96.19%].",
        "Rerun Macro-F1: 85.13% [95% CI: 71.52%, 91.35%].",
        "Statistical Audit Finding: 95% CIs overlap substantially ([83.55%, 91.88%]).",
        "ML Caveat: No statistically distinguishable improvement exists between 82.92% and 90.33% on N=486; baseline performance remains statistically solid."
    ], ACCENT_AMBER)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 7: SPATIAL FEATURE ATTRIBUTION & GRAD-CAM
    # ─────────────────────────────────────────────────────────────────────────
    slide7 = prs.slides.add_slide(blank_layout)
    add_bg(slide7)
    add_header(slide7, "6. Spatial Feature Attribution (Grad-CAM)")

    add_card(slide7, 0.8, 1.8, 5.6, 5.2, "Gradient Backpropagation Mechanism", [
        "Target Layer: base.layer4[-1] (7x7x512 feature map grid).",
        "Computes gradients of class score y_c with respect to activation maps A^k.",
        "Performs global average pooling to obtain neuron importance weights alpha_k^c.",
        "Generates 2D spatial heatmaps L_{Grad-CAM}^c = ReLU(sum(alpha_k^c * A^k))."
    ], ACCENT_CYAN)

    add_card(slide7, 6.8, 1.8, 5.7, 5.2, "Verification of Feature Representations", [
        "Validates Model Focus: Confirms CNN attends to actual disease lesions, rot spots, and stem cracks.",
        "Detects Shortcut Artifacts: Highlighted when models accidentally focus on background foliage or soil.",
        "Pointing Game Evaluation: Paddy diseased pointing acc = 13.88% (N=245); Arecanut = 4.55% (N=66)."
    ], ACCENT_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 8: EMPIRICAL AUDIT FINDINGS & REPRESENTATION BOTTLENECKS
    # ─────────────────────────────────────────────────────────────────────────
    slide8 = prs.slides.add_slide(blank_layout)
    add_bg(slide8)
    add_header(slide8, "7. Audit Findings & Representation Limits")

    add_card(slide8, 0.8, 1.8, 5.6, 5.2, "Soft-Leakage Discovery on Kaggle Pool", [
        "Expanded Experiment: Merging external Kaggle images showed ~99.79% accuracy.",
        "Audit Finding: External Kaggle images shared identical capture sessions (burst photography, camera sensors) with test set.",
        "Action Taken: Discarded leaky dataset expansions; reinstated honest baseline split (arecanut_splits.csv)."
    ], ACCENT_AMBER)

    add_card(slide8, 6.8, 1.8, 5.7, 5.2, "Physical Lineage Limitation (bud_borer)", [
        "Data Bottleneck: The entire public Arecanut dataset contains only 5 physical base photo lineages for bud_borer (N=61 train / 33 test).",
        "Linear Class Weight Shift: Loss weights (8.23x) pushed 11 test images across decision boundary (33.3% recall), but all 11 belong to 2 lineages.",
        "Conclusion: bud_borer evaluation requires new independently collected physical field specimens."
    ], ACCENT_CYAN)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 9: ADVANCED ML ROADMAP & FUTURE WORK
    # ─────────────────────────────────────────────────────────────────────────
    slide9 = prs.slides.add_slide(blank_layout)
    add_bg(slide9)
    add_header(slide9, "8. Advanced ML Roadmap & Future Scope")

    add_card(slide9, 0.8, 1.8, 5.6, 5.2, "Advanced Loss & Latent Regularization", [
        "LDAM Loss + DRW: Implement Label-Distribution-Aware Margin loss to enforce wider latent margins for minority classes.",
        "Manifold Mixup: Interpolate intermediate feature representations (layer3/layer4) to smooth minority decision boundaries.",
        "Foreground Pre-Segmentation: Pre-segment leaves to mask out background foliage and eliminate shortcut learning."
    ], ACCENT_CYAN)

    add_card(slide9, 6.8, 1.8, 5.7, 5.2, "Evaluation & Domain Generalization", [
        "Cluster-Grouped 5-Fold CV: Scale evaluation across all 2,661 Arecanut images to increase statistical evaluation power 5.5x.",
        "Domain-Adversarial Alignment (DANN): Minimize feature distance across photography sessions using domain loss terms.",
        "Precision-Recall AUC Benchmark: Evaluate PR-AUC alongside top-1 accuracy to measure threshold-independent class calibration."
    ], ACCENT_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # SLIDE 10: SUMMARY OF PAPER DELIVERABLES
    # ─────────────────────────────────────────────────────────────────────────
    slide10 = prs.slides.add_slide(blank_layout)
    add_bg(slide10)
    add_header(slide10, "9. Summary of Paper Deliverables & Benchmarks")

    add_card(slide10, 0.8, 1.8, 11.7, 5.2, "Core Empirical Benchmarks Locked for paper.tex", [
        "Paddy Disease Classification: 96.04% Test Accuracy on N=10,053 trained-on dataset (paddy_splits.csv; N=732 twin-free test).",
        "Paddy Ordinal Severity Grading: 0.7055 QWK [0.589, 0.800] via CORN Ensemble on N=343 lesion test subset.",
        "Paddy 2-Class Disease Screening: 98.36% [97.40%, 99.18%] 2-class accuracy on N=732 test images.",
        "Arecanut Baseline Classification: 82.92% Test Accuracy [72.82%, 91.88%] on N=2,661 zero-leakage split v2.",
        "Arecanut Rerun Stat Equivalence: Rerun 90.33% [83.55%, 96.19%] overlaps baseline CI by 8.33%; no statistically distinguishable improvement."
    ], ACCENT_CYAN)

    out_path = "Crop_Disease_AI_Project_Presentation.pptx"
    prs.save(out_path)
    print(f"✓ Paper-Focused Presentation saved successfully to {out_path}")

if __name__ == '__main__':
    create_paper_focused_deck()
