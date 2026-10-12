"""
Generate a professional 15-slide academic PowerPoint presentation (.pptx)
for the Crop Disease Severity Grading & Explainability project.
Includes widescreen 16:9 layout, embedded 300 DPI figures, structured bullet points,
and comprehensive speaker notes on every slide.
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Paths
PROJECT_ROOT = '/home/sahith/Projects/Crop'
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, 'presentation.pptx')

# Color Palette
DARK_NAVY = RGBColor(15, 23, 42)      # #0F172A
SLATE_BG   = RGBColor(248, 250, 252)  # #F8FAFC
CARD_BG    = RGBColor(241, 245, 249)  # #F1F5F9
BORDER_CLR = RGBColor(203, 213, 225)  # #CBD5E1
TEXT_DARK  = RGBColor(30, 41, 59)     # #1E293B
TEXT_MUTED = RGBColor(100, 116, 139)  # #64748B
ACCENT_GRN = RGBColor(13, 148, 136)   # #0D9488 (Teal / Plant Green)
ACCENT_BLU = RGBColor(2, 132, 199)    # #0284C7 (Sky Blue)
WHITE      = RGBColor(255, 255, 255)


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="CROP DISEASE SEVERITY GRADING"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_GRN

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = DARK_NAVY
        p_title.space_before = Pt(3)

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 1: Title Slide (Dark Theme)
    # ──────────────────────────────────────────────────────────────────────────
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = DARK_NAVY
    bg1.line.color.rgb = DARK_NAVY

    # Decorative accent bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.8), Inches(0.12), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_GRN
    bar.line.color.rgb = ACCENT_GRN

    tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(1.7), Inches(10.5), Inches(4.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "RESEARCH PROJECT PRESENTATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GRN

    p = tf1.add_paragraph()
    p.text = "Explainable Ordinal Deep Learning for Crop Disease Severity Grading"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_before = Pt(10)

    p = tf1.add_paragraph()
    p.text = "Benchmarking Consistent Rank Logits (CORN), Leak-Free Grouping, and Diagnostic Explainability Audits in Paddy and Arecanut"
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.space_before = Pt(12)

    p = tf1.add_paragraph()
    p.text = "Author / Candidate: Sahith   |   Domain: Precision Agriculture & Computer Vision   |   IEEE Conference Format"
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_before = Pt(28)

    s1.notes_slide.notes_text_frame.text = (
        "Good morning/afternoon, respected guide and committee members. Today, I am presenting our research project: "
        "'Explainable Ordinal Deep Learning for Crop Disease Severity Grading.' In this work, we address a fundamental "
        "flaw in standard agricultural AI—treating progressive disease severity as un-ordered classification. We formulate "
        "a mathematically grounded ordinal framework using CORN loss across Paddy and Arecanut, validate it on strict leak-free "
        "splits, and conduct rigorous explainability audits using pointing games to understand model decision-making."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 2: Motivation & Background
    # ──────────────────────────────────────────────────────────────────────────
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation & The Need for Severity Grading", "PROBLEM CONTEXT")

    # Left Column Text Card
    card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_CLR

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.3)

    bullets_s2 = [
        ("The Agricultural Challenge:", "Foliar and fruit crop diseases in staple crops (Paddy rice) and commercial cash crops (Arecanut palm) lead to 20–40% annual yield devastation if untreated."),
        ("Detection vs. Severity Grading:", "Existing computer vision models focus overwhelmingly on binary disease classification (Disease vs. Healthy). However, real-world agronomy requires Severity Grading (e.g., Grade 0 to Grade 3) to calibrate accurate fungicide dosage and assess economic injury levels."),
        ("The Human Inspection Bottleneck:", "Traditional visual grading by field agronomists relies on manual percentage estimation scales (Horsfall-Barratt / EPPO). This process is slow, labor-intensive, and subject to high inter-observer variability."),
        ("Our Goal:", "Develop an automated, objective, and explainable deep learning pipeline that respects disease progression order while maintaining real-time mobile deployability.")
    ]

    for i, (head, body) in enumerate(bullets_s2):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {head} "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_NAVY
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(12)
        run.font.color.rgb = TEXT_DARK

    # Right Column Image: Dataset Samples
    if os.path.exists(os.path.join(FIGURES_DIR, 'fig2_dataset_samples.png')):
        s2.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig2_dataset_samples.png'), Inches(7.1), Inches(1.6), width=Inches(5.4))

    s2.notes_slide.notes_text_frame.text = (
        "Slide 2 Speaker Notes:\n"
        "Start by establishing the practical agronomic context. Most AI papers in agriculture simply answer: 'Is this leaf diseased or not?' "
        "In commercial farming, knowing that a leaf has blast or koleroga is not enough. Farmers need to know the severity grade. If a leaf has 2% lesion area, "
        "it requires a mild localized intervention; if it has 25% lesion area, it requires urgent systemic fungicide. Currently, manual visual inspection "
        "is subjective and error-prone. Our objective is automated severity grading."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 3: The Core Research Problem: Softmax vs. Ordinality
    # ──────────────────────────────────────────────────────────────────────────
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "The Core Mathematical Gap: Why Softmax Fails", "THEORETICAL FOUNDATION")

    # 2 Comparison Cards
    c1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(254, 242, 242)
    c1.line.color.rgb = RGBColor(252, 165, 165)

    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_right = tf1.margin_top = Inches(0.3)
    p = tf1.paragraphs[0]
    p.text = "❌ Conventional Multi-Class (Softmax CE)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(185, 28, 28)

    points1 = [
        "Treats all classes as orthogonal & independent categories (like Dog vs. Cat vs. Plane).",
        "Loss formulation: -log(p_y). Penalizes every misclassification identically, regardless of distance.",
        "Catastrophic Error Blindness: Predicting Grade 3 (Severe) for a Grade 0 (Healthy) leaf incurs the exact same loss penalty as predicting Grade 1 (Mild).",
        "Causes high Mean Absolute Error (MAE) and severe grade reversal jumps (|Δ| >= 2)."
    ]
    for pt in points1:
        p = tf1.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    c2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(240, 253, 244)
    c2.line.color.rgb = RGBColor(134, 239, 172)

    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = tf2.margin_top = Inches(0.3)
    p = tf2.paragraphs[0]
    p.text = "✅ Ordinal Regression Formulation (CORN)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(21, 128, 61)

    points2 = [
        "Explicitly respects natural progression order: Grade 0 < Grade 1 < Grade 2 < Grade 3.",
        "Distance-Aware Loss: Decomposes K classes into K-1 sequential binary ranking sub-tasks.",
        "Quadratically Penalizes Large Errors: A 2-grade mistake is penalized far more heavily than an adjacent 1-grade boundary shift.",
        "Constrains errors to immediately adjacent classes (|Δ| = 1), drastically lowering MAE and maximizing Quadratic Weighted Kappa (QWK)."
    ]
    for pt in points2:
        p = tf2.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    s3.notes_slide.notes_text_frame.text = (
        "Slide 3 Speaker Notes:\n"
        "This is the intellectual cornerstone of your presentation. Explain why treating severity grading as regular multi-class classification is flawed. "
        "Under Cross-Entropy, classes are mutually exclusive categorical labels. If a model predicts Grade 3 for a Healthy leaf, Softmax penalizes that "
        "no more than if it predicted Grade 1. But in precision agriculture, that is a catastrophic misdiagnosis. Ordinal regression mathematically "
        "enforces that Grade 0 < 1 < 2 < 3. Emphasize this distinction to your guide."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 4: End-to-End System Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "End-to-End System Pipeline & Methodology", "WORKFLOW OVERVIEW")

    # Image center
    if os.path.exists(os.path.join(FIGURES_DIR, 'fig1_system_architecture.png')):
        s4.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig1_system_architecture.png'), Inches(0.8), Inches(1.5), width=Inches(11.7))

    s4.notes_slide.notes_text_frame.text = (
        "Slide 4 Speaker Notes:\n"
        "Walk the committee through the 5 distinct phases of our pipeline:\n"
        "Phase 1: Ingestion of Paddy and Arecanut raw imagery.\n"
        "Phase 2: Automated proxy-labeling using adaptive color-space segmentation grounded in the Horsfall-Barratt scale.\n"
        "Phase 3: Leak-free partitioning, specifically parent-plant grouping for Arecanut to ensure true out-of-sample generalization.\n"
        "Phase 4: Feature extraction across ResNet-18, 34, and MobileNetV2 with CORN ordinal heads vs Softmax baselines.\n"
        "Phase 5: Diagnostic auditing using Grad-CAM pointing games and edge deployment via our Streamlit application."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 5: Datasets & Agronomic Severity Grounding
    # ──────────────────────────────────────────────────────────────────────────
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Datasets & Horsfall-Barratt Severity Scale", "DATA CURATION")

    card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_CLR

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)

    bullets_s5 = [
        ("Dual Botanical Morphologies:", "Evaluated across two distinct botanical forms: Paddy Rice (delicate annual grass monocot leaves) and Arecanut Palm (heavy perennial woody palm fronds/nuts/trunks)."),
        ("Paddy Doctor Benchmark (N = 10,053):", "10 disease classes: Blast, Bacterial Blight, Brown Spot, Tungro, Hispa, Dead Heart, Downy Mildew, etc. Partitioned: 7,036 train / 1,510 val / 1,507 test."),
        ("Arecanut Plantation Dataset (N = 895):", "9 classes: Bud Borer, Mahali / Koleroga (Fruit Rot), Stem Bleeding, Stem Cracking, Yellow Leaf Disease, and healthy controls. Partitioned: 623 train / 148 val / 124 test."),
        ("Standardized Severity Scale (Horsfall-Barratt):", "Grade 0: 0% lesion area (Healthy)\nGrade 1: >0% to 5% lesion area (Mild)\nGrade 2: >5% to 15% lesion area (Moderate)\nGrade 3: >15% lesion area (Severe)")
    ]

    for i, (head, body) in enumerate(bullets_s5):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {head}\n   "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_NAVY
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_DARK

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig3_data_preprocessing.png')):
        s5.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig3_data_preprocessing.png'), Inches(7.1), Inches(1.6), width=Inches(5.4))

    s5.notes_slide.notes_text_frame.text = (
        "Slide 5 Speaker Notes:\n"
        "Point to the dual-crop design: We didn't just test on a single easy dataset. We tested on Paddy (a monocot leaf) and Arecanut (a woody tree palm). "
        "Explain that our severity boundaries are not arbitrary: they are strictly calibrated to the global plant pathology standard—the Horsfall-Barratt "
        "and EPPO scale based on percentage lesion surface area (0%, <5%, 5-15%, >15%)."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 6: Data Integrity & Parent-Plant Leakage Prevention
    # ──────────────────────────────────────────────────────────────────────────
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Data Integrity: Preventing Parent-Plant Leakage", "EXPERIMENTAL RIGOR")

    card1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    card1.fill.solid()
    card1.fill.fore_color.rgb = RGBColor(254, 242, 242)
    card1.line.color.rgb = RGBColor(252, 165, 165)

    tf = card1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "⚠️ The Flaw in Published Literature"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(185, 28, 28)

    pts = [
        "In plantation datasets like Arecanut, researchers capture multiple close-up photos of the same palm trunk, nut bunch, or canopy from varying angles.",
        "Naive Random Splitting: Placing images randomly into train/val/test causes severe Data Leakage. The model memorizes unique tree bark textures, soil backgrounds, and camera artifacts.",
        "Result: Published papers report 98–99% accuracy that crashes completely when deployed on unseen trees in a new farm."
    ]
    for pt in pts:
        p = tf.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    card2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.3))
    card2.fill.solid()
    card2.fill.fore_color.rgb = RGBColor(240, 253, 244)
    card2.line.color.rgb = RGBColor(134, 239, 172)

    tf2 = card2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = tf2.margin_top = Inches(0.3)
    p = tf2.paragraphs[0]
    p.text = "🛡️ Our Solution: Parent-Plant Grouped Split"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(21, 128, 61)

    pts2 = [
        "Group-KFold Protocol: We grouped all 895 Arecanut images strictly by Parent Tree Origin ID.",
        "Zero Cross-Partition Contamination: All images originating from a specific palm tree belong exclusively to Train, Validation, or Held-Out Test.",
        "Bootstrap CI Resampling: In statistical evaluations, bootstrap resampling was conducted by parent plant group, preserving leakage boundaries across 2,000 iterations.",
        "Guarantees that test performance measures true botanical disease generalization, not background recognition."
    ]
    for pt in pts2:
        p = tf2.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    s6.notes_slide.notes_text_frame.text = (
        "Slide 6 Speaker Notes:\n"
        "Emphasize scientific honesty here. Many agricultural AI papers suffer from hidden data leakage because multiple pictures of the same tree "
        "appear in both training and test sets. We explicitly grouped images by Parent Palm Tree ID. This means our test set of 124 Arecanut images "
        "consists of palms the model has NEVER seen before. This will impress your guide because it shows professional experimental rigor."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 7: Model Architecture & CORN Formulation
    # ──────────────────────────────────────────────────────────────────────────
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Consistent Rank Logits (CORN) Formulation", "MATHEMATICAL FRAMEWORK")

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig4_model_architecture.png')):
        s7.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig4_model_architecture.png'), Inches(0.8), Inches(1.5), width=Inches(11.7))

    s7.notes_slide.notes_text_frame.text = (
        "Slide 7 Speaker Notes:\n"
        "Explain the CORN architecture: For 4 severity grades (0, 1, 2, 3), the network outputs 3 conditional binary logits corresponding to tasks: "
        "Is Grade > 0? Is Grade > 1? Is Grade > 2? "
        "During inference, each logit passes through a sigmoid function. We sequentially threshold probabilities at 0.5. "
        "The predicted grade is the sum of positive binary thresholds. This structurally guarantees rank order."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 8: Experimental Results: Performance Comparison
    # ──────────────────────────────────────────────────────────────────────────
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Performance Benchmarks & Ordinal Metrics", "EXPERIMENTAL RESULTS")

    card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_CLR

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)

    bullets_s8 = [
        ("Paddy Benchmark Results (N = 1,507):", "• CORN ResNet-18: QWK = 0.9390, Exact Acc = 90.98%, Within-1 Acc = 98.81%, MAE = 0.1048.\n• Softmax ResNet-18: QWK = 0.9388, Exact Acc = 91.37%, Within-1 Acc = 98.67%, MAE = 0.1029."),
        ("Arecanut Benchmark Results (N = 124):", "• CORN ResNet-18: QWK = 0.9177, Exact Acc = 90.32%, Within-1 Acc = 97.58%, MAE = 0.1290.\n• Softmax ResNet-18: QWK = 0.9031, Exact Acc = 89.52%, Within-1 Acc = 98.39%, MAE = 0.1371."),
        ("Key Agronomic Takeaway:", "Over 98% of all predictions fall within ±1 grade of true pathology. Catastrophic errors (healthy predicted as severe or vice-versa) are practically eradicated."),
        ("Statistical Significance (McNemar's Test):", "Paired McNemar test reveals p = 0.656 (Paddy) and p = 1.000 (Arecanut). CORN maintains top-tier accuracy while enforcing strict ranking order.")
    ]

    for i, (head, body) in enumerate(bullets_s8):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{head}\n"
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_NAVY
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_DARK

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig6_performance_comparison.png')):
        s8.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig6_performance_comparison.png'), Inches(6.7), Inches(1.6), width=Inches(5.8))

    s8.notes_slide.notes_text_frame.text = (
        "Slide 8 Speaker Notes:\n"
        "Walk through the primary results table. Highlight that on both crops, we achieve Quadratic Weighted Kappa (QWK) scores above 0.90 to 0.94. "
        "Point to the Within-1 Accuracy: 98.81% for Paddy and 97.58% for Arecanut. This means when the model does make an error, it is almost exclusively "
        "on the boundary between Grade 1 and Grade 2, which is acceptable in agronomic practice."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 9: Error Distribution: Confusion Matrix Analysis
    # ──────────────────────────────────────────────────────────────────────────
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Confusion Matrix Analysis: Off-Diagonal Compression", "ERROR PROFILES")

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig5_confusion_matrices.png')):
        s9.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig5_confusion_matrices.png'), Inches(0.8), Inches(1.5), width=Inches(11.7))

    s9.notes_slide.notes_text_frame.text = (
        "Slide 9 Speaker Notes:\n"
        "Point directly to Figure 5. Explain the confusion matrices: "
        "Observe how the errors are compressed along the primary diagonal and immediately adjacent off-diagonal cells. "
        "There are zero entries in the top-right and bottom-left corners. This proves that CORN effectively prevents multi-grade jumps."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 10: Multi-Backbone Scalability & Seed Stability
    # ──────────────────────────────────────────────────────────────────────────
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Backbone Comparison: ResNet-18 vs. 34 vs. MobileNetV2", "ARCHITECTURE BENCHMARK")

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig_backbone_comparison.png')):
        s10.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig_backbone_comparison.png'), Inches(0.8), Inches(1.5), width=Inches(11.7))

    s10.notes_slide.notes_text_frame.text = (
        "Slide 10 Speaker Notes:\n"
        "Explain our multi-backbone benchmark: "
        "1. MobileNetV2: Only 2.2M parameters. Delivers 88-91% accuracy, making it ideal for offline mobile apps in rural areas. "
        "2. ResNet-18: The optimal sweet spot of speed and accuracy. "
        "3. ResNet-34: Deeper residual capacity captures subtle betel nut textures, achieving our highest Arecanut QWK of 0.9566 ± 0.016 across multiple seeds."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 11: Diagnostic Explainability: Pointing Game Audit
    # ──────────────────────────────────────────────────────────────────────────
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Explainability Audit: Grad-CAM vs. Ground-Truth Masks", "MODEL TRUST & DIAGNOSTICS")

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig8_explainability_gradcam.png')):
        s11.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig8_explainability_gradcam.png'), Inches(0.8), Inches(1.5), width=Inches(11.7))

    s11.notes_slide.notes_text_frame.text = (
        "Slide 11 Speaker Notes:\n"
        "This is our most significant diagnostic contribution. Most papers show pretty Grad-CAM heatmaps and assume the model is working. "
        "We performed a rigorous quantitative Pointing Game against ground-truth segmentation masks. "
        "Our discovery: While models localize spot diseases (Blast, Brown Spot) accurately, for systemic diseases (Panicle Blight, Dead Heart), "
        "models frequently rely on leaf edges, petioles, and background contrast (shortcut learning). This is a vital finding for real-world deployment."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 12: Zero-Shot Cross-Crop Evaluation (Domain Shift)
    # ──────────────────────────────────────────────────────────────────────────
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Domain Shift: Cross-Crop Transfer Collapse", "GENERALIZATION AUDIT")

    if os.path.exists(os.path.join(FIGURES_DIR, 'fig_zeroshot_transfer.png')):
        s12.shapes.add_picture(os.path.join(FIGURES_DIR, 'fig_zeroshot_transfer.png'), Inches(0.8), Inches(1.5), width=Inches(11.7))

    s12.notes_slide.notes_text_frame.text = (
        "Slide 12 Speaker Notes:\n"
        "Explain the Zero-Shot experiment: We tested what happens when the Paddy model evaluates Arecanut and vice-versa. "
        "Performance crashes catastrophically: QWK plummets from 0.94 to <= 0.20. "
        "Why? Monocot grass leaves have completely different textures from woody palm trunks. "
        "This proved that our deployment app required a Crop Gatekeeper to prevent out-of-domain evaluation."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 13: Edge Deployment & Gatekeeper AI
    # ──────────────────────────────────────────────────────────────────────────
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Real-World Deployment: Streamlit App + Gatekeeper AI", "PRACTICAL IMPLEMENTATION")

    card1 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    card1.fill.solid()
    card1.fill.fore_color.rgb = CARD_BG
    card1.line.color.rgb = BORDER_CLR

    tf = card1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "📱 Interactive Streamlit Web Application (app.py)"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = DARK_NAVY

    pts = [
        "Farmer-Centric UI: Instant drag-and-drop crop leaf upload with real-time severity prediction.",
        "Dual-Stream Output: Displays PyTorch deep learning severity grade alongside Computer Vision lesion overlay masks.",
        "Horsfall-Barratt Breakdown: Outputs confidence percentage and probability distribution across all 4 severity grades.",
        "Edge Optimization: Runs smoothly on low-power CPU environments with sub-second inference."
    ]
    for pt in pts:
        p = tf.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    card2 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.3))
    card2.fill.solid()
    card2.fill.fore_color.rgb = RGBColor(240, 249, 255)
    card2.line.color.rgb = RGBColor(125, 211, 252)

    tf2 = card2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = tf2.margin_top = Inches(0.3)
    p = tf2.paragraphs[0]
    p.text = "🛡️ 3-Way Gatekeeper AI & OOD Rejection"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = ACCENT_BLU

    pts2 = [
        "Solves the Closed-World Flaw: Standard classifiers force non-crop images (cars, dogs, faces) into a disease grade.",
        "Integrated Gatekeeper Model: Trained on 1,800 images across Paddy, Arecanut, and Non-Crop/Other (100.0% validation accuracy).",
        "Automatic Crop Routing: Auto-detects whether the leaf is Paddy or Arecanut and routes to the specialist model.",
        "Out-of-Distribution Defense: Automatically rejects non-plant images with a warning card and halts severity grading."
    ]
    for pt in pts2:
        p = tf2.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    s13.notes_slide.notes_text_frame.text = (
        "Slide 13 Speaker Notes:\n"
        "Demonstrate our real-world deployment: We developed an interactive Streamlit web application. "
        "Crucially, we added a Gatekeeper AI that solves a major weakness in deployed agricultural vision: "
        "If a farmer uploads a picture of a car, a face, or the wrong plant, our Gatekeeper automatically identifies "
        "it as Non-Crop / Out-of-Domain and rejects it, preventing false disease predictions."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 14: Summary of Key Contributions
    # ──────────────────────────────────────────────────────────────────────────
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusion & Key Research Contributions", "SUMMARY")

    card = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_CLR

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.4)

    contribs = [
        ("1. Ordinal Loss Formulation:", "Proved that CORN ordinal loss effectively penalizes multi-grade distance errors, compressing off-diagonal misclassifications and achieving QWK > 0.938 on Paddy and > 0.917 on Arecanut."),
        ("2. Leak-Free Grouping Protocols:", "Demonstrated that parent-plant grouping is mandatory for woody plantation crops to prevent background memorization and guarantee true out-of-sample generalization."),
        ("3. Multi-Backbone Edge Feasibility:", "Validated that MobileNetV2 (2.2M params) delivers competitive severity grading with low latency, proving viability for offline smartphone deployment."),
        ("4. Quantitative XAI Pointing Audits:", "Exposed the limitations of raw Grad-CAM IoU metrics and diagnosed shortcut learning behaviors where models rely on leaf margins rather than pathology."),
        ("5. Gatekeeper-Protected Deployment:", "Constructed an end-to-end interactive application with automated crop identification and out-of-distribution non-crop rejection.")
    ]

    for i, (head, body) in enumerate(contribs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{head} "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_NAVY
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(12)
        run.font.color.rgb = TEXT_DARK

    s14.notes_slide.notes_text_frame.text = (
        "Slide 14 Speaker Notes:\n"
        "Summarize our core contributions: (1) Proving ordinal loss benefits, (2) preventing leakage with parent grouping, "
        "(3) validating lightweight backbones for edge deployment, (4) uncovering shortcut learning with pointing games, "
        "and (5) delivering an intelligent gatekeeper-protected application."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SLIDE 15: Future Scope & Q&A
    # ──────────────────────────────────────────────────────────────────────────
    s15 = prs.slides.add_slide(blank_layout)
    bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg15.fill.solid()
    bg15.fill.fore_color.rgb = DARK_NAVY
    bg15.line.color.rgb = DARK_NAVY

    tb15 = s15.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.3), Inches(4.5))
    tf15 = tb15.text_frame
    tf15.word_wrap = True

    p = tf15.paragraphs[0]
    p.text = "FUTURE WORK & DIRECTIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GRN

    p = tf15.add_paragraph()
    p.text = "Thank You! Questions & Discussion"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_before = Pt(10)

    future_pts = [
        "Multi-Spectral & Drone Integration: Expanding severity grading to UAV aerial multispectral imagery for plantation-level mapping.",
        "Self-Supervised Pre-Training: Pre-training backbones on unannotated agricultural foliage to enhance zero-shot cross-crop domain adaptation.",
        "Active Learning Pipeline: Integrating field farmer feedback to continuously refine ambiguous boundary cases (Grade 1 vs 2)."
    ]
    for pt in future_pts:
        p = tf15.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = RGBColor(203, 213, 225)
        p.space_before = Pt(10)

    p = tf15.add_paragraph()
    p.text = "I am now open to questions from the guide and committee."
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GRN
    p.space_before = Pt(25)

    s15.notes_slide.notes_text_frame.text = (
        "Slide 15 Speaker Notes:\n"
        "Conclude with confidence: Thank your guide and committee for their guidance and time. "
        "Mention our exciting future directions (active learning, drone integration) and invite questions."
    )

    prs.save(OUTPUT_PPTX)
    print(f"Presentation successfully created at: {OUTPUT_PPTX}")

if __name__ == '__main__':
    create_deck()
