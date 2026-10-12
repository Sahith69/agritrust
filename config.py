"""
Central configuration for the Explainable ML Framework for Ordinal Disease Severity Grading.
All paths, hyperparameters, and constants in one place.
"""
import os

# ─── Project Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Data Paths ─────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Arecanut
ARECANUT_DIR = os.path.join(DATA_DIR, "arecanut", "dataset", "train")
ARECANUT_ANNOTATIONS = os.path.join(PROJECT_ROOT, "arecanut_annotations.json")

# Paddy Doctor
PADDY_TRAIN_DIR = os.path.join(DATA_DIR, "paddy", "paddy-disease-classification", "train_images")
PADDY_TRAIN_CSV = os.path.join(DATA_DIR, "paddy", "paddy-disease-classification", "train.csv")

# ─── Output Paths ───────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
NEEDS_REVIEW_DIR = os.path.join(OUTPUTS_DIR, "needs_review")
DEBUG_GALLERY_DIR = os.path.join(OUTPUTS_DIR, "debug_gallery")

# Processed data outputs
PADDY_SEVERITY_CSV = os.path.join(OUTPUTS_DIR, "paddy_severity_labels.csv")
PADDY_REVIEW_CSV = os.path.join(OUTPUTS_DIR, "needs_review", "paddy_low_confidence.csv")
ARECANUT_SPLIT_CSV = os.path.join(OUTPUTS_DIR, "arecanut_splits.csv")
PADDY_SPLIT_CSV = os.path.join(OUTPUTS_DIR, "paddy_splits.csv")

# ─── Arecanut Class Mapping ────────────────────────────────────
ARECANUT_CLASSES = {
    0: "bud_borer",
    1: "healthy_foot",
    2: "healthy_leaf",
    3: "healthy_nut",
    4: "healthy_trunk",
    5: "mahali_koleroga",
    6: "stem_cracking",
    7: "stem_bleeding",
    8: "yellow_leaf_disease",
}

# Which arecanut classes are "healthy" (severity grade 0 by definition)
ARECANUT_HEALTHY_CLASSES = {1, 2, 3, 4}  # healthy_foot, healthy_leaf, healthy_nut, healthy_trunk

# Arecanut classes in Group A (lesion-based segmentation)
ARECANUT_GROUP_A = {"mahali_koleroga", "stem_bleeding", "yellow_leaf_disease"}

# ─── Paddy Class Mapping ───────────────────────────────────────
PADDY_CLASSES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

# Group A: lesion/spot diseases (segmentation-based severity)
PADDY_GROUP_A = [
    "blast",
    "brown_spot",
    "tungro",
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "downy_mildew",
]

# Group B: structurally different diseases (custom proxies)
PADDY_GROUP_B_DEAD_HEART = "dead_heart"
PADDY_GROUP_B_HISPA = "hispa"
PADDY_NORMAL = "normal"

# ─── Apple Class Mapping ─────────────────────────────────────────
APPLE_CLASSES = [
    "apple_scab",
    "black_rot",
    "cedar_apple_rust",
    "healthy",
]
APPLE_HEALTHY_CLASSES = {"healthy"}
APPLE_GROUP_A = {"apple_scab", "black_rot", "cedar_apple_rust"}

# ─── Phase 1: Severity Labeling ────────────────────────────────
# Horsfall-Barratt / EPPO standard thresholds
SEVERITY_THRESHOLDS = {
    # ratio boundaries: (lower_bound, upper_bound) → grade
    # 0%           → Grade 0
    # 0% – 5%      → Grade 1
    # 5% – 15%     → Grade 2
    # > 15%         → Grade 3
    "grade_0_max": 0.0,    # exactly 0
    "grade_1_max": 0.05,   # up to 5%
    "grade_2_max": 0.15,   # up to 15%
    # above 15% → grade 3
}

# Segmentation parameters
LEAF_MASK_EROSION_PX = 9         # Erode leaf mask to avoid boundary artifacts
MIN_LESION_BLOB_PX = 15          # Minimum connected component size for lesions

# Confidence scoring parameters
LEAF_AREA_MIN_PCT = 0.05         # Minimum expected leaf area (5% of image)
LEAF_AREA_MAX_PCT = 0.75         # Maximum expected leaf area (75% of image)
CONFIDENCE_AUTO_ACCEPT = 80      # Auto-accept threshold (0-100%)

# ─── Phase 2: Data Splitting ───────────────────────────────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ─── Phase 3: Model Training ───────────────────────────────────
NUM_SEVERITY_GRADES = 4          # 0, 1, 2, 3
NUM_ORDINAL_TASKS = 3            # K-1 binary tasks for CORN
BACKBONE = "resnet18"
BACKBONE_NAME = "resnet18"
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation settings (Phase 3)
AUGMENTATION = {
    "rotation_degrees": 30,
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.3,
    "brightness_jitter": 0.15,
    "contrast_jitter": 0.15,
    # NO heavy color jitter — color is a meaningful disease signal
}

# ─── Phase 4: Explainability ───────────────────────────────────
GRADCAM_TARGET_LAYER = "layer4"  # Last conv block of ResNet
GRADCAM_NUM_SAMPLES = 50         # Number of test images to visualize


def ensure_dirs():
    """Create all output directories."""
    for d in [OUTPUTS_DIR, MODELS_DIR, GRADCAM_DIR, REPORTS_DIR,
              NEEDS_REVIEW_DIR, DEBUG_GALLERY_DIR]:
        os.makedirs(d, exist_ok=True)
