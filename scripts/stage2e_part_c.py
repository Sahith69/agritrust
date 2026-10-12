import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage
import cv2

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, detect_lesions_group_a, compute_ratio, otsu_threshold, get_disk
from src.severity_grader import ratio_to_grade

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2e - PART C: PADDY LABEL GENERATOR UNFORCED & SEGMENTATION COMPARISON")
print("=" * 80)

# Load Paddy splits
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
paddy_normal = df_paddy[df_paddy['disease_class'] == 'normal'].copy()

# Sample 100 images with fixed seed 42
sample_100 = paddy_normal.sample(n=100, random_state=42).reset_index(drop=True)

# Define fast_detect_lesions as in scripts/prepare_kaggle_merged_cache.py
DISK_STRUCT = get_disk(config.LEAF_MASK_EROSION_PX)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def fast_detect_lesions(img_array, leaf_mask):
    eroded_mask = scipy.ndimage.binary_erosion(leaf_mask, structure=DISK_STRUCT)
    img_float = img_array.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    thresh = otsu_threshold(gray, mask=eroded_mask)
    lesion_mask = (gray < thresh) & eroded_mask
    labeled, num_features = scipy.ndimage.label(lesion_mask)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        valid_labels = np.where(counts >= MIN_LESION_BLOB)[0]
        valid_labels = valid_labels[valid_labels > 0]
        lesion_mask = np.isin(labeled, valid_labels)
    else:
        lesion_mask = np.zeros_like(lesion_mask, dtype=bool)
    return lesion_mask

results_actual_grader = []
results_fast = []
mask_diffs = []
ratio_diffs = []
grade_diffs = []

print("Running actual Paddy label generator (detect_lesions_group_a) and fast_detect_lesions on 100 Paddy normal images...")

for idx, row in sample_100.iterrows():
    img_path = row['image_path']
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    
    # Organ segmentation (common to both)
    leaf_mask = segment_leaf_adaptive(arr)
    
    # 1. Actual unforced label_paddy.py / grade_image path (detect_lesions_group_a)
    lesion_mask_orig = detect_lesions_group_a(arr, leaf_mask)
    ratio_orig = compute_ratio(lesion_mask_orig, leaf_mask)
    grade_orig = ratio_to_grade(ratio_orig)
    
    # 2. fast_detect_lesions path
    lesion_mask_fast = fast_detect_lesions(arr, leaf_mask)
    ratio_fast = compute_ratio(lesion_mask_fast, leaf_mask)
    grade_fast = ratio_to_grade(ratio_fast)
    
    # Compare
    px_diff = np.sum(lesion_mask_orig != lesion_mask_fast)
    r_diff = abs(ratio_orig - ratio_fast)
    g_diff = abs(grade_orig - grade_fast)
    
    results_actual_grader.append({
        'filename': row['filename'],
        'ratio': float(ratio_orig),
        'grade': int(grade_orig)
    })
    results_fast.append({
        'filename': row['filename'],
        'ratio': float(ratio_fast),
        'grade': int(grade_fast)
    })
    mask_diffs.append(int(px_diff))
    ratio_diffs.append(float(r_diff))
    grade_diffs.append(int(g_diff))

r_orig_arr = np.array([r['ratio'] for r in results_actual_grader]) * 100.0
g_orig_arr = np.array([r['grade'] for r in results_actual_grader])

r_fast_arr = np.array([r['ratio'] for r in results_fast]) * 100.0
g_fast_arr = np.array([r['grade'] for r in results_fast])

print(f"\n1. Actual Paddy label generator (UNFORCED) on 100 Paddy normal images:")
print(f"   Min Ratio:    {np.min(r_orig_arr):.2f}%")
print(f"   25th %:       {np.percentile(r_orig_arr, 25):.2f}%")
print(f"   Median Ratio: {np.median(r_orig_arr):.2f}%")
print(f"   Mean Ratio:   {np.mean(r_orig_arr):.2f}%")
print(f"   75th %:       {np.percentile(r_orig_arr, 75):.2f}%")
print(f"   Max Ratio:    {np.max(r_orig_arr):.2f}%")
print(f"   Unforced Grades Produced: G0={int((g_orig_arr==0).sum())}, G1={int((g_orig_arr==1).sum())}, G2={int((g_orig_arr==2).sum())}, G3={int((g_orig_arr==3).sum())}")

print(f"\n2. Comparison between detect_lesions_group_a and fast_detect_lesions across 100 images:")
print(f"   Total pixel mask differences across all 100 images: {sum(mask_diffs)} pixels")
print(f"   Max ratio difference: {max(ratio_diffs):.6e}")
print(f"   Grade differences: {sum(grade_diffs)} / 100")

# Explanation of Kaggle labels
kaggle_explanation = (
    "CONFIRMATION OF KAGGLE LABEL GENERATION:\n"
    "1. Did Kaggle labels use fast_detect_lesions or segment_leaf_adaptive?\n"
    "   Kaggle labeling in scripts/prepare_kaggle_merged_cache.py used BOTH in sequence as distinct stages of the pipeline:\n"
    "   - Stage 1: segment_leaf_adaptive(arr) extracts the organ boundary mask (ExG Otsu in Lab/RGB space).\n"
    "   - Stage 2: fast_detect_lesions(arr, leaf_mask) extracts the lesion mask INSIDE that organ boundary mask\n"
    "     using eroded grayscale Otsu thresholding followed by minimum blob size filtering.\n"
    "   They are not alternative algorithms; segment_leaf_adaptive produces the leaf mask, and fast_detect_lesions\n"
    "   detects lesions on the leaf mask.\n"
    "2. Does fast_detect_lesions differ from detect_lesions_group_a (in src/segmentation.py)?\n"
    "   - Algorithmically, fast_detect_lesions is a vectorized drop-in for detect_lesions_group_a.\n"
    "   - In detect_lesions_group_a, filter_small_components uses a pure-python BFS queue (which is prohibitively slow\n"
    "     on multi-megapixel images).\n"
    "   - In fast_detect_lesions, filter_small_components is replaced by scipy.ndimage.label + np.bincount.\n"
    "   - The mathematical logic (grayscale weights 0.2989/0.5870/0.1140, Otsu threshold, erosion disk, and blob threshold 100 px)\n"
    "     is completely identical."
)
print("\n" + kaggle_explanation)

output_json = {
    'unforced_paddy_normal_100_eval': {
        'N': 100,
        'min_pct': float(np.min(r_orig_arr)),
        'p25_pct': float(np.percentile(r_orig_arr, 25)),
        'median_pct': float(np.median(r_orig_arr)),
        'mean_pct': float(np.mean(r_orig_arr)),
        'p75_pct': float(np.percentile(r_orig_arr, 75)),
        'max_pct': float(np.max(r_orig_arr)),
        'grades_produced': {
            'G0': int((g_orig_arr == 0).sum()),
            'G1': int((g_orig_arr == 1).sum()),
            'G2': int((g_orig_arr == 2).sum()),
            'G3': int((g_orig_arr == 3).sum())
        }
    },
    'comparison_actual_vs_fast': {
        'total_pixel_differences': int(sum(mask_diffs)),
        'max_ratio_difference': float(max(ratio_diffs)),
        'grade_discrepancies_count': int(sum(grade_diffs))
    },
    'kaggle_pipeline_explanation': kaggle_explanation
}

with open('outputs/final_v2/stage2e_part_c.json', 'w') as f:
    json.dump(output_json, f, indent=2)

print("\nSaved Stage 2e Part C results to outputs/final_v2/stage2e_part_c.json")
