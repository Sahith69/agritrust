import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import scipy.ndimage
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2d - SECTION B: RATIO VALIDITY & HSV COMPARISON")
print("=" * 80)

np.random.seed(42)

DISK_STRUCT = get_disk(config.LEAF_MASK_EROSION_PX)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def compute_lesion_mask_and_ratio(img_rgb, leaf_mask):
    eroded = scipy.ndimage.binary_erosion(leaf_mask, structure=DISK_STRUCT)
    img_float = img_rgb.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    thresh = otsu_threshold(gray, mask=eroded)
    lesion_mask = (gray < thresh) & eroded
    labeled, num_features = scipy.ndimage.label(lesion_mask)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        valid_labels = np.where(counts >= MIN_LESION_BLOB)[0]
        valid_labels = valid_labels[valid_labels > 0]
        lesion_mask = np.isin(labeled, valid_labels)
    else:
        lesion_mask = np.zeros_like(lesion_mask, dtype=bool)
    ratio = compute_ratio(lesion_mask, leaf_mask)
    return lesion_mask, ratio

def ratio_to_grade_eq2(r):
    if r <= 0.0: return 0
    elif r <= 0.05: return 1
    elif r <= 0.15: return 2
    else: return 3

# ==============================================================================
# B.1 MEASURING HEALTHY CLASSES WITHOUT FORCING ZERO
# ==============================================================================
print("\n--- B.1 RUNNING EQ. 2 PIPELINE ON HEALTHY IMAGES WITHOUT FORCING ZERO ---")
areca_df = pd.read_csv('outputs/final_v2/arecanut_clean_v2.csv')
paddy_df = pd.read_csv('outputs/paddy_splits.csv')

healthy_classes_to_test = ['healthy_trunk', 'healthy_leaf', 'healthy_nut', 'healthy_foot']
healthy_results = {}

for hc in healthy_classes_to_test:
    sub = areca_df[areca_df['disease_class'] == hc]
    n_sample = min(100, len(sub))
    sampled = sub.sample(n=n_sample, random_state=42).copy()
    
    ratios = []
    grades = []
    for idx, row in sampled.iterrows():
        im_bgr = cv2.imread(row['image_path'])
        if im_bgr is None:
            continue
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        leaf_mask = segment_leaf_adaptive(im_rgb)
        _, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
        ratios.append(float(r) * 100.0)
        grades.append(ratio_to_grade_eq2(r))
        
    r_arr = np.array(ratios)
    healthy_results[hc] = {
        'N_tested': len(r_arr),
        'min_pct': float(np.min(r_arr)),
        'p25_pct': float(np.percentile(r_arr, 25)),
        'median_pct': float(np.median(r_arr)),
        'mean_pct': float(np.mean(r_arr)),
        'std_pct': float(np.std(r_arr)),
        'p75_pct': float(np.percentile(r_arr, 75)),
        'max_pct': float(np.max(r_arr)),
        'grades_produced': {
            'G0_0pct': int((r_arr <= 0.0).sum()),
            'G1_0to5pct': int(((r_arr > 0.0) & (r_arr <= 5.0)).sum()),
            'G2_5to15pct': int(((r_arr > 5.0) & (r_arr <= 15.0)).sum()),
            'G3_gt15pct': int((r_arr > 15.0).sum())
        }
    }
    print(f"Class: {hc:15} (N={len(r_arr)}) | Median={healthy_results[hc]['median_pct']:.2f}% | Mean={healthy_results[hc]['mean_pct']:.2f}% | Grades: {healthy_results[hc]['grades_produced']}")

# Paddy normal:
paddy_normal_sub = paddy_df[paddy_df['disease_class'] == 'normal']
paddy_normal_sampled = paddy_normal_sub.sample(n=100, random_state=42).copy()
paddy_norm_ratios = []
paddy_norm_grades = []

for idx, row in paddy_normal_sampled.iterrows():
    im_bgr = cv2.imread(row['image_path'])
    if im_bgr is None:
        continue
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    _, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
    paddy_norm_ratios.append(float(r) * 100.0)
    paddy_norm_grades.append(ratio_to_grade_eq2(r))

p_arr = np.array(paddy_norm_ratios)
healthy_results['paddy_normal'] = {
    'N_tested': len(p_arr),
    'min_pct': float(np.min(p_arr)),
    'p25_pct': float(np.percentile(p_arr, 25)),
    'median_pct': float(np.median(p_arr)),
    'mean_pct': float(np.mean(p_arr)),
    'std_pct': float(np.std(p_arr)),
    'p75_pct': float(np.percentile(p_arr, 75)),
    'max_pct': float(np.max(p_arr)),
    'grades_produced': {
        'G0_0pct': int((p_arr <= 0.0).sum()),
        'G1_0to5pct': int(((p_arr > 0.0) & (p_arr <= 5.0)).sum()),
        'G2_5to15pct': int(((p_arr > 5.0) & (p_arr <= 15.0)).sum()),
        'G3_gt15pct': int((p_arr > 15.0).sum())
    }
}
print(f"Class: {'paddy_normal':15} (N={len(p_arr)}) | Median={healthy_results['paddy_normal']['median_pct']:.2f}% | Mean={healthy_results['paddy_normal']['mean_pct']:.2f}% | Grades: {healthy_results['paddy_normal']['grades_produced']}")

# Explanation of Stage 2c Part B:
provenance_explanation = (
    "In Stage 2c Part B (scripts/stage2c_part_b.py, lines 68-71), healthy classes were explicitly hardcoded to 0.00%:\n"
    "    if cls in HEALTHY_CLASSES:\n"
    "        orig_ratios.append(0.0)\n"
    "        orig_eq2_grades.append(0)\n"
    "        continue\n"
    "Thus, the 0.00% values reported in Part B were FORCED by design to reflect biological ground truth (healthy organs have no disease),\n"
    "rather than raw algorithmic output from the Otsu pipeline. When the classical Eq. 2 pipeline is run unconstrained on healthy organs,\n"
    "Otsu thresholding partitions natural shadow gradients, bark crevices, and chlorophyll gradients into pseudo-lesions."
)
print("\nProvenance of Stage 2c Part B 0.00% healthy values:")
print(provenance_explanation)

# ==============================================================================
# B.2 SECOND MEASURE: FRACTION OF ORGAN PIXELS IN BROWN/DARK HSV RANGE
# ==============================================================================
print("\n--- B.2 SECOND MEASURE: BROWN/DARK HSV RANGE CORRELATION ---")
# Definition:
# In HSV color space:
# 1. Necrotic Brown/Rust: Hue in [8, 32], Saturation in [40, 255], Value in [25, 210]
# 2. Dark/Black Necrosis: Value < 45, Saturation >= 20
# Total necrotic pixels = (Brown | Dark) & leaf_mask
hsv_definition = {
    'space': 'HSV (OpenCV H:[0,179], S:[0,255], V:[0,255])',
    'brown_range': {'H': [8, 32], 'S': [40, 255], 'V': [25, 210]},
    'dark_range': {'V_max': 45, 'S_min': 20},
    'formula': 'count((is_brown | is_dark) & leaf_mask) / count(leaf_mask)'
}

def compute_hsv_necrotic_ratio(img_rgb, leaf_mask):
    im_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = im_hsv[:, :, 0]
    s = im_hsv[:, :, 1]
    v = im_hsv[:, :, 2]
    
    brown_mask = (h >= 8) & (h <= 32) & (s >= 40) & (v >= 25) & (v <= 210)
    dark_mask = (v < 45) & (s >= 20)
    necrotic = (brown_mask | dark_mask) & leaf_mask
    
    leaf_px = np.count_nonzero(leaf_mask)
    if leaf_px == 0:
        return np.zeros_like(leaf_mask, dtype=bool), 0.0
    ratio = np.count_nonzero(necrotic) / leaf_px
    return necrotic, ratio

lesion_classes = ['mahali_koleroga', 'stem_cracking', 'stem_bleeding', 'bud_borer']
hsv_spearman_results = {}
cache_lesion_ratios_path = 'outputs/final_v2/arecanut_lesion_ratios_eq2_and_hsv.csv'

# Compute or load ratios
if os.path.exists(cache_lesion_ratios_path):
    print(f"Loading cached lesion ratios from {cache_lesion_ratios_path}...")
    lesion_data_df = pd.read_csv(cache_lesion_ratios_path)
else:
    print("Computing Eq. 2 and HSV ratios for all 4 lesion classes in arecanut_clean_v2.csv...")
    lesion_df_all = areca_df[areca_df['disease_class'].isin(lesion_classes)].copy().reset_index(drop=True)
    
    r_eq2 = []
    r_hsv = []
    for idx, row in lesion_df_all.iterrows():
        im_bgr = cv2.imread(row['image_path'])
        if im_bgr is None:
            r_eq2.append(0.0)
            r_hsv.append(0.0)
            continue
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        leaf_mask = segment_leaf_adaptive(im_rgb)
        _, r1 = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
        _, r2 = compute_hsv_necrotic_ratio(im_rgb, leaf_mask)
        r_eq2.append(float(r1))
        r_hsv.append(float(r2))
        
    lesion_df_all['eq2_ratio'] = r_eq2
    lesion_df_all['hsv_ratio'] = r_hsv
    lesion_df_all.to_csv(cache_lesion_ratios_path, index=False)
    lesion_data_df = lesion_df_all

for cls in lesion_classes:
    cls_sub = lesion_data_df[lesion_data_df['disease_class'] == cls]
    rho, pval = spearmanr(cls_sub['eq2_ratio'], cls_sub['hsv_ratio'])
    hsv_spearman_results[cls] = {
        'N': len(cls_sub),
        'spearman_rho': float(rho),
        'p_value': float(pval),
        'median_eq2_ratio_pct': float(np.median(cls_sub['eq2_ratio']) * 100.0),
        'median_hsv_ratio_pct': float(np.median(cls_sub['hsv_ratio']) * 100.0)
    }
    print(f"Class: {cls:18} (N={len(cls_sub):4d}) | Spearman rho = {rho:.4f} (p = {pval:.4e})")

# Overall across all 4 lesion classes:
overall_rho, overall_pval = spearmanr(lesion_data_df['eq2_ratio'], lesion_data_df['hsv_ratio'])
hsv_spearman_results['overall_all_lesion'] = {
    'N': len(lesion_data_df),
    'spearman_rho': float(overall_rho),
    'p_value': float(overall_pval)
}
print(f"Overall All Lesions (N={len(lesion_data_df):4d}) | Spearman rho = {overall_rho:.4f} (p = {overall_pval:.4e})")

# ==============================================================================
# B.3 MONTAGE: 12 LOW (<10%) AND 12 HIGH (>40%) RATIO IMAGES IN MAHALI_KOLEROGA
# ==============================================================================
print("\n--- B.3 GENERATING MONTAGE: 12 LOW (<10%) & 12 HIGH (>40%) RATIO IMAGES ---")
mk_data = lesion_data_df[lesion_data_df['disease_class'] == 'mahali_koleroga'].copy()

low_pool = mk_data[mk_data['eq2_ratio'] < 0.10].copy()
high_pool = mk_data[mk_data['eq2_ratio'] > 0.40].copy()

print(f"Mahali koleroga pool with ratio < 10%: {len(low_pool)}")
print(f"Mahali koleroga pool with ratio > 40%: {len(high_pool)}")

np.random.seed(42)
sample_low = low_pool.sample(n=min(12, len(low_pool)), random_state=42)
sample_high = high_pool.sample(n=min(12, len(high_pool)), random_state=42)

# Generate 24-image montage (6 rows x 4 columns)
# Each column: 2 pairs of [Raw | Mask]
fig = plt.figure(figsize=(18, 26))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.25)

# First 12 tiles: Low (<10%)
# Next 12 tiles: High (>40%)
all_tiles = []
for _, r in sample_low.iterrows():
    all_tiles.append(('Low (<10%)', r))
for _, r in sample_high.iterrows():
    all_tiles.append(('High (>40%)', r))

for i, (group_label, row) in enumerate(all_tiles):
    row_i = i // 4
    col_i = i % 4
    
    im_bgr = cv2.imread(row['image_path'])
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    lf = segment_leaf_adaptive(im_rgb)
    ls_mask, r_val = compute_lesion_mask_and_ratio(im_rgb, lf)
    
    overlay = im_rgb.copy()
    overlay[ls_mask] = [255, 0, 0] # bright red
    vis = cv2.addWeighted(im_rgb, 0.65, overlay, 0.35, 0)
    
    raw_res = cv2.resize(im_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    fname = os.path.basename(row['image_path'])
    if len(fname) > 26:
        fname = fname[:23] + "..."
    cap = f"[{group_label}]\nRatio: {r_val*100:.2f}% (Eq2 G{ratio_to_grade_eq2(r_val)})\n{fname}"
    ax.set_title(cap, fontsize=7.5, pad=3)
    ax.axis('off')

fig.suptitle("Mahali Koleroga Lesion Ratio Extremes Inspection Montage (24 Images)\n[Top 12: Ratio < 10% | Bottom 12: Ratio > 40%]\n[Left: Raw RGB | Right: Eq. 2 Red Mask Overlay] (Seed=42)", fontsize=13, fontweight='bold', y=0.995)
montage_path = 'outputs/final_v2/ratio_validity_low_high_montage.png'
plt.savefig(montage_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved montage to: {montage_path}")

stage2d_b_output = {
    'b1_healthy_unconstrained_eval': healthy_results,
    'b1_provenance_explanation': provenance_explanation,
    'b2_hsv_definition': hsv_definition,
    'b2_hsv_spearman_results': hsv_spearman_results,
    'b3_montage_path': montage_path
}

with open('outputs/final_v2/stage2d_section_b.json', 'w') as f:
    json.dump(stage2d_b_output, f, indent=2)

print("\nSaved Stage 2d Section B results to outputs/final_v2/stage2d_section_b.json")
