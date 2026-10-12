import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, detect_lesions_group_a, compute_ratio, otsu_threshold, get_disk
from src.severity_grader import ratio_to_grade

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2f - SECTION B: PADDY NOISE FLOOR & LESION RATIO ANALYSIS")
print("=" * 80)

df_paddy = pd.read_csv('outputs/paddy_splits.csv')

DISK_STRUCT = get_disk(config.LEAF_MASK_EROSION_PX).astype(np.uint8)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def compute_lesion_mask_and_ratio(img_rgb, leaf_mask):
    eroded = cv2.erode(leaf_mask.astype(np.uint8), DISK_STRUCT).astype(bool)
    img_float = img_rgb.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    thresh = otsu_threshold(gray, mask=eroded)
    lesion_mask = (gray < thresh) & eroded
    num_features, labeled = cv2.connectedComponents(lesion_mask.astype(np.uint8))
    if num_features > 1:
        counts = np.bincount(labeled.ravel())
        valid_labels = np.where(counts >= MIN_LESION_BLOB)[0]
        valid_labels = valid_labels[valid_labels > 0]
        lesion_mask = np.isin(labeled, valid_labels)
    else:
        lesion_mask = np.zeros_like(lesion_mask, dtype=bool)
    ratio = compute_ratio(lesion_mask, leaf_mask)
    return lesion_mask, ratio

# 1. Unforced Normal-Leaf 95th Percentile Noise Floor
paddy_normal = df_paddy[df_paddy['disease_class'] == 'normal'].copy()
np.random.seed(42)
normal_sample = paddy_normal.sample(n=min(200, len(paddy_normal)), random_state=42)

normal_ratios = []
normal_over15_rows = []

print("Computing unforced lesion ratios on normal paddy leaves...")
for idx, row in normal_sample.iterrows():
    img_bgr = cv2.imread(row['image_path'])
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(img_rgb)
    l_mask, r = compute_lesion_mask_and_ratio(img_rgb, leaf_mask)
    r_pct = r * 100.0
    normal_ratios.append(r_pct)
    if r_pct > 15.0:
        normal_over15_rows.append((row, img_rgb, leaf_mask, l_mask, r_pct))

p95_noise_floor = float(np.percentile(normal_ratios, 95))
p90_noise_floor = float(np.percentile(normal_ratios, 90))

print(f"Normal-Leaf Unforced Ratio Distribution (N={len(normal_ratios)}):")
print(f"  • Median: {np.median(normal_ratios):.2f}%")
print(f"  • Mean:   {np.mean(normal_ratios):.2f}%")
print(f"  • P90 Noise Floor: {p90_noise_floor:.2f}%")
print(f"  • P95 Noise Floor: {p95_noise_floor:.2f}%")
print(f"  • Normal Leaves with Ratio > 15%: {len(normal_over15_rows)}")

# 2. Lesion Classes Analysis
cache_ratios_path = 'outputs/final_v2/paddy_lesion_ratios_cache.csv'
if os.path.exists(cache_ratios_path):
    print(f"Loading precomputed Paddy lesion ratios from {cache_ratios_path}...")
    lesion_df = pd.read_csv(cache_ratios_path)
else:
    print("Computing lesion ratios across Paddy lesion classes in paddy_splits.csv...")
    lesion_sub = df_paddy[df_paddy['disease_class'] != 'normal'].copy()
    ratios_list = []
    for idx, row in lesion_sub.iterrows():
        img_bgr = cv2.imread(row['image_path'])
        if img_bgr is None:
            ratios_list.append(0.0)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        leaf_mask = segment_leaf_adaptive(img_rgb)
        _, r = compute_lesion_mask_and_ratio(img_rgb, leaf_mask)
        ratios_list.append(r * 100.0)
    lesion_sub['lesion_ratio_pct'] = ratios_list
    lesion_sub.to_csv(cache_ratios_path, index=False)
    lesion_df = lesion_sub

# Metrics per lesion class
lesion_classes = sorted(df_paddy[df_paddy['disease_class'] != 'normal']['disease_class'].unique())

class_noise_stats = {}
g1_over15_rows = []

for cls in lesion_classes:
    cls_df = lesion_df[lesion_df['disease_class'] == cls]
    r_arr = cls_df['lesion_ratio_pct'].values
    
    med = float(np.median(r_arr))
    q25 = float(np.percentile(r_arr, 25))
    q75 = float(np.percentile(r_arr, 75))
    iqr = q75 - q25
    
    below_p95 = float(np.mean(r_arr <= p95_noise_floor) * 100.0)
    
    # Grade 1 subset
    g1_df = cls_df[cls_df['severity_grade'] == 1]
    if len(g1_df) > 0:
        g1_arr = g1_df['lesion_ratio_pct'].values
        g1_med = float(np.median(g1_arr))
        g1_q25 = float(np.percentile(g1_arr, 25))
        g1_q75 = float(np.percentile(g1_arr, 75))
        g1_iqr = g1_q75 - g1_q25
        g1_below_p95 = float(np.mean(g1_arr <= p95_noise_floor) * 100.0)
    else:
        g1_med, g1_iqr, g1_below_p95 = 0.0, 0.0, 0.0
        
    class_noise_stats[cls] = {
        'total_n': len(cls_df),
        'median_pct': med,
        'iqr_pct': iqr,
        'p25_pct': q25,
        'p75_pct': q75,
        'share_below_p95_noise_floor': below_p95,
        'g1_n': len(g1_df),
        'g1_median_pct': g1_med,
        'g1_iqr_pct': g1_iqr,
        'g1_share_below_p95_noise_floor': g1_below_p95
    }

# 3. Save 24-Image Montage
print("\nGenerating 24-image montage: 12 normal leaves (>15%) and 12 Grade 1 lesion leaves...")

np.random.seed(42)
sampled_normal_over15 = normal_over15_rows[:12] if len(normal_over15_rows) >= 12 else normal_over15_rows

g1_pool = lesion_df[lesion_df['severity_grade'] == 1].copy()
g1_sample = g1_pool.sample(n=min(12, len(g1_pool)), random_state=42)

fig = plt.figure(figsize=(18, 26))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.25)

# First 12: Normal > 15%
for i, item in enumerate(sampled_normal_over15):
    row_i = i // 4
    col_i = i % 4
    
    row, img_rgb, leaf_mask, l_mask, r_pct = item
    
    overlay = img_rgb.copy()
    overlay[l_mask] = [255, 0, 0]
    vis = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)
    
    raw_res = cv2.resize(img_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    fname = os.path.basename(row['image_path'])
    if len(fname) > 24: fname = fname[:21] + "..."
    cap = f"[Normal Unforced >15%]\nRatio: {r_pct:.2f}%\n{fname}"
    ax.set_title(cap, fontsize=7.5, pad=3)
    ax.axis('off')

# Next 12: Grade 1 Lesion Images
for j, (_, row) in enumerate(g1_sample.iterrows()):
    idx = 12 + j
    row_i = idx // 4
    col_i = idx % 4
    
    img_bgr = cv2.imread(row['image_path'])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(img_rgb)
    l_mask, r = compute_lesion_mask_and_ratio(img_rgb, leaf_mask)
    r_pct = r * 100.0
    
    overlay = img_rgb.copy()
    overlay[l_mask] = [255, 0, 0]
    vis = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)
    
    raw_res = cv2.resize(img_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    fname = os.path.basename(row['image_path'])
    if len(fname) > 24: fname = fname[:21] + "..."
    cap = f"[{row['disease_class']} G1]\nRatio: {r_pct:.2f}%\n{fname}"
    ax.set_title(cap, fontsize=7.5, pad=3)
    ax.axis('off')

fig.suptitle("Paddy Noise Floor Inspection Montage (24 Images)\n[Top 12: Normal Leaves with Unforced Ratio >15% | Bottom 12: Grade 1 Lesion Leaves]\n[Left: Raw RGB | Right: Red Lesion Mask Overlay] (Seed=42)", fontsize=13, fontweight='bold', y=0.995)

montage_path = 'outputs/final_v2/paddy_noise_floor_montage.png'
plt.savefig(montage_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved montage to: {montage_path}")

stage2f_b_output = {
    'normal_unforced_sample_size': len(normal_ratios),
    'normal_median_pct': float(np.median(normal_ratios)),
    'normal_mean_pct': float(np.mean(normal_ratios)),
    'p90_noise_floor_pct': float(p90_noise_floor),
    'p95_noise_floor_pct': float(p95_noise_floor),
    'class_noise_stats': class_noise_stats,
    'montage_path': montage_path
}

with open('outputs/final_v2/stage2f_section_b.json', 'w') as f:
    json.dump(stage2f_b_output, f, indent=2)

print("Saved Stage 2f Section B results to outputs/final_v2/stage2f_section_b.json")
