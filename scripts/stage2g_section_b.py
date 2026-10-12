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
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2g - SECTION B: PADDY NOISE FLOOR & GRADE 1 RECHECK")
print("=" * 80)

# 1. Load Paddy Severity Labels
df_labels = pd.read_csv('outputs/paddy_severity_labels.csv')
print(f"Loaded paddy_severity_labels.csv (N={len(df_labels)})")

# Grade counts per class
grade_counts = df_labels.groupby(['disease_class', 'severity_grade']).size().unstack(fill_value=0)
for g in [0, 1, 2, 3]:
    if g not in grade_counts.columns:
        grade_counts[g] = 0
grade_counts = grade_counts[[0, 1, 2, 3]]
grade_counts['Total'] = grade_counts.sum(axis=1)

# Ground-truth Severity Labels Grade 1 ratio min / median / max
g1_labels = df_labels[df_labels['severity_grade'] == 1]
g1_stats_labels = g1_labels.groupby('disease_class')['lesion_ratio'].agg(
    count='count',
    min_pct=lambda x: float(x.min() * 100.0),
    median_pct=lambda x: float(x.median() * 100.0),
    max_pct=lambda x: float(x.max() * 100.0)
).reset_index()

# Recomputed unforced segmentation ratios from cache
cache_ratios_path = 'outputs/final_v2/paddy_lesion_ratios_cache.csv'
if os.path.exists(cache_ratios_path):
    df_cache = pd.read_csv(cache_ratios_path)
    g1_cache = df_cache[df_cache['severity_grade'] == 1]
    g1_stats_cache = g1_cache.groupby('disease_class')['lesion_ratio_pct'].agg(
        count='count',
        min_pct=lambda x: float(x.min()),
        median_pct=lambda x: float(x.median()),
        max_pct=lambda x: float(x.max())
    ).reset_index()
else:
    g1_stats_cache = None

print("\n--- GRADE COUNTS PER CLASS (paddy_severity_labels.csv) ---")
print(grade_counts.to_string())

print("\n--- GRADE 1 RATIO DISTRIBUTION (from paddy_severity_labels.csv) ---")
print(g1_stats_labels.to_string(index=False))

if g1_stats_cache is not None:
    print("\n--- GRADE 1 UNFORCED RECOMPUTED RATIOS (from paddy_lesion_ratios_cache.csv) ---")
    print(g1_stats_cache.to_string(index=False))

# True ratio thresholds explanation
thresholds_summary = (
    "TRUE RATIO THRESHOLDS USED IN PIPELINE:\n"
    "1. Standard Horsfall-Barratt / EPPO Thresholds (config.py / severity_grader.py):\n"
    "   - Grade 0: exactly 0.0% lesion area (assigned via 'auto_healthy' for normal leaves)\n"
    "   - Grade 1: > 0.0% to <= 5.0% lesion area (0.0 < ratio <= 0.05)\n"
    "   - Grade 2: > 5.0% to <= 15.0% lesion area (0.05 < ratio <= 0.15)\n"
    "   - Grade 3: > 15.0% lesion area (ratio > 0.15)\n"
    "2. Disease-Specific Biological Fallbacks:\n"
    "   - 'hispa': Assigned fixed proxy ratio = 0.0300 (3.00%) -> 100% Grade 1 across all 1,594 images.\n"
    "   - 'dead_heart': Tiller drying segmentation -> ratios range 31.84% to 85.23% (median 59.20%) -> 100% Grade 3 across all 1,442 images.\n"
    "   - 'normal': Auto-healthy short-circuit -> forced ratio 0.00% -> 100% Grade 0 across all 1,764 images."
)
print("\n" + thresholds_summary)

# 2. Normal Leaves with Unforced Ratio > 15% & 12 Hand-Picked Healthy Leaves
DISK_STRUCT = get_disk(config.LEAF_MASK_EROSION_PX).astype(np.uint8)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def compute_lesion_mask_and_details(img_rgb, leaf_mask):
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
    
    leaf_px = int(np.sum(leaf_mask))
    lesion_px = int(np.sum(lesion_mask))
    ratio_pct = (lesion_px / float(leaf_px) * 100.0) if leaf_px > 0 else 0.0
    return lesion_mask, leaf_px, lesion_px, ratio_pct

paddy_normal = df_labels[df_labels['disease_class'] == 'normal'].copy()
normal_sample = paddy_normal.sample(n=min(200, len(paddy_normal)), random_state=42)

normal_over15_list = []
print("\nScanning normal paddy leaves for unforced ratio > 15%...")
for idx, row in normal_sample.iterrows():
    img_bgr = cv2.imread(row['image_path'])
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(img_rgb)
    l_mask, leaf_px, lesion_px, r_pct = compute_lesion_mask_and_details(img_rgb, leaf_mask)
    if r_pct > 15.0:
        normal_over15_list.append({
            'row': row,
            'img_rgb': img_rgb,
            'leaf_mask': leaf_mask,
            'lesion_mask': l_mask,
            'leaf_pixels': leaf_px,
            'lesion_pixels': lesion_px,
            'lesion_pct': r_pct
        })

print(f"Found {len(normal_over15_list)} normal leaves with unforced ratio > 15%.")

# Pick top 12 normal leaves > 15%
top12_normal_over15 = normal_over15_list[:12]

# Hand-picked 12 healthy leaves table:
healthy_12_records = []
print("\n--- 12 HAND-PICKED HEALTHY LEAVES (UNFORCED SEGMENTATION MASK) ---")
for i, item in enumerate(top12_normal_over15):
    fname = item['row']['filename']
    healthy_12_records.append({
        'rank': i + 1,
        'filename': fname,
        'disease_class': 'normal',
        'ground_truth_grade': 0,
        'leaf_pixels': item['leaf_pixels'],
        'lesion_pixels': item['lesion_pixels'],
        'lesion_pct': item['lesion_pct']
    })
    print(f"  [{i+1:2d}] {fname:<15} Leaf Px: {item['leaf_pixels']:8d} | Lesion Px: {item['lesion_pixels']:7d} | Unforced Lesion %: {item['lesion_pct']:6.2f}%")

df_healthy_12 = pd.DataFrame(healthy_12_records)

# Also select 12 Grade 1 images for montage
g1_sub = df_labels[(df_labels['severity_grade'] == 1) & (df_labels['disease_class'] != 'hispa')].copy()
np.random.seed(42)
g1_sample = g1_sub.sample(n=12, random_state=42)

# Generate 24-Image Montage
print("\nRendering 24-image montage...")
fig = plt.figure(figsize=(18, 26))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.25)

# First 12: Normal leaves with ratio > 15%
for i, item in enumerate(top12_normal_over15):
    row_i = i // 4
    col_i = i % 4
    img_rgb = item['img_rgb']
    l_mask = item['lesion_mask']
    r_pct = item['lesion_pct']
    fname = item['row']['filename']
    
    overlay = img_rgb.copy()
    overlay[l_mask] = [255, 0, 0]
    vis = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)
    
    raw_res = cv2.resize(img_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    if len(fname) > 24: fname = fname[:21] + "..."
    cap = f"[Normal Unforced >15%]\nMask Lesion: {r_pct:.2f}%\n{fname}"
    ax.set_title(cap, fontsize=7.5, pad=3)
    ax.axis('off')

# Next 12: Grade 1 Lesion Images
g1_records = []
for j, (_, row) in enumerate(g1_sample.iterrows()):
    idx = 12 + j
    row_i = idx // 4
    col_i = idx % 4
    
    img_bgr = cv2.imread(row['image_path'])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(img_rgb)
    l_mask, leaf_px, lesion_px, r_pct = compute_lesion_mask_and_details(img_rgb, leaf_mask)
    
    g1_records.append({
        'filename': row['filename'],
        'disease_class': row['disease_class'],
        'ground_truth_grade': 1,
        'leaf_pixels': leaf_px,
        'lesion_pixels': lesion_px,
        'lesion_pct': r_pct
    })
    
    overlay = img_rgb.copy()
    overlay[l_mask] = [255, 0, 0]
    vis = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)
    
    raw_res = cv2.resize(img_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    fname = row['filename']
    if len(fname) > 24: fname = fname[:21] + "..."
    cap = f"[{row['disease_class']} G1]\nMask Lesion: {r_pct:.2f}%\n{fname}"
    ax.set_title(cap, fontsize=7.5, pad=3)
    ax.axis('off')

fig.suptitle("Paddy Noise Floor Recheck Montage (24 Images)\n[Top 12: Normal Leaves with Unforced Ratio >15% | Bottom 12: Grade 1 Lesion Leaves]\n[Left: Raw RGB | Right: Red Lesion Mask Overlay] (Seed=42)", fontsize=13, fontweight='bold', y=0.995)

montage_path = 'outputs/final_v2/paddy_noise_floor_montage_v2.png'
plt.savefig(montage_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved montage to: {montage_path}")

# Also save JSON report
stage2g_b_output = {
    'thresholds_summary': thresholds_summary,
    'grade_counts_per_class': grade_counts.to_dict(orient='index'),
    'grade_1_stats_labels': g1_stats_labels.to_dict(orient='records'),
    'healthy_12_leaves': healthy_12_records,
    'grade_1_12_images': g1_records,
    'montage_path': montage_path
}

with open('outputs/final_v2/stage2g_section_b.json', 'w') as f:
    json.dump(stage2g_b_output, f, indent=2)

print("Saved Stage 2g Section B results to outputs/final_v2/stage2g_section_b.json")
