import os
import sys
import json
import time
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
print("STAGE 2c - PART B: ARECANUT LABEL SCHEMES ON ORIGINAL 871")
print("=" * 80)

# Load Arecanut splits
df_all = pd.read_csv('outputs/arecanut_splits.csv')
is_kaggle = df_all['image_path'].str.contains('kaggle_merged')
df_orig = df_all[~is_kaggle].copy().reset_index(drop=True)
df_kaggle = df_all[is_kaggle].copy().reset_index(drop=True)

print(f"Original 871 count: {len(df_orig)}")
print(f"Kaggle-derived count: {len(df_kaggle)}")

# Healthy classes
HEALTHY_CLASSES = {'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'}
DISK_STRUCT = get_disk(config.LEAF_MASK_EROSION_PX)  # radius 9 circular disk
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

# Compute ratios for original 871 images
print("Computing Eq. 2 lesion ratios for original 871 images...")
orig_ratios = []
orig_eq2_grades = []

for idx, row in df_orig.iterrows():
    cls = row['disease_class']
    if cls in HEALTHY_CLASSES:
        orig_ratios.append(0.0)
        orig_eq2_grades.append(0)
        continue
    
    im_bgr = cv2.imread(row['image_path'])
    if im_bgr is None:
        orig_ratios.append(0.0)
        orig_eq2_grades.append(0)
        continue
    
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    _, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
    orig_ratios.append(float(r))
    orig_eq2_grades.append(ratio_to_grade_eq2(r))

df_orig['ratio'] = orig_ratios
df_orig['ratio_pct'] = np.array(orig_ratios) * 100.0
df_orig['eq2_grade'] = orig_eq2_grades

# B.1 Statistics per class and per original (JSON) grade
stats_by_class_grade = {}
spearman_by_class = {}

for cls in sorted(df_orig['disease_class'].unique()):
    sub_c = df_orig[df_orig['disease_class'] == cls]
    stats_by_class_grade[cls] = {}
    
    for g in sorted(sub_c['severity_grade'].unique()):
        sub_g = sub_c[sub_c['severity_grade'] == g]
        r_pct = sub_g['ratio_pct'].values
        n = len(r_pct)
        med = float(np.median(r_pct))
        q25 = float(np.percentile(r_pct, 25))
        q75 = float(np.percentile(r_pct, 75))
        iqr = q75 - q25
        stats_by_class_grade[cls][f"Grade_{g}"] = {
            'N': int(n),
            'median_ratio_pct': med,
            'q25_pct': q25,
            'q75_pct': q75,
            'iqr_pct': iqr
        }
        
    # Spearman correlation between JSON grade and ratio
    if cls not in HEALTHY_CLASSES and cls != 'yellow_leaf_disease':
        # Need variance in JSON grade
        if len(sub_c['severity_grade'].unique()) > 1:
            rho, pval = spearmanr(sub_c['severity_grade'], sub_c['ratio'])
            spearman_by_class[cls] = {
                'N': len(sub_c),
                'rho': float(rho),
                'p_value': float(pval)
            }
        else:
            spearman_by_class[cls] = {
                'N': len(sub_c),
                'rho': None,
                'note': f"Single grade present ({sub_c['severity_grade'].unique()[0]})"
            }

# Overall Spearman correlation (excluding YLD and Healthy)
non_yld_lesion = df_orig[
    (~df_orig['disease_class'].isin(HEALTHY_CLASSES)) & 
    (df_orig['disease_class'] != 'yellow_leaf_disease')
]
overall_rho, overall_pval = spearmanr(non_yld_lesion['severity_grade'], non_yld_lesion['ratio'])
spearman_overall = {
    'N': len(non_yld_lesion),
    'rho': float(overall_rho),
    'p_value': float(overall_pval),
    'description': "Spearman rank correlation between original JSON severity grade and Eq. 2 lesion ratio across all lesion classes (excluding YLD)"
}

print(f"\nOverall Spearman correlation (excluding YLD): rho = {overall_rho:.4f} (p = {overall_pval:.4e}, N = {len(non_yld_lesion)})")

# B.2 Montage for mahali_koleroga: 6 random images per JSON grade (G0-G3)
print("\nGenerating mahali_koleroga montage (6 random images per JSON grade G0-G3)...")
np.random.seed(42)
mk_orig = df_orig[df_orig['disease_class'] == 'mahali_koleroga']

sampled_mk_orig = []
for g in [0, 1, 2, 3]:
    g_pool = mk_orig[mk_orig['severity_grade'] == g]
    k = min(6, len(g_pool))
    sample = g_pool.sample(n=k, random_state=42)
    sampled_mk_orig.append(sample)

df_sampled_mk = pd.concat(sampled_mk_orig).reset_index(drop=True)

# Generate 24-image side-by-side montage
# Each tile: [Raw RGB | Mask Overlay]
fig = plt.figure(figsize=(18, 26))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.25)

for i, row in df_sampled_mk.iterrows():
    row_i = i % 6
    col_i = i // 6
    
    im_bgr = cv2.imread(row['image_path'])
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    lesion_mask, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
    
    # Overlay: red tint for lesions, green boundary for leaf
    overlay = im_rgb.copy()
    overlay[lesion_mask] = [255, 0, 0] # bright red
    vis = cv2.addWeighted(im_rgb, 0.65, overlay, 0.35, 0)
    
    # Side by side: Raw | Vis
    raw_res = cv2.resize(im_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    fname = os.path.basename(row['image_path'])
    if len(fname) > 28:
        fname = fname[:25] + "..."
    eq2_g = ratio_to_grade_eq2(r)
    cap = f"JSON G{row['severity_grade']} -> Eq2 G{eq2_g}\nRatio: {r*100:.2f}%\n{fname}"
    ax.set_title(cap, fontsize=8, pad=3)
    ax.axis('off')

fig.suptitle("Original 871 Mahali Koleroga: 6 Images per Original JSON Grade (G0–G3)\n[Raw RGB | Eq. 2 Mask Overlay (Red)] (Seed = 42)", fontsize=13, fontweight='bold', y=0.995)
mk_montage_path = 'outputs/final_v2/mahali_koleroga_json_grades_montage.png'
plt.savefig(mk_montage_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved mahali_koleroga montage: {mk_montage_path}")

# B.3 Montage for Kaggle-derived images (Eq. 2 grades, 6 per grade G0-G3)
print("\nGenerating Kaggle-derived montage (6 random images per Eq. 2 grade G0-G3)...")
mk_kaggle = df_kaggle[df_kaggle['disease_class'] == 'mahali_koleroga'].copy()

# Compute ratios for Kaggle mahali_koleroga
kaggle_ratios = []
kaggle_eq2_grades = []
for _, row in mk_kaggle.iterrows():
    im_bgr = cv2.imread(row['image_path'])
    if im_bgr is None:
        kaggle_ratios.append(0.0)
        kaggle_eq2_grades.append(0)
        continue
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    _, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
    kaggle_ratios.append(float(r))
    kaggle_eq2_grades.append(ratio_to_grade_eq2(r))

mk_kaggle['ratio'] = kaggle_ratios
mk_kaggle['eq2_grade'] = kaggle_eq2_grades

sampled_mk_kaggle = []
for g in [0, 1, 2, 3]:
    g_pool = mk_kaggle[mk_kaggle['eq2_grade'] == g]
    k = min(6, len(g_pool))
    sample = g_pool.sample(n=k, random_state=42)
    sampled_mk_kaggle.append(sample)

df_sampled_kaggle = pd.concat(sampled_mk_kaggle).reset_index(drop=True)

fig = plt.figure(figsize=(18, 26))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.25)

for i, row in df_sampled_kaggle.iterrows():
    row_i = i % 6
    col_i = i // 6
    
    im_bgr = cv2.imread(row['image_path'])
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    lesion_mask, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
    
    overlay = im_rgb.copy()
    overlay[lesion_mask] = [255, 0, 0]
    vis = cv2.addWeighted(im_rgb, 0.65, overlay, 0.35, 0)
    
    raw_res = cv2.resize(im_rgb, (180, 180))
    vis_res = cv2.resize(vis, (180, 180))
    combined = np.hstack([raw_res, vis_res])
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    fname = os.path.basename(row['image_path'])
    if len(fname) > 28:
        fname = fname[:25] + "..."
    cap = f"Kaggle Eq2 G{row['eq2_grade']}\nRatio: {r*100:.2f}%\n{fname}"
    ax.set_title(cap, fontsize=8, pad=3)
    ax.axis('off')

fig.suptitle("Kaggle-Derived Mahali Koleroga: 6 Images per Eq. 2 Grade (G0–G3)\n[Raw RGB | Eq. 2 Mask Overlay (Red)] (Seed = 42)", fontsize=13, fontweight='bold', y=0.995)
kaggle_montage_path = 'outputs/final_v2/kaggle_eq2_grades_montage.png'
plt.savefig(kaggle_montage_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved Kaggle montage: {kaggle_montage_path}")

# B.4 Single Documented Recomputation of YLD Ratio Distribution (Erosion 9)
print("\nRecomputing YLD ratio distribution under documented setting (erosion 9)...")
df_yld = df_all[df_all['disease_class'] == 'yellow_leaf_disease'].copy().reset_index(drop=True)
yld_ratios = []

for _, row in df_yld.iterrows():
    im_bgr = cv2.imread(row['image_path'])
    if im_bgr is None:
        continue
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    _, r = compute_lesion_mask_and_ratio(im_rgb, leaf_mask)
    yld_ratios.append(float(r) * 100.0)

yld_ratios = np.array(yld_ratios)
yld_dist_summary = {
    'function_used': 'segment_leaf_adaptive + scipy.ndimage.binary_erosion(disk(9)) + otsu_threshold + min_blob_px(50)',
    'config_setting': 'config.LEAF_MASK_EROSION_PX = 9 (radius 9 circular disk = 19x19 footprint)',
    'total_yld_images': len(yld_ratios),
    'min_pct': float(np.min(yld_ratios)),
    'p25_pct': float(np.percentile(yld_ratios, 25)),
    'median_pct': float(np.median(yld_ratios)),
    'mean_pct': float(np.mean(yld_ratios)),
    'std_pct': float(np.std(yld_ratios)),
    'p75_pct': float(np.percentile(yld_ratios, 75)),
    'max_pct': float(np.max(yld_ratios)),
    'grade_counts_under_eq2': {
        'G0_0pct': int((yld_ratios <= 0.0).sum()),
        'G1_0to5pct': int(((yld_ratios > 0.0) & (yld_ratios <= 5.0)).sum()),
        'G2_5to15pct': int(((yld_ratios > 5.0) & (yld_ratios <= 15.0)).sum()),
        'G3_gt15pct': int((yld_ratios > 15.0).sum())
    }
}

print(f"YLD Ratio Distribution (N={len(yld_ratios)}):")
print(f"  Min:    {yld_dist_summary['min_pct']:.2f}%")
print(f"  p25:    {yld_dist_summary['p25_pct']:.2f}%")
print(f"  Median: {yld_dist_summary['median_pct']:.2f}%")
print(f"  Mean:   {yld_dist_summary['mean_pct']:.2f}% +/- {yld_dist_summary['std_pct']:.2f}%")
print(f"  p75:    {yld_dist_summary['p75_pct']:.2f}%")
print(f"  Max:    {yld_dist_summary['max_pct']:.2f}%")
print(f"  Grades under Eq. 2: {yld_dist_summary['grade_counts_under_eq2']}")

part_b_summary = {
    'b1_stats_per_class_grade': stats_by_class_grade,
    'b1_spearman_per_class': spearman_by_class,
    'b1_spearman_overall_non_yld': spearman_overall,
    'b2_mk_orig_montage': mk_montage_path,
    'b3_mk_kaggle_montage': kaggle_montage_path,
    'b4_yld_documented_distribution': yld_dist_summary
}

with open('outputs/final_v2/stage2c_part_b.json', 'w') as f:
    json.dump(part_b_summary, f, indent=2)

print("\nSaved Stage 2c Part B results to outputs/final_v2/stage2c_part_b.json")
