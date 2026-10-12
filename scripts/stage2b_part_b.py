import os
import sys
import json
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2b - PART B: LABEL PROVENANCE & STABILITY")
print("=" * 80)

# ---------------------------------------------------------
# B.1 arecanut_annotations.json provenance
# ---------------------------------------------------------
ann_path = 'arecanut_annotations.json'
stat = os.stat(ann_path)
with open(ann_path) as f:
    ann_data = json.load(f)

ann_summary = {
    'file_path': os.path.abspath(ann_path),
    'file_size_bytes': stat.st_size,
    'file_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
    'entry_count': len(ann_data),
    'fields': list(ann_data[0].keys()),
    'sample_entry': ann_data[0],
    'class_distribution': pd.Series([x['original_label'] for x in ann_data]).value_counts().to_dict(),
    'grade_distribution': pd.Series([x['severity_grade'] for x in ann_data]).value_counts().to_dict(),
    'records_in_docs': {
        'PROJECT_PLAN.md': (
            "Line 11: 'Primary: arecanut_annotations.json (895 images, 9 classes, already has severity_grade 0-3)'. "
            "Contrasted with Paddy where 'No severity labels exist — must be generated'."
        ),
        'PROGRESS.md': (
            "No record of human grading for Arecanut annotations. All progress notes document parent-plant deduplication "
            "and active learning for Paddy, never human annotation for Arecanut."
        ),
        'manuscript_draft.md': (
            "Mentions weakly supervised proxy labeling using classical computer vision (Otsu, color-space filtering). "
            "Notes that ground truth divergence accounts for majority of severe errors."
        )
    },
    'derivation_conclusion': (
        "arecanut_annotations.json was an external pre-existing JSON provided with the dataset at project start "
        "(mtime 2026-07-24 18:30:06). There is no script that created it, and no protocol or author record for human grading. "
        "It must be treated as an unverified external pseudo-ground-truth."
    )
}

# ---------------------------------------------------------
# B.2 Source-Stratified Label Changes (871 vs 1,896)
# ---------------------------------------------------------
df_all = pd.read_csv('outputs/arecanut_splits.csv')
n_all = len(df_all)
is_kaggle = df_all['image_path'].str.contains('kaggle_merged')
df_all['source'] = np.where(is_kaggle, 'kaggle_1896', 'original_871')

HEALTHY_CLASSES = {'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'}
DISK_RADIUS_BASE = config.LEAF_MASK_EROSION_PX  # 9
DISK_STRUCT_BASE = get_disk(DISK_RADIUS_BASE)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def fast_detect(img_rgb, leaf_mask, disk_struct, otsu_offset=0):
    eroded = scipy.ndimage.binary_erosion(leaf_mask, structure=disk_struct)
    img_float = img_rgb.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    thresh = otsu_threshold(gray, mask=eroded)
    thresh = int(np.clip(thresh + otsu_offset, 0, 255))
    lesion_mask = (gray < thresh) & eroded
    labeled, num_features = scipy.ndimage.label(lesion_mask)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        valid_labels = np.where(counts >= MIN_LESION_BLOB)[0]
        valid_labels = valid_labels[valid_labels > 0]
        lesion_mask = np.isin(labeled, valid_labels)
    else:
        lesion_mask = np.zeros_like(lesion_mask, dtype=bool)
    return lesion_mask

def ratio_to_grade_eq2(r):
    if r <= 0.0: return 0
    elif r <= 0.05: return 1
    elif r <= 0.15: return 2
    else: return 3

# Worker to process base and perturbations
def process_stability(item):
    idx, p, cls, curr_g = item
    if cls in HEALTHY_CLASSES:
        return idx, curr_g, 0, 0, 0, 0, 0, 0, 0
    
    im_bgr = cv2.imread(p)
    if im_bgr is None:
        return idx, curr_g, curr_g, curr_g, curr_g, curr_g, curr_g, curr_g, curr_g
    
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(im_rgb)
    
    # 1. Base (radius=9, offset=0)
    l_base = fast_detect(im_rgb, leaf_mask, DISK_STRUCT_BASE, otsu_offset=0)
    g_base = ratio_to_grade_eq2(compute_ratio(l_base, leaf_mask))
    
    # 2. Otsu offset -1 (radius=9)
    l_om1 = fast_detect(im_rgb, leaf_mask, DISK_STRUCT_BASE, otsu_offset=-1)
    g_om1 = ratio_to_grade_eq2(compute_ratio(l_om1, leaf_mask))
    
    # 3. Otsu offset +1 (radius=9)
    l_op1 = fast_detect(im_rgb, leaf_mask, DISK_STRUCT_BASE, otsu_offset=1)
    g_op1 = ratio_to_grade_eq2(compute_ratio(l_op1, leaf_mask))

    # 4. Otsu offset -2 (radius=9)
    l_om2 = fast_detect(im_rgb, leaf_mask, DISK_STRUCT_BASE, otsu_offset=-2)
    g_om2 = ratio_to_grade_eq2(compute_ratio(l_om2, leaf_mask))

    # 5. Otsu offset +2 (radius=9)
    l_op2 = fast_detect(im_rgb, leaf_mask, DISK_STRUCT_BASE, otsu_offset=2)
    g_op2 = ratio_to_grade_eq2(compute_ratio(l_op2, leaf_mask))
    
    # 6. Erosion radius = 8 (offset=0)
    struct_r8 = get_disk(8)
    l_r8 = fast_detect(im_rgb, leaf_mask, struct_r8, otsu_offset=0)
    g_r8 = ratio_to_grade_eq2(compute_ratio(l_r8, leaf_mask))
    
    # 7. Erosion radius = 10 (offset=0)
    struct_r10 = get_disk(10)
    l_r10 = fast_detect(im_rgb, leaf_mask, struct_r10, otsu_offset=0)
    g_r10 = ratio_to_grade_eq2(compute_ratio(l_r10, leaf_mask))
    
    return idx, curr_g, g_base, g_om1, g_op1, g_om2, g_op2, g_r8, g_r10

print("Processing all 2,767 images for base recomputation and stability perturbations with Pool(8)...")
tasks = [
    (i, row['image_path'], row['disease_class'], int(row['severity_grade']))
    for i, row in df_all.iterrows()
]

with mp.get_context('fork').Pool(8) as pool:
    results = pool.map(process_stability, tasks, chunksize=50)

# Unpack
g_base_arr = np.zeros(n_all, dtype=int)
g_om1_arr = np.zeros(n_all, dtype=int)
g_op1_arr = np.zeros(n_all, dtype=int)
g_om2_arr = np.zeros(n_all, dtype=int)
g_op2_arr = np.zeros(n_all, dtype=int)
g_r8_arr = np.zeros(n_all, dtype=int)
g_r10_arr = np.zeros(n_all, dtype=int)

for res in results:
    idx, curr_g, g_base, g_om1, g_op1, g_om2, g_op2, g_r8, g_r10 = res
    g_base_arr[idx] = g_base
    g_om1_arr[idx] = g_om1
    g_op1_arr[idx] = g_op1
    g_om2_arr[idx] = g_om2
    g_op2_arr[idx] = g_op2
    g_r8_arr[idx] = g_r8
    g_r10_arr[idx] = g_r10

df_all['recomputed_grade'] = g_base_arr
df_all['g_om1'] = g_om1_arr
df_all['g_op1'] = g_op1_arr
df_all['g_om2'] = g_om2_arr
df_all['g_op2'] = g_op2_arr
df_all['g_r8'] = g_r8_arr
df_all['g_r10'] = g_r10_arr

# Split by source
df_871 = df_all[df_all['source'] == 'original_871'].copy()
df_1896 = df_all[df_all['source'] == 'kaggle_1896'].copy()

# Label changes breakdown
def get_source_breakdown(sub_df):
    changed_mask = sub_df['severity_grade'] != sub_df['recomputed_grade']
    by_cls = {}
    for cls in sorted(sub_df['disease_class'].unique()):
        cls_sub = sub_df[sub_df['disease_class'] == cls]
        c_mask = cls_sub['severity_grade'] != cls_sub['recomputed_grade']
        ct = pd.crosstab(cls_sub['severity_grade'], cls_sub['recomputed_grade'])
        by_cls[cls] = {
            'total': len(cls_sub),
            'changed': int(c_mask.sum()),
            'pct_changed': float(c_mask.sum() / len(cls_sub) * 100.0) if len(cls_sub) > 0 else 0.0,
            'crosstab_current_rows_vs_recomputed_cols': ct.to_dict(orient='index')
        }
    return {
        'total': len(sub_df),
        'total_changed': int(changed_mask.sum()),
        'pct_changed': float(changed_mask.sum() / len(sub_df) * 100.0),
        'by_class': by_cls
    }

b2_orig_871 = get_source_breakdown(df_871)
b2_kaggle_1896 = get_source_breakdown(df_1896)

# Overall 4x4 confusion matrix for original 871
cm_871 = np.zeros((4, 4), dtype=int)
for _, row in df_871.iterrows():
    cm_871[int(row['severity_grade']), int(row['recomputed_grade'])] += 1

b2_orig_871['overall_4x4_confusion_matrix'] = {
    'description': 'rows = current_severity_grade in CSV, columns = recomputed_grade under Eq. 2',
    'matrix': cm_871.tolist()
}

# ---------------------------------------------------------
# B.3 Label Stability Perturbations
# ---------------------------------------------------------
perturbations = {
    'otsu_offset_minus_1': 'g_om1',
    'otsu_offset_plus_1': 'g_op1',
    'otsu_offset_minus_2': 'g_om2',
    'otsu_offset_plus_2': 'g_op2',
    'erosion_radius_8': 'g_r8',
    'erosion_radius_10': 'g_r10'
}

stability_results = {}
for p_name, col_name in perturbations.items():
    p_by_class = {}
    for cls in sorted(df_all['disease_class'].unique()):
        cls_sub = df_all[df_all['disease_class'] == cls]
        # Compare perturbation against base recomputed_grade
        diff = (cls_sub[col_name] != cls_sub['recomputed_grade'])
        p_by_class[cls] = {
            'total': len(cls_sub),
            'changed_vs_base': int(diff.sum()),
            'pct_changed': float(diff.sum() / len(cls_sub) * 100.0) if len(cls_sub) > 0 else 0.0
        }
    overall_diff = (df_all[col_name] != df_all['recomputed_grade'])
    stability_results[p_name] = {
        'total_images': n_all,
        'overall_changed_vs_base': int(overall_diff.sum()),
        'overall_pct_changed': float(overall_diff.sum() / n_all * 100.0),
        'by_class': p_by_class
    }

section_b_results = {
    'b1_arecanut_annotations_provenance': ann_summary,
    'b2_source_breakdown': {
        'original_871': b2_orig_871,
        'kaggle_1896': b2_kaggle_1896
    },
    'b3_stability_perturbations': stability_results
}

with open('outputs/final_v2/stage2b_part_b.json', 'w') as f:
    json.dump(section_b_results, f, indent=2)

print("\nPart B complete! Saved to outputs/final_v2/stage2b_part_b.json")
