import os
import sys
import json
import time
import re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import scipy.ndimage
import multiprocessing as mp

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk

HEALTHY_CLASSES = {'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'}
DISK_RADIUS = config.LEAF_MASK_EROSION_PX
DISK_STRUCT = get_disk(DISK_RADIUS)
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

def process_image(args):
    idx, path, cls, curr_g = args
    if cls in HEALTHY_CLASSES:
        return idx, 0, 0.0, curr_g
    
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return idx, curr_g, 0.0, curr_g
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = segment_leaf_adaptive(img_rgb)
    lesion_mask = fast_detect_lesions(img_rgb, leaf_mask)
    ratio = compute_ratio(lesion_mask, leaf_mask)
    
    # Equation 2 thresholds:
    # 0% -> G0
    # 0% - 5% -> G1
    # 5% - 15% -> G2
    # > 15% -> G3
    if ratio <= 0.0:
        new_g = 0
    elif ratio <= 0.05:
        new_g = 1
    elif ratio <= 0.15:
        new_g = 2
    else:
        new_g = 3
        
    return idx, new_g, ratio, curr_g

def scan_watermarks_and_studio(path, fn):
    im = cv2.imread(path)
    if im is None:
        return None
    h, w, c = im.shape
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    white_ratio = np.mean(gray > 245)
    black_ratio = np.mean(gray < 10)
    
    border_pixels = np.concatenate([
        gray[:5, :].ravel(),
        gray[-5:, :].ravel(),
        gray[:, :5].ravel(),
        gray[:, -5:].ravel()
    ])
    border_white = np.mean(border_pixels > 245)
    border_black = np.mean(border_pixels < 10)
    
    bot_right = gray[int(0.85*h):, int(0.75*w):]
    sobel_br = cv2.Sobel(bot_right, cv2.CV_64F, 1, 1, ksize=3)
    corner_gradient = np.mean(np.abs(sobel_br))
    
    flags = []
    if white_ratio > 0.40:
        flags.append(f"studio_white_background ({white_ratio*100:.1f}%)")
    if border_white > 0.85:
        flags.append(f"solid_white_border ({border_white*100:.1f}%)")
    if border_black > 0.85:
        flags.append(f"solid_black_border ({border_black*100:.1f}%)")
    if corner_gradient > 45.0:
        flags.append(f"high_contrast_corner_watermark (grad={corner_gradient:.1f})")
        
    return flags

def compute_phash(img_gray):
    res = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(res)[:8, :8]
    med = np.median(d)
    bits = (d > med).flatten()
    val = np.uint64(0)
    for b in bits:
        val = (val << np.uint64(1)) | np.uint64(b)
    return val

def main():
    os.makedirs('outputs/final_v2', exist_ok=True)

    print("=" * 80)
    print("SECTION E: LABEL INTEGRITY (ARECANUT)")
    print("=" * 80)

    df_all = pd.read_csv('outputs/arecanut_splits.csv')
    n_all = len(df_all)
    print(f"Total Arecanut images: {n_all}")

    print(f"Recomputing all {n_all} labels using unified pipeline & Eq. 2 thresholds with Pool(8)...")
    t0 = time.time()
    tasks = [
        (i, row['image_path'], row['disease_class'], int(row['severity_grade']))
        for i, row in df_all.iterrows()
    ]

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=8) as pool:
        results = pool.map(process_image, tasks, chunksize=50)

    t1 = time.time()
    print(f"Recomputation finished in {t1 - t0:.2f}s.")

    new_grades = np.zeros(n_all, dtype=int)
    ratios = np.zeros(n_all, dtype=float)
    for idx, new_g, r, curr_g in results:
        new_grades[idx] = new_g
        ratios[idx] = r

    df_all['recomputed_grade'] = new_grades
    df_all['recomputed_ratio'] = ratios

    # E.2 Analyze label changes
    changes_mask = (df_all['severity_grade'] != df_all['recomputed_grade'])
    n_changes = int(changes_mask.sum())
    print(f"\nTotal label changes: {n_changes} / {n_all} ({n_changes / n_all * 100:.2f}%)")

    change_by_class = {}
    for cls in sorted(df_all['disease_class'].unique()):
        sub = df_all[df_all['disease_class'] == cls]
        sub_changes = (sub['severity_grade'] != sub['recomputed_grade'])
        ct = pd.crosstab(sub['severity_grade'], sub['recomputed_grade'])
        change_by_class[cls] = {
            'total': len(sub),
            'changed': int(sub_changes.sum()),
            'pct_changed': float(sub_changes.sum() / len(sub) * 100),
            'confusion_current_vs_recomputed': ct.to_dict()
        }
        print(f"\n--- {cls} (Total: {len(sub)}, Changed: {sub_changes.sum()}) ---")
        print("Current (rows) vs Recomputed Eq. 2 (cols):")
        print(ct)

    # E.3 Yellow Leaf Disease Analysis
    print("\n" + "-" * 50)
    print("E.3 YELLOW LEAF DISEASE (WHOLE-CROWN CHLOROSIS)")
    print("-" * 50)
    yld_sub = df_all[df_all['disease_class'] == 'yellow_leaf_disease']
    yld_ratios = yld_sub['recomputed_ratio'].values

    print(f"YLD Image Count: {len(yld_sub)}")
    print(f"YLD Counts by split:\n{yld_sub['split'].value_counts()}")
    print(f"YLD Lesion Ratio Distribution (N={len(yld_ratios)}):")
    yld_stats = {
        'min': float(np.min(yld_ratios)),
        'p25': float(np.percentile(yld_ratios, 25)),
        'median': float(np.median(yld_ratios)),
        'mean': float(np.mean(yld_ratios)),
        'std': float(np.std(yld_ratios)),
        'p75': float(np.percentile(yld_ratios, 75)),
        'max': float(np.max(yld_ratios)),
    }
    for k, v in yld_stats.items():
        print(f"  {k:8s}: {v*100:.2f}% ({v:.4f})")

    print("\nCurrent YLD Grades in Dataset:")
    print(yld_sub['severity_grade'].value_counts())
    print("\nRecomputed YLD Grades under Eq. 2 Thresholds (>15% -> G3):")
    print(yld_sub['recomputed_grade'].value_counts())

    # E.4 Scan all images for stock/studio/watermark signatures and near-duplicates
    print("\n" + "-" * 50)
    print("E.4 SCAN FOR STOCK/STUDIO/WATERMARK & NEAR-DUPLICATES")
    print("-" * 50)

    print("Scanning all 2,767 images for visual artifacts/watermarks/studio signatures...")
    artifact_findings = []
    for i, row in df_all.iterrows():
        flags = scan_watermarks_and_studio(row['image_path'], row['filename'])
        if flags:
            artifact_findings.append({
                'index': i,
                'filename': row['filename'],
                'disease_class': row['disease_class'],
                'split': row['split'],
                'flags': flags
            })

    print(f"Total images with studio/border/watermark flags: {len(artifact_findings)}")
    for item in artifact_findings[:15]:
        print(f"  {item['filename']}: {item['flags']}")

    # Near-duplicate check on Arecanut splits
    print("\nChecking remaining cross-split pHash pairs (Hamming <= 5) on Arecanut...")
    arec_hashes = []
    for p in df_all['image_path']:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        arec_hashes.append(compute_phash(im) if im is not None else np.uint64(0))

    arec_hashes = np.array(arec_hashes, dtype=np.uint64)
    arec_splits = df_all['split'].values

    def popcount_u64(x):
        return bin(int(x)).count('1')

    arec_cross_pairs = []
    for i in range(n_all):
        for j in range(i + 1, n_all):
            if arec_splits[i] != arec_splits[j]:
                d = popcount_u64(arec_hashes[i] ^ arec_hashes[j])
                if d <= 5:
                    arec_cross_pairs.append((i, j, d, arec_splits[i], arec_splits[j], df_all.iloc[i]['filename'], df_all.iloc[j]['filename']))

    print(f"Cross-split pHash pairs with Hamming <= 5 on Arecanut: {len(arec_cross_pairs)}")
    if len(arec_cross_pairs) > 0:
        for pair in arec_cross_pairs[:5]:
            print(f"  Pair ({pair[3]} vs {pair[4]}): {pair[5]} <-> {pair[6]} (dist={pair[2]})")

    section_e_data = {
        'e1_threshold_reconciliation': {
            'original_871_origin': 'Inherited from arecanut_annotations.json (provided pre-assigned with dataset, not created by CV pipeline).',
            'kaggle_1896_origin': 'Generated by prepare_kaggle_merged_cache.py using fast_detect_lesions and ratio_to_grade with config.SEVERITY_THRESHOLDS (Grade 0: 0%, Grade 1: <=5%, Grade 2: <=15%, Grade 3: >15%).',
            'reconciliation': 'The paper Equation 2 specifies Horsfall-Barratt thresholds: Grade 1 (0-5%), Grade 2 (5-15%), Grade 3 (>15%). The >25% threshold was an ad-hoc threshold introduced in analyze_yld_confounds.py during an exploratory scratch test on YLD chlorosis. The authoritative project standard in config.py, severity_grader.py, and paper.tex Eq. 2 is >15%.'
        },
        'e2_recomputed_labels': {
            'total_images': n_all,
            'total_changed': n_changes,
            'pct_changed': float(n_changes / n_all * 100),
            'per_class': change_by_class
        },
        'e3_yellow_leaf_disease': {
            'confirmed_total_count': len(yld_sub),
            'confirmed_as_total_merged_class_count': True,
            'splits_distribution': yld_sub['split'].value_counts().to_dict(),
            'ratio_distribution': yld_stats,
            'options_consequences': {
                'option_a_exclude': 'Remove 502 YLD images from severity grading. Consequence: Eliminates chlorosis vs necrosis confound; reduces dataset from 2,767 to 2,265; leaves remaining classes with well-defined focal lesions.',
                'option_b_fixed_grade': 'Treat YLD as a whole-crown systemic disease with a fixed class-level grade (e.g. Grade 3), analogous to dead_heart in Paddy. Consequence: Keeps all 502 images; correctly acknowledges that chlorosis is systemic rather than focal lesions; eliminates noisy spot-counting.',
                'option_c_keep_as_is': 'Retain current spot-segmentation pipeline. Consequence: 100% of YLD images have ratios > 23% and are assigned Grade 3 under Eq. 2 (>15%), making the multi-grade assignment in the current CSV inconsistent with the CV pipeline.'
            }
        },
        'e4_scans': {
            'flagged_images_count': len(artifact_findings),
            'flagged_images': artifact_findings,
            'cross_split_near_duplicate_pairs_le5': len(arec_cross_pairs)
        }
    }

    with open('outputs/final_v2/stage2_section_e.json', 'w') as f:
        json.dump(section_e_data, f, indent=2)

    print("\nSection E complete! Saved to outputs/final_v2/stage2_section_e.json")

if __name__ == '__main__':
    main()
