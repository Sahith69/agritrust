import os
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import scipy.ndimage

import config
from src.segmentation import segment_leaf_adaptive, compute_ratio
from src.severity_grader import ratio_to_grade

DISK_STRUCT = scipy.ndimage.generate_binary_structure(2, 1)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def otsu_threshold(gray, mask):
    vals = gray[mask]
    if len(vals) == 0:
        return 0
    hist, bin_edges = np.histogram(vals, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0
    current_max = 0
    threshold = 0
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0
    weight_b = 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between_class_var = weight_b * weight_f * ((mean_b - mean_f) ** 2)
        if between_class_var > current_max:
            current_max = between_class_var
            threshold = t
    return threshold

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

def process_yld(p):
    im_pil = Image.open(p).convert('RGB')
    arr = np.array(im_pil)
    leaf_mask = segment_leaf_adaptive(arr)
    lesion_mask = fast_detect_lesions(arr, leaf_mask)
    r = float(compute_ratio(lesion_mask, leaf_mask))
    g = int(ratio_to_grade(r))
    return p, r, g

def main():
    splits_df = pd.read_csv('outputs/improvements/arecanut_splits_merged_expanded.csv')
    yld_df = splits_df[splits_df['disease_class'] == 'yellow_leaf_disease'].reset_index(drop=True)
    paths = yld_df['image_path'].tolist()
    print(f"Processing {len(paths)} YLD images with Pool(8)...", flush=True)
    
    with mp.get_context('fork').Pool(8) as pool:
        results = pool.map(process_yld, paths)
        
    ratios = np.array([res[1] for res in results])
    grades = np.array([res[2] for res in results])
    
    print(f"\n=== YLD Lesion Ratio Distribution (N={len(ratios)}) ===")
    print(f"Min:    {np.min(ratios)*100:.2f}%")
    print(f"25%:    {np.percentile(ratios, 25)*100:.2f}%")
    print(f"Median: {np.median(ratios)*100:.2f}%")
    print(f"Mean:   {np.mean(ratios)*100:.2f}% +/- {np.std(ratios)*100:.2f}%")
    print(f"75%:    {np.percentile(ratios, 75)*100:.2f}%")
    print(f"Max:    {np.max(ratios)*100:.2f}%")
    
    print("\nRecomputed Grade Distribution from ratios:")
    for g in range(4):
        print(f"  Grade {g}: {np.sum(grades == g)} ({np.mean(grades == g)*100:.2f}%)")
        
    print("\nAssigned Severity Grade in CSV:")
    print(yld_df['severity_grade'].value_counts())
    
    # 12 Random G3 images overlay
    g3_items = [res for res in results if res[2] == 3 or yld_df.loc[paths.index(res[0]), 'severity_grade'] == 3]
    np.random.seed(42)
    selected_idx = np.random.choice(len(g3_items), size=min(12, len(g3_items)), replace=False)
    selected = [g3_items[i] for i in selected_idx]
    
    os.makedirs('outputs/figures', exist_ok=True)
    grid_imgs = []
    for p, r, g in selected:
        im = Image.open(p).convert('RGB')
        im = im.resize((256, 256))
        arr = np.array(im)
        leaf_mask = segment_leaf_adaptive(arr)
        lesion_mask = fast_detect_lesions(arr, leaf_mask)
        
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()
        overlay[lesion_mask > 0] = [0, 0, 255] # Red lesion
        cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0, img_bgr)
        
        cv2.putText(img_bgr, f"Ratio: {r*100:.1f}% (G{g})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        grid_imgs.append(img_bgr)
        
    rows = []
    for r_i in range(3):
        rows.append(np.hstack(grid_imgs[r_i*4 : (r_i+1)*4]))
    montage = np.vstack(rows)
    cv2.imwrite('outputs/figures/yld_grade3_12_overlays.png', montage)
    print("\nSaved 12 overlay montage to outputs/figures/yld_grade3_12_overlays.png")

if __name__ == '__main__':
    main()
