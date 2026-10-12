import os
import sys
import re
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import scipy.fftpack
import scipy.ndimage
import multiprocessing as mp

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk
from src.severity_grader import ratio_to_grade

PROJECT_ROOT = "/home/sahith/Projects/Crop"
MERGED_IMG_DIR = os.path.join(PROJECT_ROOT, "data/arecanut/dataset/kaggle_merged")
IMPROVEMENTS_DIR = os.path.join(PROJECT_ROOT, "outputs/improvements")
OUTPUT_CSV = os.path.join(IMPROVEMENTS_DIR, "kaggle_extracted_1896.csv")
OUTPUT_NPZ = os.path.join(IMPROVEMENTS_DIR, "kaggle_extracted_1896_phashes.npz")

DISK_RADIUS = config.LEAF_MASK_EROSION_PX
DISK_STRUCT = get_disk(DISK_RADIUS)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def compute_phash_64(image_l):
    resized = image_l.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

def extract_timestamp(fn):
    m = re.search(r'IMG_(\d{8})_(\d{6})', fn)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None

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

HEALTHY_CLASSES = {'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'}

def process_single_image(args):
    path, cls, fn = args
    base_id = extract_base_id(fn)
    ts = extract_timestamp(fn)
    
    # 1. Hashes
    im_pil = Image.open(path)
    im_l = im_pil.convert('L')
    ho = compute_phash_64(im_l)
    hh = compute_phash_64(ImageOps.mirror(im_l))
    hv = compute_phash_64(ImageOps.flip(im_l))
    
    # 2. Severity Grade
    if cls in HEALTHY_CLASSES:
        g = 0
        r = 0.0
    else:
        im_rgb = im_pil.convert('RGB')
        arr = np.array(im_rgb)
        leaf_mask = segment_leaf_adaptive(arr)
        lesion_mask = fast_detect_lesions(arr, leaf_mask)
        r = compute_ratio(lesion_mask, leaf_mask)
        g = ratio_to_grade(r)
        
    return {
        'image_path': path,
        'filename': fn,
        'disease_class': cls,
        'severity_grade': int(g),
        'parent_id': base_id,
        'timestamp': ts if ts else '',
        'h_orig': ho,
        'h_hflip': hh,
        'h_vflip': hv
    }

def main():
    print(f"Collecting images from {MERGED_IMG_DIR}...")
    tasks = []
    for root, dirs, files in os.walk(MERGED_IMG_DIR):
        cls = os.path.basename(root)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(root, f)
                tasks.append((p, cls, f))
                
    tasks.sort(key=lambda x: x[0])
    print(f"Total images found: {len(tasks)}")
    assert len(tasks) == 1896, f"Expected 1896 images, found {len(tasks)}"
    
    t0 = time.time()
    print("Processing images (hashes + grading) with fork Pool(8)...", flush=True)
    with mp.get_context('fork').Pool(8) as pool:
        records = pool.map(process_single_image, tasks, chunksize=25)
        
    print(f"Completed in {time.time()-t0:.2f}s!", flush=True)
    
    # Prepare DataFrame
    csv_rows = []
    h_orig_list = []
    h_hflip_list = []
    h_vflip_list = []
    
    for r in records:
        csv_rows.append({
            'image_path': r['image_path'],
            'filename': r['filename'],
            'disease_class': r['disease_class'],
            'severity_grade': r['severity_grade'],
            'parent_id': r['parent_id'],
            'timestamp': r['timestamp']
        })
        h_orig_list.append(r['h_orig'])
        h_hflip_list.append(r['h_hflip'])
        h_vflip_list.append(r['h_vflip'])
        
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV to {OUTPUT_CSV}")
    
    np.savez_compressed(
        OUTPUT_NPZ,
        h_orig=np.array(h_orig_list, dtype=bool),
        h_hflip=np.array(h_hflip_list, dtype=bool),
        h_vflip=np.array(h_vflip_list, dtype=bool)
    )
    print(f"Saved hashes to {OUTPUT_NPZ}")
    
    print("\nBreakdown of extracted dataset:")
    print(pd.crosstab(df['disease_class'], df['severity_grade'], margins=True))

if __name__ == '__main__':
    main()
