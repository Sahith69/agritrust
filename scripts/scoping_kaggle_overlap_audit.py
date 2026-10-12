import os
import sys
import re
import io
import time
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import zipfile
import scipy.fftpack

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    compute_ratio
)
from src.severity_grader import ratio_to_grade

PROJECT_ROOT = "/home/sahith/Projects/Crop"
CLEAN_CSV = os.path.join(PROJECT_ROOT, "outputs/improvements/arecanut_splits_clean_871.csv")
KAGGLE_ZIP = os.path.join(PROJECT_ROOT, "data/external/arecanut_kaggle_tejpaviraj/archive.zip")
IMPROVEMENTS_DIR = os.path.join(PROJECT_ROOT, "outputs/improvements")
os.makedirs(IMPROVEMENTS_DIR, exist_ok=True)

print("=" * 75)
print("  KAGGLE TEJPAVIRAJ DATASET SCOPING & OVERLAP AUDIT")
print("=" * 75)

# ─── 1. LOAD CLEAN 871 DATASET & COMPUTE PHASHES ───────────────────────
print("\n[1/6] Loading clean 871-image dataset and computing reference pHashes...")
df_clean = pd.read_csv(CLEAN_CSV)
print(f"Loaded {len(df_clean)} clean images.")

def compute_phash_64(image_l):
    resized = image_l.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

# We need orig, hflip, and vflip for our clean images
clean_hashes_orig = []
clean_hashes_hflip = []
clean_hashes_vflip = []

for p in df_clean['image_path']:
    img_l = Image.open(p).convert('L')
    clean_hashes_orig.append(compute_phash_64(img_l))
    clean_hashes_hflip.append(compute_phash_64(ImageOps.mirror(img_l)))
    clean_hashes_vflip.append(compute_phash_64(ImageOps.flip(img_l)))

H_clean_orig = np.array(clean_hashes_orig, dtype=bool)
H_clean_hflip = np.array(clean_hashes_hflip, dtype=bool)
H_clean_vflip = np.array(clean_hashes_vflip, dtype=bool)
print(f"Reference clean pHashes prepared: shape {H_clean_orig.shape}")


# ─── 2. READ & HASH ALL 12,292 KAGGLE IMAGES ──────────────────────────
print("\n[2/6] Reading and hashing all Kaggle images from archive.zip...")
t_kag0 = time.time()

CLASS_MAP = {
    'healthy_leaf': 'healthy_leaf',
    'healthy_nut': 'healthy_nut',
    'healthy_trunk': 'healthy_trunk',
    'healthy_foot': 'healthy_foot',
    'mahali_koleroga': 'mahali_koleroga',
    'stem_bleeding': 'stem_bleeding',
    'bud borer': 'bud_borer',
    'stem cracking': 'stem_cracking',
    'yellow leaf disease': 'yellow_leaf_disease'
}

def extract_class(path):
    parts = path.split('/')
    folder = parts[-2].lower()
    if folder in CLASS_MAP:
        return CLASS_MAP[folder]
    fn = parts[-1].lower()
    for k, v in CLASS_MAP.items():
        if fn.startswith(k) or fn.startswith(v):
            return v
    return 'unknown'

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

with zipfile.ZipFile(KAGGLE_ZIP, 'r') as z:
    all_names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total image files in zip: {len(all_names)}")

    def worker_process_image(name):
        data = z.read(name)
        fn = name.split('/')[-1]
        cls = extract_class(name)
        base = extract_base_id(fn)
        ts = extract_timestamp(fn)
        
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        h, w = arr.shape[:2]
        
        # Border check for studio cutout
        borders = np.concatenate([
            arr[:10, :, :].reshape(-1, 3),
            arr[-10:, :, :].reshape(-1, 3),
            arr[:, :10, :].reshape(-1, 3),
            arr[:, -10:, :].reshape(-1, 3)
        ], axis=0)
        bm = borders.mean(axis=0)
        bs = borders.std(axis=0)
        is_white_bg = bool(np.all(bm > 235) and np.all(bs < 20))
        is_black_bg = bool(np.all(bm < 20) and np.all(bs < 15))
        
        # Metadata check
        info_str = str(im.info).lower()
        has_stock_meta = any(k in info_str for k in ['dreamstime', 'shutterstock', 'istock', 'getty', 'adobe', 'depositphotos', 'alamy'])
        
        # Filename text check
        has_text_overlay = ('contrast enhanced' in fn.lower()) or ('dreamstime' in fn.lower()) or ('alamy' in fn.lower())
        
        is_stock_suspect = is_white_bg or is_black_bg or has_stock_meta or has_text_overlay
        
        # Hash
        gray = im.convert('L')
        h_orig = compute_phash_64(gray)
        h_hflip = compute_phash_64(ImageOps.mirror(gray))
        h_vflip = compute_phash_64(ImageOps.flip(gray))
        
        return {
            'zip_path': name,
            'filename': fn,
            'disease_class': cls,
            'base_id': base,
            'timestamp': ts,
            'is_stock_suspect': is_stock_suspect,
            'stock_reasons': {
                'white_bg': is_white_bg,
                'black_bg': is_black_bg,
                'stock_meta': has_stock_meta,
                'text_overlay': has_text_overlay
            },
            'h_orig': h_orig,
            'h_hflip': h_hflip,
            'h_vflip': h_vflip
        }

    print("Extracting features, hashes, and stock checks with ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        kaggle_records = list(ex.map(worker_process_image, all_names))

print(f"Processed {len(kaggle_records)} Kaggle images in {time.time()-t_kag0:.2f}s.")

# Stack Kaggle hashes
H_k_orig = np.array([r['h_orig'] for r in kaggle_records], dtype=bool)
H_k_hflip = np.array([r['h_hflip'] for r in kaggle_records], dtype=bool)
H_k_vflip = np.array([r['h_vflip'] for r in kaggle_records], dtype=bool)


# ─── 3. OVERLAP MATCHING AGAINST OUR CLEAN 871 SET ─────────────────────
print("\n[3/6] Computing multi-orientation Hamming distance against clean 871 set...")
t_match0 = time.time()

# Compare each Kaggle image against all clean images
# For orientation symmetry: kaggle(orig) vs clean(orig), clean(hflip), clean(vflip)
# is mathematically equivalent to:
# d_orig = min(kaggle_orig vs clean_orig)
# d_hflip = min(kaggle_hflip vs clean_orig)
# d_vflip = min(kaggle_vflip vs clean_orig)

chunk_size = 1000
min_dists = []

for start in range(0, len(H_k_orig), chunk_size):
    chunk_orig = H_k_orig[start:start+chunk_size]
    chunk_hflip = H_k_hflip[start:start+chunk_size]
    chunk_vflip = H_k_vflip[start:start+chunk_size]
    
    d1 = (chunk_orig[:, None, :] != H_clean_orig[None, :, :]).sum(axis=-1).min(axis=1)
    d2 = (chunk_hflip[:, None, :] != H_clean_orig[None, :, :]).sum(axis=-1).min(axis=1)
    d3 = (chunk_vflip[:, None, :] != H_clean_orig[None, :, :]).sum(axis=-1).min(axis=1)
    
    d_min = np.minimum(d1, np.minimum(d2, d3))
    min_dists.append(d_min)

min_dists = np.concatenate(min_dists)
print(f"Pairwise match completed in {time.time()-t_match0:.2f}s.")

# Match condition: Hamming distance <= 5 bits
matches = min_dists <= 5

for i, r in enumerate(kaggle_records):
    r['min_hamming_to_clean'] = int(min_dists[i])
    r['is_duplicate_of_clean'] = bool(matches[i])

total_kaggle_files = len(kaggle_records)
matched_files = int(matches.sum())
new_files = total_kaggle_files - matched_files
overlap_pct_files = (matched_files / total_kaggle_files) * 100.0

# Base-photo level overlap
all_base_ids = set(r['base_id'] for r in kaggle_records)
matched_base_ids = set(kaggle_records[i]['base_id'] for i in range(total_kaggle_files) if matches[i])
new_base_ids = all_base_ids - matched_base_ids
overlap_pct_bases = (len(matched_base_ids) / len(all_base_ids)) * 100.0

print(f"\n--- Overlap Matching Results (Threshold <= 5 bits) ---")
print(f"File Level:")
print(f"  Total Kaggle image files: {total_kaggle_files}")
print(f"  Files WITH a match in clean set (Duplicates): {matched_files} ({overlap_pct_files:.2f}%)")
print(f"  Files with NO match in clean set (Genuinely New): {new_files} ({100.0 - overlap_pct_files:.2f}%)")
print(f"\nUnique Base Photo Level:")
print(f"  Total unique base photos in Kaggle: {len(all_base_ids)}")
print(f"  Base photos WITH match in clean set: {len(matched_base_ids)} ({overlap_pct_bases:.2f}%)")
print(f"  Base photos with NO match in clean set (Genuinely New Bases): {len(new_base_ids)} ({100.0 - overlap_pct_bases:.2f}%)")


# ─── 4. STOCK PHOTO / STUDIO CUTOUT AUDIT ON NEW IMAGES ────────────────
print("\n[4/6] Running stock-photo/studio-cutout audit on candidate new images...")

new_records = [r for r in kaggle_records if not r['is_duplicate_of_clean']]
stock_excluded_records = [r for r in new_records if r['is_stock_suspect']]
clean_new_records = [r for r in new_records if not r['is_stock_suspect']]

import scipy.ndimage

def fast_filter_small_components(mask, min_size=15):
    labeled, num_features = scipy.ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    valid_labels = np.where(counts >= min_size)[0]
    valid_labels = valid_labels[valid_labels > 0]
    return np.isin(labeled, valid_labels)

def fast_detect_lesions_group_a(img_array, leaf_mask):
    from src.segmentation import binary_erosion, otsu_threshold
    eroded_mask = binary_erosion(leaf_mask, config.LEAF_MASK_EROSION_PX)
    img_float = img_array.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    thresh = otsu_threshold(gray, mask=eroded_mask)
    lesion_mask = (gray < thresh) & eroded_mask
    lesion_mask = fast_filter_small_components(lesion_mask, config.MIN_LESION_BLOB_PX)
    return lesion_mask

# Base level stock exclusion
new_bases_stock_excluded = set(r['base_id'] for r in stock_excluded_records)
new_bases_clean = new_base_ids - new_bases_stock_excluded

print(f"Candidate new image files: {len(new_records)}")
print(f"  Stock / Studio cutout excluded files: {len(stock_excluded_records)}")
print(f"  Remaining usable clean new files: {len(clean_new_records)}")
print(f"\nCandidate new unique base photos: {len(new_base_ids)}")
print(f"  Stock / Studio cutout excluded bases: {len(new_bases_stock_excluded)}")
print(f"  Remaining usable clean new base photos: {len(new_bases_clean)}")


# ─── 5. SEVERITY AUTO-LABELING ON CLEAN NEW IMAGES ─────────────────────
print("\n[5/6] Running HSV/Lab severity auto-labeling pipeline on clean new images...")
t_lbl0 = time.time()

HEALTHY_CLASSES = {'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'}

with zipfile.ZipFile(KAGGLE_ZIP, 'r') as z:
    def grade_worker(rec):
        cls = rec['disease_class']
        if cls in HEALTHY_CLASSES:
            return 0, 0.0, 100.0
        
        data = z.read(rec['zip_path'])
        im = Image.open(io.BytesIO(data)).convert('RGB')
        arr = np.array(im)
        
        leaf_mask = segment_leaf_adaptive(arr)
        lesion_mask = fast_detect_lesions_group_a(arr, leaf_mask)
        r = compute_ratio(lesion_mask, leaf_mask)
        g = ratio_to_grade(r)
        return g, r, 95.0

    print("Grading clean new image files...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        grading_results = list(ex.map(grade_worker, clean_new_records))

for i, (g, ratio, conf) in enumerate(grading_results):
    clean_new_records[i]['severity_grade'] = g
    clean_new_records[i]['lesion_ratio'] = round(float(ratio), 4)
    clean_new_records[i]['grade_confidence'] = round(float(conf), 2)

print(f"Graded {len(clean_new_records)} images in {time.time()-t_lbl0:.2f}s.")

# Represent each clean new base photo by its primary/first clean sample
base_to_clean_samples = defaultdict(list)
for r in clean_new_records:
    base_to_clean_samples[r['base_id']].append(r)

clean_new_base_records = [base_to_clean_samples[b][0] for b in sorted(new_bases_clean)]


# ─── 6. YIELD ANALYSIS & TABLES ────────────────────────────────────────
print("\n[6/6] Generating Disease Class x Severity Grade yield tables...")

df_new_files = pd.DataFrame(clean_new_records)
df_new_bases = pd.DataFrame(clean_new_base_records)

crosstab_files = pd.crosstab(df_new_files['disease_class'], df_new_files['severity_grade'], margins=True)
crosstab_bases = pd.crosstab(df_new_bases['disease_class'], df_new_bases['severity_grade'], margins=True)

print("\n" + "=" * 70)
print("  FINAL NEW-IMAGE YIELD TABLE: ALL USABLE CLEAN FILES")
print("=" * 70)
print(crosstab_files)

print("\n" + "=" * 70)
print("  FINAL NEW-IMAGE YIELD TABLE: UNIQUE PHYSICAL BASE PHOTOS")
print("=" * 70)
print(crosstab_bases)


# ─── 7. SESSION DIVERSITY & PHOTOGRAPHY CONTEXT ANALYSIS ──────────────
print("\n" + "=" * 70)
print("  SESSION DIVERSITY & ORIGINAL PHOTOGRAPHY ANALYSIS")
print("=" * 70)

# Extract timestamp date & time patterns
new_timestamps = [r['timestamp'] for r in clean_new_base_records if r['timestamp'] is not None]
our_timestamps = [extract_timestamp(fn) for fn in df_clean['filename'] if extract_timestamp(fn) is not None]

new_dates = Counter([ts.split('_')[0] for ts in new_timestamps])
our_dates = Counter([ts.split('_')[0] for ts in our_timestamps])

print("\nCapture Dates in Existing Clean 871 Set:")
for d, cnt in sorted(our_dates.items()):
    print(f"  Date {d}: {cnt} timestamped images")

print("\nCapture Dates in Genuinely New Kaggle Base Photos:")
for d, cnt in sorted(new_dates.items()):
    print(f"  Date {d}: {cnt} timestamped base photos")

# Time interval check: are the new photos interleaved in the same walks?
all_ts_tuples = []
for ts in our_timestamps:
    all_ts_tuples.append((ts, 'our_dataset'))
for ts in new_timestamps:
    all_ts_tuples.append((ts, 'new_kaggle'))

all_ts_tuples.sort()

# Check interleaving
interleaved_count = 0
for idx in range(1, len(all_ts_tuples)):
    prev_ts, prev_src = all_ts_tuples[idx-1]
    curr_ts, curr_src = all_ts_tuples[idx]
    if prev_src != curr_src and prev_ts[:8] == curr_ts[:8]:
        # Same day, different source
        t1 = int(prev_ts[9:11])*3600 + int(prev_ts[11:13])*60 + int(prev_ts[13:15])
        t2 = int(curr_ts[9:11])*3600 + int(curr_ts[11:13])*60 + int(curr_ts[13:15])
        if abs(t2 - t1) <= 120: # within 2 minutes of each other
            interleaved_count += 1

print(f"\nCamera Burst Interleaving Analysis:")
print(f"  New Kaggle photos captured within <= 120s of an existing photo: {interleaved_count}")


# ─── 8. SAVE REPORT JSON ───────────────────────────────────────────────
report_data = {
    "total_kaggle_files": total_kaggle_files,
    "file_overlap": {
        "matched_files": matched_files,
        "new_files": new_files,
        "overlap_pct": round(overlap_pct_files, 2)
    },
    "base_photo_overlap": {
        "total_base_photos": len(all_base_ids),
        "matched_base_photos": len(matched_base_ids),
        "new_base_photos": len(new_base_ids),
        "overlap_pct": round(overlap_pct_bases, 2)
    },
    "stock_exclusion": {
        "excluded_files": len(stock_excluded_records),
        "usable_clean_files": len(clean_new_records),
        "excluded_base_photos": len(new_bases_stock_excluded),
        "usable_clean_base_photos": len(new_bases_clean)
    },
    "yield_summary_files": {
        str(cls): {
            f"G{g}": int(crosstab_files.loc[cls, g]) if g in crosstab_files.columns and cls in crosstab_files.index else 0
            for g in range(4)
        }
        for cls in crosstab_files.index if cls != 'All'
    },
    "yield_summary_bases": {
        str(cls): {
            f"G{g}": int(crosstab_bases.loc[cls, g]) if g in crosstab_bases.columns and cls in crosstab_bases.index else 0
            for g in range(4)
        }
        for cls in crosstab_bases.index if cls != 'All'
    },
    "session_analysis": {
        "existing_dates": dict(our_dates),
        "new_dates": dict(new_dates),
        "interleaved_bursts_count": interleaved_count
    }
}

report_json_path = os.path.join(IMPROVEMENTS_DIR, "kaggle_scoping_overlap_report.json")
with open(report_json_path, 'w') as f:
    json.dump(report_data, f, indent=2)

print(f"\nSaved comprehensive scoping report JSON to: {report_json_path}")
print("=" * 75)
print("AUDIT COMPLETE.")
print("=" * 75)
