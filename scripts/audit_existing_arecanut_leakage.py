import os
import sys
import re
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import scipy.fftpack

CSV_PATH = "/home/sahith/Projects/Crop/outputs/arecanut_splits.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} images from {CSV_PATH}")

# ─── PART 1: BASE IMAGE ID EXTRACTION & GROUP AUDIT ───────────────
def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1)
    return fn

df['base_id'] = df['filename'].apply(extract_base_id)

print(f"\n--- PART 1: Base Image ID Grouping Audit ---")
print(f"Total images: {len(df)}")
print(f"Unique base IDs: {df['base_id'].nunique()}")
print(f"Unique parent_ids in CSV: {df['parent_id'].nunique()}")

base_leakage = []
for base_id, group in df.groupby('base_id'):
    splits = group['split'].unique()
    if len(splits) > 1:
        base_leakage.append({
            'base_id': base_id,
            'splits': list(splits),
            'count': len(group),
            'records': group[['filename', 'split', 'disease_class', 'severity_grade']].to_dict('records')
        })

print(f"Base IDs spanning multiple splits: {len(base_leakage)}")
if base_leakage:
    print("CRITICAL: Base ID Leakage Detected!")
    for item in base_leakage:
        print(f"  Base ID: {item['base_id']} | Splits: {item['splits']} | Count: {item['count']}")
        for r in item['records']:
            print(f"    [{r['split']}] {r['filename']} (Grade {r['severity_grade']}, {r['disease_class']})")
else:
    print("CLEAN: Zero base image IDs span across splits. Every single base image ID is strictly isolated to one split.")

# Breakdown of images per base ID
counts_per_base = df['base_id'].value_counts()
print(f"Min images per base ID: {counts_per_base.min()}")
print(f"Max images per base ID: {counts_per_base.max()}")
print(f"Mean images per base ID: {counts_per_base.mean():.2f}")
print(f"Number of base IDs with >1 image: {(counts_per_base > 1).sum()}")


# ─── PART 2: PERCEPTUAL HASHING (pHash) ALL-PAIRS AUDIT ───────────
print(f"\n--- PART 2: Perceptual Hashing (pHash) All-Pairs Audit ---")

def compute_phash_64(img_gray):
    # Standard 32x32 DCT pHash
    resized = img_gray.resize((32, 32), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:]) # Exclude DC component
    bits = (dct_low > med).flatten()
    return bits # 64 boolean values

print("Computing pHash (standard, horizontal-flip, and vertical-flip) for all 895 images...")
t0 = time.time()
hashes_orig = []
hashes_hflip = []
hashes_vflip = []

valid_indices = []
for idx, row in df.iterrows():
    p = row['image_path']
    if not os.path.exists(p):
        print(f"Warning: File not found: {p}")
        continue
    img = Image.open(p).convert('L')
    h_orig = compute_phash_64(img)
    h_hflip = compute_phash_64(ImageOps.mirror(img))
    h_vflip = compute_phash_64(ImageOps.flip(img))

    hashes_orig.append(h_orig)
    hashes_hflip.append(h_hflip)
    hashes_vflip.append(h_vflip)
    valid_indices.append(idx)

H_orig = np.array(hashes_orig)   # (895, 64)
H_hflip = np.array(hashes_hflip) # (895, 64)
H_vflip = np.array(hashes_vflip) # (895, 64)
print(f"Computed hashes in {time.time() - t0:.2f}s.")

# Compute pairwise Hamming distances (bitwise XOR sum)
# Distance matrix (895, 895)
# dist[i, j] = min(Hamming(orig_i, orig_j), Hamming(orig_i, hflip_j), Hamming(orig_i, vflip_j))
print("Computing pairwise minimum Hamming distance matrix (accounting for flips)...")
t1 = time.time()
N = len(valid_indices)

# Vectorized Hamming distance:
# H is (N, 64) boolean.
# dist = (H[:, None, :] != H[None, :, :]).sum(axis=-1)
dist_orig = (H_orig[:, None, :] != H_orig[None, :, :]).sum(axis=-1)
dist_hflip = (H_orig[:, None, :] != H_hflip[None, :, :]).sum(axis=-1)
dist_vflip = (H_orig[:, None, :] != H_vflip[None, :, :]).sum(axis=-1)

min_dist = np.minimum(dist_orig, np.minimum(dist_hflip, dist_vflip))
print(f"Computed {N*(N-1)//2} pairwise distances in {time.time() - t1:.2f}s.")

# Check for near-duplicates across splits at multiple thresholds
splits = df.loc[valid_indices, 'split'].values
filenames = df.loc[valid_indices, 'filename'].values
base_ids = df.loc[valid_indices, 'base_id'].values
classes = df.loc[valid_indices, 'disease_class'].values
grades = df.loc[valid_indices, 'severity_grade'].values
paths = df.loc[valid_indices, 'image_path'].values

for threshold in [0, 2, 5, 8, 10]:
    print(f"\n=======================================================")
    print(f"  HAMMING DISTANCE THRESHOLD <= {threshold} BITS (out of 64)")
    print(f"=======================================================")
    
    total_pairs = 0
    cross_split_pairs = []
    within_split_pairs = []
    
    for i in range(N):
        for j in range(i + 1, N):
            d = min_dist[i, j]
            if d <= threshold:
                total_pairs += 1
                pair_info = {
                    'img1': filenames[i], 'split1': splits[i], 'class1': classes[i], 'grade1': grades[i], 'base1': base_ids[i], 'path1': paths[i],
                    'img2': filenames[j], 'split2': splits[j], 'class2': classes[j], 'grade2': grades[j], 'base2': base_ids[j], 'path2': paths[j],
                    'distance': int(d),
                    'same_base_id': bool(base_ids[i] == base_ids[j])
                }
                if splits[i] != splits[j]:
                    cross_split_pairs.append(pair_info)
                else:
                    within_split_pairs.append(pair_info)
                    
    print(f"Total matching pairs (d <= {threshold}): {total_pairs}")
    print(f"  Within-split pairs: {len(within_split_pairs)}")
    print(f"  Cross-split pairs:  {len(cross_split_pairs)}")
    
    if cross_split_pairs:
        print(f"  CRITICAL LEAKAGE FOUND at threshold <= {threshold}: {len(cross_split_pairs)} cross-split pairs!")
        for p_idx, p in enumerate(cross_split_pairs[:15]):
            print(f"    Pair {p_idx+1}: [{p['split1']}] {p['img1']} <---> [{p['split2']}] {p['img2']}")
            print(f"      Dist: {p['distance']} bits | Classes: {p['class1']} vs {p['class2']} | Same Base ID: {p['same_base_id']}")
        if len(cross_split_pairs) > 15:
            print(f"    ... and {len(cross_split_pairs) - 15} more cross-split pairs.")
    else:
        print(f"  CLEAN: 0 cross-split pairs at threshold <= {threshold}!")
