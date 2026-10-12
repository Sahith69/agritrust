"""
Independent cross-check of arecanut_splits_v2.csv leak-freedom.
Method 1: Perceptual hashing (pHash, Hamming ≤ 8, direct + hflip + vflip)
Method 2: Burst-window grouping (timestamp within 30-60s, same disease class)
No files modified. Verification only.
"""
import sys, os, time
sys.path.insert(0, '/home/sahith/Projects/Crop')

import cv2
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

# ─── Load the new split ───────────────────────────────────────────────────────
df = pd.read_csv('outputs/arecanut_splits_v2.csv')
print(f"Loaded arecanut_splits_v2.csv: {len(df)} rows")
print(f"Splits: {df['split'].value_counts().to_dict()}")
print()

# ─── Helper: pHash of a numpy image ──────────────────────────────────────────
def phash(img_bgr, hash_size=8):
    """DCT-based perceptual hash. Returns 64-bit integer."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32)
    # DCT on each row then each col (separable 2D DCT approximation)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    # Pack 64 bits into a Python int
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def hamming(a, b):
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

# ─── METHOD 1: pHash across all 2,661 images ──────────────────────────────────
print("=" * 70)
print("METHOD 1: Perceptual Hash (pHash, Hamming ≤ 8, + hflip + vflip)")
print("=" * 70)
print(f"Computing pHash for {len(df)} images (3 orientations each)...")
t0 = time.time()

hash_records = []
errors = []
for i, row in df.iterrows():
    path = row['image_path']
    img = cv2.imread(path)
    if img is None:
        errors.append(path)
        continue
    h_orig  = phash(img)
    h_hflip = phash(cv2.flip(img, 1))
    h_vflip = phash(cv2.flip(img, 0))
    hash_records.append({
        'filename': row['filename'],
        'image_path': path,
        'split': row['split'],
        'disease_class': row['disease_class'],
        'h_orig':  h_orig,
        'h_hflip': h_hflip,
        'h_vflip': h_vflip,
    })
    if (i + 1) % 500 == 0:
        print(f"  ... {i+1}/{len(df)} done ({time.time()-t0:.0f}s)", flush=True)

print(f"  Completed {len(hash_records)} images in {time.time()-t0:.1f}s. Errors: {len(errors)}")
if errors:
    print(f"  Missing: {errors[:5]}")
print()

# Vectorized Hamming distance using popcount trick
def hamming_vec(a_arr, b_arr):
    """a_arr, b_arr: 1D uint64 arrays. Returns Hamming distances as int array."""
    x = np.bitwise_xor(a_arr, b_arr)
    # Brian Kernighan popcount via lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = bin(i).count('1')
    # Process 8 bytes per element
    result = np.zeros(len(x), dtype=np.int32)
    for byte_shift in range(8):
        result += lut[(x >> (byte_shift * 8)) & 0xFF].astype(np.int32)
    return result

THRESHOLD = 8
cross_pairs = []

print(f"Comparing all pairs across different splits (Hamming ≤ {THRESHOLD})...")
t1 = time.time()

# Build arrays per split for all 3 orientations
split_data = {}
for sp in ['train', 'val', 'test']:
    rows = [r for r in hash_records if r['split'] == sp]
    split_data[sp] = {
        'rows': rows,
        'h_orig':  np.array([r['h_orig']  for r in rows], dtype=np.uint64),
        'h_hflip': np.array([r['h_hflip'] for r in rows], dtype=np.uint64),
        'h_vflip': np.array([r['h_vflip'] for r in rows], dtype=np.uint64),
    }

split_pairs_to_check = [('train', 'val'), ('train', 'test'), ('val', 'test')]

for sp_a, sp_b in split_pairs_to_check:
    da = split_data[sp_a]
    db = split_data[sp_b]
    found = 0
    na, nb = len(da['rows']), len(db['rows'])
    orients = ['h_orig', 'h_hflip', 'h_vflip']

    # For each pair of orientations, compute Hamming matrix row-by-row
    # to avoid materializing na×nb×9 arrays at once
    min_ham = np.full((na, nb), 64, dtype=np.int32)
    for oa in orients:
        for ob in orients:
            for i in range(na):
                row_xor = np.bitwise_xor(da[oa][i], db[ob])
                # Hamming via lookup
                lut = np.zeros(256, dtype=np.int32)
                for byte_i in range(256): lut[byte_i] = bin(byte_i).count('1')
                h = np.zeros(nb, dtype=np.int32)
                for byte_shift in range(8):
                    h += lut[(row_xor >> (byte_shift * 8)) & np.uint64(0xFF)]
                min_ham[i] = np.minimum(min_ham[i], h)

    # Find pairs below threshold
    rows_a, cols_b = np.where(min_ham <= THRESHOLD)
    for idx_a, idx_b in zip(rows_a, cols_b):
        cross_pairs.append({
            'split_A': sp_a,
            'file_A':  da['rows'][idx_a]['image_path'],
            'class_A': da['rows'][idx_a]['disease_class'],
            'split_B': sp_b,
            'file_B':  db['rows'][idx_b]['image_path'],
            'class_B': db['rows'][idx_b]['disease_class'],
            'hamming': int(min_ham[idx_a, idx_b]),
        })
        found += 1

    label = f"{sp_a}-{sp_b}"
    print(f"  {label:>12}: {found} cross-split near-duplicate pairs found")

print(f"  Comparison done in {time.time()-t1:.1f}s")
print()

if cross_pairs:
    print(f"⚠️  TOTAL CROSS-SPLIT PAIRS FOUND BY pHash: {len(cross_pairs)}")
    df_pairs = pd.DataFrame(cross_pairs).sort_values(['split_A','split_B','hamming'])
    print(df_pairs[['split_A','class_A','file_A','split_B','class_B','file_B','hamming']].to_string(index=False))
    df_pairs.to_csv('outputs/final_v2/v2_phash_cross_split_pairs.csv', index=False)
    print("Saved: outputs/final_v2/v2_phash_cross_split_pairs.csv")
else:
    print("✓ METHOD 1 RESULT: ZERO cross-split near-duplicate pairs found via pHash.")
    print("  (Checked Hamming ≤ 8 across orig + hflip + vflip for all 2,661 × 2,661 combos)")

print()

# ─── METHOD 2: Burst-window grouping (timestamp within 60s, same class) ───────
print("=" * 70)
print("METHOD 2: Burst-Window Grouping (timestamp within 60s, same disease class)")
print("=" * 70)

# Check if timestamp columns are available
df_clusters = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
print(f"Cluster file columns: {df_clusters.columns.tolist()}")

if 'sec_of_day' not in df_clusters.columns and 'timestamp_str' not in df_clusters.columns:
    print("  No timestamp data available in cluster file. Skipping Method 2.")
else:
    # Merge timestamp into new split
    ts_cols = ['filename', 'disease_class', 'robust_cluster_id']
    if 'sec_of_day' in df_clusters.columns:
        ts_cols.append('sec_of_day')
    if 'timestamp_str' in df_clusters.columns:
        ts_cols.append('timestamp_str')

    df_ts = df.merge(df_clusters[ts_cols], on=['filename','disease_class'], how='left')

    if 'sec_of_day' in df_ts.columns:
        df_ts_valid = df_ts.dropna(subset=['sec_of_day']).copy()
        print(f"  Images with sec_of_day: {len(df_ts_valid)} / {len(df_ts)}")
        print()

        WINDOW_30  = 30
        WINDOW_60  = 60
        burst_pairs_30  = []
        burst_pairs_60  = []

        for cls in df_ts_valid['disease_class'].unique():
            cls_df = df_ts_valid[df_ts_valid['disease_class']==cls].sort_values('sec_of_day').reset_index(drop=True)
            for i in range(len(cls_df)):
                for j in range(i+1, len(cls_df)):
                    dt = abs(cls_df.loc[j,'sec_of_day'] - cls_df.loc[i,'sec_of_day'])
                    # Wrap-around for midnight (e.g. 86399 vs 1)
                    dt = min(dt, 86400 - dt)
                    sp_i = cls_df.loc[i,'split']
                    sp_j = cls_df.loc[j,'split']
                    if sp_i == sp_j:
                        continue
                    if dt <= WINDOW_30:
                        burst_pairs_30.append({
                            'delta_s': dt, 'class': cls,
                            'split_A': sp_i, 'file_A': cls_df.loc[i,'image_path'],
                            'split_B': sp_j, 'file_B': cls_df.loc[j,'image_path'],
                        })
                    if dt <= WINDOW_60:
                        burst_pairs_60.append({
                            'delta_s': dt, 'class': cls,
                            'split_A': sp_i, 'file_A': cls_df.loc[i,'image_path'],
                            'split_B': sp_j, 'file_B': cls_df.loc[j,'image_path'],
                        })

        print(f"  Cross-split burst pairs within 30s: {len(burst_pairs_30)}")
        print(f"  Cross-split burst pairs within 60s: {len(burst_pairs_60)}")

        if burst_pairs_60:
            df_burst = pd.DataFrame(burst_pairs_60).sort_values('delta_s')
            print()
            print(f"  ⚠️  BURST PAIRS (≤60s, cross-split):")
            print(df_burst.to_string(index=False))
            df_burst.to_csv('outputs/final_v2/v2_burst_cross_split_pairs.csv', index=False)
            print("  Saved: outputs/final_v2/v2_burst_cross_split_pairs.csv")
        else:
            print("  ✓ METHOD 2 RESULT: ZERO cross-split burst pairs within 60s.")

        # Also check 30s window explicitly
        if burst_pairs_30:
            print(f"  ⚠️  Stricter 30s window: {len(burst_pairs_30)} pairs")
        else:
            print("  ✓ Also zero at the stricter 30s window.")
    else:
        print("  sec_of_day column missing despite being listed. Check cluster file.")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
phash_clean = len(cross_pairs) == 0
print(f"  Method 1 (pHash Hamming ≤ {THRESHOLD}): {'✓ ZERO cross-split pairs' if phash_clean else f'⚠️  {len(cross_pairs)} pairs found'}")
if 'burst_pairs_60' in dir():
    burst_clean = len(burst_pairs_60) == 0
    print(f"  Method 2 (burst ≤60s):               {'✓ ZERO cross-split pairs' if burst_clean else f'⚠️  {len(burst_pairs_60)} pairs found'}")
    if phash_clean and burst_clean:
        print()
        print("  ✅ INDEPENDENT CONFIRMATION: arecanut_splits_v2.csv is leak-free")
        print("     by both perceptual hashing AND timestamp burst analysis.")
    else:
        print()
        print("  ⚠️  Leak(s) detected — see output above for full details.")
else:
    if phash_clean:
        print()
        print("  ✅ pHash confirms: arecanut_splits_v2.csv is leak-free.")
