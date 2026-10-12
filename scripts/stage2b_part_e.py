import os
import sys
import json
import re
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.sparse.csgraph as csgraph
from scipy.sparse import csr_matrix
import cv2

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2b - PART E: CLUSTER SIZES & BURST WINDOW ANALYSIS")
print("=" * 80)

df_all = pd.read_csv('outputs/arecanut_splits.csv')
M = len(df_all)
print(f"Total Arecanut images: {M}")

# 1. Base ID extraction
def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

# 2. Timestamp extraction
def extract_timestamp(fn):
    m = re.search(r'IMG_(\d{8})_(\d{6})', fn)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None

df_all['base_id'] = df_all['filename'].apply(extract_base_id)
df_all['timestamp_str'] = df_all['filename'].apply(extract_timestamp)

# Convert timestamps to date and seconds of day
dates = []
seconds_of_day = []
for ts in df_all['timestamp_str']:
    if isinstance(ts, str) and len(ts) >= 15:
        dates.append(ts[:8])
        sec = int(ts[9:11]) * 3600 + int(ts[11:13]) * 60 + int(ts[13:15])
        seconds_of_day.append(sec)
    else:
        dates.append(None)
        seconds_of_day.append(None)

df_all['date'] = dates
df_all['sec_of_day'] = seconds_of_day

# 3. Base ID edges
print("Building Base ID edges...")
base_edges = []
base_groups = df_all.groupby('base_id').indices
for b_id, idxs in base_groups.items():
    if len(idxs) > 1:
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                base_edges.append((idxs[i], idxs[j]))
print(f"Found {len(base_edges)} base-ID edges across {df_all['base_id'].nunique()} unique base IDs.")

# 4. pHash edges (Hamming <= 5, same class)
print("Computing pHashes for all images...")
def compute_phash_uint64(gray_img):
    resized = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    bit_array = (dct_low > med).flatten()
    val = np.uint64(0)
    for b in bit_array:
        val = (val << np.uint64(1)) | np.uint64(b)
    return val

hashes_orig = np.zeros(M, dtype=np.uint64)
hashes_hflip = np.zeros(M, dtype=np.uint64)
hashes_vflip = np.zeros(M, dtype=np.uint64)

for i, p in enumerate(df_all['image_path']):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is not None:
        hashes_orig[i] = compute_phash_uint64(im)
        hashes_hflip[i] = compute_phash_uint64(cv2.flip(im, 1))
        hashes_vflip[i] = compute_phash_uint64(cv2.flip(im, 0))

def popcount(x):
    return bin(int(x)).count('1')

def min_phash_dist(i, j):
    h_i = hashes_orig[i]
    d1 = popcount(h_i ^ hashes_orig[j])
    d2 = popcount(h_i ^ hashes_hflip[j])
    d3 = popcount(h_i ^ hashes_vflip[j])
    return min(d1, d2, d3)

classes = df_all['disease_class'].values

print("Building pHash edges (same class, Hamming <= 5)...")
phash_edges = []
# Fast check within same class
class_groups = df_all.groupby('disease_class').indices
for cls, idxs in class_groups.items():
    n_c = len(idxs)
    for ii in range(n_c):
        i = idxs[ii]
        for jj in range(ii + 1, n_c):
            j = idxs[jj]
            if min_phash_dist(i, j) <= 5:
                phash_edges.append((i, j))
print(f"Found {len(phash_edges)} pHash edges.")

# 5. Graph clustering under different burst windows W
windows = [120, 30, 10]
cluster_results = {}

for W in windows:
    print(f"\nEvaluating burst window W = {W}s...")
    adj = np.zeros((M, M), dtype=bool)
    
    # Add base edges
    for u, v in base_edges:
        adj[u, v] = True
        adj[v, u] = True
        
    # Add pHash edges
    for u, v in phash_edges:
        adj[u, v] = True
        adj[v, u] = True
        
    # Add temporal burst edges (same class, same date, |delta t| <= W)
    time_edges_count = 0
    for cls, idxs in class_groups.items():
        n_c = len(idxs)
        for ii in range(n_c):
            i = idxs[ii]
            d_i = dates[i]
            s_i = seconds_of_day[i]
            if d_i is None or s_i is None:
                continue
            for jj in range(ii + 1, n_c):
                j = idxs[jj]
                d_j = dates[j]
                s_j = seconds_of_day[j]
                if d_j is None or s_j is None:
                    continue
                if d_i == d_j and abs(s_i - s_j) <= W:
                    if not adj[i, j]:
                        time_edges_count += 1
                    adj[i, j] = True
                    adj[j, i] = True
                    
    print(f"  Added {time_edges_count} temporal edges (|dt| <= {W}s).")
    
    n_components, labels = csgraph.connected_components(csr_matrix(adj), directed=False)
    print(f"  Total clusters formed: {n_components}")
    
    df_all[f'cluster_{W}s'] = labels
    
    # Cluster sizes
    sizes = pd.Series(labels).value_counts().values
    stats = {
        'total_clusters': int(n_components),
        'min_size': int(np.min(sizes)),
        'p25_size': float(np.percentile(sizes, 25)),
        'median_size': float(np.median(sizes)),
        'mean_size': float(np.mean(sizes)),
        'p75_size': float(np.percentile(sizes, 75)),
        'p90_size': float(np.percentile(sizes, 90)),
        'p95_size': float(np.percentile(sizes, 95)),
        'p99_size': float(np.percentile(sizes, 99)),
        'max_size': int(np.max(sizes))
    }
    
    # Largest clusters per class with time spans
    largest_per_class = {}
    for cls in sorted(df_all['disease_class'].unique()):
        cls_sub = df_all[df_all['disease_class'] == cls]
        cls_cluster_counts = cls_sub[f'cluster_{W}s'].value_counts()
        top_clusters = []
        for c_lbl, c_size in cls_cluster_counts.head(3).items():
            c_members = cls_sub[cls_sub[f'cluster_{W}s'] == c_lbl]
            c_secs = c_members['sec_of_day'].dropna().values
            c_dates = c_members['date'].dropna().unique().tolist()
            if len(c_secs) > 1:
                t_span_sec = int(np.max(c_secs) - np.min(c_secs))
            else:
                t_span_sec = 0
            top_clusters.append({
                'cluster_label': int(c_lbl),
                'size': int(c_size),
                'dates': c_dates,
                'time_span_seconds': t_span_sec,
                'time_span_minutes': round(t_span_sec / 60.0, 2)
            })
        largest_per_class[cls] = top_clusters
        
    cluster_results[f'{W}s'] = {
        'distribution': stats,
        'largest_clusters_per_class': largest_per_class
    }

# ---------------------------------------------------------
# 6. Detailed Anatomy of cluster_0078 under 30s and 10s
# ---------------------------------------------------------
print("\nAnalyzing cluster_0078 breakdown...")
c0078_mask = (df_all['cluster_id'] == 'cluster_0078')
c0078_df = df_all[c0078_mask].copy()
print(f"Total images currently in cluster_0078: {len(c0078_df)}")
print(f"Disease class of cluster_0078: {c0078_df['disease_class'].unique().tolist()}")
print(f"Current splits of cluster_0078: {c0078_df['split'].value_counts().to_dict()}")

# Breakdown under 120s, 30s, 10s
c0078_breakdown = {}
for W in [120, 30, 10]:
    sub_clusters = c0078_df[f'cluster_{W}s'].value_counts()
    pieces = []
    for sc_lbl, sc_size in sub_clusters.items():
        mems = c0078_df[c0078_df[f'cluster_{W}s'] == sc_lbl]
        m_secs = mems['sec_of_day'].dropna().values
        m_dates = mems['date'].dropna().unique().tolist()
        span_s = int(np.max(m_secs) - np.min(m_secs)) if len(m_secs) > 1 else 0
        earliest = mems['timestamp_str'].dropna().min()
        latest = mems['timestamp_str'].dropna().max()
        pieces.append({
            'sub_cluster_id': int(sc_lbl),
            'size': int(sc_size),
            'dates': m_dates,
            'time_span_seconds': span_s,
            'earliest_timestamp': earliest,
            'latest_timestamp': latest,
            'unique_base_ids': int(mems['base_id'].nunique())
        })
    c0078_breakdown[f'{W}s'] = {
        'num_sub_clusters': len(sub_clusters),
        'pieces': pieces
    }
    print(f"Under W = {W}s: cluster_0078 breaks into {len(sub_clusters)} pieces (sizes: {sub_clusters.values.tolist()})")

part_e_summary = {
    'windows_analyzed': [120, 30, 10],
    'cluster_results_by_window': cluster_results,
    'cluster_0078_anatomy': {
        'total_images': len(c0078_df),
        'current_cluster_id': 'cluster_0078',
        'disease_class': 'healthy_trunk',
        'breakdown_by_window': c0078_breakdown
    }
}

with open('outputs/final_v2/stage2b_part_e.json', 'w') as f:
    json.dump(part_e_summary, f, indent=2)

print("\nPart E complete! Saved to outputs/final_v2/stage2b_part_e.json")
