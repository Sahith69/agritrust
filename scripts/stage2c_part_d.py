import os
import sys
import json
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2c - PART D: ROBUST DUPLICATE CLUSTERING ON ARECANUT_CLEAN_V2.CSV")
print("=" * 80)

# 1. Load clean manifest v2
manifest_path = 'outputs/final_v2/arecanut_clean_v2.csv'
df = pd.read_csv(manifest_path)
n = len(df)
print(f"Total clean images: {n}")

# Base ID extraction
def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

df['base_id'] = df['filename'].apply(extract_base_id)

# Timestamp extraction
def extract_timestamp(fn):
    m = re.search(r'(?:IMG_)?(20\d{6})_(\d{6})', fn)
    if m:
        try:
            dt_str = f"{m.group(1)}_{m.group(2)}"
            return datetime.strptime(dt_str, "%Y%m%d_%H%M%S").timestamp()
        except Exception:
            return None
    return None

df['timestamp'] = df['filename'].apply(extract_timestamp)

# Sets of undirected edges: set of (min(i,j), max(i,j))
edges_base_id = set()
edges_temporal = set()
edges_phash = set()
edges_resnet50 = set()

# -------------------------------------------------------------
# Edge Set 1: Same Base ID
# -------------------------------------------------------------
print("Computing Base ID edges...")
base_groups = df.groupby('base_id').indices
for b_id, indices in base_groups.items():
    if len(indices) > 1:
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                u, v = int(indices[i_idx]), int(indices[j_idx])
                edges_base_id.add((min(u, v), max(u, v)))

print(f"Base ID edges: {len(edges_base_id)}")

# -------------------------------------------------------------
# Edge Set 2: Temporal Proximity within 30 s (Same Class)
# -------------------------------------------------------------
print("Computing Temporal Proximity edges (<= 30 s, same class)...")
class_groups = df.groupby('disease_class').indices
for cls, indices in class_groups.items():
    ts_list = [(idx, df.iloc[idx]['timestamp']) for idx in indices if df.iloc[idx]['timestamp'] is not None]
    ts_list.sort(key=lambda x: x[1])
    
    for i_pos in range(len(ts_list)):
        u, t_u = ts_list[i_pos]
        for j_pos in range(i_pos + 1, len(ts_list)):
            v, t_v = ts_list[j_pos]
            diff = t_v - t_u
            if diff <= 30.0:
                edges_temporal.add((min(u, v), max(u, v)))
            else:
                break

print(f"Temporal edges (<= 30 s): {len(edges_temporal)}")

# -------------------------------------------------------------
# Edge Set 3: Multi-resize & Multi-orientation pHash (Hamming <= 8)
# -------------------------------------------------------------
print("Computing Multi-resize (AREA, LANCZOS, LINEAR) & Multi-orientation pHashes...")
RESIZE_METHODS = [cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR]

def compute_phash_64(gray, interp):
    res = cv2.resize(gray, (32, 32), interpolation=interp).astype(np.float32)
    dct = cv2.dct(res)[:8, :8]
    med = np.median(dct)
    bits = (dct > med).flatten()
    val = np.uint64(0)
    for b in bits:
        val = (val << np.uint64(1)) | np.uint64(b)
    return val

# 3 resize methods x 3 orientations = 9 hashes per image
all_hashes = np.zeros((n, 9), dtype=np.uint64)

for i, p in enumerate(df['image_path']):
    gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    gray_h = cv2.flip(gray, 1)
    gray_v = cv2.flip(gray, 0)
    
    col_idx = 0
    for interp in RESIZE_METHODS:
        all_hashes[i, col_idx] = compute_phash_64(gray, interp)
        all_hashes[i, col_idx + 1] = compute_phash_64(gray_h, interp)
        all_hashes[i, col_idx + 2] = compute_phash_64(gray_v, interp)
        col_idx += 3

def fast_popcount(m):
    m = m - ((m >> np.uint64(1)) & np.uint64(0x5555555555555555))
    m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
    m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return m.astype(np.int32)

print("Comparing pHashes across all image pairs (chunked vectorized)...")
chunk_sz = 300
for i_start in range(0, n, chunk_sz):
    i_end = min(i_start + chunk_sz, n)
    h_chunk_i = all_hashes[i_start:i_end] # (B, 9)
    
    for j_start in range(i_start, n, chunk_sz):
        j_end = min(j_start + chunk_sz, n)
        h_chunk_j = all_hashes[j_start:j_end] # (B2, 9)
        
        # Compare each orientation/interp combo
        # We can find minimum distance across the 9 hash representations:
        # min over k1, k2 of popcount(h_chunk_i[:, k1, None] ^ h_chunk_j[None, k2, :])
        min_dist_block = np.full((i_end - i_start, j_end - j_start), 64, dtype=np.int32)
        
        for k1 in range(9):
            col_k1 = h_chunk_i[:, k1, None]
            for k2 in range(9):
                col_k2 = h_chunk_j[:, k2][None, :]
                xor_m = col_k1 ^ col_k2
                d = fast_popcount(xor_m)
                min_dist_block = np.minimum(min_dist_block, d)
                
        # Find pairs with min_dist <= 8
        hits = np.where(min_dist_block <= 8)
        for r_i, c_j in zip(hits[0], hits[1]):
            u = i_start + r_i
            v = j_start + c_j
            if u < v:
                edges_phash.add((u, v))

print(f"Multi-resize/orientation pHash edges (Hamming <= 8): {len(edges_phash)}")

# -------------------------------------------------------------
# Edge Set 4: ImageNet ResNet-50 Embedding Cosine Similarity
# -------------------------------------------------------------
print("Extracting ImageNet ResNet-50 feature embeddings...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity() # 2048-dim output
resnet50.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

class FastArecanutDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(im)

ds = FastArecanutDataset(df['image_path'].tolist(), transform)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

embeddings = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        feats = resnet50(batch)
        embeddings.append(feats.cpu().numpy())

embeddings = np.vstack(embeddings) # (N, 2048)
# L2-normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
embeddings_norm = embeddings / norms

print("Computing pairwise cosine similarity...")
# Block matrix multiplication to avoid huge RAM spike
cos_sim = np.zeros((n, n), dtype=np.float32)
for b_start in range(0, n, 500):
    b_end = min(b_start + 500, n)
    cos_sim[b_start:b_end] = np.dot(embeddings_norm[b_start:b_end], embeddings_norm.T)

# Identify pairs in intervals: [0.90, 0.95), [0.95, 0.98), [0.98, 1.0]
pairs_90_95 = []
pairs_95_98 = []
pairs_gt_98 = []

for u in range(n):
    for v in range(u + 1, n):
        s = float(cos_sim[u, v])
        if s >= 0.95:
            edges_resnet50.add((u, v))
            
        if 0.90 <= s < 0.95:
            pairs_90_95.append((u, v, s))
        elif 0.95 <= s < 0.98:
            pairs_95_98.append((u, v, s))
        elif s >= 0.98:
            pairs_gt_98.append((u, v, s))

print(f"ResNet-50 Cosine Similarity >= 0.95 edges: {len(edges_resnet50)}")
print(f"Pair counts in intervals:")
print(f"  [0.90, 0.95): {len(pairs_90_95)}")
print(f"  [0.95, 0.98): {len(pairs_95_98)}")
print(f"  >= 0.98:      {len(pairs_gt_98)}")

# Save 24-pair montage (8 from each interval)
np.random.seed(42)
sampled_montage_pairs = []

def sample_k(pool, k):
    if len(pool) >= k:
        idx = np.random.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]
    return pool

sample_90_95 = sample_k(pairs_90_95, 8)
sample_95_98 = sample_k(pairs_95_98, 8)
sample_gt_98 = sample_k(pairs_gt_98, 8)

montage_triplets = [
    ("Cosine in [0.90, 0.95)", sample_90_95),
    ("Cosine in [0.95, 0.98)", sample_95_98),
    ("Cosine >= 0.98", sample_gt_98)
]

print("\nGenerating 24-pair ResNet-50 embedding similarity inspection montage...")
fig = plt.figure(figsize=(16, 26))
gs = GridSpec(6, 4, figure=fig, hspace=0.38, wspace=0.25)

tile_idx = 0
for group_label, pair_list in montage_triplets:
    for u, v, s in pair_list:
        if tile_idx >= 24:
            break
        row_i = tile_idx // 4
        col_i = tile_idx % 4
        
        im_u = Image.open(df.iloc[u]['image_path']).convert('RGB').resize((180, 180))
        im_v = Image.open(df.iloc[v]['image_path']).convert('RGB').resize((180, 180))
        
        combined = Image.new('RGB', (370, 180))
        combined.paste(im_u, (0, 0))
        combined.paste(im_v, (190, 0))
        
        ax = fig.add_subplot(gs[row_i, col_i])
        ax.imshow(combined)
        c_u = df.iloc[u]['disease_class']
        c_v = df.iloc[v]['disease_class']
        fn_u = df.iloc[u]['filename'][:22]
        fn_v = df.iloc[v]['filename'][:22]
        cap = f"{group_label} (sim={s:.4f})\n{c_u} vs {c_v}\n{fn_u}... | {fn_v}..."
        ax.set_title(cap, fontsize=7.5, pad=3)
        ax.axis('off')
        tile_idx += 1

fig.suptitle("ImageNet ResNet-50 Embedding Cosine Similarity Inspection Montage (24 Pairs)\n[8 in 0.90–0.95 | 8 in 0.95–0.98 | 8 in >=0.98]", fontsize=13, fontweight='bold', y=0.995)
embed_montage_path = 'outputs/final_v2/resnet50_embedding_similarity_montage.png'
plt.savefig(embed_montage_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved embedding montage: {embed_montage_path}")

# -------------------------------------------------------------
# Graph Clustering: Combine All 4 Edge Sets
# -------------------------------------------------------------
print("\nBuilding connected components graph combining all 4 edge sets...")
all_edges = edges_base_id | edges_temporal | edges_phash | edges_resnet50
print(f"Total Unique Undirected Edges: {len(all_edges)}")

# Build adjacency matrix
rows = []
cols = []
for u, v in all_edges:
    rows.append(u)
    cols.append(v)
    rows.append(v)
    cols.append(u)

data = np.ones(len(rows), dtype=int)
adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

num_clusters, labels = connected_components(adj, directed=False)
df['cluster_id'] = labels
print(f"Total Connected Clusters formed: {num_clusters}")

# Cluster size statistics
cluster_sizes = pd.Series(labels).value_counts().values
c_stats = {
    'total_clusters': int(num_clusters),
    'min': int(np.min(cluster_sizes)),
    'p25': float(np.percentile(cluster_sizes, 25)),
    'median': float(np.median(cluster_sizes)),
    'mean': float(np.mean(cluster_sizes)),
    'p75': float(np.percentile(cluster_sizes, 75)),
    'p90': float(np.percentile(cluster_sizes, 90)),
    'p95': float(np.percentile(cluster_sizes, 95)),
    'p99': float(np.percentile(cluster_sizes, 99)),
    'max': int(np.max(cluster_sizes))
}

print(f"\nCluster Size Distribution:")
print(f"  Total Clusters: {c_stats['total_clusters']}")
print(f"  Min:    {c_stats['min']}")
print(f"  p25:    {c_stats['p25']}")
print(f"  Median: {c_stats['median']}")
print(f"  Mean:   {c_stats['mean']:.2f}")
print(f"  p75:    {c_stats['p75']}")
print(f"  p90:    {c_stats['p90']}")
print(f"  p95:    {c_stats['p95']}")
print(f"  p99:    {c_stats['p99']}")
print(f"  Max:    {c_stats['max']}")

# 10 Largest Clusters
print("\n--- 10 LARGEST CLUSTERS ---")
largest_cluster_ids = pd.Series(labels).value_counts().head(10).index.tolist()
largest_clusters_info = []

for rank, cid in enumerate(largest_cluster_ids, 1):
    c_sub = df[df['cluster_id'] == cid]
    sz = len(c_sub)
    classes_present = c_sub['disease_class'].value_counts().to_dict()
    valid_ts = c_sub['timestamp'].dropna()
    if len(valid_ts) > 1:
        span_sec = float(valid_ts.max() - valid_ts.min())
    else:
        span_sec = 0.0
        
    info = {
        'rank': rank,
        'cluster_id': int(cid),
        'size': int(sz),
        'disease_classes': classes_present,
        'time_span_seconds': span_sec,
        'sample_filenames': c_sub['filename'].head(3).tolist()
    }
    largest_clusters_info.append(info)
    print(f"Rank {rank:2d} | Cluster {cid:4d} | Size={sz:3d} | Classes={classes_present} | Time Span={span_sec:.1f}s")

# Clusters per disease class
print("\n--- CLUSTERS PER DISEASE CLASS ---")
clusters_per_class = {}
for cls in sorted(df['disease_class'].unique()):
    cls_sub = df[df['disease_class'] == cls]
    n_img = len(cls_sub)
    n_cls_clusters = cls_sub['cluster_id'].nunique()
    clusters_per_class[cls] = {
        'images': int(n_img),
        'clusters': int(n_cls_clusters),
        'avg_images_per_cluster': float(n_img / n_cls_clusters)
    }
    print(f"Class: {cls:25} | Images={n_img:4d} | Clusters={n_cls_clusters:4d} | Avg={n_img/n_cls_clusters:.2f}")

part_d_summary = {
    'total_clean_images': n,
    'edge_counts_by_type': {
        'base_id_edges': len(edges_base_id),
        'temporal_edges_30s': len(edges_temporal),
        'phash_edges_le8_multi_interp': len(edges_phash),
        'resnet50_cosine_ge_095': len(edges_resnet50),
        'total_unique_undirected_edges': len(all_edges)
    },
    'resnet50_similarity_pair_counts': {
        'interval_090_095': len(pairs_90_95),
        'interval_095_098': len(pairs_95_98),
        'interval_ge_098': len(pairs_gt_98)
    },
    'montage_embedding_path': embed_montage_path,
    'cluster_size_distribution': c_stats,
    'top_10_largest_clusters': largest_clusters_info,
    'clusters_per_class': clusters_per_class
}

with open('outputs/final_v2/stage2c_part_d.json', 'w') as f:
    json.dump(part_d_summary, f, indent=2)

print("\nSaved Stage 2c Part D results to outputs/final_v2/stage2c_part_d.json")
