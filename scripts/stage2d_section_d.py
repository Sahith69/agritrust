import os
import sys
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2d - SECTION D: ARECANUT EFFECTIVE SAMPLE SIZES & CLUSTER AUDIT")
print("=" * 80)

# 1. Load Clean Manifest v2
df = pd.read_csv('outputs/final_v2/arecanut_clean_v2.csv')
n = len(df)
print(f"Total clean images: {n}")

# Grade definition: JSON grade for original_871, Eq. 2 grade for Kaggle
# In arecanut_clean_v2.csv, 'severity_grade' already reflects this:
# for original_871 it is the original JSON grade, and for kaggle_1896 it was assigned via Eq. 2!
print(f"Source counts: {df['source'].value_counts().to_dict()}")

# Extract base IDs
def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

df['base_id'] = df['filename'].apply(extract_base_id)

# 2. Extract timestamps
def extract_timestamp(fn):
    m = re.search(r'IMG_(\d{8})_(\d{6})', fn)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None

df['timestamp_str'] = df['filename'].apply(extract_timestamp)
seconds_of_day = []
for ts in df['timestamp_str']:
    if isinstance(ts, str) and len(ts) >= 15:
        sec = int(ts[9:11]) * 3600 + int(ts[11:13]) * 60 + int(ts[13:15])
        seconds_of_day.append(sec)
    else:
        seconds_of_day.append(None)
df['sec_of_day'] = seconds_of_day

classes = df['disease_class'].values

# Edge set 1: Base ID
print("\nComputing Base ID edges...")
edges_base_id = set()
base_groups = df.groupby('base_id').indices
for b_id, idxs in base_groups.items():
    if len(idxs) > 1:
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                edges_base_id.add((idxs[i], idxs[j]))
print(f"Base ID edges: {len(edges_base_id)}")

# Edge set 2: Temporal Proximity (<= 30s, same class)
print("Computing Temporal Proximity edges (<= 30 s, same class)...")
edges_temporal = set()
for cls in np.unique(classes):
    cls_indices = np.where(classes == cls)[0]
    cls_df = df.iloc[cls_indices]
    timed_mask = cls_df['sec_of_day'].notna().values
    timed_indices = cls_indices[timed_mask]
    timed_secs = cls_df.loc[cls_df['sec_of_day'].notna(), 'sec_of_day'].values
    
    order = np.argsort(timed_secs)
    sorted_idx = timed_indices[order]
    sorted_s = timed_secs[order]
    
    for i in range(len(sorted_s)):
        for j in range(i + 1, len(sorted_s)):
            dt = sorted_s[j] - sorted_s[i]
            if dt <= 30:
                u, v = min(sorted_idx[i], sorted_idx[j]), max(sorted_idx[i], sorted_idx[j])
                edges_temporal.add((u, v))
            else:
                break
print(f"Temporal edges (<= 30 s): {len(edges_temporal)}")

# Edge set 3: Multi-resize & multi-orientation pHash
print("Computing / loading multi-resize/orientation pHashes...")
cache_phash_path = 'outputs/final_v2/arecanut_phashes_multi_cache.npy'

def compute_phash_uint64_interp(gray_img, interp):
    resized = cv2.resize(gray_img, (32, 32), interpolation=interp).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    bit_array = (dct_low > med).flatten()
    val = np.uint64(0)
    for b in bit_array:
        val = (val << np.uint64(1)) | np.uint64(b)
    return val

def fast_popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x.astype(np.int32)

if os.path.exists(cache_phash_path):
    print(f"Loading cached pHashes from {cache_phash_path}...")
    hashes_all = np.load(cache_phash_path)
else:
    interps = [cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR]
    hashes_all = np.zeros((n, 9), dtype=np.uint64)
    for i, p in enumerate(df['image_path']):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im_h = cv2.flip(im, 1)
        im_v = cv2.flip(im, 0)
        col = 0
        for interp in interps:
            hashes_all[i, col]   = compute_phash_uint64_interp(im, interp)
            hashes_all[i, col+1] = compute_phash_uint64_interp(im_h, interp)
            hashes_all[i, col+2] = compute_phash_uint64_interp(im_v, interp)
            col += 3
    np.save(cache_phash_path, hashes_all)

edges_phash = set()
chunk = 400
for i_start in range(0, n, chunk):
    i_end = min(i_start + chunk, n)
    h_chunk_i = hashes_all[i_start:i_end]
    for j_start in range(i_start, n, chunk):
        j_end = min(j_start + chunk, n)
        h_chunk_j = hashes_all[j_start:j_end]
        min_dist_block = np.full((i_end - i_start, j_end - j_start), 64, dtype=np.int32)
        for k1 in range(9):
            col_k1 = h_chunk_i[:, k1, None]
            for k2 in range(9):
                col_k2 = h_chunk_j[:, k2][None, :]
                xor_m = col_k1 ^ col_k2
                d = fast_popcount(xor_m)
                min_dist_block = np.minimum(min_dist_block, d)
        hits = np.where(min_dist_block <= 8)
        for r_i, c_j in zip(hits[0], hits[1]):
            u = i_start + r_i
            v = j_start + c_j
            if u < v:
                edges_phash.add((u, v))
print(f"pHash edges (Hamming <= 8): {len(edges_phash)}")

# Edge set 4: ResNet-50 Cosine Similarity >= 0.95
cache_embed_path = 'outputs/final_v2/arecanut_resnet50_embeddings.npy'
if os.path.exists(cache_embed_path):
    print(f"Loading cached ResNet-50 embeddings from {cache_embed_path}...")
    embeddings = np.load(cache_embed_path)
else:
    print("Extracting ImageNet ResNet-50 feature embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Identity()
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
    embed_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = resnet50(batch)
            embed_list.append(feats.cpu().numpy())
    embeddings = np.vstack(embed_list)
    np.save(cache_embed_path, embeddings)

norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
embeddings_norm = embeddings / norms

cos_sim = np.zeros((n, n), dtype=np.float32)
for b_start in range(0, n, 500):
    b_end = min(b_start + 500, n)
    cos_sim[b_start:b_end] = np.dot(embeddings_norm[b_start:b_end], embeddings_norm.T)

edges_resnet50 = set()
for u in range(n):
    for v in range(u + 1, n):
        if cos_sim[u, v] >= 0.95:
            edges_resnet50.add((u, v))
print(f"ResNet-50 Cosine Similarity >= 0.95 edges: {len(edges_resnet50)}")

# Check cross-class edges in each set
def check_cross_class(edge_set, name):
    cross = []
    for u, v in edge_set:
        if classes[u] != classes[v]:
            cross.append((u, v))
    print(f"Cross-class edges in {name}: {len(cross)}")
    return cross

cross_base = check_cross_class(edges_base_id, "Base ID")
cross_temp = check_cross_class(edges_temporal, "Temporal")
cross_phash = check_cross_class(edges_phash, "pHash")
cross_resnet = check_cross_class(edges_resnet50, "ResNet-50")

# Graph connected components combining all 4 edge sets
all_edges = edges_base_id | edges_temporal | edges_phash | edges_resnet50
rows = []
cols = []
for u, v in all_edges:
    rows.append(u); cols.append(v)
    rows.append(v); cols.append(u)
data = np.ones(len(rows), dtype=int)
adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
num_clusters, labels = connected_components(adj, directed=False)
df['robust_cluster_id'] = labels
print(f"\nTotal Connected Clusters formed: {num_clusters}")

# Identify cross-class clusters
cluster_class_counts = df.groupby('robust_cluster_id')['disease_class'].nunique()
multi_class_clusters = cluster_class_counts[cluster_class_counts > 1]
print(f"Clusters spanning multiple classes: {len(multi_class_clusters)}")

cross_cluster_details = []
for c_id in multi_class_clusters.index:
    c_images = df[df['robust_cluster_id'] == c_id]
    c_classes = c_images['disease_class'].value_counts().to_dict()
    print(f"\nCross-class Cluster #{c_id} (Size = {len(c_images)}):")
    print(f"  Classes: {c_classes}")
    
    # Find which edges inside this cluster connect different classes
    c_indices = set(c_images.index)
    c_cross_edges = []
    for u, v in all_edges:
        if u in c_indices and v in c_indices and classes[u] != classes[v]:
            # Identify which edge set it belongs to
            types = []
            if (u, v) in edges_base_id: types.append('base_id')
            if (u, v) in edges_temporal: types.append('temporal')
            if (u, v) in edges_phash: types.append('phash')
            if (u, v) in edges_resnet50: types.append('resnet50_sim')
            sim_val = float(cos_sim[u, v])
            c_cross_edges.append({
                'image_u': df.iloc[u]['filename'],
                'class_u': classes[u],
                'image_v': df.iloc[v]['filename'],
                'class_v': classes[v],
                'edge_types': types,
                'resnet50_cosine': sim_val
            })
            print(f"  Edge: {df.iloc[u]['filename']} ({classes[u]}) <---> {df.iloc[v]['filename']} ({classes[v]}) via {types} (cosine={sim_val:.4f})")
    
    cross_cluster_details.append({
        'cluster_id': int(c_id),
        'size': len(c_images),
        'class_breakdown': c_classes,
        'cross_edges': c_cross_edges,
        'all_images': c_images[['filename', 'disease_class', 'source', 'severity_grade']].to_dict(orient='records')
    })

# Save robust clusters to arecanut_clean_v2_robust_clusters.csv
df.to_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv', index=False)
print("Saved outputs/final_v2/arecanut_clean_v2_robust_clusters.csv")

# ==============================================================================
# D.1 CLASS x GRADE TABLE (IMAGES & CLUSTERS)
# ==============================================================================
# Table: class x grade: images, clusters
# JSON grades for original images, Eq. 2 for Kaggle (which severity_grade holds)
print("\n--- CLASS x GRADE TABLE OF IMAGES AND CLUSTERS ---")
class_grade_rows = []
for cls in sorted(df['disease_class'].unique()):
    cls_df = df[df['disease_class'] == cls]
    for g in sorted(cls_df['severity_grade'].unique()):
        sub = cls_df[cls_df['severity_grade'] == g]
        n_img = len(sub)
        n_clus = sub['robust_cluster_id'].nunique()
        class_grade_rows.append({
            'disease_class': cls,
            'severity_grade': int(g),
            'images': int(n_img),
            'clusters': int(n_clus)
        })
        print(f"Class: {cls:22} | Grade {g} | Images: {n_img:4d} | Clusters: {n_clus:4d}")

# How many clusters span more than one grade?
cluster_grade_counts = df.groupby('robust_cluster_id')['severity_grade'].nunique()
multi_grade_clusters = cluster_grade_counts[cluster_grade_counts > 1]
print(f"\nClusters spanning more than one severity grade: {len(multi_grade_clusters)} / {num_clusters} ({len(multi_grade_clusters)/num_clusters*100:.2f}%)")

# Within lesion classes only:
lesion_df = df[df['disease_class'].isin(['bud_borer', 'mahali_koleroga', 'stem_bleeding', 'stem_cracking'])]
lesion_clus_grades = lesion_df.groupby('robust_cluster_id')['severity_grade'].nunique()
multi_grade_lesion = lesion_clus_grades[lesion_clus_grades > 1]
print(f"Lesion clusters spanning more than one grade: {len(multi_grade_lesion)} / {lesion_df['robust_cluster_id'].nunique()} ({len(multi_grade_lesion)/lesion_df['robust_cluster_id'].nunique()*100:.2f}%)")

# ==============================================================================
# D.3 CLASSES WITH < 15 CLUSTERS & 5-FOLD GROUPKFOLD PROPOSAL
# ==============================================================================
clusters_per_class = df.groupby('disease_class')['robust_cluster_id'].nunique()
classes_lt_15 = clusters_per_class[clusters_per_class < 15].to_dict()
print(f"\nClasses with < 15 clusters:")
for c, cnt in classes_lt_15.items():
    print(f"  • {c:20}: {cnt} clusters ({len(df[df['disease_class'] == c])} images)")

# Propose (don't run) a 5-fold GroupKFold over clusters stratified by class
# Let's compute the proposal assignment to report exact fold statistics
print("\nComputing 5-fold GroupKFold proposal statistics...")
# Stratified cluster-level assignment:
# For each class, distribute its clusters across 5 folds aiming for balanced images
np.random.seed(42)
fold_assignments = {} # cluster_id -> fold (0 to 4)

for cls in sorted(df['disease_class'].unique()):
    cls_df = df[df['disease_class'] == cls]
    cls_cluster_sizes = cls_df.groupby('robust_cluster_id').size()
    # Sort clusters descending by size (greedy balanced partition)
    sorted_clusters = cls_cluster_sizes.sort_values(ascending=False)
    
    # 5 buckets tracking total images in this class
    fold_img_counts = [0] * 5
    for c_id, c_sz in sorted_clusters.items():
        if c_id not in fold_assignments:
            # assign to fold with minimum total images
            best_f = int(np.argmin(fold_img_counts))
            fold_assignments[c_id] = best_f
            fold_img_counts[best_f] += c_sz

df['proposed_fold'] = df['robust_cluster_id'].map(fold_assignments)

fold_summary = []
for f_i in range(5):
    f_df = df[df['proposed_fold'] == f_i]
    f_clusters = f_df['robust_cluster_id'].nunique()
    f_images = len(f_df)
    
    # Largest cluster share
    largest_c_size = f_df.groupby('robust_cluster_id').size().max()
    largest_c_share = (largest_c_size / f_images) * 100.0
    
    # Clusters per class
    cls_clus = f_df.groupby('disease_class')['robust_cluster_id'].nunique().to_dict()
    
    fold_summary.append({
        'fold': f_i,
        'images': f_images,
        'clusters': f_clusters,
        'largest_cluster_size': int(largest_c_size),
        'largest_cluster_share_pct': float(largest_c_share),
        'clusters_per_class': cls_clus
    })
    print(f"Fold {f_i}: Images={f_images:4d} ({f_images/n*100:.1f}%) | Clusters={f_clusters:3d} | Largest Cluster={largest_c_size} ({largest_c_share:.1f}%)")

stage2d_d_output = {
    'total_images': n,
    'total_clusters': int(num_clusters),
    'class_grade_table': class_grade_rows,
    'clusters_spanning_multi_grade': {
        'overall': int(len(multi_grade_clusters)),
        'lesion_classes_only': int(len(multi_grade_lesion))
    },
    'cross_class_clusters': cross_cluster_details,
    'classes_with_fewer_than_15_clusters': classes_lt_15,
    'groupkfold_5fold_proposal': fold_summary
}

with open('outputs/final_v2/stage2d_section_d.json', 'w') as f:
    json.dump(stage2d_d_output, f, indent=2)

print("\nSaved Stage 2d Section D results to outputs/final_v2/stage2d_section_d.json")
