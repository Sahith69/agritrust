"""
scripts/dev/prepare_apple_fgvc_splits.py - Leak-Safe Cluster Partitioning of FGVC Apple Images

Stage 3 Step 3b:
1. Filter data/apple_fgvc/train.csv: drop multiple_diseases, map scab/rust/healthy to:
   apple_scab, cedar_apple_rust, healthy.
2. Compute pHash (orig, hflip, vflip) and DINOv2 vits14 embeddings for all 1,730 images.
3. Form connected-component clusters where pHash <= 8 OR DINOv2 cosine >= 0.95.
4. Report cluster count and size distribution.
5. Partition whole clusters 60/20/20 into fgvc_train / fgvc_val / fgvc_test (stratified, seed 42).
6. Verify no fgvc_test image leaks into fgvc_train, fgvc_val, or PlantVillage apple train split
   (pHash <= 8 or DINOv2 cosine >= 0.95); drop and count any leaking test images.
7. Save split to outputs/apple_fgvc/fgvc_splits.csv.
"""

import os
import time
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
import torchvision.transforms as T
import imagehash

OUT_DIR = "outputs/apple_fgvc"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Data Ingestion & Mapping
csv_path = "data/apple_fgvc/train.csv"
df = pd.read_csv(csv_path)

# Drop multiple_diseases
df = df[df["multiple_diseases"] == 0].copy()

def get_disease_class(row):
    if row["scab"] == 1:
        return "apple_scab"
    elif row["rust"] == 1:
        return "cedar_apple_rust"
    elif row["healthy"] == 1:
        return "healthy"
    raise ValueError(f"Unknown row class: {row}")

df["disease_class"] = df.apply(get_disease_class, axis=1)
df["image_path"] = df["image_id"].apply(lambda x: os.path.join("data/apple_fgvc/images", f"{x}.jpg"))

# Check all paths exist
assert all(os.path.exists(p) for p in df["image_path"]), "Missing image file(s) in data/apple_fgvc/images/"
print(f"Total single-disease FGVC images: {len(df)}")
print("Class counts:\n", df["disease_class"].value_counts())

# ── 2. Feature Extraction (pHash + DINOv2)
feats_cache = os.path.join(OUT_DIR, "fgvc_features.npz")

dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = df["image_path"].tolist()
n_imgs = len(image_paths)

if os.path.exists(feats_cache):
    print(f"Loading cached FGVC features from {feats_cache}...")
    cached = np.load(feats_cache)
    phashes_orig = cached["phashes_orig"]
    phashes_hflip = cached["phashes_hflip"]
    phashes_vflip = cached["phashes_vflip"]
    dino_embeddings = cached["dino_embeddings"]
else:
    print("Computing pHashes and DINOv2 embeddings for FGVC images...")
    phashes_orig = []
    phashes_hflip = []
    phashes_vflip = []
    dino_list = []

    t0 = time.time()
    batch_tensors = []
    batch_indices = []

    for idx, path in enumerate(image_paths):
        with Image.open(path) as pil_img:
            rgb = pil_img.convert("RGB")
            # pHashes
            h_orig = imagehash.phash(rgb)
            h_hflip = imagehash.phash(rgb.transpose(Image.FLIP_LEFT_RIGHT))
            h_vflip = imagehash.phash(rgb.transpose(Image.FLIP_TOP_BOTTOM))
            phashes_orig.append(np.uint64(int(str(h_orig), 16)))
            phashes_hflip.append(np.uint64(int(str(h_hflip), 16)))
            phashes_vflip.append(np.uint64(int(str(h_vflip), 16)))

            tensor = transform(rgb)
            batch_tensors.append(tensor)
            batch_indices.append(idx)

        if len(batch_tensors) == 32 or idx == n_imgs - 1:
            batch_stack = torch.stack(batch_tensors)
            with torch.no_grad():
                embs = dinov2(batch_stack)
                # L2 normalize
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                dino_list.append(embs.cpu().numpy())
            batch_tensors = []
            if (idx + 1) % 256 == 0 or idx == n_imgs - 1:
                print(f"  Processed {idx + 1}/{n_imgs} images ({time.time() - t0:.1f}s)...")

    phashes_orig = np.array(phashes_orig, dtype=np.uint64)
    phashes_hflip = np.array(phashes_hflip, dtype=np.uint64)
    phashes_vflip = np.array(phashes_vflip, dtype=np.uint64)
    dino_embeddings = np.concatenate(dino_list, axis=0)

    np.savez_compressed(
        feats_cache,
        phashes_orig=phashes_orig,
        phashes_hflip=phashes_hflip,
        phashes_vflip=phashes_vflip,
        dino_embeddings=dino_embeddings
    )
    print(f"Saved FGVC features cache in {time.time() - t0:.1f}s")

# ── 3. Connected-Component Clustering (pHash <= 8 OR DINOv2 cosine >= 0.95)
print("Building clustering graph for connected components...")
adj_matrix = np.zeros((n_imgs, n_imgs), dtype=bool)

# Fast vectorized pairwise calculations
# DINO cosine similarity: dot product since L2-normalized
sim_matrix = np.dot(dino_embeddings, dino_embeddings.T)
dino_adj = sim_matrix >= 0.95

# pHash pairwise min distance across orientations
phash_adj = np.zeros((n_imgs, n_imgs), dtype=bool)
for i in range(n_imgs):
    cand_hashes = [phashes_orig[i], phashes_hflip[i], phashes_vflip[i]]
    for ch in cand_hashes:
        for split_arr in [phashes_orig, phashes_hflip, phashes_vflip]:
            diffs = np.bitwise_count(split_arr ^ ch)
            phash_adj[i] |= (diffs <= 8)

adj_matrix = dino_adj | phash_adj
np.fill_diagonal(adj_matrix, True)

graph = csr_matrix(adj_matrix)
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
df["cluster_id"] = [f"cluster_{c:04d}" for c in labels]

print(f"\n=======================================================")
print(f"CONNECTED COMPONENTS CLUSTERING REPORT")
print(f"=======================================================")
print(f"Total FGVC images: {n_imgs}")
print(f"Total connected-component clusters: {n_components}")

cluster_sizes = pd.Series(labels).value_counts()
print(f"Cluster size distribution:")
print(f"  Min size: {cluster_sizes.min()}")
print(f"  Median size: {cluster_sizes.median()}")
print(f"  Mean size: {cluster_sizes.mean():.2f}")
print(f"  Max size: {cluster_sizes.max()}")
print(f"  Singletons (size=1): {(cluster_sizes == 1).sum()} ({(cluster_sizes == 1).sum() / n_components * 100:.1f}%)")
print(f"  Clusters with size > 1: {(cluster_sizes > 1).sum()}")
print(f"  Size breakdown: {dict(cluster_sizes.value_counts().head(10))}")
print(f"=======================================================\n")

# ── 4. Split Clusters 60/20/20 Stratified by Dominant Class (seed 42)
rng = np.random.default_rng(42)

cluster_summary = []
for c_id, group in df.groupby("cluster_id"):
    dom_class = group["disease_class"].mode()[0]
    cluster_summary.append({
        "cluster_id": c_id,
        "size": len(group),
        "dominant_class": dom_class
    })
cluster_df = pd.DataFrame(cluster_summary)

train_clusters = []
val_clusters = []
test_clusters = []

for cls, c_group in cluster_df.groupby("dominant_class"):
    # Shuffle clusters deterministically
    c_indices = c_group.index.to_numpy().copy()
    rng.shuffle(c_indices)
    
    n_c = len(c_indices)
    n_train = int(round(0.60 * n_c))
    n_val = int(round(0.20 * n_c))
    
    train_c = c_group.loc[c_indices[:n_train], "cluster_id"].tolist()
    val_c = c_group.loc[c_indices[n_train:n_train + n_val], "cluster_id"].tolist()
    test_c = c_group.loc[c_indices[n_train + n_val:], "cluster_id"].tolist()
    
    train_clusters.extend(train_c)
    val_clusters.extend(val_c)
    test_clusters.extend(test_c)

split_map = {}
for c in train_clusters:
    split_map[c] = "train"
for c in val_clusters:
    split_map[c] = "val"
for c in test_clusters:
    split_map[c] = "test"

df["split"] = df["cluster_id"].map(split_map)

# ── 5. Leakage Verification for fgvc_test
# Check test images against fgvc_train, fgvc_val, and PlantVillage apple train
print("Checking for any leakage into fgvc_test...")

# Compute PlantVillage apple train features
pv_df = pd.read_csv("outputs/apple_splits.csv")
pv_train_df = pv_df[pv_df["split"] == "train"].copy()
pv_cache = os.path.join(OUT_DIR, "plantvillage_train_features.npz")

if os.path.exists(pv_cache):
    pv_cached = np.load(pv_cache)
    pv_orig = pv_cached["orig"]
    pv_hflip = pv_cached["hflip"]
    pv_vflip = pv_cached["vflip"]
    pv_dino = pv_cached["dino"]
else:
    print(f"Computing features for {len(pv_train_df)} PlantVillage apple train images...")
    pv_orig = []
    pv_hflip = []
    pv_vflip = []
    pv_dino_list = []
    pv_batch = []
    for p in pv_train_df["image_path"]:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            pv_orig.append(np.uint64(int(str(imagehash.phash(rgb)), 16)))
            pv_hflip.append(np.uint64(int(str(imagehash.phash(rgb.transpose(Image.FLIP_LEFT_RIGHT))), 16)))
            pv_vflip.append(np.uint64(int(str(imagehash.phash(rgb.transpose(Image.FLIP_TOP_BOTTOM))), 16)))
            pv_batch.append(transform(rgb))
        if len(pv_batch) == 32:
            with torch.no_grad():
                embs = dinov2(torch.stack(pv_batch))
                pv_dino_list.append(torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy())
            pv_batch = []
    if pv_batch:
        with torch.no_grad():
            embs = dinov2(torch.stack(pv_batch))
            pv_dino_list.append(torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy())
    pv_orig = np.array(pv_orig, dtype=np.uint64)
    pv_hflip = np.array(pv_hflip, dtype=np.uint64)
    pv_vflip = np.array(pv_vflip, dtype=np.uint64)
    pv_dino = np.concatenate(pv_dino_list, axis=0)
    np.savez_compressed(pv_cache, orig=pv_orig, hflip=pv_hflip, vflip=pv_vflip, dino=pv_dino)

# Get index masks for fgvc_train, fgvc_val, fgvc_test
train_mask = (df["split"] == "train").to_numpy()
val_mask = (df["split"] == "val").to_numpy()
test_mask = (df["split"] == "test").to_numpy()

test_indices = np.where(test_mask)[0]
non_test_dino = np.concatenate([dino_embeddings[train_mask | val_mask], pv_dino], axis=0)
non_test_phashes = [
    np.concatenate([phashes_orig[train_mask | val_mask], pv_orig]),
    np.concatenate([phashes_hflip[train_mask | val_mask], pv_hflip]),
    np.concatenate([phashes_vflip[train_mask | val_mask], pv_vflip])
]

dropped_test_indices = []
for idx in test_indices:
    # 1. DINO cosine
    sims = np.dot(non_test_dino, dino_embeddings[idx])
    max_sim = np.max(sims)
    if max_sim >= 0.95:
        dropped_test_indices.append((idx, f"DINO cosine {max_sim:.4f} >= 0.95"))
        continue

    # 2. pHash distance
    cand_hs = [phashes_orig[idx], phashes_hflip[idx], phashes_vflip[idx]]
    leaked = False
    for ch in cand_hs:
        for ref_arr in non_test_phashes:
            diffs = np.bitwise_count(ref_arr ^ ch)
            if np.min(diffs) <= 8:
                dropped_test_indices.append((idx, f"pHash diff {np.min(diffs)} <= 8"))
                leaked = True
                break
        if leaked:
            break

print(f"fgvc_test leakage check completed:")
print(f"  Test candidates: {len(test_indices)}")
print(f"  Dropped test images due to leakage: {len(dropped_test_indices)}")
for idx, reason in dropped_test_indices:
    print(f"    - Image {df.iloc[idx]['image_id']} dropped: {reason}")

if dropped_test_indices:
    drop_idx_set = set(idx for idx, _ in dropped_test_indices)
    df = df.drop(index=[df.index[i] for i in drop_idx_set]).reset_index(drop=True)

# Final split distribution
print(f"\nFinal Split Image Counts:")
print(df["split"].value_counts())
print("\nClass breakdown per split:")
print(pd.crosstab(df["split"], df["disease_class"]))

# Save split CSV
out_csv = os.path.join(OUT_DIR, "fgvc_splits.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved split CSV to {out_csv}")
