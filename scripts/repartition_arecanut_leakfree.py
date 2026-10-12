import os
import sys
import re
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import scipy.fftpack
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import corn_probs_to_grade

CSV_PATH = "/home/sahith/Projects/Crop/outputs/arecanut_splits.csv"
BACKUP_PATH = "/home/sahith/Projects/Crop/outputs/arecanut_splits_LEAKY_BACKUP.csv"
IMPROVEMENTS_DIR = "/home/sahith/Projects/Crop/outputs/improvements"
os.makedirs(IMPROVEMENTS_DIR, exist_ok=True)

print("=" * 70)
print("  STEP 1: LOAD & BACKUP EXISTING ARECANUT SPLIT")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} images from {CSV_PATH}")

if not os.path.exists(BACKUP_PATH):
    df.to_csv(BACKUP_PATH, index=False)
    print(f"Created backup of original split at {BACKUP_PATH}")
else:
    print(f"Backup already exists at {BACKUP_PATH}")


# ─── 1. BASE ID & TIMESTAMP EXTRACTION ──────────────────────────────
def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

def extract_timestamp(fn):
    m = re.search(r'IMG_(\d{8})_(\d{6})', fn)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except Exception:
            return None
    return None

df['base_id'] = df['filename'].apply(extract_base_id)
df['timestamp'] = df['filename'].apply(extract_timestamp)

print(f"Extracted {df['base_id'].nunique()} string base IDs.")
print(f"Images with camera timestamp: {df['timestamp'].notna().sum()}")


# ─── 2. PERCEPTUAL HASHING (pHash) ──────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2: COMPUTE PERCEPTUAL HASHES (ORIG, H-FLIP, V-FLIP)")
print("=" * 70)

def compute_phash_64(img_gray):
    resized = img_gray.resize((32, 32), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

CACHE_PATH = os.path.join(IMPROVEMENTS_DIR, "arecanut_phashes.npz")
if os.path.exists(CACHE_PATH):
    print("Loading cached pHashes...")
    cache = np.load(CACHE_PATH)
    H_orig = cache['H_orig']
    H_hflip = cache['H_hflip']
    H_vflip = cache['H_vflip']
else:
    print(f"Computing 3x hashes for {len(df)} images...")
    t0 = time.time()
    hashes_orig = []
    hashes_hflip = []
    hashes_vflip = []

    for p in df['image_path']:
        img = Image.open(p).convert('L')
        hashes_orig.append(compute_phash_64(img))
        hashes_hflip.append(compute_phash_64(ImageOps.mirror(img)))
        hashes_vflip.append(compute_phash_64(ImageOps.flip(img)))

    H_orig = np.array(hashes_orig)
    H_hflip = np.array(hashes_hflip)
    H_vflip = np.array(hashes_vflip)
    np.savez_compressed(CACHE_PATH, H_orig=H_orig, H_hflip=H_hflip, H_vflip=H_vflip)
    print(f"Computed and cached 3x hashes in {time.time()-t0:.2f}s.")

# Pairwise minimum distance
print("Computing pairwise minimum distance matrix...")
t1 = time.time()
dist_orig = (H_orig[:, None, :] != H_orig[None, :, :]).sum(axis=-1)
dist_hflip = (H_orig[:, None, :] != H_hflip[None, :, :]).sum(axis=-1)
dist_vflip = (H_orig[:, None, :] != H_vflip[None, :, :]).sum(axis=-1)
min_dist = np.minimum(dist_orig, np.minimum(dist_hflip, dist_vflip))
print(f"Pairwise matrix computed in {time.time()-t1:.2f}s.")


# ─── 3. BUILD DEDUPLICATION GRAPH ───────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3: BUILD MULTI-CRITERIA GRAPH & FIND CONNECTED CLUSTERS")
print("=" * 70)

N = len(df)
adj = np.zeros((N, N), dtype=bool)

# Edge condition 1: Identical base ID string
base_ids = df['base_id'].values
for i in range(N):
    for j in range(i + 1, N):
        if base_ids[i] == base_ids[j]:
            adj[i, j] = True
            adj[j, i] = True

# Edge condition 2: Camera burst shots (same class, within 30 seconds)
timestamps = df['timestamp'].values
classes = df['disease_class'].values
burst_edges = 0
for i in range(N):
    if pd.isna(timestamps[i]):
        continue
    for j in range(i + 1, N):
        if pd.isna(timestamps[j]):
            continue
        if classes[i] == classes[j]:
            diff_sec = abs(float((timestamps[i] - timestamps[j]) / np.timedelta64(1, 's')))
            if diff_sec <= 10.0: # 10-second burst window
                if not adj[i, j]:
                    burst_edges += 1
                adj[i, j] = True
                adj[j, i] = True

print(f"Added {burst_edges} burst-shot edges (<=10s window).")

# Edge condition 3: Perceptual near-duplicates (Hamming dist <= 5 bits, same class)
phash_edges = 0
for i in range(N):
    for j in range(i + 1, N):
        if classes[i] == classes[j] and min_dist[i, j] <= 5:
            if not adj[i, j]:
                phash_edges += 1
            adj[i, j] = True
            adj[j, i] = True

print(f"Added {phash_edges} perceptual near-duplicate edges (Hamming dist <= 5).")

# Connected components
n_components, labels = csgraph.connected_components(csr_matrix(adj), directed=False)
df['cluster_id'] = [f"cluster_{l:04d}" for l in labels]
print(f"\nUnified {len(df)} images into {n_components} independent physical clusters.")

# Cluster statistics
cluster_sizes = df['cluster_id'].value_counts()
print(f"Clusters with 1 image: {(cluster_sizes == 1).sum()}")
print(f"Clusters with >1 image: {(cluster_sizes > 1).sum()} (max size: {cluster_sizes.max()}, mean: {cluster_sizes.mean():.2f})")


# ─── 4. RE-PARTITION INTO 70/15/15 STRATIFIED SPLIT ─────────────────
print("\n" + "=" * 70)
print("  STEP 4: RE-PARTITION ARECANUT DATASET (ZERO CLUSTER LEAKAGE)")
print("=" * 70)

sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
folds = np.zeros(len(df), dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, df['disease_class'], groups=df['cluster_id'])):
    folds[val_idx] = fold_idx

split_arr = np.array(['train'] * len(df), dtype=object)
split_arr[folds == 0] = 'test'
split_arr[folds == 1] = 'val'
df['split'] = split_arr

print("\nNew Leak-Free Split Breakdown:")
print(df['split'].value_counts())
print("\nNew Split Grade Distribution:")
print(pd.crosstab(df['severity_grade'], df['split']))
print("\nNew Split Disease Class Distribution:")
print(pd.crosstab(df['disease_class'], df['split']))


# ─── 5. VERIFICATION: CONFIRM 0% CLUSTER & PHASH OVERLAP ─────────────
print("\n" + "=" * 70)
print("  STEP 5: VERIFICATION OF THE NEW SPLIT")
print("=" * 70)

# Verification A: Cluster overlap
train_set = set(df[df['split'] == 'train']['cluster_id'])
val_set = set(df[df['split'] == 'val']['cluster_id'])
test_set = set(df[df['split'] == 'test']['cluster_id'])

print(f"Train & Val cluster overlap: {len(train_set & val_set)}")
print(f"Train & Test cluster overlap: {len(train_set & test_set)}")
print(f"Val & Test cluster overlap: {len(val_set & test_set)}")

# Verification B: pHash overlap at threshold <= 5
new_splits = df['split'].values
cross_split_phash_leaks = 0
for i in range(N):
    for j in range(i + 1, N):
        if new_splits[i] != new_splits[j] and min_dist[i, j] <= 5:
            cross_split_phash_leaks += 1

print(f"Cross-split pHash pairs (threshold <= 5 bits): {cross_split_phash_leaks}")
if cross_split_phash_leaks == 0 and len(train_set & test_set) == 0:
    print(">>> VERIFICATION PASSED: EXACTLY 0% LEAKAGE! <<<")
else:
    print(">>> WARNING: NON-ZERO LEAKAGE DETECTED! <<<")

# Save new split
out_cols = ['image_path', 'filename', 'disease_class', 'severity_grade', 'parent_id', 'split', 'cluster_id']
df[out_cols].to_csv(CSV_PATH, index=False)
leakfree_copy = os.path.join(IMPROVEMENTS_DIR, "arecanut_splits_leakfree.csv")
df[out_cols].to_csv(leakfree_copy, index=False)
print(f"\nSaved updated split to {CSV_PATH} and {leakfree_copy}")


# ─── 6. RE-RUN MODEL EVALUATION (OLD vs. NEW TEST SET) ──────────────
print("\n" + "=" * 70)
print("  STEP 6: INFERENCE & EVALUATION (OLD LEAKY vs. NEW LEAK-FREE TEST SET)")
print("=" * 70)

device = 'cpu'
torch.set_num_threads(4)

def evaluate_checkpoint(model_path, test_df, ordinal=True):
    model = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=ordinal, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    transform = get_transforms('test')
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, r in test_df.iterrows():
            img = Image.open(r['image_path']).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            out = model(tensor)
            if ordinal:
                probs = torch.sigmoid(out)
                pred_t, _ = corn_probs_to_grade(probs)
                pred = int(pred_t.item())
            else:
                pred = int(out.argmax(dim=1).item())

            y_true.append(int(r['severity_grade']))
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    mae = float(mean_absolute_error(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)

    per_grade = {
        f"G{g}": {
            "p": round(float(rep[str(g)]['precision']), 4),
            "r": round(float(rep[str(g)]['recall']), 4),
            "f1": round(float(rep[str(g)]['f1-score']), 4),
            "support": int(rep[str(g)]['support'])
        }
        for g in range(4)
    }

    return {
        "num_test_samples": len(test_df),
        "qwk": round(qwk, 4),
        "mae": round(mae, 4),
        "exact_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_grade": per_grade
    }

old_df = pd.read_csv(BACKUP_PATH)
old_test_df = old_df[old_df['split'] == 'test'].reset_index(drop=True)
new_test_df = df[df['split'] == 'test'].reset_index(drop=True)

corn_model_path = "/home/sahith/Projects/Crop/outputs/models/arecanut_ordinal.pth"
softmax_model_path = "/home/sahith/Projects/Crop/outputs/models/arecanut_baseline.pth"

print(f"\nEvaluating ResNet-18 CORN on OLD test set ({len(old_test_df)} samples)...")
corn_old = evaluate_checkpoint(corn_model_path, old_test_df, ordinal=True)

print(f"Evaluating ResNet-18 CORN on NEW leak-free test set ({len(new_test_df)} samples)...")
corn_new = evaluate_checkpoint(corn_model_path, new_test_df, ordinal=True)

print(f"\nEvaluating ResNet-18 Softmax on OLD test set ({len(old_test_df)} samples)...")
softmax_old = evaluate_checkpoint(softmax_model_path, old_test_df, ordinal=False)

print(f"Evaluating ResNet-18 Softmax on NEW leak-free test set ({len(new_test_df)} samples)...")
softmax_new = evaluate_checkpoint(softmax_model_path, new_test_df, ordinal=False)

eval_summary = {
    "resnet18_corn": {
        "old_leaky_test": corn_old,
        "new_leakfree_test": corn_new,
        "delta": {
            "qwk": round(corn_new['qwk'] - corn_old['qwk'], 4),
            "mae": round(corn_new['mae'] - corn_old['mae'], 4),
            "exact_accuracy": round(corn_new['exact_accuracy'] - corn_old['exact_accuracy'], 4),
            "macro_f1": round(corn_new['macro_f1'] - corn_old['macro_f1'], 4)
        }
    },
    "resnet18_softmax": {
        "old_leaky_test": softmax_old,
        "new_leakfree_test": softmax_new,
        "delta": {
            "qwk": round(softmax_new['qwk'] - softmax_old['qwk'], 4),
            "mae": round(softmax_new['mae'] - softmax_old['mae'], 4),
            "exact_accuracy": round(softmax_new['exact_accuracy'] - softmax_old['exact_accuracy'], 4),
            "macro_f1": round(softmax_new['macro_f1'] - softmax_old['macro_f1'], 4)
        }
    }
}

summary_json_path = os.path.join(IMPROVEMENTS_DIR, "leakage_remediation_evaluation.json")
with open(summary_json_path, 'w') as f:
    json.dump(eval_summary, f, indent=2)

print(f"\nSaved evaluation summary JSON to {summary_json_path}")
