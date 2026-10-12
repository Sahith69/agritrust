import os
import sys
import json
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import corn_probs_to_grade

torch.set_num_threads(8)
device = torch.device('cpu')

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("SECTION C: PADDY CHECKS (NO TRAINING)")
print("=" * 80)

# C.1 pHash all 10,053 Paddy images (direct, h-flip, v-flip)
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_total = len(df_paddy)
print(f"Total Paddy images to hash: {n_total}")

def compute_phash_uint64(gray_img):
    # Resize to 32x32
    resized = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    bit_array = (dct_low > med).flatten()
    val = np.uint64(0)
    for b in bit_array:
        val = (val << np.uint64(1)) | np.uint64(b)
    return val

print("Computing pHashes (direct, h-flip, v-flip)...")
t0 = time.time()
hashes_orig = np.zeros(n_total, dtype=np.uint64)
hashes_hflip = np.zeros(n_total, dtype=np.uint64)
hashes_vflip = np.zeros(n_total, dtype=np.uint64)

for i, p in enumerate(df_paddy['image_path']):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        continue
    h_orig = compute_phash_uint64(im)
    h_h = compute_phash_uint64(cv2.flip(im, 1))
    h_v = compute_phash_uint64(cv2.flip(im, 0))
    hashes_orig[i] = h_orig
    hashes_hflip[i] = h_h
    hashes_vflip[i] = h_v

t1 = time.time()
print(f"Computed pHashes for {n_total} images in {t1 - t0:.2f}s")

# Multi-index hashing to find all pairs with Hamming distance <= 5
# Partition 64 bits into 6 non-overlapping chunks: 11, 11, 11, 11, 10, 10 bits
# By pigeonhole principle, any pair with <= 5 bit flips MUST match on at least one chunk.
def get_chunks(val):
    v = int(val)
    return (
        (v >> 53) & 0x7FF,
        (v >> 42) & 0x7FF,
        (v >> 31) & 0x7FF,
        (v >> 20) & 0x7FF,
        (v >> 10) & 0x3FF,
        v & 0x3FF
    )

print("Building multi-index tables...")
table = [{} for _ in range(6)]
for i in range(n_total):
    c_orig = get_chunks(hashes_orig[i])
    for chunk_idx, c_val in enumerate(c_orig):
        table[chunk_idx].setdefault(c_val, []).append(i)

print("Querying near-duplicate pairs (Hamming <= 5)...")
splits = df_paddy['split'].values

def popcount(x):
    return bin(int(x)).count('1')

def min_hamming(i, j):
    # Check all flip combinations between i and j
    # (orig vs orig, orig vs hflip, orig vs vflip)
    h_i = hashes_orig[i]
    d1 = popcount(h_i ^ hashes_orig[j])
    d2 = popcount(h_i ^ hashes_hflip[j])
    d3 = popcount(h_i ^ hashes_vflip[j])
    return min(d1, d2, d3)

matching_pairs = set()

for i in range(n_total):
    candidates = set()
    # Check candidates matching in any of the 3 orientations
    for h in [hashes_orig[i], hashes_hflip[i], hashes_vflip[i]]:
        chunks = get_chunks(h)
        for chunk_idx, c_val in enumerate(chunks):
            if c_val in table[chunk_idx]:
                for j in table[chunk_idx][c_val]:
                    if i < j:
                        candidates.add(j)
    for j in candidates:
        d = min_hamming(i, j)
        if d <= 5:
            matching_pairs.add((i, j, d))

print(f"Found {len(matching_pairs)} total pairs with Hamming <= 5.")

# Analyze cross-split vs within-split
cross_split_pairs = []
train_test_pairs = []
train_val_pairs = []
val_test_pairs = []
within_train_pairs = []
within_val_pairs = []
within_test_pairs = []

for i, j, d in matching_pairs:
    s_i, s_j = splits[i], splits[j]
    if s_i != s_j:
        cross_split_pairs.append((i, j, d, s_i, s_j))
        pair_set = {s_i, s_j}
        if pair_set == {'train', 'test'}:
            train_test_pairs.append((i, j, d))
        elif pair_set == {'train', 'val'}:
            train_val_pairs.append((i, j, d))
        elif pair_set == {'val', 'test'}:
            val_test_pairs.append((i, j, d))
    else:
        if s_i == 'train':
            within_train_pairs.append((i, j, d))
        elif s_i == 'val':
            within_val_pairs.append((i, j, d))
        elif s_i == 'test':
            within_test_pairs.append((i, j, d))

print(f"\nCross-Split Pairs with Hamming <= 5: {len(cross_split_pairs)}")
print(f"  Train - Test: {len(train_test_pairs)}")
print(f"  Train - Val:  {len(train_val_pairs)}")
print(f"  Val - Test:   {len(val_test_pairs)}")

print(f"\nWithin-Split Pairs with Hamming <= 5:")
print(f"  Within Train: {len(within_train_pairs)}")
print(f"  Within Val:   {len(within_val_pairs)}")
print(f"  Within Test:  {len(within_test_pairs)}")

# Compute connected components (duplicate clusters) per split
def compute_clusters(n_split, edges):
    adj = {i: [] for i in range(n_split)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    clusters = []
    for node in range(n_split):
        if node not in visited:
            comp = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(comp)
    return clusters

train_idx_map = {idx: i for i, idx in enumerate(np.where(splits == 'train')[0])}
val_idx_map = {idx: i for i, idx in enumerate(np.where(splits == 'val')[0])}
test_idx_map = {idx: i for i, idx in enumerate(np.where(splits == 'test')[0])}

train_edges = [(train_idx_map[i], train_idx_map[j]) for i, j, d in within_train_pairs]
val_edges = [(val_idx_map[i], val_idx_map[j]) for i, j, d in within_val_pairs]
test_edges = [(test_idx_map[i], test_idx_map[j]) for i, j, d in within_test_pairs]

train_clusters = compute_clusters(len(train_idx_map), train_edges)
val_clusters = compute_clusters(len(val_idx_map), val_edges)
test_clusters = compute_clusters(len(test_idx_map), test_edges)

train_dup_clusters = [c for c in train_clusters if len(c) > 1]
val_dup_clusters = [c for c in val_clusters if len(c) > 1]
test_dup_clusters = [c for c in test_clusters if len(c) > 1]

print(f"\nPer-Split Duplicate Clusters (components with size > 1):")
print(f"  Train: {len(train_dup_clusters)} duplicate clusters (involving {sum(len(c) for c in train_dup_clusters)} images out of {len(train_idx_map)})")
print(f"  Val:   {len(val_dup_clusters)} duplicate clusters (involving {sum(len(c) for c in val_dup_clusters)} images out of {len(val_idx_map)})")
print(f"  Test:  {len(test_dup_clusters)} duplicate clusters (involving {sum(len(c) for c in test_dup_clusters)} images out of {len(test_idx_map)})")

# C.2 Restricted evaluation on the 757 lesion-class test images only
print("\n" + "-" * 50)
print("C.2 EVALUATION ON THE 757 LESION-CLASS TEST IMAGES")
print("-" * 50)

train_df = df_paddy[df_paddy['split'] == 'train']
test_df = df_paddy[df_paddy['split'] == 'test'].reset_index(drop=True)

# Lesion classes (exclude normal, hispa, dead_heart)
lesion_classes = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "downy_mildew",
    "tungro"
]

lesion_test_mask = test_df['disease_class'].isin(lesion_classes).values
lesion_test_df = test_df[lesion_test_mask].reset_index(drop=True)
n_lesion = len(lesion_test_df)
print(f"Number of lesion-class test images: {n_lesion}")

y_lesion = lesion_test_df['severity_grade'].values

# Mode baseline restricted to the 757 images
mode_paddy_train = train_df.groupby('disease_class')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
base_preds = lesion_test_df['disease_class'].map(mode_paddy_train).values

# Load models and predict
tf = get_transforms('test')
tensors_lesion = torch.stack([tf(Image.open(p).convert('RGB')) for p in lesion_test_df['image_path']]).to(device)

def load_paddy_model(path, ordinal=True):
    m = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=ordinal, pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

m_corn = load_paddy_model('outputs/models/paddy_ordinal.pth', ordinal=True)
m_softmax = load_paddy_model('outputs/models/paddy_baseline.pth', ordinal=False)

with torch.no_grad():
    corn_preds = corn_probs_to_grade(torch.sigmoid(m_corn(tensors_lesion)))[0].cpu().numpy()
    softmax_preds = torch.argmax(m_softmax(tensors_lesion), dim=1).cpu().numpy()

def calc_metrics(y_true, y_pred):
    q = float(cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[0,1,2,3]))
    mae = float(mean_absolute_error(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    w1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    return {'qwk': q, 'mae': mae, 'accuracy': acc, 'within_1': w1}

def bootstrap_metrics_ci(y_true, y_pred, B=2000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    q_vals, mae_vals, acc_vals, w1_vals = [], [], [], []
    for _ in range(B):
        idx = np.random.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        try:
            q = cohen_kappa_score(yt, yp, weights='quadratic', labels=[0,1,2,3])
        except Exception:
            q = 0.0
        q_vals.append(q)
        mae_vals.append(mean_absolute_error(yt, yp))
        acc_vals.append(accuracy_score(yt, yp))
        w1_vals.append(np.mean(np.abs(yt - yp) <= 1))
        
    return {
        'qwk_ci': [float(np.percentile(q_vals, 2.5)), float(np.percentile(q_vals, 97.5))],
        'mae_ci': [float(np.percentile(mae_vals, 2.5)), float(np.percentile(mae_vals, 97.5))],
        'acc_ci': [float(np.percentile(acc_vals, 2.5)), float(np.percentile(acc_vals, 97.5))],
        'w1_ci':  [float(np.percentile(w1_vals, 2.5)), float(np.percentile(w1_vals, 97.5))]
    }

m_base = calc_metrics(y_lesion, base_preds)
ci_base = bootstrap_metrics_ci(y_lesion, base_preds)

m_corn_res = calc_metrics(y_lesion, corn_preds)
ci_corn = bootstrap_metrics_ci(y_lesion, corn_preds)

m_soft_res = calc_metrics(y_lesion, softmax_preds)
ci_soft = bootstrap_metrics_ci(y_lesion, softmax_preds)

print("\n--- RESULTS ON 757 LESION-CLASS TEST IMAGES ---")
print(f"Mode Baseline:")
print(f"  QWK:      {m_base['qwk']:.4f} {ci_base['qwk_ci']}")
print(f"  MAE:      {m_base['mae']:.4f} {ci_base['mae_ci']}")
print(f"  Accuracy: {m_base['accuracy']*100:.2f}% {[round(x*100, 2) for x in ci_base['acc_ci']]}")
print(f"  Within-1: {m_base['within_1']*100:.2f}% {[round(x*100, 2) for x in ci_base['w1_ci']]}")

print(f"\nCORN Production Checkpoint:")
print(f"  QWK:      {m_corn_res['qwk']:.4f} {ci_corn['qwk_ci']}")
print(f"  MAE:      {m_corn_res['mae']:.4f} {ci_corn['mae_ci']}")
print(f"  Accuracy: {m_corn_res['accuracy']*100:.2f}% {[round(x*100, 2) for x in ci_corn['acc_ci']]}")
print(f"  Within-1: {m_corn_res['within_1']*100:.2f}% {[round(x*100, 2) for x in ci_corn['w1_ci']]}")

print(f"\nSoftmax Production Checkpoint:")
print(f"  QWK:      {m_soft_res['qwk']:.4f} {ci_soft['qwk_ci']}")
print(f"  MAE:      {m_soft_res['mae']:.4f} {ci_soft['mae_ci']}")
print(f"  Accuracy: {m_soft_res['accuracy']*100:.2f}% {[round(x*100, 2) for x in ci_soft['acc_ci']]}")
print(f"  Within-1: {m_soft_res['within_1']*100:.2f}% {[round(x*100, 2) for x in ci_soft['w1_ci']]}")

# Save full results to json
c_report = {
    'c1_phash_audit': {
        'total_images': n_total,
        'cross_split_hamming_le5': {
            'total': len(cross_split_pairs),
            'train_test': len(train_test_pairs),
            'train_val': len(train_val_pairs),
            'val_test': len(val_test_pairs),
        },
        'within_split_hamming_le5': {
            'within_train': len(within_train_pairs),
            'within_val': len(within_val_pairs),
            'within_test': len(within_test_pairs),
        },
        'per_split_duplicate_clusters': {
            'train': {'num_clusters': len(train_dup_clusters), 'num_images_involved': sum(len(c) for c in train_dup_clusters)},
            'val':   {'num_clusters': len(val_dup_clusters),   'num_images_involved': sum(len(c) for c in val_dup_clusters)},
            'test':  {'num_clusters': len(test_dup_clusters),  'num_images_involved': sum(len(c) for c in test_dup_clusters)},
        }
    },
    'c2_restricted_metrics_757': {
        'n_images': n_lesion,
        'baseline': {**m_base, **ci_base},
        'corn_production': {**m_corn_res, **ci_corn},
        'softmax_production': {**m_soft_res, **ci_soft}
    }
}

with open('outputs/final_v2/stage2_section_c.json', 'w') as f:
    json.dump(c_report, f, indent=2)

print("\nSaved Section C results to outputs/final_v2/stage2_section_c.json")
