import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2d - SECTION C: PADDY STATISTICS & COMPONENT-GROUPED BOOTSTRAP")
print("=" * 80)

# ==============================================================================
# C.1 RECONCILING PAIR COUNTS
# ==============================================================================
print("\n--- C.1 PAIR COUNT RECONCILIATION ---")
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_paddy = len(df_paddy)
print(f"Total Paddy images: {n_paddy}")

splits = df_paddy['split'].values
test_idx = np.where(splits == 'test')[0]
train_idx = np.where(splits == 'train')[0]
val_idx = np.where(splits == 'val')[0]
train_val_idx = np.where(splits != 'test')[0]

print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

# Load precomputed pHashes
cache_phash_path = 'outputs/final_v2/paddy_phashes_cache.npz'
print(f"Loading cached pHashes from {cache_phash_path}...")
npz = np.load(cache_phash_path)
hashes_orig = npz['orig']
hashes_hflip = npz['hflip']
hashes_vflip = npz['vflip']

def fast_popcount_matrix(m):
    m = m - ((m >> np.uint64(1)) & np.uint64(0x5555555555555555))
    m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
    m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return m.astype(np.int32)

def count_cross_pairs(idx_a, idx_b, max_hamming=5):
    h_a = hashes_orig[idx_a]
    h_b_orig = hashes_orig[idx_b]
    h_b_hflip = hashes_hflip[idx_b]
    h_b_vflip = hashes_vflip[idx_b]
    
    pairs = []
    chunk_a = 500
    chunk_b = 2500
    for a_start in range(0, len(idx_a), chunk_a):
        a_end = min(a_start + chunk_a, len(idx_a))
        sub_a = h_a[a_start:a_end]
        for b_start in range(0, len(idx_b), chunk_b):
            b_end = min(b_start + chunk_b, len(idx_b))
            
            xor1 = sub_a[:, None] ^ h_b_orig[b_start:b_end][None, :]
            xor2 = sub_a[:, None] ^ h_b_hflip[b_start:b_end][None, :]
            xor3 = sub_a[:, None] ^ h_b_vflip[b_start:b_end][None, :]
            
            d1 = fast_popcount_matrix(xor1)
            d2 = fast_popcount_matrix(xor2)
            d3 = fast_popcount_matrix(xor3)
            min_d = np.minimum(d1, np.minimum(d2, d3))
            
            hits = np.where(min_d <= max_hamming)
            for r_i, c_j in zip(hits[0], hits[1]):
                pairs.append((idx_a[a_start + r_i], idx_b[b_start + c_j], int(min_d[r_i, c_j])))
    return pairs

print("Counting cross-split pairs at Hamming <= 5...")
pairs_train_test = count_cross_pairs(test_idx, train_idx, max_hamming=5)
pairs_val_test = count_cross_pairs(test_idx, val_idx, max_hamming=5)
pairs_train_val = count_cross_pairs(val_idx, train_idx, max_hamming=5)

pairs_tv_test = count_cross_pairs(test_idx, train_val_idx, max_hamming=5)

print(f"Pair counts at Hamming <= 5:")
print(f"  • Train-Test pairs: {len(pairs_train_test)}")
print(f"  • Val-Test pairs:   {len(pairs_val_test)}")
print(f"  • Train-Val pairs:  {len(pairs_train_val)}")
print(f"  • Total All Cross-Split Pairs (Stage 2 sum): {len(pairs_train_test) + len(pairs_val_test) + len(pairs_train_val)} (Expected 1226)")
print(f"  • Total (Train+Val)-to-Test Pairs (Stage 2c Part A): {len(pairs_tv_test)} (Expected 713)")

reconciliation_summary = {
    'stage2_all_cross_split_pairs': {
        'train_test': len(pairs_train_test),
        'val_test': len(pairs_val_test),
        'train_val': len(pairs_train_val),
        'total_cross_split': len(pairs_train_test) + len(pairs_val_test) + len(pairs_train_val),
        'description': "Evaluated all 3 cross-split boundaries: Train-Test (602) + Val-Test (111) + Train-Val (513) = 1,226 total pairs."
    },
    'stage2c_test_leakage_pairs': {
        'test_vs_train_val': len(pairs_tv_test),
        'test_unique_images_with_twin_le5': len(set(p[0] for p in pairs_tv_test)),
        'description': "Evaluated test set leakage only: Test images matched against (Train + Val) = 602 + 111 = 713 pairs, involving 362 unique test images."
    },
    'hash_and_resize_method': "64-bit DCT pHash via cv2.resize((32, 32), cv2.INTER_AREA) on 8-bit grayscale, 8x8 low frequency DCT, median thresholding, evaluated under direct, horizontal flip, and vertical flip."
}

# ==============================================================================
# C.2 PHASH CONNECTED COMPONENTS OVER ALL PADDY
# ==============================================================================
print("\n--- C.2 PHASH CONNECTED COMPONENTS OVER ALL 10,053 PADDY IMAGES ---")
# Build adjacency matrix for <= 5 and <= 8 across all 10,053 images
def build_components(max_hamming):
    print(f"Building connected components graph for Hamming <= {max_hamming}...")
    edges = set()
    chunk = 500
    for i_start in range(0, n_paddy, chunk):
        i_end = min(i_start + chunk, n_paddy)
        h_chunk = hashes_orig[i_start:i_end]
        for j_start in range(i_start, n_paddy, chunk):
            j_end = min(j_start + chunk, n_paddy)
            
            xor1 = h_chunk[:, None] ^ hashes_orig[j_start:j_end][None, :]
            xor2 = h_chunk[:, None] ^ hashes_hflip[j_start:j_end][None, :]
            xor3 = h_chunk[:, None] ^ hashes_vflip[j_start:j_end][None, :]
            
            d1 = fast_popcount_matrix(xor1)
            d2 = fast_popcount_matrix(xor2)
            d3 = fast_popcount_matrix(xor3)
            min_d = np.minimum(d1, np.minimum(d2, d3))
            
            hits = np.where(min_d <= max_hamming)
            for r_i, c_j in zip(hits[0], hits[1]):
                u = i_start + r_i
                v = j_start + c_j
                if u < v:
                    edges.add((u, v))
                    
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(rows), dtype=int)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_paddy, n_paddy))
    n_comp, labels = connected_components(adj, directed=False)
    return n_comp, labels, edges

n_comp_5, labels_5, edges_5 = build_components(5)
n_comp_8, labels_8, edges_8 = build_components(8)

df_paddy['comp_le5'] = labels_5
df_paddy['comp_le8'] = labels_8

# Test images in multi-image components
test_df = df_paddy.iloc[test_idx].copy()

comp_sizes_5 = pd.Series(labels_5).value_counts()
comp_sizes_8 = pd.Series(labels_8).value_counts()

multi_comps_5 = set(comp_sizes_5[comp_sizes_5 > 1].index)
multi_comps_8 = set(comp_sizes_8[comp_sizes_8 > 1].index)

test_in_multi_5 = test_df['comp_le5'].isin(multi_comps_5).sum()
test_in_multi_8 = test_df['comp_le8'].isin(multi_comps_8).sum()

print(f"\nAll-Paddy Components at Hamming <= 5:")
print(f"  • Total components: {n_comp_5}")
print(f"  • Total edges: {len(edges_5)}")
print(f"  • Test images in multi-image components (size > 1): {test_in_multi_5} / {len(test_df)} ({test_in_multi_5/len(test_df)*100:.2f}%)")

print(f"\nAll-Paddy Components at Hamming <= 8:")
print(f"  • Total components: {n_comp_8}")
print(f"  • Total edges: {len(edges_8)}")
print(f"  • Test images in multi-image components (size > 1): {test_in_multi_8} / {len(test_df)} ({test_in_multi_8/len(test_df)*100:.2f}%)")

# ==============================================================================
# C.3 COMPONENT-GROUPED BOOTSTRAP RECOMPUTATION
# ==============================================================================
print("\n--- C.3 COMPONENT-GROUPED BOOTSTRAP FOR LESION-ONLY TEST SUBSETS ---")
# Load model predictions
cache_preds_path = 'outputs/final_v2/paddy_test_predictions_cache.csv'
if os.path.exists(cache_preds_path):
    print(f"Loading test predictions from {cache_preds_path}...")
    preds_df = pd.read_csv(cache_preds_path)
else:
    print("Computing test predictions with CORN and Softmax models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    model_c = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
    ckpt_c = torch.load('outputs/models/paddy_ordinal.pth', map_location='cpu')
    model_c.load_state_dict(ckpt_c if 'model_state_dict' not in ckpt_c else ckpt_c['model_state_dict'])
    model_c.to(device).eval()
    
    model_s = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=False)
    ckpt_s = torch.load('outputs/models/paddy_baseline.pth', map_location='cpu')
    model_s.load_state_dict(ckpt_s if 'model_state_dict' not in ckpt_s else ckpt_s['model_state_dict'])
    model_s.to(device).eval()
    
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            im = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(im)
            
    ds = TestDataset(test_df['image_path'].tolist(), transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    pred_c_list, pred_s_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_c = model_c(batch)
            p_c = torch.sum(torch.sigmoid(out_c) > 0.5, dim=1)
            pred_c_list.extend(p_c.cpu().numpy().tolist())
            
            out_s = model_s(batch)
            p_s = torch.argmax(out_s, dim=1)
            pred_s_list.extend(p_s.cpu().numpy().tolist())
            
    preds_df = pd.DataFrame({
        'filename': test_df['filename'].values,
        'true_grade': test_df['severity_grade'].values,
        'pred_corn': pred_c_list,
        'pred_soft': pred_s_list
    })
    preds_df.to_csv(cache_preds_path, index=False)

test_df['pred_corn'] = preds_df['pred_corn'].values
test_df['pred_soft'] = preds_df['pred_soft'].values

# Mode baseline: mode training grade per disease class
train_df = df_paddy.iloc[train_idx]
class_modes = train_df.groupby('disease_class')['severity_grade'].apply(lambda s: s.mode()[0]).to_dict()
test_df['pred_mode'] = test_df['disease_class'].map(class_modes).values

# Lesion classes only: config.PADDY_GROUP_A
lesion_classes = set(config.PADDY_GROUP_A)
test_lesion = test_df[test_df['disease_class'].isin(lesion_classes)].copy().reset_index(drop=True)

# Subsets:
# 1. Full Lesion-Only (N=757)
# 2. Twin-Free <= 5 Lesion-Only (N=556)
# 3. Twin-Free <= 8 Lesion-Only (N=343)
# Note: In Part A, twin-free meant no twin in Train/Val!
# Test images that have a twin in Train/Val at <=5:
test_has_twin_le5 = set(p[0] for p in pairs_tv_test)
pairs_tv_test_le8 = count_cross_pairs(test_idx, train_val_idx, max_hamming=8)
test_has_twin_le8 = set(p[0] for p in pairs_tv_test_le8)

test_lesion['has_twin_le5'] = [test_idx[i] in test_has_twin_le5 for i in test_lesion.index] # wait! index mapping
# Let's map filename to has_twin:
test_idx_to_fn = {test_idx[i]: test_df.iloc[i]['filename'] for i in range(len(test_idx))}
twin_fns_le5 = set(test_idx_to_fn[t_i] for t_i in test_has_twin_le5)
twin_fns_le8 = set(test_idx_to_fn[t_i] for t_i in test_has_twin_le8)

test_lesion['is_twin_free_le5'] = ~test_lesion['filename'].isin(twin_fns_le5)
test_lesion['is_twin_free_le8'] = ~test_lesion['filename'].isin(twin_fns_le8)

print(f"\nLesion-Only Test Sets:")
print(f"  • Full Lesion:          N = {len(test_lesion)}")
print(f"  • Twin-Free <=5 Lesion: N = {test_lesion['is_twin_free_le5'].sum()}")
print(f"  • Twin-Free <=8 Lesion: N = {test_lesion['is_twin_free_le8'].sum()}")

# Component-Grouped Bootstrap function
def compute_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = float(np.mean(np.abs(y_true - y_pred)))
    acc = float(np.mean(y_true == y_pred) * 100.0)
    w1 = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100.0)
    return qwk, mae, acc, w1

def grouped_bootstrap_eval(sub_df, cluster_col, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    unique_clusters = sub_df[cluster_col].unique()
    k_clusters = len(unique_clusters)
    
    # Pre-group indices by cluster
    cluster_indices = sub_df.groupby(cluster_col).indices
    
    models = ['pred_mode', 'pred_corn', 'pred_soft']
    boot_stats = {m: {'qwk': [], 'mae': [], 'acc': [], 'w1': []} for m in models}
    diff_corn_soft = []
    
    for _ in range(n_boot):
        # Sample clusters with replacement
        sampled_c = rng.choice(unique_clusters, size=k_clusters, replace=True)
        boot_idx = []
        for c in sampled_c:
            boot_idx.extend(cluster_indices[c])
            
        b_df = sub_df.iloc[boot_idx]
        y_t = b_df['severity_grade'].values
        
        # Check variance in true labels
        if len(np.unique(y_t)) < 2:
            continue
            
        for m in models:
            y_p = b_df[m].values
            qwk, mae, acc, w1 = compute_metrics(y_t, y_p)
            boot_stats[m]['qwk'].append(qwk)
            boot_stats[m]['mae'].append(mae)
            boot_stats[m]['acc'].append(acc)
            boot_stats[m]['w1'].append(w1)
            
        # Paired difference
        q_c = boot_stats['pred_corn']['qwk'][-1]
        q_s = boot_stats['pred_soft']['qwk'][-1]
        diff_corn_soft.append(q_c - q_s)
        
    results = {}
    for m in models:
        results[m] = {}
        for metric in ['qwk', 'mae', 'acc', 'w1']:
            arr = np.array(boot_stats[m][metric])
            results[m][metric] = {
                'mean': float(np.mean(arr)),
                'ci_lower': float(np.percentile(arr, 2.5)),
                'ci_upper': float(np.percentile(arr, 97.5))
            }
            
    diff_arr = np.array(diff_corn_soft)
    p_corn_gt_soft = float(np.mean(diff_arr > 0))
    p_soft_gt_corn = float(np.mean(diff_arr < 0))
    diff_results = {
        'mean_diff': float(np.mean(diff_arr)),
        'ci_lower': float(np.percentile(diff_arr, 2.5)),
        'ci_upper': float(np.percentile(diff_arr, 97.5)),
        'p_corn_gt_soft': p_corn_gt_soft,
        'p_soft_gt_corn': p_soft_gt_corn
    }
    
    return results, diff_results

# Run grouped bootstrap on all 3 lesion subsets using component_le8 as grouping cluster
subsets_data = {
    'Full Lesion-Only': test_lesion,
    'Twin-Free <=5 Lesion-Only': test_lesion[test_lesion['is_twin_free_le5']].copy().reset_index(drop=True),
    'Twin-Free <=8 Lesion-Only': test_lesion[test_lesion['is_twin_free_le8']].copy().reset_index(drop=True)
}

grouped_table_results = {}
paired_diff_results = {}

print("\n--- COMPONENT-GROUPED BOOTSTRAP RESULTS (2,000 RESAMPLES) ---")
for s_name, s_df in subsets_data.items():
    # Number of clusters
    n_clus = s_df['comp_le8'].nunique()
    print(f"\nSubset: {s_name} (N = {len(s_df)}, Unique Clusters = {n_clus})")
    
    res, diff_res = grouped_bootstrap_eval(s_df, 'comp_le8', n_boot=2000, seed=42)
    grouped_table_results[s_name] = res
    paired_diff_results[s_name] = diff_res
    
    for m_label, m_key in [('Mode Baseline', 'pred_mode'), ('CORN (Production)', 'pred_corn'), ('Softmax (Production)', 'pred_soft')]:
        q = res[m_key]['qwk']
        m = res[m_key]['mae']
        a = res[m_key]['acc']
        w = res[m_key]['w1']
        print(f"  {m_label:22} | QWK={q['mean']:.4f} [{q['ci_lower']:.4f}, {q['ci_upper']:.4f}] | MAE={m['mean']:.4f} [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}] | Acc={a['mean']:.2f}% [{a['ci_lower']:.2f}%, {a['ci_upper']:.2f}%] | W1={w['mean']:.2f}% [{w['ci_lower']:.2f}%, {w['ci_upper']:.2f}%]")
        
    print(f"  --> Paired Diff QWK(CORN) - QWK(Softmax): {diff_res['mean_diff']:+.4f} [{diff_res['ci_lower']:+.4f}, {diff_res['ci_upper']:+.4f}] (P(CORN > Softmax) = {diff_res['p_corn_gt_soft']:.3f}, P(Softmax > CORN) = {diff_res['p_soft_gt_corn']:.3f})")

stage2d_c_output = {
    'c1_reconciliation': reconciliation_summary,
    'c2_component_counts': {
        'hamming_le5': {
            'total_components': int(n_comp_5),
            'total_edges': len(edges_5),
            'test_images_in_multi_image_components': int(test_in_multi_5)
        },
        'hamming_le8': {
            'total_components': int(n_comp_8),
            'total_edges': len(edges_8),
            'test_images_in_multi_image_components': int(test_in_multi_8)
        }
    },
    'c3_earlier_table_bootstrap_type': "The earlier Stage 2c Part A table used an UNGROUPED IID IMAGE-LEVEL bootstrap (2,000 resamples of individual test images with replacement). The recomputed table here uses a COMPONENT-GROUPED bootstrap (resampling entire connected components with replacement).",
    'c3_grouped_bootstrap_table': grouped_table_results,
    'c3_paired_qwk_differences': paired_diff_results
}

with open('outputs/final_v2/stage2d_section_c.json', 'w') as f:
    json.dump(stage2d_c_output, f, indent=2)

print("\nSaved Stage 2d Section C results to outputs/final_v2/stage2d_section_c.json")
