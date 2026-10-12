import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2e - PART A: CORN DISCREPANCY & RE-EVALUATION")
print("=" * 80)

# 1. Load Paddy splits & hash cache
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_paddy = len(df_paddy)
splits = df_paddy['split'].values
test_idx = np.where(splits == 'test')[0]
train_idx = np.where(splits == 'train')[0]
val_idx = np.where(splits == 'val')[0]
train_val_idx = np.where(splits != 'test')[0]

df_test = df_paddy.iloc[test_idx].copy().reset_index(drop=True)
y_true = df_test['severity_grade'].values

# Load hashes
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

print("Finding test images with twins in train/val at Hamming <= 5 and <= 8...")
pairs_tv_test_le5 = count_cross_pairs(test_idx, train_val_idx, max_hamming=5)
pairs_tv_test_le8 = count_cross_pairs(test_idx, train_val_idx, max_hamming=8)

test_has_twin_le5 = set(p[0] for p in pairs_tv_test_le5)
test_has_twin_le8 = set(p[0] for p in pairs_tv_test_le8)

# Map back to test df rows
df_test['has_twin_le5'] = [test_idx[i] in test_has_twin_le5 for i in range(len(test_idx))]
df_test['has_twin_le8'] = [test_idx[i] in test_has_twin_le8 for i in range(len(test_idx))]

# Build components for cluster-grouped bootstrap (at Hamming <= 8)
print("Building connected components graph for all 10,053 Paddy images at Hamming <= 8...")
edges_8 = set()
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
        
        hits = np.where(min_d <= 8)
        for r_i, c_j in zip(hits[0], hits[1]):
            u = i_start + r_i
            v = j_start + c_j
            if u < v:
                edges_8.add((u, v))

rows = [e[0] for e in edges_8] + [e[1] for e in edges_8]
cols = [e[1] for e in edges_8] + [e[0] for e in edges_8]
data = np.ones(len(rows), dtype=int)
adj = sp.csr_matrix((data, (rows, cols)), shape=(n_paddy, n_paddy))
n_comp_8, labels_8 = connected_components(adj, directed=False)
df_paddy['comp_le8'] = labels_8
df_test['comp_le8'] = df_paddy.iloc[test_idx]['comp_le8'].values

# 2. Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

# Paddy CORN Model
ckpt_c_path = 'outputs/models/paddy_ordinal.pth'
model_corn = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
ckpt_c = torch.load(ckpt_c_path, map_location='cpu')
model_corn.load_state_dict(ckpt_c if 'model_state_dict' not in ckpt_c else ckpt_c['model_state_dict'])
model_corn.to(device).eval()

# Paddy Softmax Model
ckpt_s_path = 'outputs/models/paddy_baseline.pth'
model_soft = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=False)
ckpt_s = torch.load(ckpt_s_path, map_location='cpu')
model_soft.load_state_dict(ckpt_s if 'model_state_dict' not in ckpt_s else ckpt_s['model_state_dict'])
model_soft.to(device).eval()

# Dataset and DataLoader
class FastDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(im)

test_ds = FastDataset(df_test['image_path'].tolist(), transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

all_logits_c = []
all_probs_c = []
pred_corn_sum = []
pred_corn_chained = []
pred_soft = []

print("Running inference on 1,507 Paddy test set images...")
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        
        # CORN
        out_c = model_corn(batch)
        sig_c = torch.sigmoid(out_c)
        all_logits_c.append(out_c.cpu().numpy())
        all_probs_c.append(sig_c.cpu().numpy())
        
        # Rule (i): sum of (sigmoid > 0.5) -- matches paper Eq. 9 / Eq. 150
        p_sum = (sig_c > 0.5).sum(dim=1)
        pred_corn_sum.extend(p_sum.cpu().numpy().tolist())
        
        # Rule (ii): chained-product decode (corn_probs_to_grade)
        p_chained, _ = corn_probs_to_grade(sig_c)
        pred_corn_chained.extend(p_chained.cpu().numpy().tolist())
        
        # Softmax
        out_s = model_soft(batch)
        p_s = torch.argmax(out_s, dim=1)
        pred_soft.extend(p_s.cpu().numpy().tolist())

all_logits_c = np.vstack(all_logits_c)
all_probs_c = np.vstack(all_probs_c)
pred_corn_sum = np.array(pred_corn_sum)
pred_corn_chained = np.array(pred_corn_chained)
pred_soft = np.array(pred_soft)

# Save predictions to CSV
pred_df = pd.DataFrame({
    'image_path': df_test['image_path'],
    'filename': df_test['filename'],
    'disease_class': df_test['disease_class'],
    'severity_grade': df_test['severity_grade'],
    'has_twin_le5': df_test['has_twin_le5'],
    'has_twin_le8': df_test['has_twin_le8'],
    'component_le8': df_test['comp_le8'],
    'corn_logit_0': all_logits_c[:, 0],
    'corn_logit_1': all_logits_c[:, 1],
    'corn_logit_2': all_logits_c[:, 2],
    'corn_prob_0': all_probs_c[:, 0],
    'corn_prob_1': all_probs_c[:, 1],
    'corn_prob_2': all_probs_c[:, 2],
    'pred_corn_sum_rule_i': pred_corn_sum,
    'pred_corn_chained_rule_ii': pred_corn_chained,
    'pred_softmax': pred_soft
})

csv_out_path = 'outputs/final_v2/paddy_corn_predictions_both_decodes.csv'
pred_df.to_csv(csv_out_path, index=False)
print(f"Saved full predictions CSV to {csv_out_path}")

diff_count = (pred_corn_sum != pred_corn_chained).sum()
print(f"\nDiscrepancy between Rule (i) and Rule (ii) on Paddy test set: {diff_count} / {len(df_test)} ({diff_count/len(df_test)*100:.2f}%)")

# Define subsets
lesion_classes = set(config.PADDY_GROUP_A)
is_lesion = df_test['disease_class'].isin(lesion_classes).values
mask_full_lesion = is_lesion
mask_twinfree_le5_lesion = is_lesion & (~df_test['has_twin_le5'].values)
mask_twinfree_le8_lesion = is_lesion & (~df_test['has_twin_le8'].values)

# Mode baseline per class (from training set)
df_train = df_paddy.iloc[train_idx]
mode_per_class = df_train.groupby('disease_class')['severity_grade'].agg(lambda x: x.mode()[0]).to_dict()
y_pred_mode = df_test['disease_class'].map(mode_per_class).values

def compute_metrics(y_t, y_p):
    qwk = cohen_kappa_score(y_t, y_p, weights='quadratic')
    mae = mean_absolute_error(y_t, y_p)
    acc = accuracy_score(y_t, y_p) * 100.0
    w1 = np.mean(np.abs(y_t - y_p) <= 1) * 100.0
    return {'qwk': float(qwk), 'mae': float(mae), 'accuracy': float(acc), 'within_1': float(w1)}

subsets = {
    'Full Lesion-Only': mask_full_lesion,
    'Twin-Free <=5 Lesion-Only': mask_twinfree_le5_lesion,
    'Twin-Free <=8 Lesion-Only': mask_twinfree_le8_lesion
}

results_summary = {}

for s_name, s_mask in subsets.items():
    y_t_sub = y_true[s_mask]
    clus_sub = df_test.loc[s_mask, 'comp_le8'].values
    
    m_mode = compute_metrics(y_t_sub, y_pred_mode[s_mask])
    m_corn_sum = compute_metrics(y_t_sub, pred_corn_sum[s_mask])
    m_corn_chain = compute_metrics(y_t_sub, pred_corn_chained[s_mask])
    m_soft = compute_metrics(y_t_sub, pred_soft[s_mask])
    
    print(f"\n--- SUBSET: {s_name} (N={len(y_t_sub)}, Clusters={len(np.unique(clus_sub))}) ---")
    print(f"  Mode Baseline:        QWK={m_mode['qwk']:.4f} | MAE={m_mode['mae']:.4f} | Acc={m_mode['accuracy']:.2f}% | W1={m_mode['within_1']:.2f}%")
    print(f"  CORN Rule (i) [Sum]:  QWK={m_corn_sum['qwk']:.4f} | MAE={m_corn_sum['mae']:.4f} | Acc={m_corn_sum['accuracy']:.2f}% | W1={m_corn_sum['within_1']:.2f}%")
    print(f"  CORN Rule (ii) [Chn]: QWK={m_corn_chain['qwk']:.4f} | MAE={m_corn_chain['mae']:.4f} | Acc={m_corn_chain['accuracy']:.2f}% | W1={m_corn_chain['within_1']:.2f}%")
    print(f"  Softmax (prod):       QWK={m_soft['qwk']:.4f} | MAE={m_soft['mae']:.4f} | Acc={m_soft['accuracy']:.2f}% | W1={m_soft['within_1']:.2f}%")
    
    results_summary[s_name] = {
        'N': int(len(y_t_sub)),
        'N_clusters': int(len(np.unique(clus_sub))),
        'mode_baseline': m_mode,
        'corn_rule_i_sum': m_corn_sum,
        'corn_rule_ii_chained': m_corn_chain,
        'softmax': m_soft
    }

# 3. Paired Grouped Bootstrap (2,000 resamples) for BOTH decodes
print("\nRunning Paired Grouped Bootstrap (2,000 resamples) across all 3 lesion subsets...")
np.random.seed(42)
B = 2000

bootstrap_results = {}

for s_name, s_mask in subsets.items():
    y_t_sub = y_true[s_mask]
    y_c_sum_sub = pred_corn_sum[s_mask]
    y_c_chn_sub = pred_corn_chained[s_mask]
    y_s_sub = pred_soft[s_mask]
    clus_sub = df_test.loc[s_mask, 'comp_le8'].values
    
    unique_clus = np.unique(clus_sub)
    n_clus = len(unique_clus)
    
    clus_to_indices = {c: np.where(clus_sub == c)[0] for c in unique_clus}
    
    boot_qwk_c_sum = []
    boot_mae_c_sum = []
    boot_acc_c_sum = []
    boot_w1_c_sum = []
    
    boot_qwk_c_chn = []
    boot_mae_c_chn = []
    boot_acc_c_chn = []
    boot_w1_c_chn = []
    
    boot_qwk_s = []
    boot_mae_s = []
    boot_acc_s = []
    boot_w1_s = []
    
    diff_sum = []
    diff_chn = []
    
    for b in range(B):
        sample_clus = np.random.choice(unique_clus, size=n_clus, replace=True)
        idx_sample = np.concatenate([clus_to_indices[c] for c in sample_clus])
        
        yb = y_t_sub[idx_sample]
        yc_sum_b = y_c_sum_sub[idx_sample]
        yc_chn_b = y_c_chn_sub[idx_sample]
        ys_b = y_s_sub[idx_sample]
        
        if len(np.unique(yb)) < 2:
            continue
            
        # Rule (i)
        q_sum = cohen_kappa_score(yb, yc_sum_b, weights='quadratic')
        boot_qwk_c_sum.append(q_sum)
        boot_mae_c_sum.append(mean_absolute_error(yb, yc_sum_b))
        boot_acc_c_sum.append(accuracy_score(yb, yc_sum_b) * 100.0)
        boot_w1_c_sum.append(np.mean(np.abs(yb - yc_sum_b) <= 1) * 100.0)
        
        # Rule (ii)
        q_chn = cohen_kappa_score(yb, yc_chn_b, weights='quadratic')
        boot_qwk_c_chn.append(q_chn)
        boot_mae_c_chn.append(mean_absolute_error(yb, yc_chn_b))
        boot_acc_c_chn.append(accuracy_score(yb, yc_chn_b) * 100.0)
        boot_w1_c_chn.append(np.mean(np.abs(yb - yc_chn_b) <= 1) * 100.0)
        
        # Softmax
        qs = cohen_kappa_score(yb, ys_b, weights='quadratic')
        boot_qwk_s.append(qs)
        boot_mae_s.append(mean_absolute_error(yb, ys_b))
        boot_acc_s.append(accuracy_score(yb, ys_b) * 100.0)
        boot_w1_s.append(np.mean(np.abs(yb - ys_b) <= 1) * 100.0)
        
        diff_sum.append(q_sum - qs)
        diff_chn.append(q_chn - qs)
        
    diff_sum = np.array(diff_sum)
    diff_chn = np.array(diff_chn)
    
    ci = lambda arr: [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    
    bootstrap_results[s_name] = {
        'corn_rule_i_sum': {
            'qwk_ci': ci(boot_qwk_c_sum),
            'mae_ci': ci(boot_mae_c_sum),
            'acc_ci': ci(boot_acc_c_sum),
            'w1_ci': ci(boot_w1_c_sum),
            'diff_vs_softmax': {
                'mean_diff': float(np.mean(diff_sum)),
                'ci': ci(diff_sum),
                'p_corn_gt_softmax': float(np.mean(diff_sum > 0))
            }
        },
        'corn_rule_ii_chained': {
            'qwk_ci': ci(boot_qwk_c_chn),
            'mae_ci': ci(boot_mae_c_chn),
            'acc_ci': ci(boot_acc_c_chn),
            'w1_ci': ci(boot_w1_c_chn),
            'diff_vs_softmax': {
                'mean_diff': float(np.mean(diff_chn)),
                'ci': ci(diff_chn),
                'p_corn_gt_softmax': float(np.mean(diff_chn > 0))
            }
        },
        'softmax': {
            'qwk_ci': ci(boot_qwk_s),
            'mae_ci': ci(boot_mae_s),
            'acc_ci': ci(boot_acc_s),
            'w1_ci': ci(boot_w1_s)
        }
    }
    
    print(f"\nBootstrap Results for {s_name}:")
    print(f"  Rule (i) [Sum]:    QWK={results_summary[s_name]['corn_rule_i_sum']['qwk']:.4f} {ci(boot_qwk_c_sum)} | Diff vs Soft={np.mean(diff_sum):+.4f} {ci(diff_sum)} (P={np.mean(diff_sum > 0):.3f})")
    print(f"  Rule (ii) [Chain]: QWK={results_summary[s_name]['corn_rule_ii_chained']['qwk']:.4f} {ci(boot_qwk_c_chn)} | Diff vs Soft={np.mean(diff_chn):+.4f} {ci(diff_chn)} (P={np.mean(diff_chn > 0):.3f})")
    print(f"  Softmax:           QWK={results_summary[s_name]['softmax']['qwk']:.4f} {ci(boot_qwk_s)}")

output_json = {
    'checkpoint_used': ckpt_c_path,
    'softmax_checkpoint_used': ckpt_s_path,
    'code_path_rule_i': 'sum(sigmoid(logits) > 0.5) [matches paper Eq. 150 / Eq. 9]',
    'code_path_rule_ii': 'corn_probs_to_grade(sigmoid(logits)) in src/corn_loss.py [chained-product decode]',
    'paper_eq_reference': 'Equation 150 (label: eq:decode) in paper.tex',
    'predictions_csv': csv_out_path,
    'total_test_discrepancies_count': int(diff_count),
    'point_estimates': results_summary,
    'bootstrap_grouped_results': bootstrap_results
}

with open('outputs/final_v2/stage2e_part_a.json', 'w') as f:
    json.dump(output_json, f, indent=2)

print("\nSaved Stage 2e Part A results to outputs/final_v2/stage2e_part_a.json")
