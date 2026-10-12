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
print("STAGE 2f - SECTION A: CORN DECODE AUDIT & SELECTION")
print("=" * 80)

# Load Paddy dataset & hashes
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_paddy = len(df_paddy)
splits = df_paddy['split'].values

val_idx = np.where(splits == 'val')[0]
test_idx = np.where(splits == 'test')[0]
train_idx = np.where(splits == 'train')[0]
train_val_idx = np.where(splits != 'test')[0]

df_val = df_paddy.iloc[val_idx].copy().reset_index(drop=True)
df_test = df_paddy.iloc[test_idx].copy().reset_index(drop=True)

# Load cached pHashes for twin identification & graph clustering
cache_phash_path = 'outputs/final_v2/paddy_phashes_cache.npz'
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

pairs_tv_test_le5 = count_cross_pairs(test_idx, train_val_idx, max_hamming=5)
pairs_tv_test_le8 = count_cross_pairs(test_idx, train_val_idx, max_hamming=8)

test_has_twin_le5 = set(p[0] for p in pairs_tv_test_le5)
test_has_twin_le8 = set(p[0] for p in pairs_tv_test_le8)

df_test['has_twin_le5'] = [test_idx[i] in test_has_twin_le5 for i in range(len(test_idx))]
df_test['has_twin_le8'] = [test_idx[i] in test_has_twin_le8 for i in range(len(test_idx))]

# Build components for cluster-grouped bootstrap at Hamming <= 8
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

# Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

model_corn = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
ckpt_c = torch.load('outputs/models/paddy_ordinal.pth', map_location='cpu')
model_corn.load_state_dict(ckpt_c if 'model_state_dict' not in ckpt_c else ckpt_c['model_state_dict'])
model_corn.to(device).eval()

model_soft = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=False)
ckpt_s = torch.load('outputs/models/paddy_baseline.pth', map_location='cpu')
model_soft.load_state_dict(ckpt_s if 'model_state_dict' not in ckpt_s else ckpt_s['model_state_dict'])
model_soft.to(device).eval()

class FastDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))

val_loader = torch.utils.data.DataLoader(FastDataset(df_val['image_path'].tolist(), transform), batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(FastDataset(df_test['image_path'].tolist(), transform), batch_size=64, shuffle=False)

def decode_rule_i_sum(sig_probs):
    # Rule (i): sum(sigmoid(logits) > 0.5)
    return (sig_probs > 0.5).sum(dim=1)

def decode_rule_ii_chained(sig_probs):
    # Rule (ii): corn_probs_to_grade (chained argmax)
    pred_g, _ = corn_probs_to_grade(sig_probs)
    return pred_g

def decode_rule_iii_standard(sig_probs):
    # Rule (iii): Standard CORN decode: p = cumprod(sigmoid(logits)); grade = sum(p > 0.5)
    p_cum = torch.cumprod(sig_probs, dim=1)
    return (p_cum > 0.5).sum(dim=1)

# A.2 VALIDATION SET SELECTION
print("\n--- A.2 MODEL SELECTION ON PADDY VALIDATION DATA ONLY (N=1,510) ---")
val_sigmoids = []
val_y_true = df_val['severity_grade'].values

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out = model_corn(batch)
        val_sigmoids.append(torch.sigmoid(out))
val_sigmoids = torch.cat(val_sigmoids, dim=0)

val_pred_rule_i = decode_rule_i_sum(val_sigmoids).cpu().numpy()
val_pred_rule_ii = decode_rule_ii_chained(val_sigmoids).cpu().numpy()
val_pred_rule_iii = decode_rule_iii_standard(val_sigmoids).cpu().numpy()

val_qwk_i = cohen_kappa_score(val_y_true, val_pred_rule_i, weights='quadratic')
val_qwk_ii = cohen_kappa_score(val_y_true, val_pred_rule_ii, weights='quadratic')
val_qwk_iii = cohen_kappa_score(val_y_true, val_pred_rule_iii, weights='quadratic')

print(f"Validation QWK - Rule (i)   [Sum Sigmoid > 0.5]:     {val_qwk_i:.6f}")
print(f"Validation QWK - Rule (ii)  [Chained Argmax]:        {val_qwk_ii:.6f}")
print(f"Validation QWK - Rule (iii) [Standard cumprod > 0.5]: {val_qwk_iii:.6f}")

best_rule_name = ""
if val_qwk_iii >= max(val_qwk_i, val_qwk_ii):
    best_rule_name = "Rule (iii) [Standard cumprod > 0.5]"
elif val_qwk_i >= val_qwk_ii:
    best_rule_name = "Rule (i) [Sum Sigmoid > 0.5]"
else:
    best_rule_name = "Rule (ii) [Chained Argmax]"

print(f"\nOFFICIAL VALIDATION CHOICE: {best_rule_name} (Selected strictly by highest Validation QWK)")

# A.1 TEST SET EVALUATION
print("\n--- A.1 TEST SET EVALUATION ACROSS ALL DECODE RULES ---")
test_sigmoids = []
test_softmax_preds = []
test_y_true = df_test['severity_grade'].values

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out_c = model_corn(batch)
        test_sigmoids.append(torch.sigmoid(out_c))
        out_s = model_soft(batch)
        test_softmax_preds.extend(torch.argmax(out_s, dim=1).cpu().numpy())

test_sigmoids = torch.cat(test_sigmoids, dim=0)
test_softmax_preds = np.array(test_softmax_preds)

test_pred_i = decode_rule_i_sum(test_sigmoids).cpu().numpy()
test_pred_ii = decode_rule_ii_chained(test_sigmoids).cpu().numpy()
test_pred_iii = decode_rule_iii_standard(test_sigmoids).cpu().numpy()

# Save predictions CSV
pred_df = pd.DataFrame({
    'filename': df_test['filename'],
    'disease_class': df_test['disease_class'],
    'severity_grade': test_y_true,
    'has_twin_le5': df_test['has_twin_le5'],
    'has_twin_le8': df_test['has_twin_le8'],
    'comp_le8': df_test['comp_le8'],
    'pred_rule_i_sum': test_pred_i,
    'pred_rule_ii_chained': test_pred_ii,
    'pred_rule_iii_standard': test_pred_iii,
    'pred_softmax': test_softmax_preds
})
pred_df.to_csv('outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv', index=False)

def compute_metrics(y_t, y_p):
    qwk = cohen_kappa_score(y_t, y_p, weights='quadratic')
    mae = mean_absolute_error(y_t, y_p)
    acc = accuracy_score(y_t, y_p) * 100.0
    w1 = np.mean(np.abs(y_t - y_p) <= 1) * 100.0
    return {'qwk': float(qwk), 'mae': float(mae), 'accuracy': float(acc), 'within_1': float(w1)}

lesion_classes = set(config.PADDY_GROUP_A)
is_lesion = df_test['disease_class'].isin(lesion_classes).values

subsets = {
    'Full Test (All 1,507)': np.ones(len(df_test), dtype=bool),
    'Full Lesion-Only (N=757)': is_lesion,
    'Twin-Free <=5 Lesion (N=556)': is_lesion & (~df_test['has_twin_le5'].values),
    'Twin-Free <=8 Lesion (N=343)': is_lesion & (~df_test['has_twin_le8'].values)
}

point_results = {}
for s_name, s_mask in subsets.items():
    yt = test_y_true[s_mask]
    point_results[s_name] = {
        'N': int(s_mask.sum()),
        'rule_i_sum': compute_metrics(yt, test_pred_i[s_mask]),
        'rule_ii_chained': compute_metrics(yt, test_pred_ii[s_mask]),
        'rule_iii_standard': compute_metrics(yt, test_pred_iii[s_mask]),
        'softmax': compute_metrics(yt, test_softmax_preds[s_mask])
    }

# Paired Grouped Bootstrap (2,000 resamples) for Rule (iii) vs Softmax
print("\nRunning Paired Grouped Bootstrap (2,000 resamples) for Rule (iii) vs Softmax...")
np.random.seed(42)
B = 2000
bootstrap_results = {}

for s_name, s_mask in subsets.items():
    yt = test_y_true[s_mask]
    yp_iii = test_pred_iii[s_mask]
    yp_s = test_softmax_preds[s_mask]
    clus = df_test.loc[s_mask, 'comp_le8'].values
    
    unique_clus = np.unique(clus)
    n_clus = len(unique_clus)
    clus_map = {c: np.where(clus == c)[0] for c in unique_clus}
    
    b_qwk_iii, b_mae_iii, b_acc_iii, b_w1_iii = [], [], [], []
    b_qwk_s, b_mae_s, b_acc_s, b_w1_s = [], [], [], []
    diffs = []
    
    for _ in range(B):
        sample_c = np.random.choice(unique_clus, size=n_clus, replace=True)
        idx_sample = np.concatenate([clus_map[c] for c in sample_c])
        yb = yt[idx_sample]
        yiii_b = yp_iii[idx_sample]
        ys_b = yp_s[idx_sample]
        
        if len(np.unique(yb)) < 2:
            continue
            
        q_iii = cohen_kappa_score(yb, yiii_b, weights='quadratic')
        q_s = cohen_kappa_score(yb, ys_b, weights='quadratic')
        
        b_qwk_iii.append(q_iii)
        b_mae_iii.append(mean_absolute_error(yb, yiii_b))
        b_acc_iii.append(accuracy_score(yb, yiii_b) * 100.0)
        b_w1_iii.append(np.mean(np.abs(yb - yiii_b) <= 1) * 100.0)
        
        b_qwk_s.append(q_s)
        b_mae_s.append(mean_absolute_error(yb, ys_b))
        b_acc_s.append(accuracy_score(yb, ys_b) * 100.0)
        b_w1_s.append(np.mean(np.abs(yb - ys_b) <= 1) * 100.0)
        
        diffs.append(q_iii - q_s)
        
    diffs = np.array(diffs)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    
    bootstrap_results[s_name] = {
        'rule_iii_standard': {
            'qwk_ci': ci(b_qwk_iii),
            'mae_ci': ci(b_mae_iii),
            'acc_ci': ci(b_acc_iii),
            'w1_ci': ci(b_w1_iii)
        },
        'softmax': {
            'qwk_ci': ci(b_qwk_s),
            'mae_ci': ci(b_mae_s),
            'acc_ci': ci(b_acc_s),
            'w1_ci': ci(b_w1_s)
        },
        'diff_rule_iii_vs_softmax': {
            'mean_diff': float(np.mean(diffs)),
            'ci': ci(diffs),
            'p_corn_gt_softmax': float(np.mean(diffs > 0))
        }
    }

# A.3 HISTORICAL CORN DECODE AUDIT
historical_audit = {
    'paper_table_1_paddy': 'Rule (iii) / Rule (i) [sum(sigmoid>0.5)] - reported 0.9390 QWK',
    'paper_table_1_arecanut': 'Rule (ii) [corn_probs_to_grade in src/corn_loss.py] - reported 0.8261 QWK',
    'arecanut_multiseed': 'Rule (ii) [corn_probs_to_grade in src/corn_loss.py]',
    '502_test_evaluation': 'Rule (ii) [corn_probs_to_grade in src/corn_loss.py]',
    'clean_146_test_evaluation': 'Rule (ii) [corn_probs_to_grade in src/corn_loss.py]',
    'stage2c_part_a_paddy': 'Rule (ii) [corn_probs_to_grade in src/corn_loss.py]',
    'stage2d_section_c_paddy': 'Rule (i) [sum(sigmoid>0.5) in scripts/stage2d_section_c.py]'
}

stage2f_a_output = {
    'val_set_size': len(df_val),
    'val_qwk_rule_i_sum': float(val_qwk_i),
    'val_qwk_rule_ii_chained': float(val_qwk_ii),
    'val_qwk_rule_iii_standard': float(val_qwk_iii),
    'official_validation_chosen_decode': best_rule_name,
    'point_estimates': point_results,
    'bootstrap_grouped_results': bootstrap_results,
    'historical_decode_audit': historical_audit
}

with open('outputs/final_v2/stage2f_section_a.json', 'w') as f:
    json.dump(stage2f_a_output, f, indent=2)

print("\nSaved Stage 2f Section A results to outputs/final_v2/stage2f_section_a.json")
