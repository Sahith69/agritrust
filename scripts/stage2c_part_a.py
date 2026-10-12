import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2c - PART A: PADDY TWIN-FREE & HAMMING 6-8 ANALYSIS")
print("=" * 80)

# 1. Load Paddy splits
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_total = len(df_paddy)
print(f"Total Paddy images: {n_total}")

# 64-bit DCT pHash
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

print("Computing pHashes for all 10,053 Paddy images (direct, h-flip, v-flip)...")
t0 = time.time()
hashes_orig = np.zeros(n_total, dtype=np.uint64)
hashes_hflip = np.zeros(n_total, dtype=np.uint64)
hashes_vflip = np.zeros(n_total, dtype=np.uint64)

for i, p in enumerate(df_paddy['image_path']):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        continue
    hashes_orig[i] = compute_phash_uint64(im)
    hashes_hflip[i] = compute_phash_uint64(cv2.flip(im, 1))
    hashes_vflip[i] = compute_phash_uint64(cv2.flip(im, 0))

print(f"Computed pHashes in {time.time() - t0:.2f}s")

# Vectorized Hamming distance computation
splits = df_paddy['split'].values
test_indices = np.where(splits == 'test')[0]
train_indices = np.where(splits == 'train')[0]
val_indices = np.where(splits == 'val')[0]
train_val_indices = np.concatenate([train_indices, val_indices])

print(f"Test size: {len(test_indices)}, Train size: {len(train_indices)}, Val size: {len(val_indices)}")

def fast_popcount_matrix(m):
    m = m - ((m >> np.uint64(1)) & np.uint64(0x5555555555555555))
    m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
    m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return m.astype(np.int32)

print("Finding all exact train/val matches for test images up to Hamming 8...")
test_has_twin_le5 = set()
test_has_twin_6to8 = set()
test_has_twin_le8 = set()

# Pairs: (test_idx, train_val_idx, dist, same_class)
pairs_le5 = []
pairs_6to8 = []
train_test_pairs_6to8 = []

classes = df_paddy['disease_class'].values

H_test = hashes_orig[test_indices]
H_tv_orig = hashes_orig[train_val_indices]
H_tv_hflip = hashes_hflip[train_val_indices]
H_tv_vflip = hashes_vflip[train_val_indices]

# Chunked search
chunk_test = 200
chunk_tv = 2000

for b_start in range(0, len(test_indices), chunk_test):
    b_end = min(b_start + chunk_test, len(test_indices))
    t_sub = test_indices[b_start:b_end]
    h_sub = hashes_orig[t_sub]
    
    for tv_start in range(0, len(train_val_indices), chunk_tv):
        tv_end = min(tv_start + chunk_tv, len(train_val_indices))
        tv_sub_idx = train_val_indices[tv_start:tv_end]
        
        xor1 = h_sub[:, None] ^ H_tv_orig[tv_start:tv_end][None, :]
        xor2 = h_sub[:, None] ^ H_tv_hflip[tv_start:tv_end][None, :]
        xor3 = h_sub[:, None] ^ H_tv_vflip[tv_start:tv_end][None, :]
        
        d1 = fast_popcount_matrix(xor1)
        d2 = fast_popcount_matrix(xor2)
        d3 = fast_popcount_matrix(xor3)
        min_d = np.minimum(d1, np.minimum(d2, d3))
        
        # Matches <= 5
        hits_le5 = np.where(min_d <= 5)
        for r_i, c_j in zip(hits_le5[0], hits_le5[1]):
            t_idx = t_sub[r_i]
            tv_idx = tv_sub_idx[c_j]
            dist = int(min_d[r_i, c_j])
            same_c = (classes[t_idx] == classes[tv_idx])
            test_has_twin_le5.add(t_idx)
            test_has_twin_le8.add(t_idx)
            pairs_le5.append((t_idx, tv_idx, dist, same_c))
            
        # Matches 6 to 8
        hits_6to8 = np.where((min_d >= 6) & (min_d <= 8))
        for r_i, c_j in zip(hits_6to8[0], hits_6to8[1]):
            t_idx = t_sub[r_i]
            tv_idx = tv_sub_idx[c_j]
            dist = int(min_d[r_i, c_j])
            same_c = (classes[t_idx] == classes[tv_idx])
            test_has_twin_6to8.add(t_idx)
            test_has_twin_le8.add(t_idx)
            pairs_6to8.append((t_idx, tv_idx, dist, same_c))
            if splits[tv_idx] == 'train':
                train_test_pairs_6to8.append((t_idx, tv_idx, dist, same_c))

test_only_6to8 = test_has_twin_6to8 - test_has_twin_le5

print("\n--- PADDY LEAKAGE PAIR & ENTITY BREAKDOWN ---")
print(f"Total test images: {len(test_indices)}")
print(f"Test images with twin at Hamming <= 5: {len(test_has_twin_le5)} ({len(test_has_twin_le5)/len(test_indices)*100:.2f}%)")
print(f"Test images with twin at Hamming 6-8: {len(test_has_twin_6to8)} ({len(test_has_twin_6to8)/len(test_indices)*100:.2f}%)")
print(f"Test images with twin at Hamming 6-8 only (no twin <=5): {len(test_only_6to8)} ({len(test_only_6to8)/len(test_indices)*100:.2f}%)")
print(f"Total test images with twin at Hamming <= 8: {len(test_has_twin_le8)} ({len(test_has_twin_le8)/len(test_indices)*100:.2f}%)")

pairs_le5_same = sum(1 for p in pairs_le5 if p[3])
pairs_6to8_same = sum(1 for p in pairs_6to8 if p[3])

print(f"\nPairs with Hamming <= 5: Total={len(pairs_le5)}, Same Class={pairs_le5_same} ({pairs_le5_same/len(pairs_le5)*100:.2f}%), Different Class={len(pairs_le5)-pairs_le5_same}")
print(f"Pairs with Hamming 6-8: Total={len(pairs_6to8)}, Same Class={pairs_6to8_same} ({pairs_6to8_same/len(pairs_6to8)*100:.2f}%), Different Class={len(pairs_6to8)-pairs_6to8_same}")

# Save 24 random train-test pairs at Hamming 6-8 montage
np.random.seed(42)
sampled_24_pairs = []
if len(train_test_pairs_6to8) >= 24:
    s_indices = np.random.choice(len(train_test_pairs_6to8), size=24, replace=False)
    sampled_24_pairs = [train_test_pairs_6to8[idx] for idx in s_indices]
else:
    sampled_24_pairs = train_test_pairs_6to8

print(f"\nSaving montage of 24 train-test pairs at Hamming 6-8...")
fig = plt.figure(figsize=(16, 24))
gs = GridSpec(6, 4, figure=fig, hspace=0.38, wspace=0.25)

for p_i, (t_idx, tr_idx, d, same_c) in enumerate(sampled_24_pairs):
    row_i = p_i // 4
    col_i = p_i % 4
    
    im_t = Image.open(df_paddy.iloc[t_idx]['image_path']).convert('RGB').resize((180, 180))
    im_tr = Image.open(df_paddy.iloc[tr_idx]['image_path']).convert('RGB').resize((180, 180))
    
    combined = Image.new('RGB', (370, 180))
    combined.paste(im_t, (0, 0))
    combined.paste(im_tr, (190, 0))
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined)
    c_t = df_paddy.iloc[t_idx]['disease_class']
    c_tr = df_paddy.iloc[tr_idx]['disease_class']
    g_t = df_paddy.iloc[t_idx]['severity_grade']
    g_tr = df_paddy.iloc[tr_idx]['severity_grade']
    match_str = "MATCH" if same_c else "MISMATCH"
    cap = f"Pair #{p_i+1} | Dist={d} | {match_str}\nTest: {c_t} G{g_t} ({df_paddy.iloc[t_idx]['filename']})\nTrain: {c_tr} G{g_tr} ({df_paddy.iloc[tr_idx]['filename']})"
    ax.set_title(cap, fontsize=7.5, pad=3)
    ax.axis('off')

fig.suptitle("Paddy Train-Test Near-Duplicate Pairs at Hamming Distance 6–8 bits\n[Left: Test Image | Right: Train Image]", fontsize=14, fontweight='bold', y=0.995)
montage_6to8_path = 'outputs/final_v2/paddy_train_test_hamming6_8_montage.png'
plt.savefig(montage_6to8_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {montage_6to8_path}")

# Model evaluation across all subsets
print("\nLoading models and running inference on test set...")
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

df_test = df_paddy[df_paddy['split'] == 'test'].copy().reset_index()
y_true = df_test['severity_grade'].values
test_orig_indices = df_test['index'].values

# Fast batch inference
class FastPaddyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(im)

test_dataset = FastPaddyDataset(df_test['image_path'].tolist(), transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

y_pred_corn = []
y_pred_soft = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out_c = model_corn(batch)
        p_c = torch.sigmoid(out_c)
        pred_c, _ = corn_probs_to_grade(p_c)
        y_pred_corn.extend(pred_c.cpu().numpy().tolist())
        
        out_s = model_soft(batch)
        pred_s = torch.argmax(out_s, dim=1)
        y_pred_soft.extend(pred_s.cpu().numpy().tolist())

y_pred_corn = np.array(y_pred_corn)
y_pred_soft = np.array(y_pred_soft)

# Masks
is_twin_le5 = np.array([idx in test_has_twin_le5 for idx in test_orig_indices])
is_twin_le8 = np.array([idx in test_has_twin_le8 for idx in test_orig_indices])
is_twin_free_le5 = ~is_twin_le5
is_twin_free_le8 = ~is_twin_le8

lesion_classes = set(config.PADDY_GROUP_A)
is_lesion = df_test['disease_class'].isin(lesion_classes).values

is_lesion_twin_free_le5 = is_twin_free_le5 & is_lesion
is_lesion_twin_free_le8 = is_twin_free_le8 & is_lesion

subsets = {
    'Full Test': np.ones(len(df_test), dtype=bool),
    'Twin-Free <=5': is_twin_free_le5,
    'Twin-Free <=8': is_twin_free_le8,
    'Full Lesion-Only': is_lesion,
    'Twin-Free <=5 Lesion-Only': is_lesion_twin_free_le5,
    'Twin-Free <=8 Lesion-Only': is_lesion_twin_free_le8,
}

# Training modes per class
train_df = df_paddy[df_paddy['split'] == 'train']
train_modes = train_df.groupby('disease_class')['severity_grade'].agg(lambda x: x.mode()[0]).to_dict()
base_preds = np.array([train_modes[c] for c in df_test['disease_class']])

def compute_metrics_boot(yt, yp, n_boot=2000, seed=42):
    n = len(yt)
    qwk = cohen_kappa_score(yt, yp, weights='quadratic', labels=[0,1,2,3])
    mae = mean_absolute_error(yt, yp)
    acc = accuracy_score(yt, yp)
    w1 = np.mean(np.abs(yt - yp) <= 1)
    
    np.random.seed(seed)
    boot_qwk, boot_mae, boot_acc, boot_w1 = [], [], [], []
    for _ in range(n_boot):
        b_idx = np.random.choice(n, size=n, replace=True)
        yt_b, yp_b = yt[b_idx], yp[b_idx]
        boot_qwk.append(cohen_kappa_score(yt_b, yp_b, weights='quadratic', labels=[0,1,2,3]))
        boot_mae.append(mean_absolute_error(yt_b, yp_b))
        boot_acc.append(accuracy_score(yt_b, yp_b))
        boot_w1.append(np.mean(np.abs(yt_b - yp_b) <= 1))
        
    return {
        'n': int(n),
        'qwk': float(qwk),
        'qwk_ci': [float(np.percentile(boot_qwk, 2.5)), float(np.percentile(boot_qwk, 97.5))],
        'mae': float(mae),
        'mae_ci': [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
        'accuracy': float(acc),
        'accuracy_ci': [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
        'within_1': float(w1),
        'within_1_ci': [float(np.percentile(boot_w1, 2.5)), float(np.percentile(boot_w1, 97.5))]
    }

results_table = []
full_results_dict = {}

print("\n" + "=" * 120)
print("SECTION A: PLAIN TEXT RESULTS TABLE (ONE LINE PER ROW)")
print("=" * 120)

for s_name, s_mask in subsets.items():
    yt_sub = y_true[s_mask]
    n_sub = len(yt_sub)
    
    models = {
        'Mode Baseline': base_preds[s_mask],
        'CORN (Production)': y_pred_corn[s_mask],
        'Softmax (Production)': y_pred_soft[s_mask]
    }
    
    full_results_dict[s_name] = {}
    for m_name, yp_sub in models.items():
        res = compute_metrics_boot(yt_sub, yp_sub)
        full_results_dict[s_name][m_name] = res
        
        line_str = (
            f"{s_name:25} | {m_name:20} | N={res['n']:4d} | "
            f"QWK={res['qwk']:.4f} [{res['qwk_ci'][0]:.4f}, {res['qwk_ci'][1]:.4f}] | "
            f"MAE={res['mae']:.4f} [{res['mae_ci'][0]:.4f}, {res['mae_ci'][1]:.4f}] | "
            f"Acc={res['accuracy']*100:.2f}% [{res['accuracy_ci'][0]*100:.2f}%, {res['accuracy_ci'][1]*100:.2f}%] | "
            f"W1={res['within_1']*100:.2f}% [{res['within_1_ci'][0]*100:.2f}%, {res['within_1_ci'][1]*100:.2f}%]"
        )
        results_table.append(line_str)
        print(line_str)

# Within-class QWK per lesion class on <=8 twin-free lesion subset
sub_mask_le8_lesion = is_lesion_twin_free_le8
df_sub_le8 = df_test[sub_mask_le8_lesion].copy()
yt_le8_lesion = y_true[sub_mask_le8_lesion]
corn_le8_lesion = y_pred_corn[sub_mask_le8_lesion]
soft_le8_lesion = y_pred_soft[sub_mask_le8_lesion]
base_le8_lesion = base_preds[sub_mask_le8_lesion]

print("\n" + "=" * 120)
print("WITHIN-CLASS QWK PER LESION CLASS ON <=8 TWIN-FREE LESION SUBSET (N=" + str(len(df_sub_le8)) + ")")
print("=" * 120)

within_class_results = {}
for cls in sorted(lesion_classes):
    cls_mask = (df_sub_le8['disease_class'] == cls).values
    yt_cls = yt_le8_lesion[cls_mask]
    n_cls = len(yt_cls)
    if n_cls == 0:
        continue
    
    unique_g = [int(x) for x in sorted(list(set(yt_cls)))]
    base_qwk = cohen_kappa_score(yt_cls, base_le8_lesion[cls_mask], weights='quadratic', labels=[0,1,2,3])
    corn_qwk = cohen_kappa_score(yt_cls, corn_le8_lesion[cls_mask], weights='quadratic', labels=[0,1,2,3])
    soft_qwk = cohen_kappa_score(yt_cls, soft_le8_lesion[cls_mask], weights='quadratic', labels=[0,1,2,3])
    
    within_class_results[cls] = {
        'N': int(n_cls),
        'grades_present': unique_g,
        'baseline_qwk': float(base_qwk),
        'corn_qwk': float(corn_qwk),
        'softmax_qwk': float(soft_qwk)
    }
    
    print(f"Class: {cls:25} | N={n_cls:3d} | Grades={str(unique_g):12} | Baseline QWK={base_qwk:.4f} | CORN QWK={corn_qwk:.4f} | Softmax QWK={soft_qwk:.4f}")

part_a_summary = {
    'leakage_stats': {
        'total_test_images': len(test_indices),
        'twin_le5_count': len(test_has_twin_le5),
        'twin_le5_pct': float(len(test_has_twin_le5)/len(test_indices)*100),
        'twin_6to8_count': len(test_has_twin_6to8),
        'twin_6to8_pct': float(len(test_has_twin_6to8)/len(test_indices)*100),
        'twin_only_6to8_count': len(test_only_6to8),
        'twin_only_6to8_pct': float(len(test_only_6to8)/len(test_indices)*100),
        'twin_le8_count': len(test_has_twin_le8),
        'twin_le8_pct': float(len(test_has_twin_le8)/len(test_indices)*100),
        'pairs_le5': {
            'total': len(pairs_le5),
            'same_class': pairs_le5_same,
            'same_class_pct': float(pairs_le5_same/len(pairs_le5)*100) if len(pairs_le5)>0 else 0,
            'diff_class': len(pairs_le5) - pairs_le5_same
        },
        'pairs_6to8': {
            'total': len(pairs_6to8),
            'same_class': pairs_6to8_same,
            'same_class_pct': float(pairs_6to8_same/len(pairs_6to8)*100) if len(pairs_6to8)>0 else 0,
            'diff_class': len(pairs_6to8) - pairs_6to8_same
        }
    },
    'montage_6to8_path': montage_6to8_path,
    'results_table_lines': results_table,
    'full_results_dict': full_results_dict,
    'within_class_results_le8_twin_free': within_class_results
}

with open('outputs/final_v2/stage2c_part_a.json', 'w') as f:
    json.dump(part_a_summary, f, indent=2)

print("\nSaved Stage 2c Part A results to outputs/final_v2/stage2c_part_a.json")
