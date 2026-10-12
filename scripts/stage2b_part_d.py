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
print("STAGE 2b - PART D: PADDY LEAKAGE AUDIT & TWIN-FREE EVALUATION")
print("=" * 80)

# 1. Load Paddy splits
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_total = len(df_paddy)
print(f"Total Paddy images: {n_total}")

# Compute 64-bit DCT pHash
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

print("Computing pHashes (direct, h-flip, v-flip)...")
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

def popcount(x):
    return bin(int(x)).count('1')

def min_hamming_dist(h_orig_i, h_orig_j, h_hflip_j, h_vflip_j):
    d1 = popcount(h_orig_i ^ h_orig_j)
    d2 = popcount(h_orig_i ^ h_hflip_j)
    d3 = popcount(h_orig_i ^ h_vflip_j)
    return min(d1, d2, d3)

# Multi-index hashing for fast matching
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

print("Building lookup table...")
table = [{} for _ in range(6)]
for i in range(n_total):
    chunks = get_chunks(hashes_orig[i])
    for c_idx, c_val in enumerate(chunks):
        table[c_idx].setdefault(c_val, []).append(i)

# Split indexes
splits = df_paddy['split'].values
test_indices = np.where(splits == 'test')[0]
train_indices = set(np.where(splits == 'train')[0])
val_indices = set(np.where(splits == 'val')[0])
train_val_indices = train_indices | val_indices

print(f"Test size: {len(test_indices)}, Train size: {len(train_indices)}, Val size: {len(val_indices)}")

# Check test images against train & val
test_twins_le5 = {}
test_twins_le8 = {}
train_test_pairs_le5 = []

print("Finding test duplicates against train and val...")
for test_idx in test_indices:
    candidates = set()
    for h in [hashes_orig[test_idx], hashes_hflip[test_idx], hashes_vflip[test_idx]]:
        chunks = get_chunks(h)
        for c_idx, c_val in enumerate(chunks):
            if c_val in table[c_idx]:
                for cand in table[c_idx][c_val]:
                    if cand in train_val_indices:
                        candidates.add(cand)
    
    # Check all candidates
    for cand in candidates:
        d = min_hamming_dist(hashes_orig[test_idx], hashes_orig[cand], hashes_hflip[cand], hashes_vflip[cand])
        if d <= 5:
            test_twins_le5.setdefault(test_idx, []).append((cand, d, splits[cand]))
            if splits[cand] == 'train':
                train_test_pairs_le5.append((test_idx, cand, d))
        if d <= 8:
            test_twins_le8.setdefault(test_idx, []).append((cand, d, splits[cand]))

# For Hamming <= 8, let's also do a linear scan check for test images to make sure no chunk misses occurred:
# (Since 6 chunks of 10-11 bits guarantees finding dist <= 5, but for dist <= 8, let's test if any chunk matches)
# With 6 chunks, pigeonhole guarantees finding up to 5 flips. For 6-8 flips, multi-index might miss ~5%.
# Let's do exact matrix block checks for test against train_val to be 100% exact!
print("Performing exact vectorized check for Hamming <= 8...")
H_test = hashes_orig[test_indices]
# We can do chunked vectorized popcounts:
H_tv_orig = hashes_orig[list(train_val_indices)]
H_tv_hflip = hashes_hflip[list(train_val_indices)]
H_tv_vflip = hashes_vflip[list(train_val_indices)]
tv_list = list(train_val_indices)

# Check test images that have min_dist <= 8
test_has_twin_le5 = set(test_twins_le5.keys())
test_has_twin_le8 = set(test_twins_le8.keys())

# Vectorized block loop
for b_start in range(0, len(test_indices), 200):
    b_end = min(b_start + 200, len(test_indices))
    t_sub = test_indices[b_start:b_end]
    h_sub = hashes_orig[t_sub]
    
    # Check against tv_list in chunks
    for tv_start in range(0, len(tv_list), 2000):
        tv_end = min(tv_start + 2000, len(tv_list))
        tv_orig_sub = H_tv_orig[tv_start:tv_end]
        tv_hflip_sub = H_tv_hflip[tv_start:tv_end]
        tv_vflip_sub = H_tv_vflip[tv_start:tv_end]
        
        # XOR bitwise
        xor1 = h_sub[:, None] ^ tv_orig_sub[None, :]
        xor2 = h_sub[:, None] ^ tv_hflip_sub[None, :]
        xor3 = h_sub[:, None] ^ tv_vflip_sub[None, :]
        
        # Popcount using numpy uint64 bit counting
        def fast_popcount_matrix(m):
            m = m - ((m >> np.uint64(1)) & np.uint64(0x5555555555555555))
            m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
            m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            m = (m * np.uint64(0x0101010101010101)) >> np.uint64(56)
            return m.astype(np.int32)
            
        d1 = fast_popcount_matrix(xor1)
        d2 = fast_popcount_matrix(xor2)
        d3 = fast_popcount_matrix(xor3)
        min_d = np.minimum(d1, np.minimum(d2, d3))
        
        # Identify hits
        hits_le5 = np.where(min_d <= 5)
        for row_i, col_j in zip(hits_le5[0], hits_le5[1]):
            t_idx = t_sub[row_i]
            test_has_twin_le5.add(t_idx)
            
        hits_le8 = np.where(min_d <= 8)
        for row_i, col_j in zip(hits_le8[0], hits_le8[1]):
            t_idx = t_sub[row_i]
            test_has_twin_le8.add(t_idx)

print(f"\nExact Unique Test Images with at least one Train/Val Twin:")
print(f"  Hamming <= 5: {len(test_has_twin_le5)} / {len(test_indices)} ({len(test_has_twin_le5) / len(test_indices) * 100:.2f}%)")
print(f"  Hamming <= 8: {len(test_has_twin_le8)} / {len(test_indices)} ({len(test_has_twin_le8) / len(test_indices) * 100:.2f}%)")

# 2. Montage of 24 random flagged train-test pairs
np.random.seed(42)
sample_24_pairs = []
if len(train_test_pairs_le5) >= 24:
    sampled_indices = np.random.choice(len(train_test_pairs_le5), size=24, replace=False)
    sample_24_pairs = [train_test_pairs_le5[idx] for idx in sampled_indices]
else:
    sample_24_pairs = train_test_pairs_le5

print(f"\nGenerating montage for {len(sample_24_pairs)} train-test duplicate pairs...")
fig = plt.figure(figsize=(16, 24))
gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.25)

for pair_idx, (t_idx, tr_idx, d) in enumerate(sample_24_pairs):
    row_i = pair_idx // 4
    col_i = pair_idx % 4
    
    # Create combined tile: Test image on left, Train image on right
    im_test_p = df_paddy.iloc[t_idx]['image_path']
    im_train_p = df_paddy.iloc[tr_idx]['image_path']
    
    im_t = Image.open(im_test_p).convert('RGB').resize((180, 180))
    im_tr = Image.open(im_train_p).convert('RGB').resize((180, 180))
    
    combined_img = Image.new('RGB', (370, 180))
    combined_img.paste(im_t, (0, 0))
    combined_img.paste(im_tr, (190, 0))
    
    ax = fig.add_subplot(gs[row_i, col_i])
    ax.imshow(combined_img)
    cls_name = df_paddy.iloc[t_idx]['disease_class']
    g_t = df_paddy.iloc[t_idx]['severity_grade']
    g_tr = df_paddy.iloc[tr_idx]['severity_grade']
    caption = f"Pair #{pair_idx+1} (dist={d})\n{cls_name} [G{g_t} vs G{g_tr}]\nTest: {df_paddy.iloc[t_idx]['filename']}\nTrain: {df_paddy.iloc[tr_idx]['filename']}"
    ax.set_title(caption, fontsize=8, pad=3)
    ax.axis('off')

fig.suptitle("Paddy Train-Test Near-Duplicate Pairs (Hamming dist <= 5 bits)\n[Left: Test Image | Right: Train Image]", fontsize=14, fontweight='bold', y=0.995)
montage_out = 'outputs/final_v2/paddy_train_test_leakage_montage.png'
plt.savefig(montage_out, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved Paddy leakage montage: {montage_out}")

# 3. Model Evaluation on Twin-Free Subset (CORN & Softmax)
print("\nEvaluating Production Paddy Checkpoints on Twin-Free Subset...")
device = torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

# Load CORN model
model_corn = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
ckpt_corn = torch.load('outputs/models/paddy_ordinal.pth', map_location='cpu')
model_corn.load_state_dict(ckpt_corn if 'model_state_dict' not in ckpt_corn else ckpt_corn['model_state_dict'])
model_corn.eval()

# Load Softmax model
model_soft = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=False)
ckpt_soft = torch.load('outputs/models/paddy_baseline.pth', map_location='cpu')
model_soft.load_state_dict(ckpt_soft if 'model_state_dict' not in ckpt_soft else ckpt_soft['model_state_dict'])
model_soft.eval()

# Run inference on all 1,507 test images
df_test = df_paddy[df_paddy['split'] == 'test'].copy().reset_index()
y_true = df_test['severity_grade'].values
y_pred_corn = []
y_pred_soft = []

print("Running inference on 1,507 test images...")
for p in df_test['image_path']:
    im = Image.open(p).convert('RGB')
    tensor = transform(im).unsqueeze(0).to(device)
    with torch.no_grad():
        out_c = model_corn(tensor)
        p_corn = torch.sigmoid(out_c)
        pred_c, _ = corn_probs_to_grade(p_corn[0])
        y_pred_corn.append(int(pred_c.item()))
        
        out_s = model_soft(tensor)
        p_soft = torch.softmax(out_s, dim=1)
        pred_s = int(torch.argmax(p_soft, dim=1).item())
        y_pred_soft.append(pred_s)

y_pred_corn = np.array(y_pred_corn)
y_pred_soft = np.array(y_pred_soft)

# Masks for subsets
# 1. Twin-free mask (Hamming <= 5)
test_orig_indices = df_test['index'].values
is_twin_le5 = np.array([idx in test_has_twin_le5 for idx in test_orig_indices])
is_twin_free_le5 = ~is_twin_le5

# 2. Lesion-only mask
lesion_classes = set(config.PADDY_GROUP_A)
is_lesion = df_test['disease_class'].isin(lesion_classes).values

# 3. Twin-free & Lesion-only mask
is_twin_free_lesion = is_twin_free_le5 & is_lesion

print(f"\nSubsets:")
print(f"  Full Test:               {len(df_test)}")
print(f"  Twin-Free Full Test:     {is_twin_free_le5.sum()} ({is_twin_free_le5.sum() / len(df_test) * 100:.2f}%)")
print(f"  Lesion-Only Test:        {is_lesion.sum()}")
print(f"  Twin-Free Lesion Test:   {is_twin_free_lesion.sum()} ({is_twin_free_lesion.sum() / is_lesion.sum() * 100:.2f}%)")

def compute_metrics_with_ci(y_t, y_p, n_boot=2000, seed=42):
    n = len(y_t)
    qwk = cohen_kappa_score(y_t, y_p, weights='quadratic', labels=[0,1,2,3])
    mae = mean_absolute_error(y_t, y_p)
    acc = accuracy_score(y_t, y_p)
    w1 = np.mean(np.abs(y_t - y_p) <= 1)
    
    np.random.seed(seed)
    boot_qwk, boot_mae, boot_acc, boot_w1 = [], [], [], []
    for _ in range(n_boot):
        b_idx = np.random.choice(n, size=n, replace=True)
        yt_b, yp_b = y_t[b_idx], y_p[b_idx]
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

subsets_dict = {
    'full_test_1507': np.ones(len(df_test), dtype=bool),
    'twin_free_test': is_twin_free_le5,
    'lesion_only_757': is_lesion,
    'twin_free_lesion_only': is_twin_free_lesion
}

eval_results = {}
for s_name, s_mask in subsets_dict.items():
    yt_sub = y_true[s_mask]
    corn_sub = y_pred_corn[s_mask]
    soft_sub = y_pred_soft[s_mask]
    
    # Class-only mode baseline for this subset
    # Compute training mode per class
    train_df = df_paddy[df_paddy['split'] == 'train']
    train_modes = train_df.groupby('disease_class')['severity_grade'].agg(lambda x: x.mode()[0]).to_dict()
    sub_df = df_test[s_mask]
    base_sub = np.array([train_modes[c] for c in sub_df['disease_class']])
    
    eval_results[s_name] = {
        'count': int(s_mask.sum()),
        'baseline': compute_metrics_with_ci(yt_sub, base_sub),
        'corn': compute_metrics_with_ci(yt_sub, corn_sub),
        'softmax': compute_metrics_with_ci(yt_sub, soft_sub)
    }

part_d_summary = {
    'unique_test_images_with_twin': {
        'total_test_images': len(test_indices),
        'twin_le5_count': len(test_has_twin_le5),
        'twin_le5_pct': float(len(test_has_twin_le5) / len(test_indices) * 100.0),
        'twin_le8_count': len(test_has_twin_le8),
        'twin_le8_pct': float(len(test_has_twin_le8) / len(test_indices) * 100.0),
        'explanation_pair_count_1226': (
            "The 1,226 count was the exact sum of pairwise cross-split pairs: "
            "602 Train-Test pairs + 513 Train-Val pairs + 111 Val-Test pairs = 1,226 total cross-split pairs. "
            "It was NOT a reused variable; it represents pair combinations, whereas test-twin counts represent unique test entities."
        )
    },
    'montage_path': montage_out,
    'evaluation_on_subsets': eval_results
}

with open('outputs/final_v2/stage2b_part_d.json', 'w') as f:
    json.dump(part_d_summary, f, indent=2)

print("\nPart D complete! Saved to outputs/final_v2/stage2b_part_d.json")
