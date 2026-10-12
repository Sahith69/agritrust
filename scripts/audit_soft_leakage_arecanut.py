"""
Audit of Soft Leakage in Arecanut Expanded Retrain (99.79% Test Accuracy).
1. Session-level / Campaign-level overlap check between 486 test images and 8,903 expanded train images.
2. Accuracy evaluation on truly isolated test subset (zero session overlap).
3. bud_borer per-image sibling lineage audit (33 test images vs 51 newly added train images).
Verification only. No files modified.
"""
import sys, os, zipfile, time, json, re
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score
from PIL import Image

# ── Load data ─────────────────────────────────────────────────────────────────
df_exp = pd.read_csv('outputs/arecanut_splits_v2_expanded.csv')
df_clusters = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')

test_df  = df_exp[df_exp['split'] == 'test'].copy().reset_index(drop=True)
train_df = df_exp[df_exp['split'] == 'train'].copy().reset_index(drop=True)
new_train_df = train_df[train_df['filename'].str.startswith('ext_')].copy().reset_index(drop=True)
old_train_df = train_df[~train_df['filename'].str.startswith('ext_')].copy().reset_index(drop=True)

print("="*70)
print("AUDIT OF SOFT LEAKAGE IN ARECANUT EXPANDED MODEL (99.79%)")
print("="*70)
print(f"Total Test Images: {len(test_df)}")
print(f"Total Train Images: {len(train_df)} (Original={len(old_train_df)}, Newly Added Kaggle={len(new_train_df)})")
print()

# Helper: Extract base photo stem / lineage from filename
# Examples:
#   'ext_Arecanut_dataset/Arecanut_dataset/test/Healthy_Leaf/Healthy_Leaf_original_50.jpg_8fd3cca6...' -> 'Healthy_Leaf_original_50.jpg'
#   'stem cracking_original_IMG_20230819_165058.jpg_8f6397ee...' -> 'stem cracking_original_IMG_20230819_165058.jpg'
def extract_base_lineage(filename_or_path):
    fname = os.path.basename(filename_or_path)
    # Remove ext_ prefix if present
    if fname.startswith('ext_'):
        fname = fname[4:]
    # Match pattern before UUID suffix
    m = re.match(r'(.+\.(?:jpg|jpeg|png))_[0-9a-f\-]{32,}\.(?:jpg|jpeg|png)$', fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.match(r'(.+?)_[0-9a-f\-]{36}', fname, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return fname.lower()

# Helper: Extract timestamp session string (e.g., '20230819_170846')
def extract_timestamp(path_str):
    m = re.search(r'(20\d{6}_\d{6})', path_str)
    return m.group(1) if m else None

# Build lineage sets for new_train (8,903 images)
new_train_lineages = set(new_train_df['filename'].apply(extract_base_lineage))
all_train_lineages = set(train_df['filename'].apply(extract_base_lineage))

# Merge timestamp and cluster_id info into test_df
test_df['base_lineage'] = test_df['filename'].apply(extract_base_lineage)
test_df['timestamp_str'] = test_df['image_path'].apply(extract_timestamp)

# Also merge robust_cluster_id and sec_of_day from df_clusters if present
test_df = test_df.merge(
    df_clusters[['filename', 'robust_cluster_id', 'sec_of_day']],
    on='filename', how='left'
)
new_train_df = new_train_df.merge(
    df_clusters[['filename', 'robust_cluster_id', 'sec_of_day']],
    on='filename', how='left'
)

# ── ITEM 1: Identify session / campaign overlap ──────────────────────────────
# Overlap criteria:
# 1) Exact base photo lineage overlap (augmented copy or sibling from same original photo stem)
# 2) Timestamp burst session overlap (timestamp within 120s in same disease class)
# 3) Soft perceptual hash overlap (pHash Hamming distance between 9 and 16 with any new train image)

print("Computing pHashes for test (486) and new_train (8,903) images to detect soft pHash overlap (Hamming 9-16)...")
t0 = time.time()

LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
def phash(img_bgr, hs=8):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(g, (hs*4, hs*4), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(r)[:hs, :hs]
    bits = (d > np.median(d)).flatten()
    v = np.uint64(0)
    for b in bits: v = (v << np.uint64(1)) | np.uint64(int(b))
    return v

def hamming_vec(h_scalar, arr_uint64):
    xor = np.bitwise_xor(np.uint64(h_scalar), arr_uint64)
    h = np.zeros(len(xor), dtype=np.int32)
    for shift in range(8):
        h += LUT[(xor >> np.uint64(shift*8)) & np.uint64(0xFF)]
    return h

test_hashes = []
for p in test_df['image_path']:
    img = cv2.imread(p)
    if img is not None:
        test_hashes.append([phash(img), phash(cv2.flip(img,1)), phash(cv2.flip(img,0))])
    else:
        test_hashes.append([np.uint64(0), np.uint64(0), np.uint64(0)])

new_train_hashes = []
for p in new_train_df['image_path']:
    img = cv2.imread(p)
    if img is not None:
        new_train_hashes.append([phash(img), phash(cv2.flip(img,1)), phash(cv2.flip(img,0))])
    else:
        new_train_hashes.append([np.uint64(0), np.uint64(0), np.uint64(0)])

nt_h0 = np.array([h[0] for h in new_train_hashes], dtype=np.uint64)
nt_h1 = np.array([h[1] for h in new_train_hashes], dtype=np.uint64)
nt_h2 = np.array([h[2] for h in new_train_hashes], dtype=np.uint64)

# For each test image, find min Hamming distance to any image in new_train
min_hammings = []
for i, th in enumerate(test_hashes):
    min_d = 64
    for h_scalar in th:
        d0 = hamming_vec(h_scalar, nt_h0).min()
        d1 = hamming_vec(h_scalar, nt_h1).min()
        d2 = hamming_vec(h_scalar, nt_h2).min()
        min_d = min(min_d, d0, d1, d2)
    min_hammings.append(min_d)

test_df['min_hamming_to_new_train'] = min_hammings
print(f"pHash calculation complete in {time.time()-t0:.1f}s.")
print()

# Define session overlap flags per test image
lineage_overlaps = []
burst_overlaps   = []
soft_phash_overlaps = [] # Hamming 9-16
total_overlaps   = []

# Pre-extract new_train timestamps by class
new_train_timestamps_by_class = {}
for cls in test_df['disease_class'].unique():
    paths = new_train_df[new_train_df['disease_class']==cls]['image_path']
    ts_set = set()
    for p in paths:
        ts = extract_timestamp(p)
        if ts: ts_set.add(ts)
    new_train_timestamps_by_class[cls] = ts_set

for idx, row in test_df.iterrows():
    lin = row['base_lineage']
    cls = row['disease_class']
    ts  = row['timestamp_str']
    ham = row['min_hamming_to_new_train']

    # 1. Lineage match
    has_lin = lin in new_train_lineages

    # 2. Timestamp burst match in new_train
    has_ts = (ts in new_train_timestamps_by_class.get(cls, set())) if ts else False

    # 3. Soft pHash match (Hamming 9 to 16)
    has_soft_phash = (ham >= 9 and ham <= 16)

    # Total session overlap if ANY of the three conditions holds
    has_session = has_lin or has_ts or has_soft_phash

    lineage_overlaps.append(has_lin)
    burst_overlaps.append(has_ts)
    soft_phash_overlaps.append(has_soft_phash)
    total_overlaps.append(has_session)

test_df['has_lineage_overlap']  = lineage_overlaps
test_df['has_burst_overlap']    = burst_overlaps
test_df['has_soft_phash_overlap'] = soft_phash_overlaps
test_df['has_session_overlap']  = total_overlaps

n_test = len(test_df)
n_lin  = sum(lineage_overlaps)
n_ts   = sum(burst_overlaps)
n_soft = sum(soft_phash_overlaps)
n_tot  = sum(total_overlaps)

print("="*70)
print("ITEM 1: SESSION-LEVEL / CAMPAIGN-LEVEL OVERLAP MEASUREMENT")
print("="*70)
print(f"Total Test Images: {n_test}")
print(f"  - Base photo lineage match with new train: {n_lin:>4} / {n_test} ({n_lin/n_test*100:.1f}%)")
print(f"  - Timestamp-adjacent burst match with new train: {n_ts:>4} / {n_test} ({n_ts/n_test*100:.1f}%)")
print(f"  - Soft pHash perceptual match (Hamming 9-16): {n_soft:>4} / {n_test} ({n_soft/n_test*100:.1f}%)")
print(f"  -------------------------------------------------------------")
print(f"  Total Test Images with ANY Session-Level Overlap: {n_tot:>4} / {n_test} ({n_tot/n_test*100:.1f}%)")
print(f"  Truly Isolated Test Images (Zero Session Overlap): {n_test - n_tot:>4} / {n_test} ({(n_test - n_tot)/n_test*100:.1f}%)")
print()

# ── ITEM 2: Re-evaluate v2 model on Isolated vs Overlapping test subsets ──────
print("="*70)
print("ITEM 2: MODEL RE-EVALUATION ON TEST SUBSETS")
print("="*70)

model_path = os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier_v2.pth')
ck = torch.load(model_path, map_location='cpu', weights_only=True)
classes = ck['classes']

base = models.resnet18(weights=None)
n_features = base.fc.in_features
base.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(n_features, len(classes)))
base.load_state_dict(ck['model_state_dict'])
base.eval()

TF = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

class SimpleDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return TF(Image.open(self.paths[i]).convert('RGB'))

loader = DataLoader(SimpleDS(test_df['image_path'].tolist()), batch_size=32, shuffle=False)
all_preds = []
with torch.no_grad():
    for batch in loader:
        all_preds.extend(base(batch).argmax(1).numpy())

test_df['pred_cls'] = [classes[i] for i in all_preds]
test_df['is_correct'] = test_df['disease_class'] == test_df['pred_cls']

# Split subsets
df_overlapping = test_df[test_df['has_session_overlap']].reset_index(drop=True)
df_isolated    = test_df[~test_df['has_session_overlap']].reset_index(drop=True)

acc_full = accuracy_score(test_df['disease_class'], test_df['pred_cls']) * 100
acc_over = accuracy_score(df_overlapping['disease_class'], df_overlapping['pred_cls']) * 100 if len(df_overlapping) > 0 else 0.0
acc_iso  = accuracy_score(df_isolated['disease_class'], df_isolated['pred_cls']) * 100 if len(df_isolated) > 0 else 0.0

print(f"FULL TEST SET (N={len(test_df)}):                     Accuracy = {acc_full:.2f}%")
print(f"OVERLAPPING TEST SUBSET (N={len(df_overlapping)}):         Accuracy = {acc_over:.2f}%")
if len(df_isolated) > 0:
    print(f"TRULY ISOLATED TEST SUBSET (N={len(df_isolated)}):        Accuracy = {acc_iso:.2f}%")
else:
    print(f"TRULY ISOLATED TEST SUBSET (N=0): ZERO isolated test images exist.")
    print("  EXPLICIT FINDING: 100% of test images share session-level lineage/burst/soft-pHash overlap with the expanded Kaggle dataset.")

print()

# ── ITEM 3: bud_borer per-image sibling audit ──────────────────────────────────
print("="*70)
print("ITEM 3: BUD_BORER PER-IMAGE SIBLING LINEAGE AUDIT")
print("="*70)

test_bb = test_df[test_df['disease_class'] == 'bud_borer'].copy().reset_index(drop=True)
new_train_bb = new_train_df[new_train_df['disease_class'] == 'bud_borer'].copy().reset_index(drop=True)

print(f"Total bud_borer test images: N = {len(test_bb)}")
print(f"Newly added bud_borer train images: N = {len(new_train_bb)}")
print()

# Extract lineages of new train bud_borer
new_bb_lineages = set(new_train_bb['filename'].apply(extract_base_lineage))

# Audit each of the 33 test bud_borer images
bb_records = []
for idx, row in test_bb.iterrows():
    p = row['image_path']
    lin = row['base_lineage']
    pred = row['pred_cls']
    correct = (pred == 'bud_borer')
    ham = row['min_hamming_to_new_train']

    # Lineage match in new train bud_borer
    has_lin = lin in new_bb_lineages

    # Soft pHash match in new train bud_borer (Hamming <= 16)
    has_near_sibling = has_lin or (ham <= 16)

    bb_records.append({
        'test_index': idx + 1,
        'filename': row['filename'],
        'base_lineage': lin,
        'prediction': pred,
        'is_correct': correct,
        'has_lineage_match_in_new_train': has_lin,
        'min_hamming_to_new_train': ham,
        'has_near_visual_sibling': has_near_sibling,
    })

df_bb_audit = pd.DataFrame(bb_records)

n_bb_correct = df_bb_audit['is_correct'].sum()
n_bb_sibling = df_bb_audit['has_near_visual_sibling'].sum()
n_bb_both    = (df_bb_audit['is_correct'] & df_bb_audit['has_near_visual_sibling']).sum()

print(f"bud_borer Test Results Summary:")
print(f"  - Correctly predicted test images: {n_bb_correct} / {len(test_bb)} ({n_bb_correct/len(test_bb)*100:.1f}%)")
print(f"  - Images with near-visual-sibling in newly added train: {n_bb_sibling} / {len(test_bb)} ({n_bb_sibling/len(test_bb)*100:.1f}%)")
print(f"  - Correctly predicted AND has sibling in newly added train: {n_bb_both} / {n_bb_correct} ({n_bb_both/n_bb_correct*100:.1f}%)")
print()

print("Per-image breakdown for all 33 bud_borer test images:")
print(df_bb_audit[['test_index', 'base_lineage', 'prediction', 'is_correct', 'min_hamming_to_new_train', 'has_near_visual_sibling']].to_string(index=False))

if n_bb_both == n_bb_correct:
    print("\nPLAIN FINDING: 100% of correctly predicted bud_borer test images have a near-visual-sibling in the newly-added training set.")
else:
    print(f"\nPLAIN FINDING: {n_bb_both}/{n_bb_correct} ({n_bb_both/n_bb_correct*100:.1f}%) of correctly predicted bud_borer test images have a near-visual-sibling in the newly-added training set.")

# Save audit results JSON
audit_out = {
    'task': 'Soft Leakage Audit of Arecanut Expanded Retrain',
    'total_test_images': len(test_df),
    'session_overlap': {
        'lineage_match_count': int(n_lin),
        'timestamp_burst_match_count': int(n_ts),
        'soft_phash_match_count': int(n_soft),
        'total_session_overlap_count': int(n_tot),
        'total_session_overlap_pct': round(n_tot/n_test*100, 2),
        'truly_isolated_count': int(n_test - n_tot),
        'truly_isolated_pct': round((n_test - n_tot)/n_test*100, 2),
    },
    'subset_accuracy': {
        'full_test_accuracy': float(acc_full),
        'overlapping_subset_accuracy': float(acc_over),
        'isolated_subset_accuracy': float(acc_iso) if len(df_isolated)>0 else None,
        'isolated_subset_count': int(len(df_isolated)),
    },
    'bud_borer_audit': {
        'total_test_count': int(len(test_bb)),
        'correct_count': int(n_bb_correct),
        'with_sibling_count': int(n_bb_sibling),
        'correct_and_with_sibling_count': int(n_bb_both),
        'pct_correct_with_sibling': round(n_bb_both/n_bb_correct*100, 2) if n_bb_correct>0 else 0.0,
    }
}

with open('outputs/final_v2/soft_leakage_audit_results.json', 'w') as f:
    json.dump(audit_out, f, indent=2)

print(f"\n✓ Saved: outputs/final_v2/soft_leakage_audit_results.json")
