"""
Strict Leak-Free Kaggle Expansion for Arecanut Training Set.
1. Reads archive.zip (12,292 images).
2. Extracts images and computes pHash (Hamming <= 8, 3 orientations).
3. Compares EVERY external image against the 486 test images in arecanut_splits_v2.csv.
4. Excludes any image with pHash Hamming <= 8 or source filename overlap with test/val set.
5. Adds clean external images into the TRAINING split only.
6. Keeps val (547) and test (486) splits 100% identical and untouched.
7. Saves outputs/arecanut_splits_v2_expanded.csv
"""
import sys, os, zipfile, io, time, json
sys.path.insert(0, '/home/sahith/Projects/Crop')
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ── Load existing leak-free split v2 ──────────────────────────────────────────
df_v2 = pd.read_csv('outputs/arecanut_splits_v2.csv')
test_df = df_v2[df_v2['split'] == 'test'].copy().reset_index(drop=True)
val_df  = df_v2[df_v2['split'] == 'val'].copy().reset_index(drop=True)
train_df= df_v2[df_v2['split'] == 'train'].copy().reset_index(drop=True)

print(f"Loaded arecanut_splits_v2.csv: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# pHash helper
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

# Compute pHashes for all test + val images
print("\nComputing pHashes for test (486) and val (547) images...")
test_val_hashes = []
test_val_paths  = list(test_df['image_path']) + list(val_df['image_path'])

for path in test_val_paths:
    img = cv2.imread(path)
    if img is not None:
        test_val_hashes.append([phash(img), phash(cv2.flip(img, 1)), phash(cv2.flip(img, 0))])
    else:
        test_val_hashes.append([np.uint64(0), np.uint64(0), np.uint64(0)])

tv_h0 = np.array([h[0] for h in test_val_hashes], dtype=np.uint64)
tv_h1 = np.array([h[1] for h in test_val_hashes], dtype=np.uint64)
tv_h2 = np.array([h[2] for h in test_val_hashes], dtype=np.uint64)

# ── Extract and filter images from external Kaggle archive.zip ───────────────
ZIP_PATH = 'data/external/arecanut_kaggle_tejpaviraj/archive.zip'
EXPANDED_IMG_DIR = 'data/arecanut/dataset/expanded_kaggle_clean'
os.makedirs(EXPANDED_IMG_DIR, exist_ok=True)

class_map = {
    'bud_borer': 'bud_borer',
    'healthy_foot': 'healthy_foot',
    'healthy_leaf': 'healthy_leaf',
    'healthy_nut': 'healthy_nut',
    'healthy_trunk': 'healthy_trunk',
    'mahali_koleroga': 'mahali_koleroga',
    'stem_bleeding': 'stem_bleeding',
    'stem_cracking': 'stem_cracking',
    'yellow_leaf_disease': 'yellow_leaf_disease',
}

def get_class_from_path(path_str):
    p_lower = path_str.lower()
    if 'bud' in p_lower or 'borer' in p_lower: return 'bud_borer'
    if 'healthy_foot' in p_lower or 'healthy foot' in p_lower: return 'healthy_foot'
    if 'healthy_leaf' in p_lower or 'healthy leaf' in p_lower: return 'healthy_leaf'
    if 'healthy_nut' in p_lower or 'healthy nut' in p_lower: return 'healthy_nut'
    if 'healthy_trunk' in p_lower or 'healthy trunk' in p_lower: return 'healthy_trunk'
    if 'mahali' in p_lower or 'koleroga' in p_lower: return 'mahali_koleroga'
    if 'stem_bleeding' in p_lower or 'stem bleeding' in p_lower: return 'stem_bleeding'
    if 'stem_cracking' in p_lower or 'stem cracking' in p_lower: return 'stem_cracking'
    if 'yellow' in p_lower or 'yld' in p_lower: return 'yellow_leaf_disease'
    return None

print(f"\nScanning external archive {ZIP_PATH}...")
t0 = time.time()
new_train_records = []
excluded_count = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    all_members = [m for m in z.namelist() if m.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total image files in zip: {len(all_members)}")

    # Deduplicate against images already in v2 split by filename stem
    existing_filenames = set(df_v2['filename'].unique())

    for idx, member in enumerate(all_members):
        cls = get_class_from_path(member)
        if cls is None: continue

        fname = os.path.basename(member)
        if fname in existing_filenames:
            continue  # Already in dataset

        # Read image bytes
        img_bytes = z.read(member)
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None: continue

        # Compute pHash for candidate image
        ch0 = phash(img_bgr)
        ch1 = phash(cv2.flip(img_bgr, 1))
        ch2 = phash(cv2.flip(img_bgr, 0))

        # Check against test + val pHashes (Hamming <= 8)
        is_leak = False
        for ch in [ch0, ch1, ch2]:
            h0_dist = hamming_vec(ch, tv_h0)
            h1_dist = hamming_vec(ch, tv_h1)
            h2_dist = hamming_vec(ch, tv_h2)
            min_d   = min(h0_dist.min(), h1_dist.min(), h2_dist.min())
            if min_d <= 8:
                is_leak = True
                break

        if is_leak:
            excluded_count += 1
            continue

        # Save clean candidate image to expanded_kaggle_clean folder
        save_class_dir = os.path.join(EXPANDED_IMG_DIR, cls)
        os.makedirs(save_class_dir, exist_ok=True)
        save_path = os.path.join(save_class_dir, f"ext_{fname}")
        cv2.imwrite(save_path, img_bgr)

        new_train_records.append({
            'image_path': save_path,
            'filename': f"ext_{fname}",
            'disease_class': cls,
            'severity_grade': 1,
            'split': 'train'
        })

        if (idx + 1) % 2000 == 0:
            print(f"  ... processed {idx+1}/{len(all_members)} ({time.time()-t0:.0f}s), added clean={len(new_train_records)}, excluded_leaks={excluded_count}")

print(f"\nFinished archive scan in {time.time()-t0:.1f}s.")
print(f"  Clean external images added to train: {len(new_train_records)}")
print(f"  Excluded leak matches: {excluded_count}")

# ── Merge new clean train records into dataset ────────────────────────────────
df_new_train = pd.DataFrame(new_train_records)
df_expanded  = pd.concat([df_v2, df_new_train], ignore_index=True)

# Save expanded split CSV
df_expanded.to_csv('outputs/arecanut_splits_v2_expanded.csv', index=False)
print(f"\n✓ Saved: outputs/arecanut_splits_v2_expanded.csv ({len(df_expanded)} total rows)")

print("\nPer-class training counts (Old v2 Train -> New Expanded Train):")
for cls in sorted(df_v2['disease_class'].unique()):
    old_c = len(train_df[train_df['disease_class']==cls])
    new_c = len(df_expanded[(df_expanded['split']=='train') & (df_expanded['disease_class']==cls)])
    val_c = len(df_expanded[(df_expanded['split']=='val') & (df_expanded['disease_class']==cls)])
    test_c= len(df_expanded[(df_expanded['split']=='test') & (df_expanded['disease_class']==cls)])
    print(f"  {cls:<30} Train: {old_c:>4} -> {new_c:>4} | Val: {val_c:>3} | Test: {test_c:>3}")
