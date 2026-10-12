import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2c - PART E: HUMAN-GRADING SHEET V2 (165 SLOTS)")
print("=" * 80)

np.random.seed(42)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("Loading Arecanut clean manifest v2...")
areca_df = pd.read_csv('outputs/final_v2/arecanut_clean_v2.csv')
print(f"Total Arecanut clean v2 images: {len(areca_df)}")

# Arecanut lesion classes (excluding YLD)
ARECA_LESION_CLASSES = ['bud_borer', 'mahali_koleroga', 'stem_bleeding', 'stem_cracking']
HEALTHY_CLASSES = ['healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk']

DISK_STRUCT = get_disk(config.LEAF_MASK_EROSION_PX)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def compute_lesion_mask_and_ratio(img_rgb, leaf_mask):
    eroded = scipy.ndimage.binary_erosion(leaf_mask, structure=DISK_STRUCT)
    img_float = img_rgb.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    thresh = otsu_threshold(gray, mask=eroded)
    lesion_mask = (gray < thresh) & eroded
    labeled, num_features = scipy.ndimage.label(lesion_mask)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        valid_labels = np.where(counts >= MIN_LESION_BLOB)[0]
        valid_labels = valid_labels[valid_labels > 0]
        lesion_mask = np.isin(labeled, valid_labels)
    else:
        lesion_mask = np.zeros_like(lesion_mask, dtype=bool)
    ratio = compute_ratio(lesion_mask, leaf_mask)
    return lesion_mask, ratio

def ratio_to_grade_eq2(r):
    if r <= 0.0: return 0
    elif r <= 0.05: return 1
    elif r <= 0.15: return 2
    else: return 3

# Compute or load ratios for original 871 lesion candidates
cache_areca_orig_path = 'outputs/final_v2/arecanut_orig_lesion_eq2_cache.csv'
orig_lesion_mask = (areca_df['source'] == 'original_871') & (areca_df['disease_class'].isin(ARECA_LESION_CLASSES))
orig_lesion_df = areca_df[orig_lesion_mask].copy().reset_index(drop=True)

if os.path.exists(cache_areca_orig_path):
    print(f"\nLoading cached Eq. 2 ratios from {cache_areca_orig_path}...")
    cached_df = pd.read_csv(cache_areca_orig_path)
    orig_lesion_df = cached_df
else:
    print("\nComputing Eq. 2 ratios for original 871 lesion candidates...")
    orig_ratios = []
    orig_eq2_grades = []
    for idx, row in orig_lesion_df.iterrows():
        im_bgr = cv2.imread(row['image_path'])
        if im_bgr is None:
            orig_ratios.append(0.0)
            orig_eq2_grades.append(0)
            continue
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lf = segment_leaf_adaptive(im_rgb)
        _, r = compute_lesion_mask_and_ratio(im_rgb, lf)
        orig_ratios.append(float(r))
        orig_eq2_grades.append(ratio_to_grade_eq2(r))

    orig_lesion_df['eq2_ratio'] = orig_ratios
    orig_lesion_df['eq2_grade'] = orig_eq2_grades
    orig_lesion_df.to_csv(cache_areca_orig_path, index=False)

# Stratum 1: JSON != Eq2 (target 35)
orig_disagree = orig_lesion_df[orig_lesion_df['severity_grade'] != orig_lesion_df['eq2_grade']].copy()
# Stratum 2: JSON == Eq2 (target 15)
orig_agree = orig_lesion_df[orig_lesion_df['severity_grade'] == orig_lesion_df['eq2_grade']].copy()

print(f"Original 871 lesion images: {len(orig_lesion_df)}")
print(f"  JSON != Eq2: {len(orig_disagree)}")
print(f"  JSON == Eq2: {len(orig_agree)}")

# Function to sample with cluster diversity (max 2 per cluster)
def sample_with_cluster_limit(df_pool, target_n, max_per_cluster=2, random_state=42):
    rng = np.random.RandomState(random_state)
    shuffled = df_pool.sample(frac=1.0, random_state=rng).copy()
    selected_indices = []
    cluster_counts = {}
    
    for idx, row in shuffled.iterrows():
        c = row['cluster_id']
        cnt = cluster_counts.get(c, 0)
        if cnt < max_per_cluster:
            selected_indices.append(idx)
            cluster_counts[c] = cnt + 1
            if len(selected_indices) == target_n:
                break
                
    if len(selected_indices) < target_n:
        # Fall back to allow more if needed
        for idx, row in shuffled.iterrows():
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) == target_n:
                    break
    return df_pool.loc[selected_indices].copy()

# Sample 35 disagree
sample_areca_disagree = sample_with_cluster_limit(orig_disagree, 35, max_per_cluster=2, random_state=42)
sample_areca_disagree['stratum'] = 'arecanut_orig_disagree'

# Sample 15 agree
# Avoid reusing clusters if possible
sample_areca_agree = sample_with_cluster_limit(orig_agree, 15, max_per_cluster=2, random_state=43)
sample_areca_agree['stratum'] = 'arecanut_orig_agree'

# Stratum 3: Kaggle lesion images (target 20)
kaggle_lesion_mask = (areca_df['source'] == 'kaggle_1896') & (areca_df['disease_class'].isin(ARECA_LESION_CLASSES))
kaggle_lesion_df = areca_df[kaggle_lesion_mask].copy().reset_index(drop=True)
sample_areca_kaggle = sample_with_cluster_limit(kaggle_lesion_df, 20, max_per_cluster=2, random_state=44)
sample_areca_kaggle['stratum'] = 'arecanut_kaggle_lesion'
sample_areca_kaggle['eq2_ratio'] = np.nan
sample_areca_kaggle['eq2_grade'] = sample_areca_kaggle['severity_grade'] # in Kaggle, Eq. 2 was used to assign severity_grade

# Compute actual Eq. 2 ratio for sampled Kaggle images so answer key has exact ratios
for idx, row in sample_areca_kaggle.iterrows():
    im_bgr = cv2.imread(row['image_path'])
    if im_bgr is not None:
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lf = segment_leaf_adaptive(im_rgb)
        _, r = compute_lesion_mask_and_ratio(im_rgb, lf)
        sample_areca_kaggle.loc[idx, 'eq2_ratio'] = float(r)

print(f"Sampled Arecanut lesion images: Disagree={len(sample_areca_disagree)}, Agree={len(sample_areca_agree)}, Kaggle={len(sample_areca_kaggle)}")
total_areca_lesion = pd.concat([sample_areca_disagree, sample_areca_agree, sample_areca_kaggle]).reset_index(drop=True)
print(f"Total Arecanut lesion selected: {len(total_areca_lesion)} (Unique clusters: {total_areca_lesion['cluster_id'].nunique()})")

# ==============================================================================
# 2. PADDY LESION SELECTION (TWIN-FREE AT <= 8 ONLY)
# ==============================================================================
print("\nLoading Paddy dataset & computing twin-free at <= 8 mask...")
paddy_df = pd.read_csv('outputs/paddy_splits.csv')
paddy_merged = paddy_df.copy() # paddy_splits.csv already has lesion_ratio

# Compute pHashes for Paddy
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

def fast_popcount_matrix(m):
    m = m - ((m >> np.uint64(1)) & np.uint64(0x5555555555555555))
    m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
    m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return m.astype(np.int32)

n_paddy = len(paddy_merged)
cache_paddy_phash_path = 'outputs/final_v2/paddy_phashes_cache.npz'
if os.path.exists(cache_paddy_phash_path):
    print(f"Loading cached Paddy pHashes from {cache_paddy_phash_path}...")
    npz = np.load(cache_paddy_phash_path)
    hashes_p_orig = npz['orig']
    hashes_p_hflip = npz['hflip']
    hashes_p_vflip = npz['vflip']
else:
    print("Computing pHashes for all Paddy images...")
    hashes_p_orig = np.zeros(n_paddy, dtype=np.uint64)
    hashes_p_hflip = np.zeros(n_paddy, dtype=np.uint64)
    hashes_p_vflip = np.zeros(n_paddy, dtype=np.uint64)

    for i, p in enumerate(paddy_merged['image_path']):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            hashes_p_orig[i] = compute_phash_uint64(im)
            hashes_p_hflip[i] = compute_phash_uint64(cv2.flip(im, 1))
            hashes_p_vflip[i] = compute_phash_uint64(cv2.flip(im, 0))
    np.savez_compressed(cache_paddy_phash_path, orig=hashes_p_orig, hflip=hashes_p_hflip, vflip=hashes_p_vflip)

p_splits = paddy_merged['split'].values
p_test_idx = np.where(p_splits == 'test')[0]
p_tv_idx = np.where(p_splits != 'test')[0]

H_p_test = hashes_p_orig[p_test_idx]
H_p_tv_orig = hashes_p_orig[p_tv_idx]
H_p_tv_hflip = hashes_p_hflip[p_tv_idx]
H_p_tv_vflip = hashes_p_vflip[p_tv_idx]

paddy_test_has_twin_le8 = set()
chunk_t = 250
chunk_tv = 2500

for b_start in range(0, len(p_test_idx), chunk_t):
    b_end = min(b_start + chunk_t, len(p_test_idx))
    t_sub = p_test_idx[b_start:b_end]
    h_sub = hashes_p_orig[t_sub]
    
    for tv_start in range(0, len(p_tv_idx), chunk_tv):
        tv_end = min(tv_start + chunk_tv, len(p_tv_idx))
        tv_sub_idx = p_tv_idx[tv_start:tv_end]
        
        xor1 = h_sub[:, None] ^ H_p_tv_orig[tv_start:tv_end][None, :]
        xor2 = h_sub[:, None] ^ H_p_tv_hflip[tv_start:tv_end][None, :]
        xor3 = h_sub[:, None] ^ H_p_tv_vflip[tv_start:tv_end][None, :]
        
        d1 = fast_popcount_matrix(xor1)
        d2 = fast_popcount_matrix(xor2)
        d3 = fast_popcount_matrix(xor3)
        min_d = np.minimum(d1, np.minimum(d2, d3))
        
        hits = np.where(min_d <= 8)
        for r_i in hits[0]:
            paddy_test_has_twin_le8.add(t_sub[r_i])

paddy_test_df = paddy_merged.iloc[p_test_idx].copy().reset_index(drop=True)
paddy_test_orig_indices = p_test_idx

is_twin_free_le8 = np.array([idx not in paddy_test_has_twin_le8 for idx in paddy_test_orig_indices])
paddy_test_df['twin_free_le8'] = is_twin_free_le8

PADDY_LESION_CLASSES = set(config.PADDY_GROUP_A)
paddy_twin_free_lesion = paddy_test_df[paddy_test_df['twin_free_le8'] & paddy_test_df['disease_class'].isin(PADDY_LESION_CLASSES)].copy().reset_index(drop=True)
print(f"Paddy twin-free at <= 8 lesion test images available: {len(paddy_twin_free_lesion)}")

# Define boundary/G1:
# Severity grade 1 OR lesion ratio close to grade boundaries [0.03, 0.07] or [0.13, 0.17]
is_boundary_g1 = (
    (paddy_twin_free_lesion['severity_grade'] == 1) |
    ((paddy_twin_free_lesion['lesion_ratio'] >= 0.03) & (paddy_twin_free_lesion['lesion_ratio'] <= 0.07)) |
    ((paddy_twin_free_lesion['lesion_ratio'] >= 0.13) & (paddy_twin_free_lesion['lesion_ratio'] <= 0.17))
)

paddy_pool_boundary = paddy_twin_free_lesion[is_boundary_g1].copy()
paddy_pool_other = paddy_twin_free_lesion[~is_boundary_g1].copy()

print(f"Paddy Boundary/G1 pool: {len(paddy_pool_boundary)}")
print(f"Paddy other lesion pool: {len(paddy_pool_other)}")

# Sample 25 boundary/G1
sample_paddy_boundary = paddy_pool_boundary.sample(n=min(25, len(paddy_pool_boundary)), random_state=42).copy()
sample_paddy_boundary['stratum'] = 'paddy_boundary_g1'

# Sample 25 random from remaining
remaining_paddy_pool = paddy_twin_free_lesion[~paddy_twin_free_lesion['filename'].isin(sample_paddy_boundary['filename'])].copy()
sample_paddy_random = remaining_paddy_pool.sample(n=25, random_state=42).copy()
sample_paddy_random['stratum'] = 'paddy_random'

paddy_selected_50 = pd.concat([sample_paddy_boundary, sample_paddy_random]).reset_index(drop=True)
paddy_selected_50['crop_type'] = 'Paddy'
paddy_selected_50['source'] = 'paddy_test_twin_free'
paddy_selected_50['cluster_id'] = 'paddy_tf_' + paddy_selected_50['filename'].str.replace('.jpg', '', regex=False)
paddy_selected_50['original_grade'] = paddy_selected_50['severity_grade']
paddy_selected_50['eq2_grade'] = paddy_selected_50['severity_grade']
paddy_selected_50['eq2_ratio'] = paddy_selected_50['lesion_ratio']
print(f"Selected {len(paddy_selected_50)} Paddy lesion images (Boundary/G1={len(sample_paddy_boundary)}, Random={len(sample_paddy_random)})")

# ==============================================================================
# 3. HEALTHY IMAGES (30 TOTAL: 15 HEALTHY_TRUNK, 15 OTHER HEALTHY)
# ==============================================================================
print("\nSampling healthy images...")
# 15 healthy_trunk from >= 6 clusters
ht_pool = areca_df[areca_df['disease_class'] == 'healthy_trunk'].copy()
ht_clusters = ht_pool['cluster_id'].unique()
print(f"Healthy trunk clusters available: {len(ht_clusters)}")

# Sample across >= 6 clusters (sample from 8 clusters, 1-2 per cluster)
sample_ht = sample_with_cluster_limit(ht_pool, 15, max_per_cluster=2, random_state=42)
sample_ht['stratum'] = 'healthy_trunk'
sample_ht['crop_type'] = 'Arecanut'
sample_ht['eq2_ratio'] = 0.0
sample_ht['eq2_grade'] = 0
sample_ht['original_grade'] = 0
print(f"Selected {len(sample_ht)} healthy_trunk images across {sample_ht['cluster_id'].nunique()} clusters")

# 15 other healthy:
# Sample from healthy_foot, healthy_leaf, healthy_nut, and Paddy normal
hf_pool = areca_df[areca_df['disease_class'] == 'healthy_foot']
hl_pool = areca_df[areca_df['disease_class'] == 'healthy_leaf']
hn_pool = areca_df[areca_df['disease_class'] == 'healthy_nut']
pn_pool = paddy_test_df[paddy_test_df['disease_class'] == 'normal']

sample_hf = sample_with_cluster_limit(hf_pool, 4, max_per_cluster=1, random_state=42)
sample_hf['crop_type'] = 'Arecanut'
sample_hl = sample_with_cluster_limit(hl_pool, 4, max_per_cluster=1, random_state=42)
sample_hl['crop_type'] = 'Arecanut'
sample_hn = sample_with_cluster_limit(hn_pool, 4, max_per_cluster=1, random_state=42)
sample_hn['crop_type'] = 'Arecanut'

sample_pn = pn_pool.sample(n=3, random_state=42).copy()
sample_pn['crop_type'] = 'Paddy'
sample_pn['source'] = 'paddy_test_twin_free'
sample_pn['cluster_id'] = 'paddy_tf_' + sample_pn['filename'].str.replace('.jpg', '', regex=False)
sample_pn['severity_grade'] = 0
sample_pn['lesion_ratio'] = 0.0

other_healthy = pd.concat([sample_hf, sample_hl, sample_hn, sample_pn]).reset_index(drop=True)
other_healthy['stratum'] = 'other_healthy'
other_healthy['eq2_ratio'] = 0.0
other_healthy['eq2_grade'] = 0
other_healthy['original_grade'] = 0
print(f"Selected {len(other_healthy)} other healthy images (HF={len(sample_hf)}, HL={len(sample_hl)}, HN={len(sample_hn)}, PN={len(sample_pn)})")

# ==============================================================================
# 4. ASSEMBLE 150 UNIQUE IMAGES & 15 REPEATS -> 165 SLOTS
# ==============================================================================
# Harmonize columns for Arecanut lesion images
total_areca_lesion['crop_type'] = 'Arecanut'
total_areca_lesion['original_grade'] = total_areca_lesion['severity_grade']

cols_needed = ['crop_type', 'disease_class', 'filename', 'image_path', 'cluster_id', 'source', 'original_grade', 'eq2_grade', 'eq2_ratio', 'stratum']

unique_150_df = pd.concat([
    total_areca_lesion[cols_needed],
    paddy_selected_50[cols_needed],
    sample_ht[cols_needed],
    other_healthy[cols_needed]
]).reset_index(drop=True)

unique_150_df['unique_image_id'] = np.arange(1, 151)
print(f"\nAssembled {len(unique_150_df)} unique images:")
print(unique_150_df['stratum'].value_counts())

# Select 15 images to repeat (from diverse strata)
# 7 from Arecanut lesion, 5 from Paddy lesion, 3 from Healthy
np.random.seed(42)
repeat_areca_idx = np.random.choice(unique_150_df[unique_150_df['stratum'].str.startswith('arecanut')].index, size=7, replace=False)
repeat_paddy_idx = np.random.choice(unique_150_df[unique_150_df['stratum'].str.startswith('paddy')].index, size=5, replace=False)
repeat_healthy_idx = np.random.choice(unique_150_df[unique_150_df['stratum'].str.contains('healthy')].index, size=3, replace=False)

repeat_indices = np.concatenate([repeat_areca_idx, repeat_paddy_idx, repeat_healthy_idx])
print(f"Selected 15 image indices for repeat: {repeat_indices}")

# Assign initial slots: place the 150 unique images
# We want to place the 15 repeats at random positions such that |slot_orig - slot_repeat| >= 15
# Total slots = 165
slots_data = [None] * 165

# Randomly select 15 positions in the latter half (e.g. slots 50-164) for the repeats
# and ensure they don't collide
rng = np.random.RandomState(42)
repeat_positions = rng.choice(np.arange(60, 165), size=15, replace=False)
repeat_positions.sort()

# The remaining 150 positions are for the unique images
unique_positions = [i for i in range(165) if i not in repeat_positions]
rng.shuffle(unique_positions)

# Place unique images
unique_slot_map = {} # unique_image_id -> slot_id
for pos, (_, row) in zip(unique_positions, unique_150_df.iterrows()):
    slot_id = pos + 1
    u_id = int(row['unique_image_id'])
    unique_slot_map[u_id] = slot_id
    slots_data[pos] = {
        'slot_id': slot_id,
        'unique_image_id': u_id,
        'is_repeat': False,
        'repeat_of_slot_id': None,
        'crop_type': row['crop_type'],
        'disease_class': row['disease_class'],
        'filename': row['filename'],
        'image_path': row['image_path'],
        'cluster_id': row['cluster_id'],
        'source': row['source'],
        'original_grade': int(row['original_grade']),
        'eq2_grade': int(row['eq2_grade']),
        'lesion_ratio_pct': float(row['eq2_ratio']) * 100.0,
        'sampling_stratum': row['stratum']
    }

# Now place the 15 repeats in repeat_positions
for rep_pos, orig_u_idx in zip(repeat_positions, repeat_indices):
    orig_row = unique_150_df.iloc[orig_u_idx]
    orig_u_id = int(orig_row['unique_image_id'])
    orig_slot_id = unique_slot_map[orig_u_id]
    
    slot_id = rep_pos + 1
    slots_data[rep_pos] = {
        'slot_id': slot_id,
        'unique_image_id': orig_u_id,
        'is_repeat': True,
        'repeat_of_slot_id': orig_slot_id,
        'crop_type': orig_row['crop_type'],
        'disease_class': orig_row['disease_class'],
        'filename': orig_row['filename'],
        'image_path': orig_row['image_path'],
        'cluster_id': orig_row['cluster_id'],
        'source': orig_row['source'],
        'original_grade': int(orig_row['original_grade']),
        'eq2_grade': int(orig_row['eq2_grade']),
        'lesion_ratio_pct': float(orig_row['eq2_ratio']) * 100.0,
        'sampling_stratum': orig_row['stratum']
    }

slots_df = pd.DataFrame(slots_data)
print(f"Total slots prepared: {len(slots_df)} (Repeats: {slots_df['is_repeat'].sum()})")

# Verify distance between original and repeat
min_dist = 999
for _, row in slots_df[slots_df['is_repeat']].iterrows():
    dist = abs(row['slot_id'] - row['repeat_of_slot_id'])
    if dist < min_dist:
        min_dist = dist
print(f"Minimum distance between duplicate pairs: {min_dist} slots (Target >= 15 slots)")

# ==============================================================================
# 5. SAVE BLANK CSV & ANSWER KEY CSV/JSON
# ==============================================================================
blank_df = pd.DataFrame({
    'slot_id': slots_df['slot_id'],
    'crop_type': slots_df['crop_type'],
    'disease_class': slots_df['disease_class'],
    'human_grade': [''] * len(slots_df),
    'confidence_1_to_5': [''] * len(slots_df),
    'notes': [''] * len(slots_df)
})
blank_csv_path = 'outputs/final_v2/human_grading_sheet_v2_blank.csv'
blank_df.to_csv(blank_csv_path, index=False)
print(f"\nSaved blank grading sheet to: {blank_csv_path}")

key_csv_path = 'outputs/final_v2/human_grading_answer_key_v2.csv'
slots_df.to_csv(key_csv_path, index=False)
print(f"Saved answer key CSV to: {key_csv_path}")

key_json_path = 'outputs/final_v2/human_grading_answer_key_v2.json'
slots_dict = slots_df.to_dict(orient='records')
with open(key_json_path, 'w') as f:
    json.dump({
        'total_slots': len(slots_df),
        'unique_images': len(unique_150_df),
        'exact_repeats': int(slots_df['is_repeat'].sum()),
        'slots': slots_dict
    }, f, indent=2)
print(f"Saved answer key JSON to: {key_json_path}")

# ==============================================================================
# 6. GENERATE PDF BOOKLET WITH 1-PAGE INSTRUCTION CARD & 165 SLOTS
# ==============================================================================
pdf_path = 'outputs/final_v2/human_grading_sheet_v2.pdf'
print(f"\nGenerating multi-page PDF booklet: {pdf_path}...")

with PdfPages(pdf_path) as pdf:
    # -------------------------------------------------------------
    # PAGE 1: INSTRUCTION CARD WITH CALIBRATION AREA DIAGRAMS
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    
    # Title & Header
    fig.text(0.5, 0.95, "AGRICULTURAL DISEASE SEVERITY GRADING PROTOCOL", 
             ha='center', va='top', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.92, "Standardized Protocol & Visual Calibration Card for Human Raters", 
             ha='center', va='top', fontsize=11, fontstyle='italic', color='#333333')
    
    # Instructions Box
    instr_text = (
        "OBJECTIVE:\n"
        "Accurately estimate disease severity by evaluating the proportion of leaf/organ area covered by visible necrotic\n"
        "or chlorotic lesions. Severity is assessed on an ordinal 4-grade scale [Grade 0 to Grade 3].\n\n"
        "SEVERITY SCALE DEFINITIONS (Standard Equation 2 Thresholds):\n"
        "  • Grade 0 (Healthy):   0% lesion area. Organ shows healthy pigmentation with zero necrotic symptoms.\n"
        "  • Grade 1 (Mild):      > 0% and ≤ 5% lesion area. Sparse, localized lesion specks or tiny spots.\n"
        "  • Grade 2 (Moderate):  > 5% and ≤ 15% lesion area. Noticeable lesions coalescing across significant leaf area.\n"
        "  • Grade 3 (Severe):    > 15% lesion area. Extensive blighting, necrosis, cracking, or tissue desiccation.\n\n"
        "GRADING INSTRUCTIONS:\n"
        "1. Examine the raw crop image carefully in each slot.\n"
        "2. Visually calibrate the lesion area against the reference diagrams below before selecting a grade.\n"
        "3. Check exactly ONE grade box [0, 1, 2, or 3] and rate your grading confidence from 1 (Low) to 5 (High).\n"
        "4. Note any image blur, occlusion, or ambiguous boundaries in the 'Notes' field."
    )
    fig.text(0.08, 0.88, instr_text, ha='left', va='top', fontsize=9.2, family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4F6F7', edgecolor='#BDC3C7'))
    
    # Calibration Area Diagrams: 2%, 5%, 10%, 15%, 25%
    fig.text(0.5, 0.53, "VISUAL AREA CALIBRATION REFERENCE (Standardized Organ Coverage)", 
             ha='center', va='top', fontsize=11, fontweight='bold', color='#2C3E50')
    
    calib_pcts = [2.0, 5.0, 10.0, 15.0, 25.0]
    calib_grades = ["Grade 1 (2% Mild)", "Threshold G1/G2 (5%)", "Grade 2 (10% Moderate)", "Threshold G2/G3 (15%)", "Grade 3 (25% Severe)"]
    
    # 5 calibration boxes side by side
    gs_calib = GridSpec(1, 5, left=0.08, right=0.92, bottom=0.18, top=0.50, wspace=0.35)
    
    rng_calib = np.random.RandomState(101)
    
    for c_i, (pct, g_label) in enumerate(zip(calib_pcts, calib_grades)):
        ax = fig.add_subplot(gs_calib[0, c_i])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        
        # Draw background organ boundary (rounded rectangle representing leaf tissue)
        bg = patches.FancyBboxPatch((5, 5), 90, 90, boxstyle="round,pad=2", 
                                    facecolor="#E8F8F5", edgecolor="#16A085", linewidth=2)
        ax.add_patch(bg)
        
        # Simulate lesions totaling approx `pct` percent of total organ area
        # Total organ area = ~90x90 = 8100
        target_lesion_area = 8100 * (pct / 100.0)
        curr_area = 0.0
        
        # Draw scattered spots
        while curr_area < target_lesion_area:
            r = rng_calib.uniform(2.5, 5.5)
            area_blob = np.pi * (r ** 2)
            if curr_area + area_blob > target_lesion_area * 1.15:
                # adjust radius for exact match
                rem = max(0.5, target_lesion_area - curr_area)
                r = np.sqrt(rem / np.pi)
                area_blob = np.pi * (r ** 2)
            
            cx = rng_calib.uniform(15, 85)
            cy = rng_calib.uniform(15, 85)
            spot = patches.Circle((cx, cy), r, facecolor="#C0392B", edgecolor="#78281F", alpha=0.9)
            ax.add_patch(spot)
            curr_area += area_blob
            if curr_area >= target_lesion_area:
                break
                
        ax.set_title(f"{pct:.0f}% Coverage\n{g_label}", fontsize=8, fontweight='bold', pad=4)
        ax.axis('off')
        
    # Footer notice
    footer_text = (
        "CONFIDENTIALITY NOTICE: This evaluation sheet is strictly blind. Image identifiers, filenames, algorithmic predictions,\n"
        "and duplicated calibration controls are tracked solely in the central answer key to evaluate inter- and intra-rater concordance."
    )
    fig.text(0.5, 0.08, footer_text, ha='center', va='bottom', fontsize=8, fontstyle='italic', color='#7F8C8D')
    
    pdf.savefig(fig, dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # PAGES 2+: IMAGE SLOTS (6 SLOTS PER PAGE: 2 COLUMNS x 3 ROWS)
    # -------------------------------------------------------------
    slots_per_page = 6
    total_pages = int(np.ceil(len(slots_df) / slots_per_page))
    
    for page_i in range(total_pages):
        fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
        page_slots = slots_df.iloc[page_i * slots_per_page : (page_i + 1) * slots_per_page]
        
        for slot_idx_in_page, ax in enumerate(axes.flat):
            if slot_idx_in_page < len(page_slots):
                item = page_slots.iloc[slot_idx_in_page]
                img_path = item['image_path']
                try:
                    im = Image.open(img_path).convert('RGB')
                    ax.imshow(im)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center', fontsize=9)
                    
                slot_header = f"Slot #{item['slot_id']:03d}   [{item['crop_type']} — {item['disease_class']}]"
                ax.set_title(slot_header, fontsize=9.5, fontweight='bold', pad=4)
                
                # Grading checkboxes under the image
                caption = (
                    "Grade:       [  ] 0     [  ] 1     [  ] 2     [  ] 3\n"
                    "Confidence:  [  ] 1     [  ] 2     [  ] 3     [  ] 4     [  ] 5\n"
                    "Notes: __________________________________________"
                )
                ax.set_xlabel(caption, fontsize=8, family='monospace', labelpad=4)
            else:
                ax.axis('off')
                
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#BDC3C7')
                spine.set_linewidth(1)
                
        fig.suptitle(f"Blind Agricultural Disease Severity Grading Sheet — Page {page_i+2} of {total_pages+1}", 
                     fontsize=9.5, y=0.985, color='#555555')
        plt.tight_layout(rect=[0.04, 0.02, 0.96, 0.97])
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

print(f"\nSuccessfully generated {total_pages+1}-page PDF booklet: {pdf_path}")
print("STAGE 2c - PART E COMPLETE!")
