import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2d - SECTION E: GOLD LOCK SPECIFICATION")
print("=" * 80)

# 1. Load v3 answer key
key_v3_path = 'outputs/final_v2/human_grading_answer_key_v3.json'
if not os.path.exists(key_v3_path):
    # Fall back to v2 if v3 not yet generated
    key_v3_path = 'outputs/final_v2/human_grading_answer_key_v2.json'

with open(key_v3_path) as f:
    key_data = json.load(f)

slots_df = pd.DataFrame(key_data['slots'])
# Filter to 150 unique images (drop repeats to avoid duplicate expansion)
gold_unique = slots_df[~slots_df['is_repeat']].copy().reset_index(drop=True)
print(f"Loaded {len(gold_unique)} unique gold images from {key_v3_path}")

# Breakdown by crop
gold_areca = gold_unique[gold_unique['crop_type'] == 'Arecanut'].copy()
gold_paddy = gold_unique[gold_unique['crop_type'] == 'Paddy'].copy()

print(f"Gold Unique Images: Arecanut = {len(gold_areca)}, Paddy = {len(gold_paddy)}")

# ------------------------------------------------------------------------------
# 2. EXPAND ARECANUT: ALL CLUSTER MATES
# ------------------------------------------------------------------------------
# Load robust cluster assignments
areca_clean = pd.read_csv('outputs/final_v2/arecanut_clean_v2.csv')
if os.path.exists('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv'):
    areca_robust = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
    cluster_col = 'robust_cluster_id'
else:
    areca_robust = areca_clean
    cluster_col = 'cluster_id'

print(f"\nExpanding Arecanut cluster mates using column '{cluster_col}'...")
gold_areca_filenames = set(gold_areca['filename'])
# Find the robust cluster IDs corresponding to the gold Arecanut filenames
gold_matched = areca_robust[areca_robust['filename'].isin(gold_areca_filenames)]
gold_clusters = gold_matched[cluster_col].unique()
print(f"Matched {len(gold_matched)} / {len(gold_areca)} gold Arecanut images, covering {len(gold_clusters)} unique robust clusters.")

# Find all images in areca_robust belonging to these clusters
locked_areca_df = areca_robust[areca_robust[cluster_col].isin(gold_clusters)].copy()
locked_areca_df['lock_reason'] = 'arecanut_gold_cluster_mate'
locked_areca_df['is_direct_gold'] = locked_areca_df['filename'].isin(gold_areca_filenames)

print(f"Arecanut Locked Images: {len(locked_areca_df)} total across {locked_areca_df[cluster_col].nunique()} clusters")
print(f"  • Direct gold images: {locked_areca_df['is_direct_gold'].sum()}")
print(f"  • Additional cluster mates locked: {len(locked_areca_df) - locked_areca_df['is_direct_gold'].sum()}")

# ------------------------------------------------------------------------------
# 3. EXPAND PADDY: ALL PHASH <= 8 TWINS
# ------------------------------------------------------------------------------
print("\nExpanding Paddy twins at pHash Hamming <= 8...")
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
n_paddy = len(df_paddy)

# Load cached Paddy pHashes
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

# Find row indices of gold Paddy images in df_paddy
gold_paddy_filenames = set(gold_paddy['filename'])
gold_paddy_indices = df_paddy[df_paddy['filename'].isin(gold_paddy_filenames)].index.values
print(f"Found {len(gold_paddy_indices)} gold Paddy images in df_paddy.")

# Search all 10,053 images for any match with gold_paddy_indices at Hamming <= 8
h_gold = hashes_orig[gold_paddy_indices]

locked_paddy_indices = set(gold_paddy_indices)
chunk = 2000
for b_start in range(0, n_paddy, chunk):
    b_end = min(b_start + chunk, n_paddy)
    
    xor1 = h_gold[:, None] ^ hashes_orig[b_start:b_end][None, :]
    xor2 = h_gold[:, None] ^ hashes_hflip[b_start:b_end][None, :]
    xor3 = h_gold[:, None] ^ hashes_vflip[b_start:b_end][None, :]
    
    d1 = fast_popcount_matrix(xor1)
    d2 = fast_popcount_matrix(xor2)
    d3 = fast_popcount_matrix(xor3)
    min_d = np.minimum(d1, np.minimum(d2, d3))
    
    hits = np.where(min_d <= 8)
    for c_j in hits[1]:
        locked_paddy_indices.add(b_start + c_j)

locked_paddy_df = df_paddy.iloc[list(locked_paddy_indices)].copy()
locked_paddy_df['crop_type'] = 'Paddy'
locked_paddy_df['lock_reason'] = 'paddy_gold_phash_le8_twin'
locked_paddy_df['is_direct_gold'] = locked_paddy_df['filename'].isin(gold_paddy_filenames)

print(f"Paddy Locked Images: {len(locked_paddy_df)} total")
print(f"  • Direct gold images: {locked_paddy_df['is_direct_gold'].sum()}")
print(f"  • Additional <=8 twins locked: {len(locked_paddy_df) - locked_paddy_df['is_direct_gold'].sum()}")
print(f"  • Split breakdown of locked Paddy images: {locked_paddy_df['split'].value_counts().to_dict()}")

# ------------------------------------------------------------------------------
# 4. CONSOLIDATE AND WRITE outputs/final_v2/gold_lock.csv
# ------------------------------------------------------------------------------
cols_lock = ['crop_type', 'disease_class', 'filename', 'image_path', 'is_direct_gold', 'lock_reason']
areca_lock_clean = locked_areca_df.copy()
areca_lock_clean['crop_type'] = 'Arecanut'

gold_lock_df = pd.concat([
    areca_lock_clean[cols_lock],
    locked_paddy_df[cols_lock]
]).reset_index(drop=True)

gold_lock_path = 'outputs/final_v2/gold_lock.csv'
gold_lock_df.to_csv(gold_lock_path, index=False)
print(f"\nSaved consolidated Gold Lock file: {gold_lock_path} (N = {len(gold_lock_df)})")

# Summary breakdown
summary = {
    'total_locked_images': len(gold_lock_df),
    'direct_gold_images': int(gold_lock_df['is_direct_gold'].sum()),
    'additional_cluster_mates_or_twins': int((~gold_lock_df['is_direct_gold']).sum()),
    'by_crop': {
        'Arecanut': {
            'total': len(areca_lock_clean),
            'direct_gold': int(areca_lock_clean['is_direct_gold'].sum()),
            'additional_mates': int((~areca_lock_clean['is_direct_gold']).sum()),
            'clusters_locked': int(areca_lock_clean[cluster_col].nunique()),
            'by_class': areca_lock_clean['disease_class'].value_counts().to_dict()
        },
        'Paddy': {
            'total': len(locked_paddy_df),
            'direct_gold': int(locked_paddy_df['is_direct_gold'].sum()),
            'additional_twins': int((~locked_paddy_df['is_direct_gold']).sum()),
            'by_split': locked_paddy_df['split'].value_counts().to_dict(),
            'by_class': locked_paddy_df['disease_class'].value_counts().to_dict()
        }
    }
}

with open('outputs/final_v2/gold_lock_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSaved Gold Lock summary to outputs/final_v2/gold_lock_summary.json")
