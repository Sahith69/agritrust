import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2e - PART D: REDRAW ARECANUT GOLD SAMPLE (CLUSTER SIZE <= 10, <= 2 PER CLUSTER)")
print("=" * 80)

# 1. Verify old lock file exists and leave it untouched
old_lock_path = 'outputs/final_v2/gold_lock.csv'
if not os.path.exists(old_lock_path):
    raise RuntimeError(f"Missing {old_lock_path}")
old_lock_df = pd.read_csv(old_lock_path)
print(f"Old lock file preserved at {old_lock_path} with {len(old_lock_df)} total locked images.")

# 2. Load Arecanut robust clusters
df_areca = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
c_sizes = df_areca.groupby('robust_cluster_id').size()
df_areca['cluster_size'] = df_areca['robust_cluster_id'].map(c_sizes)

# Filter for clusters of <= 10 images
df_eligible = df_areca[df_areca['cluster_size'] <= 10].copy()
print(f"Eligible Arecanut images (cluster size <= 10): {len(df_eligible)} / {len(df_areca)} ({len(df_eligible)/len(df_areca)*100:.1f}%)")
print(f"Eligible Arecanut clusters: {df_eligible['robust_cluster_id'].nunique()} / {df_areca['robust_cluster_id'].nunique()} ({df_eligible['robust_cluster_id'].nunique()/df_areca['robust_cluster_id'].nunique()*100:.1f}%)")

# Stratified sampling of clusters and images (Seed 42)
np.random.seed(42)

# Target allocation to reach exactly 97 images across classes:
# - stem_bleeding: 1 cluster * 2 = 2 images (all eligible clusters)
# - healthy_foot: 1 cluster * 2 = 2 images (all eligible clusters)
# - bud_borer: 4 clusters * 2 = 8 images (all eligible clusters)
# - stem_cracking: 6 clusters * 2 = 12 images
# - mahali_koleroga: 17 clusters * 2 = 34 images
# - yellow_leaf_disease: 8 clusters * 2 = 16 images
# - healthy_trunk: 8 clusters * (1 or 2) = 9 images
# - healthy_nut: 8 clusters * (1 or 2) = 8 images
# - healthy_leaf: 3 clusters * 2 = 6 images
target_clusters_per_class = {
    'stem_bleeding': 1,
    'healthy_foot': 1,
    'bud_borer': 4,
    'stem_cracking': 6,
    'mahali_koleroga': 17,
    'yellow_leaf_disease': 8,
    'healthy_trunk': 8,
    'healthy_nut': 8,
    'healthy_leaf': 3
}

sampled_gold_indices = []
sampled_clusters = set()

for cls in sorted(target_clusters_per_class.keys()):
    n_clus_target = target_clusters_per_class[cls]
    cls_pool = df_eligible[df_eligible['disease_class'] == cls]
    available_clusters = sorted(cls_pool['robust_cluster_id'].unique())
    
    selected_c = np.random.choice(available_clusters, size=min(n_clus_target, len(available_clusters)), replace=False)
    
    for c_id in selected_c:
        sampled_clusters.add(c_id)
        c_images = cls_pool[cls_pool['robust_cluster_id'] == c_id]
        
        # Sample at most 2 images from this cluster
        n_sample = min(2, len(c_images))
        sampled_rows = c_images.sample(n=n_sample, random_state=42)
        sampled_gold_indices.extend(sampled_rows.index.tolist())

# Adjust to exactly 97 if slight variation due to 1-image clusters
if len(sampled_gold_indices) < 97:
    needed = 97 - len(sampled_gold_indices)
    # Add from healthy_nut or healthy_trunk singleton clusters
    remaining_nut_clus = [c for c in df_eligible[df_eligible['disease_class'] == 'healthy_nut']['robust_cluster_id'].unique() if c not in sampled_clusters]
    for c_id in remaining_nut_clus[:needed]:
        sampled_clusters.add(c_id)
        c_images = df_eligible[df_eligible['robust_cluster_id'] == c_id]
        sampled_gold_indices.extend(c_images.sample(n=1, random_state=42).index.tolist())

df_gold_areca = df_areca.loc[sampled_gold_indices].copy()
print(f"\nRedrawn Arecanut Gold Sample: {len(df_gold_areca)} images across {len(sampled_clusters)} clusters.")

# Find all cluster mates of the sampled clusters
all_locked_areca_mask = df_areca['robust_cluster_id'].isin(sampled_clusters)
df_locked_areca = df_areca[all_locked_areca_mask].copy()

print(f"Total Arecanut images locked (Gold + all cluster mates): {len(df_locked_areca)} images.")
print(f"Lock size reduction: From 917 images (Old) to {len(df_locked_areca)} images (New) - freed {917 - len(df_locked_areca)} images!")

# Class breakdown of redrawn gold sample and locked set
print("\n--- CLASS BREAKDOWN OF REDRAWN GOLD SAMPLE & NEW LOCK ---")
class_summary = {}
for cls in sorted(df_areca['disease_class'].unique()):
    n_gold = len(df_gold_areca[df_gold_areca['disease_class'] == cls])
    n_locked = len(df_locked_areca[df_locked_areca['disease_class'] == cls])
    n_clus_locked = df_locked_areca[df_locked_areca['disease_class'] == cls]['robust_cluster_id'].nunique()
    total_in_clean = len(df_areca[df_areca['disease_class'] == cls])
    pct_locked = (n_locked / total_in_clean) * 100.0
    
    class_summary[cls] = {
        'gold_images': int(n_gold),
        'locked_images': int(n_locked),
        'locked_clusters': int(n_clus_locked),
        'total_dataset_images': int(total_in_clean),
        'share_locked_pct': float(pct_locked)
    }
    print(f"  • {cls:22}: Gold={n_gold:2d} | Locked={n_locked:3d} ({pct_locked:5.1f}% of {total_in_clean}) across {n_clus_locked:2d} clusters")

# Paddy locked images from old lock (53 gold + 7 twins at Hamming <= 8)
old_paddy_locked = old_lock_df[old_lock_df['crop_type'] == 'Paddy'].copy()
print(f"\nPreserving Paddy locked set: {len(old_paddy_locked)} images (53 gold + 7 twins at Hamming <= 8).")

# Prepare new lock dataframe
df_locked_areca['crop_type'] = 'Arecanut'
df_locked_areca['is_direct_gold'] = df_locked_areca.index.isin(sampled_gold_indices)
df_locked_areca['lock_reason'] = df_locked_areca['is_direct_gold'].map(lambda x: 'direct_gold_sample' if x else 'cluster_mate_burst')

new_lock_rows = []
for idx, r in df_locked_areca.iterrows():
    new_lock_rows.append({
        'crop_type': 'Arecanut',
        'disease_class': r['disease_class'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'is_direct_gold': r['is_direct_gold'],
        'lock_reason': r['lock_reason'],
        'cluster_id': r['robust_cluster_id']
    })

for idx, r in old_paddy_locked.iterrows():
    new_lock_rows.append({
        'crop_type': 'Paddy',
        'disease_class': r['disease_class'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'is_direct_gold': r['is_direct_gold'],
        'lock_reason': r['lock_reason'],
        'cluster_id': r.get('cluster_or_component_id', -1)
    })

df_new_lock = pd.DataFrame(new_lock_rows)
new_lock_path = 'outputs/final_v2/gold_lock_v2.csv'
df_new_lock.to_csv(new_lock_path, index=False)
print(f"\nSaved new lock file to: {new_lock_path}")
print(f"Total images locked in v2: {len(df_new_lock)} (Arecanut: {len(df_locked_areca)}, Paddy: {len(old_paddy_locked)})")

summary_v2 = {
    'old_lock_path': old_lock_path,
    'old_total_locked': len(old_lock_df),
    'old_arecanut_locked': int((old_lock_df['crop_type'] == 'Arecanut').sum()),
    'new_lock_path': new_lock_path,
    'new_total_locked': len(df_new_lock),
    'new_arecanut_locked': len(df_locked_areca),
    'new_paddy_locked': len(old_paddy_locked),
    'new_arecanut_gold_sample_size': len(df_gold_areca),
    'new_arecanut_clusters_locked': int(len(sampled_clusters)),
    'freed_arecanut_images_count': int(917 - len(df_locked_areca)),
    'arecanut_class_breakdown': class_summary
}

with open('outputs/final_v2/gold_lock_v2_summary.json', 'w') as f:
    json.dump(summary_v2, f, indent=2)

print("Saved summary to outputs/final_v2/gold_lock_v2_summary.json")
