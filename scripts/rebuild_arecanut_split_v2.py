"""
Fix #1: Rebuild Arecanut train/val/test split using robust_cluster_id as grouping key.
- Eliminates the 54-cluster train/test leak found in the audit.
- Preserves the same overall ~64/18/18 ratio per class (stratified).
- Outputs: outputs/arecanut_splits_v2.csv  (old file untouched)
- Reports: before vs after split stats, leak check on both.
"""
import sys, os, json
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
df_clusters = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
print(f"Total images: {len(df_clusters)}")
print(f"Columns: {df_clusters.columns.tolist()}")
print(f"Classes: {sorted(df_clusters['disease_class'].unique())}")
print(f"Unique robust_cluster_id: {df_clusters['robust_cluster_id'].nunique()}")
print()

# ── Verify old split leak ─────────────────────────────────────────────────────
old_splits = pd.read_csv('outputs/arecanut_splits.csv')
merged_old = old_splits.merge(df_clusters[['filename','robust_cluster_id']], on='filename', how='left')

old_train_rc = set(merged_old[merged_old['split']=='train']['robust_cluster_id'].dropna().unique())
old_val_rc   = set(merged_old[merged_old['split']=='val']['robust_cluster_id'].dropna().unique())
old_test_rc  = set(merged_old[merged_old['split']=='test']['robust_cluster_id'].dropna().unique())
old_leak_tt  = old_train_rc & old_test_rc
old_leak_vt  = old_val_rc   & old_test_rc

print("=== OLD SPLIT (arecanut_splits.csv) ===")
print(f"  Train: {(old_splits['split']=='train').sum()}  Val: {(old_splits['split']=='val').sum()}  Test: {(old_splits['split']=='test').sum()}")
print(f"  robust_cluster_id: train∩test={len(old_leak_tt)} clusters | val∩test={len(old_leak_vt)} clusters")
n_leaked_test = merged_old[(merged_old['split']=='test') & (merged_old['robust_cluster_id'].isin(old_leak_tt))].shape[0]
print(f"  Leaked test images: {n_leaked_test}/502 = {n_leaked_test/502*100:.1f}%")
print()

# ── Build new split using robust_cluster_id as group key ─────────────────────
# Strategy:
#   1. Per-class stratification using robust_cluster_id groups
#   2. Test = 18% of each class, Val = 18% of remaining, Train = rest
#   3. Groups kept intact (no cluster spans splits)

df = df_clusters.copy()
df['new_split'] = ''

all_classes = sorted(df['disease_class'].unique())
split_records = []

for cls in all_classes:
    cls_df = df[df['disease_class'] == cls].copy()
    cls_groups = cls_df['robust_cluster_id'].values
    unique_groups = cls_df['robust_cluster_id'].unique()
    n_groups = len(unique_groups)
    
    # Shuffle groups
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(unique_groups)
    
    # Split groups: ~18% test, ~18% val, rest train (by group count)
    n_test  = max(1, round(n_groups * 0.18))
    n_val   = max(1, round(n_groups * 0.18))
    n_train = n_groups - n_test - n_val
    
    test_groups  = set(shuffled[:n_test])
    val_groups   = set(shuffled[n_test:n_test+n_val])
    train_groups = set(shuffled[n_test+n_val:])
    
    for _, row in cls_df.iterrows():
        g = row['robust_cluster_id']
        if g in test_groups:
            split_label = 'test'
        elif g in val_groups:
            split_label = 'val'
        else:
            split_label = 'train'
        split_records.append({'filename': row['filename'], 'new_split': split_label})

    n_tr = sum(1 for r in split_records[-len(cls_df):] if r['new_split']=='train')
    n_vl = sum(1 for r in split_records[-len(cls_df):] if r['new_split']=='val')
    n_te = sum(1 for r in split_records[-len(cls_df):] if r['new_split']=='test')
    print(f"  {cls:<30}  groups={n_groups}  train={n_tr}  val={n_vl}  test={n_te}")

split_map = pd.DataFrame(split_records).set_index('filename')
df_new = df.copy()
df_new['split'] = df_new['filename'].map(split_map['new_split'])

# ── Verify new split has zero leak ───────────────────────────────────────────
# ── Handle cross-class cluster (robust_cluster_id=74 spans mahali_koleroga + healthy_leaf)
# Force all images in that cluster to 'train' to eliminate the ambiguity.
cross_class_clusters = df_clusters.groupby('robust_cluster_id')['disease_class'].nunique()
cross_class_ids = set(cross_class_clusters[cross_class_clusters > 1].index.tolist())
if cross_class_ids:
    print(f"  Cross-class clusters (forced to train): {cross_class_ids}")
    mask = df_new['robust_cluster_id'].isin(cross_class_ids)
    df_new.loc[mask, 'split'] = 'train'
    print(f"  Images moved to train: {mask.sum()}")

new_train_rc = set(df_new[df_new['split']=='train']['robust_cluster_id'].unique())
new_val_rc   = set(df_new[df_new['split']=='val']['robust_cluster_id'].unique())
new_test_rc  = set(df_new[df_new['split']=='test']['robust_cluster_id'].unique())
new_leak_tt  = new_train_rc & new_test_rc
new_leak_vt  = new_val_rc   & new_test_rc

print()
print("=== NEW SPLIT (robust_cluster_id-based) ===")
print(f"  Train: {(df_new['split']=='train').sum()}  Val: {(df_new['split']=='val').sum()}  Test: {(df_new['split']=='test').sum()}")
print(f"  robust_cluster_id: train∩test={len(new_leak_tt)} clusters | val∩test={len(new_leak_vt)} clusters")
assert len(new_leak_tt) == 0, f"LEAK NOT FIXED: {len(new_leak_tt)} clusters still overlap train/test"
assert len(new_leak_vt) == 0, f"LEAK NOT FIXED: {len(new_leak_vt)} clusters still overlap val/test"
print("  ✓ ZERO LEAK CONFIRMED")
print()

# ── Per-class breakdown ───────────────────────────────────────────────────────
print("Per-class split distribution (new):")
for cls in all_classes:
    sub = df_new[df_new['disease_class']==cls]
    tr = (sub['split']=='train').sum()
    vl = (sub['split']=='val').sum()
    te = (sub['split']=='test').sum()
    tot = len(sub)
    print(f"  {cls:<30}  total={tot:>4}  train={tr:>4} ({tr/tot*100:.0f}%)  val={vl:>3} ({vl/tot*100:.0f}%)  test={te:>3} ({te/tot*100:.0f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
# Keep same columns as old arecanut_splits.csv + robust_cluster_id
out_cols = ['image_path','filename','disease_class','severity_grade','parent_id',
            'split','cluster_id','robust_cluster_id']
# Keep only cols that exist
out_cols = [c for c in out_cols if c in df_new.columns]
df_out = df_new[out_cols].copy()
df_out.to_csv('outputs/arecanut_splits_v2.csv', index=False)
print()
print(f"✓ Saved: outputs/arecanut_splits_v2.csv  ({len(df_out)} rows)")
print(f"  Old file untouched: outputs/arecanut_splits.csv")

# ── Summary JSON ─────────────────────────────────────────────────────────────
summary = {
    'old_split': {
        'train': int((old_splits['split']=='train').sum()),
        'val':   int((old_splits['split']=='val').sum()),
        'test':  int((old_splits['split']=='test').sum()),
        'robust_cluster_train_test_leak': int(len(old_leak_tt)),
        'leaked_test_images': int(n_leaked_test),
    },
    'new_split': {
        'train': int((df_new['split']=='train').sum()),
        'val':   int((df_new['split']=='val').sum()),
        'test':  int((df_new['split']=='test').sum()),
        'robust_cluster_train_test_leak': 0,
        'leaked_test_images': 0,
    }
}
with open('outputs/final_v2/arecanut_split_v2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved: outputs/final_v2/arecanut_split_v2_summary.json")
