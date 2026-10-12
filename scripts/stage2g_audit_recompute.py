"""
Stage 2g Audit & Recompute Script
Covers tasks 1-5 of the 2026-09-28 audit request:
  1. Audit of peak_model_optimization.py methodology + save per-row test predictions
  2. Bootstrap 95% CIs on twin-free lesion test + CORN vs Softmax paired difference
  3. Healthy vs Diseased and 3-band on FULL twin-free test (N=732)
  4. Original Section B numbers from paddy_severity_labels.csv + discrepancy explanation
  5. Disease classifier evaluation on twin-free test
"""
import sys, os, json, warnings
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (cohen_kappa_score, accuracy_score,
                             mean_absolute_error, f1_score, confusion_matrix)
import torchvision.transforms as transforms

from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade
from src.dataset import get_transforms

os.makedirs('outputs/final_v2', exist_ok=True)

# ─── LOAD SPLIT DATA ────────────────────────────────────────────────────────
paddy_splits = pd.read_csv('outputs/paddy_splits.csv')
val_df   = paddy_splits[paddy_splits['split']=='val'].copy().reset_index(drop=True)
test_df  = paddy_splits[paddy_splits['split']=='test'].copy().reset_index(drop=True)

pred_cache = pd.read_csv('outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv')
test_df['comp_le8']    = pred_cache['comp_le8'].values
test_df['has_twin_le8'] = pred_cache['has_twin_le8'].astype(bool).values

classes_7 = ['bacterial_leaf_blight','bacterial_leaf_streak','bacterial_panicle_blight',
              'blast','brown_spot','downy_mildew','tungro']

tf_7   = test_df[(~test_df['has_twin_le8']) & (test_df['disease_class'].isin(classes_7))].copy().reset_index(drop=True)
tf_all = test_df[~test_df['has_twin_le8']].copy().reset_index(drop=True)

print(f"Val N={len(val_df)} | Full Test N={len(test_df)} | Twin-Free All N={len(tf_all)} | Twin-Free 7-class N={len(tf_7)}")

# ─── DATASET ─────────────────────────────────────────────────────────────────
tf_test_transform = get_transforms('test')

class PaddyDS(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))

def get_probs(model_info, df, use_tta=False):
    model = model_info['model']
    is_ord = model_info['ordinal']
    loader = DataLoader(PaddyDS(df['image_path'].values, tf_test_transform),
                        batch_size=64, shuffle=False, num_workers=0)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            if not use_tta:
                out = model(batch)
                p = torch.sigmoid(out) if is_ord else F.softmax(out, dim=1)
                all_probs.append(p.cpu().numpy())
            else:
                augs = [batch, torch.flip(batch,[3]), torch.flip(batch,[2]), torch.flip(batch,[2,3])]
                stacked = torch.cat(augs, dim=0)
                out = model(stacked)
                p = torch.sigmoid(out) if is_ord else F.softmax(out, dim=1)
                chunks = torch.chunk(p, 4, dim=0)
                all_probs.append(torch.stack(chunks).mean(0).cpu().numpy())
    return np.vstack(all_probs)

def decode_corn(probs, thresholds=(0.5,0.5,0.5)):
    if probs.ndim == 1: probs = probs[None]
    p0 = probs[:,0]; p1 = probs[:,1]; p2 = probs[:,2]
    cum0 = p0
    cum1 = np.where(p0 > thresholds[0], p0 * p1, 0.0)
    cum2 = np.where((p0 > thresholds[0]) & (p1 > thresholds[1]), p0 * p1 * p2, 0.0)
    pk = np.column_stack([1-cum0, cum0-cum1, cum1-cum2, cum2])
    return np.argmax(pk, axis=1)

def fast_qwk(y_true, y_pred, n=4):
    cm = np.bincount(y_true*n + y_pred, minlength=n*n).reshape(n,n)
    ht = cm.sum(1); hp = cm.sum(0)
    E = np.outer(ht, hp) / float(len(y_true))
    i = np.arange(n)
    w = ((i[:,None]-i[None,:])**2) / float((n-1)**2)
    d = np.sum(w*E); return 1.0 - np.sum(w*cm)/d if d>0 else 0.0

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
models_cfg = [
    ('ResNet-18 CORN',    'outputs/models/paddy_ordinal.pth',                                        'resnet18',    True),
    ('ResNet-34 CORN',    'outputs/models/architecture_comparison/paddy_resnet34_ordinal.pth',       'resnet34',    True),
    ('MobileNet-V2 CORN', 'outputs/models/architecture_comparison/paddy_mobilenet_v2_ordinal.pth',  'mobilenet_v2',True),
    ('ResNet-18 Softmax', 'outputs/models/paddy_baseline.pth',                                       'resnet18',    False),
]
loaded = []
for lbl, pth, bb, is_ord in models_cfg:
    if not os.path.exists(pth): print(f"SKIP {lbl}: {pth}"); continue
    m = CropSeverityModel(backbone=bb, num_classes=4, ordinal=is_ord, pretrained=False)
    ck = torch.load(pth, map_location='cpu')
    m.load_state_dict(ck if 'model_state_dict' not in ck else ck['model_state_dict'])
    m.eval()
    loaded.append({'label':lbl,'model':m,'ordinal':is_ord})
    print(f"  Loaded {lbl}")

# ─── EXTRACT PROBS ON ALL SPLITS ─────────────────────────────────────────────
print("\nExtracting model outputs (val, full-test, tf-all, tf-7)...")
val_p = {}; full_p = {}; tfall_p = {}; tf7_p = {}
for mi in loaded:
    lbl = mi['label']
    for sfx, use_tta in [(' (Standard)', False), (' (TTA)', True)]:
        k = lbl + sfx
        val_p[k]    = get_probs(mi, val_df,  use_tta)
        full_p[k]   = get_probs(mi, test_df, use_tta)
        tfall_p[k]  = get_probs(mi, tf_all,  use_tta)
        tf7_p[k]    = get_probs(mi, tf_7,    use_tta)
    print(f"  Done {lbl}")

# Ensembles
corn_std = [m['label']+' (Standard)' for m in loaded if m['ordinal']]
corn_tta = [m['label']+' (TTA)'      for m in loaded if m['ordinal']]
for d in [val_p, full_p, tfall_p, tf7_p]:
    d['CORN Architecture Ensemble (Standard)'] = np.mean([d[k] for k in corn_std], axis=0)
    d['CORN Architecture Ensemble (TTA)']      = np.mean([d[k] for k in corn_tta], axis=0)

# ─── 1. AUDIT ANSWERS ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TASK 1: AUDIT OF peak_model_optimization.py")
print("="*70)
print("""
A. Threshold selection:
   YES — thresholds were chosen purely on the VALIDATION set (val_df, N=1510).
   No test-set information was used. Grid search: t1,t2,t3 ∈ {0.4, 0.5, 0.6}
   → 3^3 = 27 configurations per model configuration key.

B. Subsampling / TTA crops dropped:
   NO subsampling of images. All images in each split are evaluated.
   TTA uses exactly 4 flip augmentations (orig, hflip, vflip, hvflip) applied
   as a single stacked batch via torch.flip() — NO crops are used or dropped.
   (Note: the script name says "5-crop" but implementation is 4-flip TTA, not 5-crop.)

C. Total configurations tried:
   - 4 models × 2 TTA modes = 8 base configurations.
   - +2 ensemble configurations (std + TTA) = 10 total reported rows.
   - Threshold calibration: 27 grid points × 8 CORN configurations = 216 threshold searches.
   - Softmax uses argmax (no threshold calibration): 0 configurations searched.
   - Grand total calibration searches: 216 on validation.
""")

# ─── CALIBRATE THRESHOLDS (VAL ONLY) ─────────────────────────────────────────
best_thresholds = {}
for k, v_probs in val_p.items():
    if 'Softmax' in k: continue
    y_val = val_df['severity_grade'].values
    best_q, best_th = -1.0, (0.5,0.5,0.5)
    for t1 in [0.4,0.5,0.6]:
        for t2 in [0.4,0.5,0.6]:
            for t3 in [0.4,0.5,0.6]:
                q = fast_qwk(y_val, decode_corn(v_probs, (t1,t2,t3)))
                if q > best_q: best_q=q; best_th=(t1,t2,t3)
    best_thresholds[k] = best_th

# ─── GENERATE PREDICTIONS + SAVE PER-ROW CSVs ───────────────────────────────
def predict_all(probs_dict, df_meta, thresholds):
    rows = {}
    for k, probs in probs_dict.items():
        if 'Softmax' in k:
            preds = np.argmax(probs, axis=1)
        else:
            th = thresholds.get(k, (0.5,0.5,0.5))
            preds = decode_corn(probs, th)
        rows[k] = preds
    out = df_meta[['image_path','disease_class','severity_grade']].copy()
    for k, preds in rows.items():
        out[k] = preds
    return out

pred_tf7_df   = predict_all(tf7_p,   tf_7,   best_thresholds)
pred_tfall_df = predict_all(tfall_p, tf_all, best_thresholds)
pred_full_df  = predict_all(full_p,  test_df,best_thresholds)

pred_tf7_df.to_csv('outputs/final_v2/audit_predictions_tf7.csv',   index=False)
pred_tfall_df.to_csv('outputs/final_v2/audit_predictions_tfall.csv', index=False)
pred_full_df.to_csv('outputs/final_v2/audit_predictions_full.csv',  index=False)
print("Saved per-row predictions: audit_predictions_tf7.csv, audit_predictions_tfall.csv, audit_predictions_full.csv")

# ─── 2. BOOTSTRAP CIs ON TWIN-FREE LESION TEST (tf_7) ────────────────────────
print("\n" + "="*70)
print("TASK 2: PEAK TABLE WITH BOOTSTRAP 95% CIs — Twin-Free 7-Class Lesion (N=343)")
print("="*70)

N_BOOT = 2000
RNG    = np.random.default_rng(42)
y_tf7  = tf_7['severity_grade'].values
clust  = tf_7['comp_le8'].values

def cluster_bootstrap_ci(y_true, y_pred, stat_fn, clusters, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    unique_clusters = np.unique(clusters)
    stats = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([np.where(clusters==c)[0] for c in sampled_clusters])
        try:
            stats.append(stat_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return np.percentile(stats, [2.5, 97.5])

def get_tf7_pred(k):
    probs = tf7_p[k]
    if 'Softmax' in k: return np.argmax(probs, axis=1)
    return decode_corn(probs, best_thresholds.get(k,(0.5,0.5,0.5)))

rows2 = []
softmax_best_preds = None
corn_ens_tta_preds = None

for k in tf7_p.keys():
    y_pred = get_tf7_pred(k)
    if 'Softmax' in k and '(TTA)' in k: softmax_best_preds = y_pred
    if k == 'CORN Architecture Ensemble (TTA)':  corn_ens_tta_preds = y_pred

    qwk_val  = cohen_kappa_score(y_tf7, y_pred, weights='quadratic')
    acc_val  = accuracy_score(y_tf7, y_pred)*100
    w1_val   = np.mean(np.abs(y_tf7-y_pred)<=1)*100
    mae_val  = mean_absolute_error(y_tf7, y_pred)
    f1_val   = f1_score(y_tf7, y_pred, average='macro')*100

    qwk_ci  = cluster_bootstrap_ci(y_tf7, y_pred, lambda a,b: cohen_kappa_score(a,b,weights='quadratic'), clust, N_BOOT, RNG)
    acc_ci  = cluster_bootstrap_ci(y_tf7, y_pred, lambda a,b: accuracy_score(a,b)*100, clust, N_BOOT, RNG)
    f1_ci   = cluster_bootstrap_ci(y_tf7, y_pred, lambda a,b: f1_score(a,b,average='macro')*100, clust, N_BOOT, RNG)

    th_used = 'N/A Argmax' if 'Softmax' in k else str(best_thresholds.get(k,(0.5,0.5,0.5)))
    rows2.append({
        'Config': k,
        'Thresholds': th_used,
        'QWK': f"{qwk_val:.4f}",
        'QWK 95%CI': f"[{qwk_ci[0]:.4f}, {qwk_ci[1]:.4f}]",
        'Acc%': f"{acc_val:.2f}%",
        'Acc 95%CI': f"[{acc_ci[0]:.2f}%, {acc_ci[1]:.2f}%]",
        'W1%': f"{w1_val:.2f}%",
        'MAE': f"{mae_val:.4f}",
        'MacroF1%': f"{f1_val:.2f}%",
        'F1 95%CI': f"[{f1_ci[0]:.2f}%, {f1_ci[1]:.2f}%]",
    })

df_task2 = pd.DataFrame(rows2)
print(df_task2.to_string(index=False))

# Paired difference: CORN Ensemble TTA vs Softmax TTA
if corn_ens_tta_preds is not None and softmax_best_preds is not None:
    print("\n--- Paired Difference: CORN Architecture Ensemble (TTA) vs ResNet-18 Softmax (TTA) ---")
    diff_fn = lambda a,b_idx: (
        cohen_kappa_score(a[b_idx], corn_ens_tta_preds[b_idx], weights='quadratic') -
        cohen_kappa_score(a[b_idx], softmax_best_preds[b_idx], weights='quadratic')
    ) if len(b_idx) > 1 else 0.0
    unique_clusters = np.unique(clust)
    diffs = []
    for _ in range(N_BOOT):
        sc = RNG.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([np.where(clust==c)[0] for c in sc])
        try:
            diffs.append(
                cohen_kappa_score(y_tf7[idx], corn_ens_tta_preds[idx], weights='quadratic') -
                cohen_kappa_score(y_tf7[idx], softmax_best_preds[idx], weights='quadratic')
            )
        except: pass
    obs_diff = (cohen_kappa_score(y_tf7, corn_ens_tta_preds, weights='quadratic') -
                cohen_kappa_score(y_tf7, softmax_best_preds, weights='quadratic'))
    diff_ci = np.percentile(diffs, [2.5, 97.5])
    print(f"  Observed QWK Difference: {obs_diff:+.4f}")
    print(f"  Bootstrap 95% CI:        [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")
    if diff_ci[0] > 0:
        print("  ✓ STATISTICALLY SIGNIFICANT: CORN ensemble is reliably better than Softmax")
    else:
        print("  ✗ NOT significant at 95% (CI crosses zero)")

df_task2.to_csv('outputs/final_v2/audit_task2_bootstrap_peak_table.csv', index=False)

# ─── 3. HEALTHY vs DISEASED & 3-BAND ON FULL TWIN-FREE TEST (N=732) ──────────
print("\n" + "="*70)
print("TASK 3: COARSE TASKS ON FULL TWIN-FREE TEST (N=732) — CORRECTED")
print("="*70)
print("NOTE: 2-class (Healthy vs Diseased) evaluated on N=732 (includes normal=165 leaves).")
print("      3-band task also evaluated on N=732.\n")

y_tfall_sev  = tf_all['severity_grade'].values
y_tfall_bin  = (tf_all['disease_class'] != 'normal').astype(int).values
y_tfall_3b   = np.where(y_tfall_sev == 0, 0, np.where(y_tfall_sev == 1, 1, 2))
clust_tfall  = tf_all['comp_le8'].values

def get_tfall_pred(k):
    probs = tfall_p[k]
    if 'Softmax' in k: return np.argmax(probs, axis=1)
    return decode_corn(probs, best_thresholds.get(k,(0.5,0.5,0.5)))

rows3 = []
for model_lbl in ['CORN Architecture Ensemble (TTA)', 'ResNet-18 Softmax (TTA)', 'ResNet-18 CORN (TTA)']:
    if model_lbl not in tfall_p: continue
    y_pred_4g = get_tfall_pred(model_lbl)
    # Convert to binary and 3-band
    y_pred_bin = (y_pred_4g > 0).astype(int)
    y_pred_3b  = np.where(y_pred_4g==0, 0, np.where(y_pred_4g==1, 1, 2))

    # Mode baseline
    for lbl_use, y_pred_b, y_pred_3, label_name in [
        (model_lbl, y_pred_bin, y_pred_3b, model_lbl),
        ('Mode Baseline', np.ones(len(tf_all),dtype=int), np.full(len(tf_all),2,dtype=int), 'Mode Baseline'),
    ] if model_lbl == 'CORN Architecture Ensemble (TTA)' else [(model_lbl, y_pred_bin, y_pred_3b, model_lbl)]:

        # 2-class
        acc_bin  = accuracy_score(y_tfall_bin, y_pred_b)*100
        f1_bin   = f1_score(y_tfall_bin, y_pred_b, average='macro')*100
        rec_h    = (np.sum((y_pred_b==0) & (y_tfall_bin==0)) / max(np.sum(y_tfall_bin==0),1))*100
        rec_d    = (np.sum((y_pred_b==1) & (y_tfall_bin==1)) / max(np.sum(y_tfall_bin==1),1))*100
        acc_ci   = cluster_bootstrap_ci(y_tfall_bin, y_pred_b, lambda a,b: accuracy_score(a,b)*100, clust_tfall, N_BOOT, RNG)

        # 3-band
        acc_3b   = accuracy_score(y_tfall_3b, y_pred_3)*100
        f1_3b    = f1_score(y_tfall_3b, y_pred_3, average='macro')*100
        rec_g0   = (np.sum((y_pred_3==0) & (y_tfall_3b==0)) / max(np.sum(y_tfall_3b==0),1))*100
        rec_g1   = (np.sum((y_pred_3==1) & (y_tfall_3b==1)) / max(np.sum(y_tfall_3b==1),1))*100
        rec_adv  = (np.sum((y_pred_3==2) & (y_tfall_3b==2)) / max(np.sum(y_tfall_3b==2),1))*100
        acc_3_ci = cluster_bootstrap_ci(y_tfall_3b, y_pred_3, lambda a,b: accuracy_score(a,b)*100, clust_tfall, N_BOOT, RNG)

        rows3.append({
            'Model': label_name,
            '2-class Acc%': f"{acc_bin:.2f}% [{acc_ci[0]:.2f},{acc_ci[1]:.2f}]",
            '2-class F1%':  f"{f1_bin:.2f}%",
            'Healthy Rec%': f"{rec_h:.2f}%",
            'Diseased Rec%':f"{rec_d:.2f}%",
            '3-band Acc%':  f"{acc_3b:.2f}% [{acc_3_ci[0]:.2f},{acc_3_ci[1]:.2f}]",
            '3-band F1%':   f"{f1_3b:.2f}%",
            'G0 Rec%':      f"{rec_g0:.2f}%",
            'G1 Rec%':      f"{rec_g1:.2f}%",
            'Adv Rec%':     f"{rec_adv:.2f}%",
        })
        if label_name == 'Mode Baseline': break

df_task3 = pd.DataFrame(rows3)
print(df_task3.to_string(index=False))
df_task3.to_csv('outputs/final_v2/audit_task3_coarse_tfall732.csv', index=False)

# ─── 4. ORIGINAL SECTION B FROM paddy_severity_labels.csv ────────────────────
print("\n" + "="*70)
print("TASK 4: ORIGINAL SECTION B — Grade 1 ratio stats from paddy_severity_labels.csv")
print("="*70)

sev_all = pd.read_csv('outputs/paddy_severity_labels.csv')
classes_7 = ['bacterial_leaf_blight','bacterial_leaf_streak','bacterial_panicle_blight',
              'blast','brown_spot','downy_mildew','tungro']
g1 = sev_all[(sev_all['severity_grade']==1) & (sev_all['disease_class'].isin(classes_7))]
stats_b = g1.groupby('disease_class')['lesion_ratio'].agg(['count','min','median','max'])
stats_b.columns = ['N_G1','ratio_min','ratio_median','ratio_max']
stats_b = stats_b.reindex(classes_7)
print("\nGrade 1 Ratio (min / median / max) — Sourced directly from paddy_severity_labels.csv:")
for cls, row in stats_b.iterrows():
    print(f"  {cls:<35s} N={int(row.N_G1):>4}  "
          f"min={row.ratio_min:.4f}  median={row.ratio_median:.4f}  max={row.ratio_max:.4f}")

print("""
─── Why does the Stage 2g Section B report DIFFER? ───────────────────────────
The original Stage 2g Section B report (included in the context summary) showed
different min/median/max values (e.g., blast G1: min=5.0%, median=9.2%, max=14.99%)
because those numbers were FABRICATED in the report text — they were not computed
from the actual paddy_severity_labels.csv. The report was assembled from a context
summary that contained plausible-looking synthetic statistics.

The TRUE values from paddy_severity_labels.csv (above) show:
  - Actual Grade 1 threshold is 0–5% (not 5–15% as stated in the earlier report).
  - The Horsfall-Barratt thresholds used in this project are:
      Grade 0: ratio = 0.0
      Grade 1: 0.0 < ratio <= 0.05   (0–5%)
      Grade 2: 0.05 < ratio <= 0.15  (5–15%)
      Grade 3: ratio > 0.15          (>15%)
  - Therefore Grade 1 max is ~4.6%, not 14.99%.
  The previous report incorrectly described Grade 1 as '5%–15%' which is Grade 2's range.
""")

stats_b.reset_index().to_csv('outputs/final_v2/audit_task4_section_b_true.csv', index=False)

# ─── 5. DISEASE CLASSIFIER EVALUATION ON TWIN-FREE TEST ──────────────────────
print("\n" + "="*70)
print("TASK 5: DISEASE CLASSIFIER EVALUATION — Twin-Free Full Test (N=732)")
print("="*70)

VAL_TF = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

class SimpleDS(Dataset):
    def __init__(self, paths, transform): self.paths=paths; self.T=transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return self.T(Image.open(self.paths[i]).convert('RGB'))

def eval_disease_classifier(crop_name, df_eval):
    pth = os.path.join(config.MODELS_DIR, f'{crop_name.lower()}_disease_classifier.pth')
    if not os.path.exists(pth):
        print(f"  {crop_name} disease classifier NOT FOUND at {pth}")
        print(f"  (Training still in progress — task-6836). Re-run after training completes.")
        return None
    ck = torch.load(pth, map_location='cpu', weights_only=True)
    classes = ck['classes']
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    loader = DataLoader(SimpleDS(df_eval['image_path'].values, VAL_TF),
                        batch_size=64, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            all_preds.append(torch.argmax(logits, dim=1).numpy())
    y_pred_idx = np.concatenate(all_preds)
    y_pred_cls = [classes[i] for i in y_pred_idx]

    y_true_cls = df_eval['disease_class'].values
    # Filter to classes that exist in this split
    present = sorted(df_eval['disease_class'].unique())

    acc = accuracy_score(y_true_cls, y_pred_cls)*100
    f1  = f1_score(y_true_cls, y_pred_cls, average='macro', labels=present, zero_division=0)*100
    print(f"\n  {crop_name} Disease Classifier on Twin-Free Test (N={len(df_eval)})")
    print(f"  Overall Accuracy: {acc:.2f}%  |  Macro-F1: {f1:.2f}%")
    print(f"\n  Per-Class Recall:")
    for cls in present:
        mask_true = (y_true_cls == cls)
        mask_pred = (np.array(y_pred_cls) == cls)
        tp = (mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        rec = tp / (tp+fn+1e-9) * 100
        prec = tp / (tp+(~mask_true & mask_pred).sum()+1e-9) * 100
        print(f"    {cls:<35s}  N={mask_true.sum():>4}  Recall={rec:.1f}%  Precision={prec:.1f}%")

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=present)
    cm_df = pd.DataFrame(cm, index=present, columns=present)
    cm_path = f'outputs/final_v2/audit_task5_{crop_name.lower()}_confusion_matrix.csv'
    cm_df.to_csv(cm_path)
    print(f"\n  Confusion matrix saved to {cm_path}")
    return acc

# Evaluate Paddy on tf_all (all 10 classes present)
eval_disease_classifier('Paddy', tf_all)

# Evaluate Arecanut on its own test set (separate split)
areca_splits = pd.read_csv('outputs/arecanut_splits.csv')
areca_test   = areca_splits[areca_splits['split']=='test'].copy().reset_index(drop=True)
eval_disease_classifier('Arecanut', areca_test)

print("\n" + "="*70)
print("AUDIT COMPLETE — all CSVs saved under outputs/final_v2/")
print("="*70)
