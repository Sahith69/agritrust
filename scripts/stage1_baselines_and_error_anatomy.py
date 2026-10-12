import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import corn_probs_to_grade

torch.set_num_threads(8)
device = torch.device('cpu')

os.makedirs('outputs/final_v2', exist_ok=True)

def safe_qwk(y_t, y_p, labels=[0, 1, 2, 3]):
    if len(np.unique(y_t)) <= 1 and len(np.unique(y_p)) <= 1:
        return 1.0 if y_t[0] == y_p[0] else 0.0
    try:
        return float(cohen_kappa_score(y_t, y_p, weights='quadratic', labels=labels))
    except Exception:
        return np.nan

def load_model(path, backbone='resnet18', num_classes=4, ordinal=True):
    m = CropSeverityModel(backbone=backbone, num_classes=num_classes, ordinal=ordinal, pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    m.eval()
    return m

def predict(model, image_tensors, ordinal=True):
    with torch.no_grad():
        logits = model(image_tensors)
        if ordinal:
            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            return preds.cpu().numpy()
        else:
            return torch.argmax(logits, dim=1).cpu().numpy()

def get_tensors(paths, crop='arecanut'):
    tf = get_transforms('test')
    tensors = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        tensors.append(tf(im))
    return torch.stack(tensors).to(device)

print("=" * 80)
print("STAGE 1.1: CLASS-ONLY MODE BASELINE VS TRAINED MODELS")
print("=" * 80)

datasets_info = [
    {
        'name': 'Old 124 Test',
        'csv': 'outputs/arecanut_splits_LEAKY_BACKUP.csv',
        'models': {
            'CORN Seed 42': ('outputs/models/architecture_comparison/arecanut_resnet18_ordinal.pth', True),
            'CORN Seed 100': ('outputs/models/architecture_comparison/arecanut_resnet18_ordinal_seed100.pth', True),
            'CORN Seed 2024': ('outputs/models/architecture_comparison/arecanut_resnet18_ordinal_seed2024.pth', True),
        }
    },
    {
        'name': 'Clean 146 Test',
        'csv': 'outputs/improvements/arecanut_splits_clean_871.csv',
        'models': {
            'CORN Seed 42': ('outputs/models/clean_retrain/arecanut_resnet18_corn_seed42.pth', True),
            'CORN Seed 100': ('outputs/models/clean_retrain/arecanut_resnet18_corn_seed100.pth', True),
            'CORN Seed 2024': ('outputs/models/clean_retrain/arecanut_resnet18_corn_seed2024.pth', True),
            'Softmax Seed 42': ('outputs/models/clean_retrain/arecanut_resnet18_softmax_seed42.pth', False),
            'Softmax Seed 100': ('outputs/models/clean_retrain/arecanut_resnet18_softmax_seed100.pth', False),
            'Softmax Seed 2024': ('outputs/models/clean_retrain/arecanut_resnet18_softmax_seed2024.pth', False),
        }
    },
    {
        'name': 'Expanded 502 Test',
        'csv': 'outputs/improvements/arecanut_splits_merged_expanded.csv',
        'models': {
            'CORN Seed 42': ('outputs/models/merged_expanded/arecanut_merged_corn_seed42.pth', True),
            'CORN Seed 100': ('outputs/models/merged_expanded/arecanut_merged_corn_seed100.pth', True),
            'CORN Seed 2024': ('outputs/models/merged_expanded/arecanut_merged_corn_seed2024.pth', True),
            'Softmax Seed 42': ('outputs/models/merged_expanded/arecanut_merged_softmax_seed42.pth', False),
            'Softmax Seed 100': ('outputs/models/merged_expanded/arecanut_merged_softmax_seed100.pth', False),
            'Softmax Seed 2024': ('outputs/models/merged_expanded/arecanut_merged_softmax_seed2024.pth', False),
        }
    },
    {
        'name': 'Paddy 1507 Test',
        'csv': 'outputs/paddy_splits.csv',
        'models': {
            'CORN Production': ('outputs/models/paddy_ordinal.pth', True),
            'Softmax Production': ('outputs/models/paddy_baseline.pth', False),
        }
    }
]

stage1_results = {}

for d in datasets_info:
    name = d['name']
    df = pd.read_csv(d['csv'])
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Class-only mode baseline
    majority_map = train_df.groupby('disease_class')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
    base_preds = test_df['disease_class'].map(majority_map).values
    targets = test_df['severity_grade'].values
    
    base_acc = accuracy_score(targets, base_preds)
    base_qwk = safe_qwk(targets, base_preds)
    
    print(f"\n--- {name} (N_test = {len(test_df)}) ---")
    print(f"Class-Only Mode Baseline -> QWK: {base_qwk:.4f} | Accuracy: {base_acc*100:.2f}%")
    
    # Evaluate models
    img_tensors = get_tensors(test_df['image_path'])
    d_res = {
        'n_test': len(test_df),
        'class_baseline_qwk': base_qwk,
        'class_baseline_acc': base_acc,
        'models': {}
    }
    
    for m_label, (m_path, is_ord) in d['models'].items():
        if os.path.exists(m_path):
            model = load_model(m_path, backbone='resnet18', num_classes=4, ordinal=is_ord)
            m_preds = predict(model, img_tensors, ordinal=is_ord)
            m_acc = accuracy_score(targets, m_preds)
            m_qwk = safe_qwk(targets, m_preds)
            print(f"  {m_label} -> QWK: {m_qwk:.4f} | Accuracy: {m_acc*100:.2f}%")
            d_res['models'][m_label] = {'qwk': m_qwk, 'acc': m_acc, 'preds': m_preds.tolist()}
            
    stage1_results[name] = d_res

print("\n" + "=" * 80)
print("STAGE 1.1b: PADDY PER-CLASS GRADE DISTRIBUTIONS")
print("=" * 80)
paddy_df = pd.read_csv('outputs/paddy_splits.csv')
paddy_test = paddy_df[paddy_df['split'] == 'test']
paddy_ct = pd.crosstab(paddy_test['disease_class'], paddy_test['severity_grade'], margins=True)
print("Paddy Test Set (Counts):")
print(paddy_ct)
print("\nPaddy Entire Dataset (Counts):")
print(pd.crosstab(paddy_df['disease_class'], paddy_df['severity_grade'], margins=True))

print("\n" + "=" * 80)
print("STAGE 1.2: WITHIN-CLASS QWK & ACCURACY WITH CLUSTER BOOTSTRAP 95% CIs")
print("=" * 80)

# Arecanut 502 Test Set Within-Class
arec_df = pd.read_csv('outputs/improvements/arecanut_splits_merged_expanded.csv')
arec_test = arec_df[arec_df['split'] == 'test'].reset_index(drop=True)
arec_tensors = get_tensors(arec_test['image_path'])
arec_targets = arec_test['severity_grade'].values
arec_clusters = arec_test['cluster_id'].values

arec_models = {
    'CORN_42': load_model('outputs/models/merged_expanded/arecanut_merged_corn_seed42.pth', ordinal=True),
    'CORN_100': load_model('outputs/models/merged_expanded/arecanut_merged_corn_seed100.pth', ordinal=True),
    'CORN_2024': load_model('outputs/models/merged_expanded/arecanut_merged_corn_seed2024.pth', ordinal=True),
    'Softmax_42': load_model('outputs/models/merged_expanded/arecanut_merged_softmax_seed42.pth', ordinal=False),
    'Softmax_100': load_model('outputs/models/merged_expanded/arecanut_merged_softmax_seed100.pth', ordinal=False),
    'Softmax_2024': load_model('outputs/models/merged_expanded/arecanut_merged_softmax_seed2024.pth', ordinal=False),
}
arec_preds = {k: predict(m, arec_tensors, ordinal='CORN' in k) for k, m in arec_models.items()}

# Bootstrap helper for a subset
def cluster_bootstrap_ci(sub_targets, sub_preds, sub_clusters, B=2000, metric='qwk'):
    u_clusters = np.unique(sub_clusters)
    n_c = len(u_clusters)
    c_to_idx = {c: np.where(sub_clusters == c)[0] for c in u_clusters}
    vals = []
    for _ in range(B):
        s_c = np.random.choice(u_clusters, size=n_c, replace=True)
        s_idx = np.concatenate([c_to_idx[c] for c in s_c])
        y_t = sub_targets[s_idx]
        y_p = sub_preds[s_idx]
        if metric == 'qwk':
            v = safe_qwk(y_t, y_p)
        else:
            v = accuracy_score(y_t, y_p)
        vals.append(v)
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

np.random.seed(42)
arec_classes_to_eval = ['mahali_koleroga', 'stem_cracking', 'stem_bleeding']

print("\n--- Arecanut (502 Test) Within-Class Metrics ---")
for cls in arec_classes_to_eval:
    c_idx = (arec_test['disease_class'] == cls).values
    sub_t = arec_targets[c_idx]
    sub_c = arec_clusters[c_idx]
    n_samples = len(sub_t)
    grades_p = np.unique(sub_t)
    
    # Class-only baseline predicts mode
    cls_mode = pd.Series(sub_t).mode()[0]
    base_sub_p = np.full(n_samples, cls_mode)
    base_qwk = safe_qwk(sub_t, base_sub_p)
    base_acc = accuracy_score(sub_t, base_sub_p)
    
    print(f"\nDisease Class: {cls} (N={n_samples}, Unique Clusters={len(np.unique(sub_c))}, Grades={grades_p})")
    print(f"  Class-only Mode Baseline: QWK = {base_qwk:.4f}, Accuracy = {base_acc*100:.2f}%")
    
    for m_k in arec_models:
        sub_p = arec_preds[m_k][c_idx]
        acc_val = accuracy_score(sub_t, sub_p)
        qwk_val = safe_qwk(sub_t, sub_p)
        qwk_mean, qwk_lo, qwk_hi = cluster_bootstrap_ci(sub_t, sub_p, sub_c, metric='qwk')
        acc_mean, acc_lo, acc_hi = cluster_bootstrap_ci(sub_t, sub_p, sub_c, metric='acc')
        print(f"  {m_k:12s} | QWK: {qwk_val:.4f} (95% CI: [{qwk_lo:.4f}, {qwk_hi:.4f}]) | Acc: {acc_val*100:.2f}% (95% CI: [{acc_lo*100:.1f}%, {acc_hi*100:.1f}%])")

# Paddy 1507 Test Set Within-Class
paddy_tensors = get_tensors(paddy_test['image_path'])
paddy_targets = paddy_test['severity_grade'].values
paddy_clusters = paddy_test['md5'].values # MD5 serves as duplicate/cluster group

paddy_m_corn = load_model('outputs/models/paddy_ordinal.pth', ordinal=True)
paddy_m_soft = load_model('outputs/models/paddy_baseline.pth', ordinal=False)
p_preds_corn = predict(paddy_m_corn, paddy_tensors, ordinal=True)
p_preds_soft = predict(paddy_m_soft, paddy_tensors, ordinal=False)

print("\n--- Paddy (1,507 Test) Within-Class Metrics (All Lesion Classes) ---")
# Lesion classes are those with > 1 grade in test set
paddy_lesion_classes = [c for c in paddy_test['disease_class'].unique() if len(np.unique(paddy_test[paddy_test['disease_class'] == c]['severity_grade'])) >= 2]

for cls in sorted(paddy_lesion_classes):
    c_idx = (paddy_test['disease_class'] == cls).values
    sub_t = paddy_targets[c_idx]
    sub_c = paddy_clusters[c_idx]
    n_samples = len(sub_t)
    grades_p = np.unique(sub_t)
    
    cls_mode = pd.Series(sub_t).mode()[0]
    base_sub_p = np.full(n_samples, cls_mode)
    base_qwk = safe_qwk(sub_t, base_sub_p)
    base_acc = accuracy_score(sub_t, base_sub_p)
    
    print(f"\nPaddy Class: {cls} (N={n_samples}, Grades={grades_p})")
    print(f"  Class-only Mode Baseline: QWK = {base_qwk:.4f}, Accuracy = {base_acc*100:.2f}%")
    
    for m_label, preds in [('CORN', p_preds_corn), ('Softmax', p_preds_soft)]:
        sub_p = preds[c_idx]
        acc_val = accuracy_score(sub_t, sub_p)
        qwk_val = safe_qwk(sub_t, sub_p)
        qwk_mean, qwk_lo, qwk_hi = cluster_bootstrap_ci(sub_t, sub_p, sub_c, metric='qwk')
        acc_mean, acc_lo, acc_hi = cluster_bootstrap_ci(sub_t, sub_p, sub_c, metric='acc')
        print(f"  {m_label:10s} | QWK: {qwk_val:.4f} (95% CI: [{qwk_lo:.4f}, {qwk_hi:.4f}]) | Acc: {acc_val*100:.2f}% (95% CI: [{acc_lo*100:.1f}%, {acc_hi*100:.1f}%])")

print("\n" + "=" * 80)
print("STAGE 1.3: |DELTA| >= 2 ERRORS & HEALTHY_TRUNK FAILURE ANALYSIS")
print("=" * 80)

# Evaluate |delta| >= 2 on Arecanut 502 test set for production CORN (Seed 100) and Softmax (Seed 2024)
for m_label, preds in [('CORN (Seed 100)', arec_preds['CORN_100']), ('Softmax (Seed 2024)', arec_preds['Softmax_2024']), ('CORN (Seed 42)', arec_preds['CORN_42'])]:
    diff = np.abs(arec_targets - preds)
    cat_mask = (diff >= 2)
    print(f"\n{m_label} - Catastrophic (|Delta| >= 2) Errors by Class (Total: {np.sum(cat_mask)}/{len(arec_targets)} = {np.mean(cat_mask)*100:.2f}%):")
    sub_df = arec_test.copy()
    sub_df['pred'] = preds
    sub_df['diff'] = diff
    cat_df = sub_df[cat_mask]
    
    by_class = cat_df.groupby('disease_class').size()
    for dc, cnt in by_class.items():
        total_cls = np.sum(sub_df['disease_class'] == dc)
        print(f"  {dc:22s}: {cnt}/{total_cls} ({cnt/total_cls*100:.1f}%)")
        # Detail transitions
        trans = cat_df[cat_df['disease_class'] == dc].groupby(['severity_grade', 'pred']).size().to_dict()
        print(f"    Transitions (true -> pred): {trans}")

# Healthy trunk severe errors (Grade 0 predicted as Grade 2 or 3)
ht_mask = (arec_test['disease_class'] == 'healthy_trunk').values
ht_df = arec_test[ht_mask].reset_index(drop=True)
ht_preds = arec_preds['CORN_100'][ht_mask] # production CORN checkpoint

severe_ht_indices = np.where(ht_preds >= 2)[0]
print(f"\nHealthy Trunk Images Predicted as Grade 2/3 (Seed 100 CORN): {len(severe_ht_indices)}/{len(ht_df)} ({len(severe_ht_indices)/len(ht_df)*100:.1f}%)")

# Extract 12 examples for visual report
np.random.seed(42)
sample_size = min(12, len(severe_ht_indices))
chosen = np.random.choice(severe_ht_indices, size=sample_size, replace=False)

grid_imgs = []
for idx in chosen:
    row = ht_df.iloc[idx]
    pred_g = ht_preds[idx]
    p = row['image_path']
    im = Image.open(p).convert('RGB').resize((256, 256))
    img_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, f"True: G0 -> Pred: G{pred_g}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(img_bgr, os.path.basename(p)[:22], (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    grid_imgs.append(img_bgr)

# Stitch 3x4
rows = []
for r_i in range(3):
    rows.append(np.hstack(grid_imgs[r_i*4 : (r_i+1)*4]))
ht_montage = np.vstack(rows)
ht_out_path = 'outputs/final_v2/healthy_trunk_severe_errors.png'
cv2.imwrite(ht_out_path, ht_out_path if ht_out_path.endswith('.png') else ht_montage)
cv2.imwrite(ht_out_path, ht_montage)
print(f"Saved 12 healthy_trunk severe error images to {ht_out_path}")

# Save JSON results
with open('outputs/final_v2/stage1_diagnostics.json', 'w') as f:
    json.dump(stage1_results, f, indent=2)
print("Saved STAGE 1 diagnostics to outputs/final_v2/stage1_diagnostics.json")
