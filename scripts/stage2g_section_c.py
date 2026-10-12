import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score

sys.path.insert(0, '/home/sahith/Projects/Crop')
from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import corn_probs_to_grade

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2g - SECTION C: ARECANUT SOURCE CONFOUND & MAHALI_KOLEROGA EVAL")
print("=" * 80)

# Load 502 splits & source mappings
df_502 = pd.read_csv('outputs/arecanut_splits.csv')
clean_v2 = pd.read_csv('outputs/final_v2/arecanut_clean_v2.csv')
path_to_source = dict(zip(clean_v2['image_path'], clean_v2['source']))
df_502['source'] = df_502['image_path'].map(path_to_source).fillna('original_871')

train_502 = df_502[df_502['split'] == 'train']
test_502 = df_502[df_502['split'] == 'test'].reset_index(drop=True)

# 1. Mode by source for stem_cracking and mahali_koleroga
mode_stats = {}
for cls in ['stem_cracking', 'mahali_koleroga']:
    mode_stats[cls] = {'train': {}, 'full': {}}
    sub_train = train_502[train_502['disease_class'] == cls]
    sub_full = df_502[df_502['disease_class'] == cls]
    
    for src in sorted(sub_full['source'].unique()):
        s_tr = sub_train[sub_train['source'] == src]['severity_grade']
        s_fu = sub_full[sub_full['source'] == src]['severity_grade']
        
        mode_tr = int(s_tr.mode()[0]) if len(s_tr) > 0 else None
        mode_fu = int(s_fu.mode()[0]) if len(s_fu) > 0 else None
        
        mode_stats[cls]['train'][src] = {
            'mode': mode_tr,
            'counts': {int(k): int(v) for k, v in s_tr.value_counts().to_dict().items()},
            'total': len(s_tr)
        }
        mode_stats[cls]['full'][src] = {
            'mode': mode_fu,
            'counts': {int(k): int(v) for k, v in s_fu.value_counts().to_dict().items()},
            'total': len(s_fu)
        }

print("\n--- C.2 MODE BY SOURCE FOR STEM_CRACKING & MAHALI_KOLEROGA ---")
for cls, d in mode_stats.items():
    print(f"\nDisease Class: {cls}")
    print("  [Train Set Mode & Grade Distribution]:")
    for src, info in d['train'].items():
        print(f"    • {src:<15}: Mode = Grade {info['mode']} (Counts: {info['counts']}, Total = {info['total']})")
    print("  [Full Dataset Mode & Grade Distribution]:")
    for src, info in d['full'].items():
        print(f"    • {src:<15}: Mode = Grade {info['mode']} (Counts: {info['counts']}, Total = {info['total']})")

# 2. Evaluation for mahali_koleroga on 502 test set
mk_test = test_502[test_502['disease_class'] == 'mahali_koleroga'].copy().reset_index(drop=True)
y_true = mk_test['severity_grade'].values
n_test = len(mk_test)
print(f"\n--- C.1 MAHALI_KOLEROGA EVALUATION ON 502 TEST (N={n_test}) ---")
print(f"Ground Truth Test Distribution: {dict(pd.Series(y_true).value_counts())}")

# Baseline: Mode by Source
mk_train = train_502[train_502['disease_class'] == 'mahali_koleroga']
mode_by_src = mk_train.groupby('source')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
pred_mode = mk_test['source'].map(mode_by_src).values

qwk_base = float(cohen_kappa_score(y_true, pred_mode, weights='quadratic'))
acc_base = float(accuracy_score(y_true, pred_mode) * 100.0)
print(f"\n1. Baseline (Grade Mode by Source):")
print(f"   Train Modes by Source: {mode_by_src}")
print(f"   QWK:      {qwk_base:.4f}")
print(f"   Accuracy: {acc_base:.2f}%")

# Load model checkpoints and evaluate
tf = get_transforms('test')
tensors = torch.stack([tf(Image.open(p).convert('RGB')) for p in mk_test['image_path']])

eval_results = {
    'baseline_mode_by_source': {
        'qwk': qwk_base,
        'accuracy': acc_base,
        'predicted_mode_per_source': mode_by_src
    },
    'corn_chained': {},
    'softmax': {}
}

seeds = [42, 100, 2024]

# Softmax Models
print(f"\n2. Softmax Models:")
soft_qwks, soft_accs = [], []
for s in seeds:
    path_soft = f'outputs/models/merged_expanded/arecanut_merged_softmax_seed{s}.pth'
    if not os.path.exists(path_soft) and s == 42:
        path_soft = 'outputs/models/arecanut_baseline.pth'
    if os.path.exists(path_soft):
        m = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=False)
        m.load_state_dict(torch.load(path_soft, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            out = m(tensors)
            p_soft = torch.argmax(out, dim=1).numpy()
        q_s = float(cohen_kappa_score(y_true, p_soft, weights='quadratic'))
        a_s = float(accuracy_score(y_true, p_soft) * 100.0)
        soft_qwks.append(q_s)
        soft_accs.append(a_s)
        eval_results['softmax'][f'seed_{s}'] = {'qwk': q_s, 'accuracy': a_s, 'checkpoint': path_soft}
        print(f"   Seed {s:4d}: QWK = {q_s:.4f} | Accuracy = {a_s:.2f}%")

eval_results['softmax']['multi_seed_mean'] = {
    'qwk': float(np.mean(soft_qwks)),
    'accuracy': float(np.mean(soft_accs))
}
print(f"   MEAN     : QWK = {np.mean(soft_qwks):.4f} | Accuracy = {np.mean(soft_accs):.2f}%")

# CORN Models (Chained Decode)
print(f"\n3. CORN Models (Chained Probability Decode: corn_probs_to_grade):")
corn_qwks, corn_accs = [], []
for s in seeds:
    path_corn = f'outputs/models/merged_expanded/arecanut_merged_corn_seed{s}.pth'
    if not os.path.exists(path_corn) and s == 42:
        path_corn = 'outputs/models/arecanut_ordinal.pth'
    if os.path.exists(path_corn):
        m = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
        m.load_state_dict(torch.load(path_corn, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            out = m(tensors)
            p_corn = corn_probs_to_grade(torch.sigmoid(out))[0].numpy()
        q_c = float(cohen_kappa_score(y_true, p_corn, weights='quadratic'))
        a_c = float(accuracy_score(y_true, p_corn) * 100.0)
        corn_qwks.append(q_c)
        corn_accs.append(a_c)
        eval_results['corn_chained'][f'seed_{s}'] = {'qwk': q_c, 'accuracy': a_c, 'checkpoint': path_corn}
        print(f"   Seed {s:4d}: QWK = {q_c:.4f} | Accuracy = {a_c:.2f}%")

eval_results['corn_chained']['multi_seed_mean'] = {
    'qwk': float(np.mean(corn_qwks)),
    'accuracy': float(np.mean(corn_accs))
}
print(f"   MEAN     : QWK = {np.mean(corn_qwks):.4f} | Accuracy = {np.mean(corn_accs):.2f}%")

# Save JSON
stage2g_c_output = {
    'mode_stats_by_source': mode_stats,
    'mahali_koleroga_502_test_eval': eval_results
}

with open('outputs/final_v2/stage2g_section_c.json', 'w') as f:
    json.dump(stage2g_c_output, f, indent=2)

print("\nSaved Stage 2g Section C results to outputs/final_v2/stage2g_section_c.json")
