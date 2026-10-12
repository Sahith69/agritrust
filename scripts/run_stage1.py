import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
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
        val = float(cohen_kappa_score(y_t, y_p, weights='quadratic', labels=labels))
        return val if not np.isnan(val) else 0.0
    except Exception:
        return 0.0

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

def get_tensors(paths):
    tf = get_transforms('test')
    tensors = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        tensors.append(tf(im))
    return torch.stack(tensors).to(device)

def cluster_bootstrap_ci(sub_targets, sub_preds, sub_clusters=None, B=2000, seed=42):
    np.random.seed(seed)
    n = len(sub_targets)
    vals = []
    if sub_clusters is not None:
        u_clusters = np.unique(sub_clusters)
        n_c = len(u_clusters)
        c_to_idx = {c: np.where(sub_clusters == c)[0] for c in u_clusters}
        for _ in range(B):
            s_c = np.random.choice(u_clusters, size=n_c, replace=True)
            s_idx = np.concatenate([c_to_idx[c] for c in s_c])
            vals.append(safe_qwk(sub_targets[s_idx], sub_preds[s_idx]))
    else:
        for _ in range(B):
            s_idx = np.random.choice(n, size=n, replace=True)
            vals.append(safe_qwk(sub_targets[s_idx], sub_preds[s_idx]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def compute_error_breakdown(y_true, y_pred, df_split):
    diff = np.abs(y_true - y_pred)
    total_g0_g3 = int(np.sum((y_true == 0) & (y_pred == 3)))
    total_g3_g0 = int(np.sum((y_true == 3) & (y_pred == 0)))
    total_cat = int(np.sum(diff >= 2))
    
    by_class = {}
    for cls in sorted(df_split['disease_class'].unique()):
        mask = (df_split['disease_class'] == cls).values
        n_c = int(np.sum(mask))
        c_y = y_true[mask]
        c_p = y_pred[mask]
        c_diff = diff[mask]
        
        by_class[cls] = {
            'N': n_c,
            'catastrophic': int(np.sum(c_diff >= 2)),
            'g0_to_g3': int(np.sum((c_y == 0) & (c_p == 3))),
            'g3_to_g0': int(np.sum((c_y == 3) & (c_p == 0)))
        }
    return {
        'total_catastrophic': total_cat,
        'total_g0_to_g3': total_g0_g3,
        'total_g3_to_g0': total_g3_g0,
        'by_class': by_class
    }

print("=" * 80)
print("STAGE 1.1: CLASS-ONLY BASELINE VS TRAINED MODELS")
print("=" * 80)

# 1. Old Leaky 124 Test
print("Loading Leaky 124 test...")
df_124 = pd.read_csv('outputs/arecanut_splits_LEAKY_BACKUP.csv')
train_124 = df_124[df_124['split'] == 'train']
test_124 = df_124[df_124['split'] == 'test'].reset_index(drop=True)
mode_124 = train_124.groupby('disease_class')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
base_p_124 = test_124['disease_class'].map(mode_124).values
y_124 = test_124['severity_grade'].values
tensors_124 = get_tensors(test_124['image_path'])

res_124 = {
    'baseline': {
        'qwk': safe_qwk(y_124, base_p_124),
        'accuracy': accuracy_score(y_124, base_p_124)
    },
    'models': {}
}
cp_124 = {
    'CORN Seed 42': ('outputs/models/architecture_comparison/arecanut_resnet18_ordinal_seed42.pth', True),
    'CORN Seed 100': ('outputs/models/architecture_comparison/arecanut_resnet18_ordinal_seed100.pth', True),
    'CORN Seed 2024': ('outputs/models/architecture_comparison/arecanut_resnet18_ordinal_seed2024.pth', True),
    'Softmax Seed 42': ('outputs/models/architecture_comparison/arecanut_resnet18_softmax_seed42.pth', False),
    'Softmax Seed 100': ('outputs/models/architecture_comparison/arecanut_resnet18_softmax_seed100.pth', False),
    'Softmax Seed 2024': ('outputs/models/architecture_comparison/arecanut_resnet18_softmax_seed2024.pth', False),
}
preds_124 = {}
for name, (path, is_ord) in cp_124.items():
    if os.path.exists(path):
        m = load_model(path, ordinal=is_ord)
        p = predict(m, tensors_124, ordinal=is_ord)
        preds_124[name] = p
        res_124['models'][name] = {
            'checkpoint': path,
            'qwk': safe_qwk(y_124, p),
            'accuracy': accuracy_score(y_124, p)
        }
    else:
        res_124['models'][name] = {
            'checkpoint': path,
            'status': 'MISSING - SKIPPED'
        }

# 2. Clean 146 Test
print("Loading Clean 146 test...")
df_146 = pd.read_csv('outputs/improvements/arecanut_splits_clean_871.csv')
train_146 = df_146[df_146['split'] == 'train']
test_146 = df_146[df_146['split'] == 'test'].reset_index(drop=True)
mode_146 = train_146.groupby('disease_class')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
base_p_146 = test_146['disease_class'].map(mode_146).values
y_146 = test_146['severity_grade'].values
clusters_146 = test_146['cluster_id'].values
tensors_146 = get_tensors(test_146['image_path'])

res_146 = {
    'baseline': {
        'qwk': safe_qwk(y_146, base_p_146),
        'accuracy': accuracy_score(y_146, base_p_146)
    },
    'models': {}
}
cp_146 = {
    'CORN Seed 42': ('outputs/models/clean_retrain/arecanut_resnet18_corn_seed42.pth', True),
    'CORN Seed 100': ('outputs/models/clean_retrain/arecanut_resnet18_corn_seed100.pth', True),
    'CORN Seed 2024': ('outputs/models/clean_retrain/arecanut_resnet18_corn_seed2024.pth', True),
    'Softmax Seed 42': ('outputs/models/clean_retrain/arecanut_resnet18_softmax_seed42.pth', False),
    'Softmax Seed 100': ('outputs/models/clean_retrain/arecanut_resnet18_softmax_seed100.pth', False),
    'Softmax Seed 2024': ('outputs/models/clean_retrain/arecanut_resnet18_softmax_seed2024.pth', False),
}
preds_146 = {}
for name, (path, is_ord) in cp_146.items():
    if os.path.exists(path):
        m = load_model(path, ordinal=is_ord)
        p = predict(m, tensors_146, ordinal=is_ord)
        preds_146[name] = p
        res_146['models'][name] = {
            'checkpoint': path,
            'qwk': safe_qwk(y_146, p),
            'accuracy': accuracy_score(y_146, p)
        }
    else:
        res_146['models'][name] = {
            'checkpoint': path,
            'status': 'MISSING - SKIPPED'
        }

# 3. Expanded 502 Test
print("Loading Expanded 502 test...")
df_502 = pd.read_csv('outputs/arecanut_splits.csv')
train_502 = df_502[df_502['split'] == 'train']
test_502 = df_502[df_502['split'] == 'test'].reset_index(drop=True)
mode_502 = train_502.groupby('disease_class')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
base_p_502 = test_502['disease_class'].map(mode_502).values
y_502 = test_502['severity_grade'].values
clusters_502 = test_502['cluster_id'].values
tensors_502 = get_tensors(test_502['image_path'])

res_502 = {
    'baseline': {
        'qwk': safe_qwk(y_502, base_p_502),
        'accuracy': accuracy_score(y_502, base_p_502)
    },
    'models': {}
}
cp_502 = {
    'CORN Seed 42': ('outputs/models/merged_expanded/arecanut_merged_corn_seed42.pth', True),
    'CORN Seed 100': ('outputs/models/merged_expanded/arecanut_merged_corn_seed100.pth', True),
    'CORN Seed 2024': ('outputs/models/merged_expanded/arecanut_merged_corn_seed2024.pth', True),
    'Softmax Seed 42': ('outputs/models/merged_expanded/arecanut_merged_softmax_seed42.pth', False),
    'Softmax Seed 100': ('outputs/models/merged_expanded/arecanut_merged_softmax_seed100.pth', False),
    'Softmax Seed 2024': ('outputs/models/merged_expanded/arecanut_merged_softmax_seed2024.pth', False),
}
preds_502 = {}
for name, (path, is_ord) in cp_502.items():
    if os.path.exists(path):
        m = load_model(path, ordinal=is_ord)
        p = predict(m, tensors_502, ordinal=is_ord)
        preds_502[name] = p
        res_502['models'][name] = {
            'checkpoint': path,
            'qwk': safe_qwk(y_502, p),
            'accuracy': accuracy_score(y_502, p)
        }
    else:
        res_502['models'][name] = {
            'checkpoint': path,
            'status': 'MISSING - SKIPPED'
        }

# 4. Paddy 1507 Test
print("Loading Paddy 1507 test...")
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
train_paddy = df_paddy[df_paddy['split'] == 'train']
test_paddy = df_paddy[df_paddy['split'] == 'test'].reset_index(drop=True)
mode_paddy = train_paddy.groupby('disease_class')['severity_grade'].agg(lambda s: s.mode()[0]).to_dict()
base_p_paddy = test_paddy['disease_class'].map(mode_paddy).values
y_paddy = test_paddy['severity_grade'].values
tensors_paddy = get_tensors(test_paddy['image_path'])

res_paddy = {
    'baseline': {
        'qwk': safe_qwk(y_paddy, base_p_paddy),
        'accuracy': accuracy_score(y_paddy, base_p_paddy)
    },
    'models': {}
}
cp_paddy = {
    'CORN Production': ('outputs/models/paddy_ordinal.pth', True),
    'Softmax Production': ('outputs/models/paddy_baseline.pth', False),
}
preds_paddy = {}
for name, (path, is_ord) in cp_paddy.items():
    if os.path.exists(path):
        m = load_model(path, ordinal=is_ord)
        p = predict(m, tensors_paddy, ordinal=is_ord)
        preds_paddy[name] = p
        res_paddy['models'][name] = {
            'checkpoint': path,
            'qwk': safe_qwk(y_paddy, p),
            'accuracy': accuracy_score(y_paddy, p)
        }
    else:
        res_paddy['models'][name] = {
            'checkpoint': path,
            'status': 'MISSING - SKIPPED'
        }

# Paddy per-class grade distributions
paddy_train_ct = pd.crosstab(train_paddy['disease_class'], train_paddy['severity_grade']).to_dict(orient='index')
paddy_test_ct = pd.crosstab(test_paddy['disease_class'], test_paddy['severity_grade']).to_dict(orient='index')

print("\n" + "=" * 80)
print("STAGE 1.2: WITHIN-CLASS SEVERITY SKILL WITH 95% CIs")
print("=" * 80)

# Arecanut within-class on 502 test: mahali_koleroga, stem_cracking, stem_bleeding
arec_within_res = {}
for cls in ['mahali_koleroga', 'stem_cracking', 'stem_bleeding']:
    c_idx = (test_502['disease_class'] == cls).values
    sub_y = y_502[c_idx]
    sub_c = clusters_502[c_idx]
    n_s = len(sub_y)
    grades_p = sorted([int(g) for g in np.unique(sub_y)])
    
    cls_train_mode = mode_502[cls]
    base_sub_preds = np.full(n_s, cls_train_mode)
    base_sub_qwk = safe_qwk(sub_y, base_sub_preds)
    base_sub_acc = accuracy_score(sub_y, base_sub_preds)
    
    cls_entry = {
        'N': n_s,
        'grades_present': grades_p,
        'unique_clusters': int(len(np.unique(sub_c))),
        'baseline_qwk': base_sub_qwk,
        'baseline_acc': base_sub_acc,
        'models': {}
    }
    
    for m_name in cp_502.keys():
        if m_name in preds_502:
            p_sub = preds_502[m_name][c_idx]
            q_val = safe_qwk(sub_y, p_sub)
            a_val = accuracy_score(sub_y, p_sub)
            q_mean, q_lo, q_hi = cluster_bootstrap_ci(sub_y, p_sub, sub_clusters=sub_c, B=2000)
            cls_entry['models'][m_name] = {
                'qwk': q_val,
                'accuracy': a_val,
                'ci_qwk': [q_lo, q_hi]
            }
    arec_within_res[cls] = cls_entry

# Clean 146 within-class (for comparison & completeness)
arec_146_within_res = {}
for cls in ['mahali_koleroga', 'stem_cracking', 'stem_bleeding']:
    c_idx = (test_146['disease_class'] == cls).values
    sub_y = y_146[c_idx]
    sub_c = clusters_146[c_idx]
    n_s = len(sub_y)
    grades_p = sorted([int(g) for g in np.unique(sub_y)])
    
    cls_train_mode = mode_146[cls]
    base_sub_preds = np.full(n_s, cls_train_mode)
    base_sub_qwk = safe_qwk(sub_y, base_sub_preds)
    base_sub_acc = accuracy_score(sub_y, base_sub_preds)
    
    cls_entry = {
        'N': n_s,
        'grades_present': grades_p,
        'unique_clusters': int(len(np.unique(sub_c))),
        'baseline_qwk': base_sub_qwk,
        'baseline_acc': base_sub_acc,
        'models': {}
    }
    for m_name in cp_146.keys():
        if m_name in preds_146:
            p_sub = preds_146[m_name][c_idx]
            q_val = safe_qwk(sub_y, p_sub)
            a_val = accuracy_score(sub_y, p_sub)
            q_mean, q_lo, q_hi = cluster_bootstrap_ci(sub_y, p_sub, sub_clusters=sub_c, B=2000)
            cls_entry['models'][m_name] = {
                'qwk': q_val,
                'accuracy': a_val,
                'ci_qwk': [q_lo, q_hi]
            }
    arec_146_within_res[cls] = cls_entry

# Paddy within-class (all lesion classes with >= 2 grades in test set)
paddy_within_res = {}
paddy_classes = sorted(test_paddy['disease_class'].unique())

for cls in paddy_classes:
    c_idx = (test_paddy['disease_class'] == cls).values
    sub_y = y_paddy[c_idx]
    grades_p = sorted([int(g) for g in np.unique(sub_y)])
    if len(grades_p) < 2:
        continue # skip normal, hispa, dead_heart (only 1 grade)
        
    n_s = len(sub_y)
    cls_train_mode = mode_paddy[cls]
    base_sub_preds = np.full(n_s, cls_train_mode)
    base_sub_qwk = safe_qwk(sub_y, base_sub_preds)
    base_sub_acc = accuracy_score(sub_y, base_sub_preds)
    
    cls_entry = {
        'N': n_s,
        'grades_present': grades_p,
        'baseline_qwk': base_sub_qwk,
        'baseline_acc': base_sub_acc,
        'bootstrap_type': 'Standard bootstrap (no cluster_id in paddy split)',
        'models': {}
    }
    
    for m_name in cp_paddy.keys():
        if m_name in preds_paddy:
            p_sub = preds_paddy[m_name][c_idx]
            q_val = safe_qwk(sub_y, p_sub)
            a_val = accuracy_score(sub_y, p_sub)
            q_mean, q_lo, q_hi = cluster_bootstrap_ci(sub_y, p_sub, sub_clusters=None, B=2000)
            cls_entry['models'][m_name] = {
                'qwk': q_val,
                'accuracy': a_val,
                'ci_qwk': [q_lo, q_hi]
            }
    paddy_within_res[cls] = cls_entry

print("\n" + "=" * 80)
print("STAGE 1.3: ERROR ANATOMY & 12 HEALTHY_TRUNK EXAMPLES")
print("=" * 80)

error_anatomy = {
    'expanded_502': {},
    'clean_146': {},
    'old_124': {},
    'paddy_1507': {}
}

for m_name, p in preds_502.items():
    error_anatomy['expanded_502'][m_name] = compute_error_breakdown(y_502, p, test_502)

for m_name, p in preds_146.items():
    error_anatomy['clean_146'][m_name] = compute_error_breakdown(y_146, p, test_146)

for m_name, p in preds_124.items():
    error_anatomy['old_124'][m_name] = compute_error_breakdown(y_124, p, test_124)

for m_name, p in preds_paddy.items():
    error_anatomy['paddy_1507'][m_name] = compute_error_breakdown(y_paddy, p, test_paddy)

# 12 healthy_trunk images predicted as Grade 2 or 3
ht_mask = (test_502['disease_class'] == 'healthy_trunk').values
ht_df = test_502[ht_mask].reset_index(drop=True)
p_ht = preds_502['CORN Seed 100'][ht_mask]
severe_ht_idx = np.where(p_ht >= 2)[0]

np.random.seed(42)
sample_12_idx = np.random.choice(severe_ht_idx, size=min(12, len(severe_ht_idx)), replace=False)

thumb_w, thumb_h = 300, 300
pad_bottom = 70
card_w = thumb_w
card_h = thumb_h + pad_bottom

montage_img = Image.new('RGB', (card_w * 4, card_h * 3), color=(245, 245, 245))
draw = ImageDraw.Draw(montage_img)

font_path = '/usr/share/fonts/TTF/JetBrainsMonoNerdFontPropo-Bold.ttf'
try:
    font_bold = ImageFont.truetype(font_path, 15)
    font_sub = ImageFont.truetype(font_path, 12)
except Exception:
    font_bold = ImageFont.load_default()
    font_sub = font_bold

for i, idx in enumerate(sample_12_idx):
    row_i = i // 4
    col_i = i % 4
    x_offset = col_i * card_w
    y_offset = row_i * card_h
    
    img_path = ht_df.iloc[idx]['image_path']
    fn = ht_df.iloc[idx]['filename']
    true_g = int(ht_df.iloc[idx]['severity_grade'])
    pred_g = int(p_ht[idx])
    
    im = Image.open(img_path).convert('RGB').resize((thumb_w, thumb_h))
    montage_img.paste(im, (x_offset, y_offset))
    
    short_fn = fn if len(fn) <= 30 else fn[:14] + '...' + fn[-13:]
    text_line1 = f"True: Grade {true_g} | Pred: Grade {pred_g}"
    text_line2 = short_fn
    
    draw.rectangle([x_offset, y_offset + thumb_h, x_offset + card_w, y_offset + card_h], fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((x_offset + 10, y_offset + thumb_h + 10), text_line1, fill=(210, 30, 30), font=font_bold)
    draw.text((x_offset + 10, y_offset + thumb_h + 38), text_line2, fill=(80, 80, 80), font=font_sub)

montage_path = 'outputs/final_v2/healthy_trunk_errors.png'
montage_img.save(montage_path)
print(f"Saved healthy_trunk error montage to {montage_path}")

final_json = {
    'stage_1_1': {
        'old_124': res_124,
        'clean_146': res_146,
        'expanded_502': res_502,
        'paddy_1507': res_paddy,
        'paddy_grade_distribution': {
            'train': paddy_train_ct,
            'test': paddy_test_ct
        }
    },
    'stage_1_2_arecanut_502': arec_within_res,
    'stage_1_2_arecanut_146': arec_146_within_res,
    'stage_1_2_paddy': paddy_within_res,
    'stage_1_3_error_anatomy': error_anatomy,
    'stage_1_3_montage_path': montage_path
}

with open('outputs/final_v2/stage1_full_report.json', 'w') as f:
    json.dump(final_json, f, indent=2)

print("\nALL STAGE 1 COMPUTATIONS COMPLETE! Saved to outputs/final_v2/stage1_full_report.json")
