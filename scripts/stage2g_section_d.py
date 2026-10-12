import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2g - SECTION D: COARSE TASKS EVALUATION (3 BANDS & 2 CLASSES)")
print("=" * 80)

# Load predictions and splits
pred_path = 'outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv'
df_preds = pd.read_csv(pred_path)
print(f"Loaded predictions: {pred_path} (N={len(df_preds)})")

df_splits = pd.read_csv('outputs/paddy_splits.csv')
train_splits = df_splits[df_splits['split'] == 'train']
mode_per_class = train_splits.groupby('disease_class')['severity_grade'].agg(lambda s: int(s.mode()[0])).to_dict()
print(f"Train mode per disease class: {mode_per_class}")

df_preds['pred_mode'] = df_preds['disease_class'].map(mode_per_class).values

# Mappings
# 3 Bands: 0 = Healthy (G0), 1 = Mild (G1), 2 = Advanced (G2+G3)
def to_3bands(g):
    return np.where(g == 0, 0, np.where(g == 1, 1, 2))

# 2 Classes: 0 = Healthy (G0), 1 = Diseased (G1+G2+G3)
def to_2classes(g):
    return np.where(g == 0, 0, 1)

classes_7 = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'downy_mildew', 'tungro'
]

# Define test sets
subsets = {
    'Full Test (All 10 Classes)': {
        'df': df_preds.copy(),
        'N': len(df_preds),
        'has_healthy': True
    },
    'Lesion-Only Twin-Free (7 Classes)': {
        'df': df_preds[(~df_preds['has_twin_le8']) & (df_preds['disease_class'].isin(classes_7))].copy().reset_index(drop=True),
        'N': len(df_preds[(~df_preds['has_twin_le8']) & (df_preds['disease_class'].isin(classes_7))]),
        'has_healthy': False
    },
    'Twin-Free Full Test (10 Classes)': {
        'df': df_preds[~df_preds['has_twin_le8']].copy().reset_index(drop=True),
        'N': len(df_preds[~df_preds['has_twin_le8']]),
        'has_healthy': True
    }
}

models = [
    ('Mode Baseline', 'pred_mode'),
    ('CORN (Chained Decode)', 'pred_rule_ii_chained'),
    ('Softmax', 'pred_softmax')
]

B = 2000
np.random.seed(42)

results = {}

for s_name, s_info in subsets.items():
    sub_df = s_info['df']
    n_sub = s_info['N']
    has_h = s_info['has_healthy']
    
    y4_true = sub_df['severity_grade'].values
    y3_true = to_3bands(y4_true)
    y2_true = to_2classes(y4_true)
    
    clusters = sub_df['comp_le8'].values
    unique_clus = np.unique(clusters)
    n_clus = len(unique_clus)
    clus_map = {c: np.where(clusters == c)[0] for c in unique_clus}
    
    print(f"\n========================================================")
    print(f"SUBSET: {s_name} (N = {n_sub}, Clusters = {n_clus})")
    print(f"========================================================")
    
    sub_results = {'3_bands': {}, '2_classes': {}}
    
    for m_label, col_name in models:
        y4_pred = sub_df[col_name].values
        y3_pred = to_3bands(y4_pred)
        y2_pred = to_2classes(y4_pred)
        
        # 1. 3 Bands Point Metrics
        acc3 = accuracy_score(y3_true, y3_pred) * 100.0
        # If healthy is absent, compute macro-F1 over present classes [1, 2]
        labels3_present = [0, 1, 2] if has_h else [1, 2]
        f1_3 = f1_score(y3_true, y3_pred, average='macro', labels=labels3_present) * 100.0
        rec3 = recall_score(y3_true, y3_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
        
        # 2. 2 Classes Point Metrics
        acc2 = accuracy_score(y2_true, y2_pred) * 100.0
        labels2_present = [0, 1] if has_h else [1]
        f1_2 = f1_score(y2_true, y2_pred, average='macro', labels=labels2_present) * 100.0
        rec2 = recall_score(y2_true, y2_pred, average=None, labels=[0, 1], zero_division=0) * 100.0
        
        # Bootstrap
        b_acc3, b_f1_3, b_rec3_0, b_rec3_1, b_rec3_2 = [], [], [], [], []
        b_acc2, b_f1_2, b_rec2_0, b_rec2_1 = [], [], [], []
        
        for _ in range(B):
            sample_c = np.random.choice(unique_clus, size=n_clus, replace=True)
            idx_sample = np.concatenate([clus_map[c] for c in sample_c])
            
            yb3_true, yb3_pred = y3_true[idx_sample], y3_pred[idx_sample]
            yb2_true, yb2_pred = y2_true[idx_sample], y2_pred[idx_sample]
            
            # 3 bands
            b_acc3.append(accuracy_score(yb3_true, yb3_pred) * 100.0)
            b_f1_3.append(f1_score(yb3_true, yb3_pred, average='macro', labels=labels3_present) * 100.0)
            r3 = recall_score(yb3_true, yb3_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
            b_rec3_0.append(r3[0])
            b_rec3_1.append(r3[1])
            b_rec3_2.append(r3[2])
            
            # 2 classes
            b_acc2.append(accuracy_score(yb2_true, yb2_pred) * 100.0)
            b_f1_2.append(f1_score(yb2_true, yb2_pred, average='macro', labels=labels2_present) * 100.0)
            r2 = recall_score(yb2_true, yb2_pred, average=None, labels=[0, 1], zero_division=0) * 100.0
            b_rec2_0.append(r2[0])
            b_rec2_1.append(r2[1])
            
        def ci(arr):
            return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        
        sub_results['3_bands'][m_label] = {
            'accuracy': {'val': float(acc3), 'ci': ci(b_acc3)},
            'macro_f1': {'val': float(f1_3), 'ci': ci(b_f1_3)},
            'recall_healthy_g0': {'val': float(rec3[0]) if has_h else None, 'ci': ci(b_rec3_0) if has_h else None},
            'recall_mild_g1': {'val': float(rec3[1]), 'ci': ci(b_rec3_1)},
            'recall_advanced_g2g3': {'val': float(rec3[2]), 'ci': ci(b_rec3_2)}
        }
        
        sub_results['2_classes'][m_label] = {
            'accuracy': {'val': float(acc2), 'ci': ci(b_acc2)},
            'macro_f1': {'val': float(f1_2), 'ci': ci(b_f1_2)},
            'recall_healthy_g0': {'val': float(rec2[0]) if has_h else None, 'ci': ci(b_rec2_0) if has_h else None},
            'recall_diseased': {'val': float(rec2[1]), 'ci': ci(b_rec2_1)}
        }
        
        print(f"\nModel: {m_label}")
        print(f"  [3 Bands: Healthy / Mild / Advanced]")
        print(f"    • Accuracy: {acc3:.2f}% (95% CI: [{ci(b_acc3)[0]:.2f}%, {ci(b_acc3)[1]:.2f}%])")
        print(f"    • Macro-F1: {f1_3:.2f}% (95% CI: [{ci(b_f1_3)[0]:.2f}%, {ci(b_f1_3)[1]:.2f}%])")
        if has_h:
            print(f"    • Recall Healthy  (G0)   : {rec3[0]:.2f}% (95% CI: [{ci(b_rec3_0)[0]:.2f}%, {ci(b_rec3_0)[1]:.2f}%])")
        else:
            print(f"    • Recall Healthy  (G0)   : N/A (0 true instances in lesion-only test)")
        print(f"    • Recall Mild     (G1)   : {rec3[1]:.2f}% (95% CI: [{ci(b_rec3_1)[0]:.2f}%, {ci(b_rec3_1)[1]:.2f}%])")
        print(f"    • Recall Advanced (G2+G3): {rec3[2]:.2f}% (95% CI: [{ci(b_rec3_2)[0]:.2f}%, {ci(b_rec3_2)[1]:.2f}%])")
        
        print(f"  [2 Classes: Healthy vs Diseased]")
        print(f"    • Accuracy: {acc2:.2f}% (95% CI: [{ci(b_acc2)[0]:.2f}%, {ci(b_acc2)[1]:.2f}%])")
        print(f"    • Macro-F1: {f1_2:.2f}% (95% CI: [{ci(b_f1_2)[0]:.2f}%, {ci(b_f1_2)[1]:.2f}%])")
        if has_h:
            print(f"    • Recall Healthy  (G0)     : {rec2[0]:.2f}% (95% CI: [{ci(b_rec2_0)[0]:.2f}%, {ci(b_rec2_0)[1]:.2f}%])")
        else:
            print(f"    • Recall Healthy  (G0)     : N/A (0 true instances in lesion-only test)")
        print(f"    • Recall Diseased (G1+G2+G3): {rec2[1]:.2f}% (95% CI: [{ci(b_rec2_1)[0]:.2f}%, {ci(b_rec2_1)[1]:.2f}%])")

    results[s_name] = sub_results

with open('outputs/final_v2/stage2g_section_d.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved Stage 2g Section D results to outputs/final_v2/stage2g_section_d.json")
