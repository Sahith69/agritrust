import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("SECTION D: INSPECTION OF EXISTING HUMAN-GRADED DATA")
print("=" * 80)

# Files to inspect
files_to_inspect = {
    'blind_grading_gt.json': 'outputs/blind_grading_gt.json',
    'blind_grading_secret.json': 'outputs/blind_grading_secret.json',
    'real_validation_gt_secret.json': 'outputs/real_validation_gt_secret.json',
    'real_validation_set_30': 'outputs/real_validation_set_30',
    'real_validation_30.pdf': 'outputs/real_validation_30.pdf',
    'manual_crop_gradings.csv': 'outputs/manual_crop_gradings.csv'
}

inspection_results = {}

# 1. blind_grading_gt.json
with open('outputs/blind_grading_gt.json') as f:
    d1 = json.load(f)
df1 = pd.DataFrame(d1)
non_null_g1 = df1['severity_grade'].dropna().tolist()
mtime1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime('outputs/blind_grading_gt.json')))
inspection_results['blind_grading_gt.json'] = {
    'what_it_is': 'JSON array intended as a blind grading collection sheet / template for Paddy images.',
    'who_what_produced': 'Automated script created template with image_path, lesion_ratio, confidence; human grader started filling in grades but stopped after 3 images.',
    'date': mtime1,
    'image_count': len(d1),
    'filled_count': len(non_null_g1),
    'is_independent_human_grades': 'No (incomplete template with only 3 non-null entries out of 90 images; remaining 87 entries are NaN).'
}

# 2. blind_grading_secret.json
with open('outputs/blind_grading_secret.json') as f:
    d2 = json.load(f)
mtime2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime('outputs/blind_grading_secret.json')))
inspection_results['blind_grading_secret.json'] = {
    'what_it_is': 'Secret answer key for the 90 blind grading images, mapping image ID to pipeline-generated ground truth.',
    'who_what_produced': 'Automated computer-vision pipeline (Phase 1 segmentation & Horsfall-Barratt thresholding).',
    'date': mtime2,
    'image_count': len(d2),
    'is_independent_human_grades': 'No (100% pipeline-generated synthetic ground truth; zero human input).'
}

# 3. real_validation_gt_secret.json
with open('outputs/real_validation_gt_secret.json') as f:
    d3 = json.load(f)
mtime3 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime('outputs/real_validation_gt_secret.json')))
inspection_results['real_validation_gt_secret.json'] = {
    'what_it_is': 'Secret answer key for the 30-image pilot validation set, recording pipeline_grade and lesion_ratio.',
    'who_what_produced': 'Automated segmentation pipeline (Phase 1).',
    'date': mtime3,
    'image_count': len(d3),
    'is_independent_human_grades': 'No (100% automated pipeline labels, stored as secret ground truth to score against future human evaluations).'
}

# 4. real_validation_set_30
rv_files = sorted(os.listdir('outputs/real_validation_set_30'))
mtime4 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime('outputs/real_validation_set_30')))
inspection_results['real_validation_set_30'] = {
    'what_it_is': 'Directory containing 30 raw Paddy leaf JPEG images numbered 01 to 30.',
    'who_what_produced': 'Extracted by split_pilot_val.py from the Paddy Doctor dataset for validation.',
    'date': mtime4,
    'image_count': len(rv_files),
    'is_independent_human_grades': 'N/A (image files only; no labels or grades attached).'
}

# 5. real_validation_30.pdf
mtime5 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime('outputs/real_validation_30.pdf')))
inspection_results['real_validation_30.pdf'] = {
    'what_it_is': 'A printable visual grading booklet displaying the 30 Paddy leaf images from real_validation_set_30.',
    'who_what_produced': 'Automated script generated booklet for human grading.',
    'date': mtime5,
    'image_count': 30,
    'is_independent_human_grades': 'N/A (unfilled blank grading booklet; no human grades were filled or recorded).'
}

# 6. manual_crop_gradings.csv
df_mc = pd.read_csv('outputs/manual_crop_gradings.csv')
mtime6 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime('outputs/manual_crop_gradings.csv')))
inspection_results['manual_crop_gradings.csv'] = {
    'what_it_is': 'Log of 22 interactive grading trials on Arecanut bud_borer images.',
    'who_what_produced': 'Interactive user logging from the Streamlit verification app (app.py) over a 4-minute window.',
    'date': mtime6,
    'image_count': len(df_mc),
    'is_independent_human_grades': 'Partially human-entered, but NOT independent (grader evaluated images while model_grade and cv_lesion_ratio were simultaneously displayed on screen, introducing severe anchoring bias; covers only 22 images of a single class, bud_borer).'
}

print("Inspection Summary:")
for k, v in inspection_results.items():
    print(f"\n[{k}]")
    for prop, val in v.items():
        print(f"  {prop}: {val}")

# Agreement analysis on manual_crop_gradings.csv (the only file with manual grades)
print("\n" + "-" * 50)
print("AGREEMENT ANALYSIS ON manual_crop_gradings.csv (N=22)")
print("-" * 50)

y_manual = df_mc['manual_grade'].values
y_model = df_mc['model_grade'].values

# Compute auto-label from cv_lesion_ratio using Eq. 2 thresholds
def ratio_to_grade_eq2(r):
    if r <= 0.0: return 0
    elif r <= 0.05: return 1
    elif r <= 0.15: return 2
    else: return 3

y_auto = np.array([ratio_to_grade_eq2(r) for r in df_mc['cv_lesion_ratio'].values])

def compute_detailed_agreement(y1, y2, K=4):
    y1 = np.array(y1, dtype=int)
    y2 = np.array(y2, dtype=int)
    n = len(y1)
    
    # Quadratic weights
    w = 1.0 - ((np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2) / ((K - 1) ** 2)
    
    conf = np.zeros((K, K), dtype=int)
    for a, b in zip(y1, y2):
        conf[a, b] += 1
        
    p = conf / n
    pa_exact = np.trace(p)
    pa_weighted = np.sum(w * p)
    
    # Marginals
    pi_bar = 0.5 * (p.sum(axis=1) + p.sum(axis=0))
    
    # Gwet AC2 chance agreement (quadratic weighted)
    pe_ac2 = np.sum(w * (pi_bar[:, None] * pi_bar[None, :]))
    ac2 = (pa_weighted - pe_ac2) / (1.0 - pe_ac2) if pe_ac2 < 1.0 else 1.0
    
    # Cohen's QWK
    qwk = cohen_kappa_score(y1, y2, weights='quadratic', labels=[0,1,2,3])
    
    return {
        'exact_accuracy': float(pa_exact),
        'weighted_agreement_pa': float(pa_weighted),
        'qwk': float(qwk),
        'gwet_ac2': float(ac2),
        'confusion_matrix': conf.tolist()
    }

ag_model = compute_detailed_agreement(y_manual, y_model)
ag_auto = compute_detailed_agreement(y_manual, y_auto)

print("Manual vs. Model Grade:")
print(f"  Exact Accuracy:     {ag_model['exact_accuracy']*100:.2f}% (7/22)")
print(f"  Weighted Agreement: {ag_model['weighted_agreement_pa']:.4f}")
print(f"  QWK:                {ag_model['qwk']:.4f}")
print(f"  Gwet AC2:           {ag_model['gwet_ac2']:.4f}")
print("  Confusion Matrix (Row=Manual [0..3], Col=Model [0..3]):")
for r_i, row in enumerate(ag_model['confusion_matrix']):
    print(f"    Grade {r_i}: {row}")

print("\nManual vs. Auto CV Pipeline Grade:")
print(f"  Exact Accuracy:     {ag_auto['exact_accuracy']*100:.2f}% (7/22)")
print(f"  Weighted Agreement: {ag_auto['weighted_agreement_pa']:.4f}")
print(f"  QWK:                {ag_auto['qwk']:.4f}")
print(f"  Gwet AC2:           {ag_auto['gwet_ac2']:.4f}")
print("  Confusion Matrix (Row=Manual [0..3], Col=Auto [0..3]):")
for r_i, row in enumerate(ag_auto['confusion_matrix']):
    print(f"    Grade {r_i}: {row}")

with open('outputs/final_v2/stage2_section_d.json', 'w') as f:
    json.dump({
        'inspected_files': inspection_results,
        'manual_crop_gradings_agreement': {
            'manual_vs_model': ag_model,
            'manual_vs_auto': ag_auto
        }
    }, f, indent=2)

print("\nSaved Section D results to outputs/final_v2/stage2_section_d.json")
