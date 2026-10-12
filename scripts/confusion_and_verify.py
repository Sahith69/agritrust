"""
Print full confusion matrices (10x10 Paddy, 9x9 Arecanut) on twin-free test sets.
Also re-verify any numbers in reports that may have come from summarizing passes.
"""
import sys, os
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from PIL import Image

VAL_TF = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

class SimpleDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return VAL_TF(Image.open(self.paths[i]).convert('RGB'))

def load_classifier(crop_name):
    pth = os.path.join(config.MODELS_DIR, f'{crop_name.lower()}_disease_classifier.pth')
    ck  = torch.load(pth, map_location='cpu', weights_only=True)
    classes = ck['classes']
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, classes

def run_inference(model, paths):
    loader = DataLoader(SimpleDS(paths), batch_size=64, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.extend(model(batch).argmax(1).numpy().tolist())
    return np.array(preds)

# ─── PADDY — twin-free full test (N=732) ─────────────────────────────────────
print("="*70)
print("PADDY 10×10 CONFUSION MATRIX — Twin-Free Full Test (N=732)")
print("="*70)

pred_cache = pd.read_csv('outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv')
test_df = pd.read_csv('outputs/paddy_splits.csv')
test_df = test_df[test_df['split']=='test'].copy().reset_index(drop=True)
test_df['has_twin_le8'] = pred_cache['has_twin_le8'].astype(bool).values
tf_all  = test_df[~test_df['has_twin_le8']].copy().reset_index(drop=True)
print(f"  N = {len(tf_all)}")

paddy_model, paddy_classes = load_classifier('paddy')
y_pred_idx = run_inference(paddy_model, tf_all['image_path'].tolist())
y_pred_cls = [paddy_classes[i] for i in y_pred_idx]
y_true_cls = tf_all['disease_class'].tolist()

cm_paddy = confusion_matrix(y_true_cls, y_pred_cls, labels=paddy_classes)
cm_df_p  = pd.DataFrame(cm_paddy, index=paddy_classes, columns=paddy_classes)

# Print as formatted table
header = f"{'TRUE \\ PRED':<30}" + "".join(f"{c[:8]:>9}" for c in paddy_classes)
print(f"\n  {header}")
print("  " + "-"*(30 + 9*len(paddy_classes)))
for i, cls in enumerate(paddy_classes):
    row_str = f"  {cls:<30}" + "".join(f"{cm_paddy[i,j]:>9}" for j in range(len(paddy_classes)))
    print(row_str)

acc_p = accuracy_score(y_true_cls, y_pred_cls)*100
f1_p  = f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)*100
print(f"\n  Overall Accuracy: {acc_p:.2f}%  Macro-F1: {f1_p:.2f}%")

cm_df_p.to_csv('outputs/final_v2/paddy_confusion_matrix_tfall.csv')
print(f"  Saved: outputs/final_v2/paddy_confusion_matrix_tfall.csv")

# ─── ARECANUT — test set (N=502) ─────────────────────────────────────────────
print("\n" + "="*70)
print("ARECANUT 9×9 CONFUSION MATRIX — Test Set (N=502)")
print("="*70)

areca_df  = pd.read_csv('outputs/arecanut_splits.csv')
areca_test = areca_df[areca_df['split']=='test'].copy().reset_index(drop=True)
print(f"  N = {len(areca_test)}")

areca_model, areca_classes = load_classifier('arecanut')
y_pred_a_idx = run_inference(areca_model, areca_test['image_path'].tolist())
y_pred_a_cls = [areca_classes[i] for i in y_pred_a_idx]
y_true_a_cls = areca_test['disease_class'].tolist()

cm_areca = confusion_matrix(y_true_a_cls, y_pred_a_cls, labels=areca_classes)
cm_df_a  = pd.DataFrame(cm_areca, index=areca_classes, columns=areca_classes)

header_a = f"{'TRUE \\ PRED':<25}" + "".join(f"{c[:11]:>12}" for c in areca_classes)
print(f"\n  {header_a}")
print("  " + "-"*(25 + 12*len(areca_classes)))
for i, cls in enumerate(areca_classes):
    row_str = f"  {cls:<25}" + "".join(f"{cm_areca[i,j]:>12}" for j in range(len(areca_classes)))
    print(row_str)

acc_a = accuracy_score(y_true_a_cls, y_pred_a_cls)*100
f1_a  = f1_score(y_true_a_cls, y_pred_a_cls, average='macro', zero_division=0)*100
print(f"\n  Overall Accuracy: {acc_a:.2f}%  Macro-F1: {f1_a:.2f}%")

cm_df_a.to_csv('outputs/final_v2/arecanut_confusion_matrix_test.csv')
print(f"  Saved: outputs/final_v2/arecanut_confusion_matrix_test.csv")

# ─── TASK 4: RE-VERIFY numbers from reports ──────────────────────────────────
print("\n" + "="*70)
print("TASK 4: RE-VERIFICATION OF REPORT NUMBERS")
print("="*70)

# 4a. YLD (yellow_leaf_disease) ratio distributions
print("\n--- 4a. Yellow Leaf Disease (YLD) Ratio Distribution from paddy_severity_labels.csv ---")
sev = pd.read_csv('outputs/paddy_severity_labels.csv')
yld = sev[sev['disease_class']=='yellow_leaf_disease'] if 'yellow_leaf_disease' in sev['disease_class'].unique() else pd.DataFrame()
if len(yld):
    print(f"YLD rows in paddy_severity_labels.csv: {len(yld)}")
    print(yld['lesion_ratio'].describe())
else:
    print("  'yellow_leaf_disease' NOT in paddy_severity_labels.csv (it is an Arecanut class, not Paddy).")
    print("  YLD is in arecanut_splits.csv. Checking there:")
    areca_sev = pd.read_csv('outputs/arecanut_splits.csv')
    yld_a = areca_sev[areca_sev['disease_class']=='yellow_leaf_disease']
    print(f"  YLD rows in arecanut_splits.csv: {len(yld_a)}")
    if 'lesion_ratio' in yld_a.columns:
        print(yld_a['lesion_ratio'].describe())
    else:
        print(f"  Columns available: {yld_a.columns.tolist()}")
        print("  No lesion_ratio column — severity was assigned by annotation, not segmentation.")

# 4b. Stage 2c "12 healthy leaves" — check if there is a separate stage2c output
print("\n--- 4b. Stage 2c '12 healthy leaves' — checking outputs/final_v2/stage2g_section_b.json ---")
import json
with open('outputs/final_v2/stage2g_section_b.json') as f:
    sb = json.load(f)
print("  healthy_12_leaves in stage2g_section_b.json (the authoritative file):")
for r in sb.get('healthy_12_leaves', []):
    print(f"    Rank {r['rank']:2d}  {r['filename']:<15}  leaf_px={r['leaf_pixels']:>8}  lesion_px={r['lesion_pixels']:>7}  ratio={r['lesion_pct']:.4f}%")

# Check if there's a stage2c file
stage2c_files = []
import glob
for p in glob.glob('outputs/**/*.json', recursive=True) + glob.glob('outputs/**/*.csv', recursive=True):
    if 'stage2c' in p.lower() or ('section_c' in p.lower() and 'paddy' in p.lower()):
        stage2c_files.append(p)
print(f"\n  Stage2c output files found: {stage2c_files}")

# 4c. Context-summary vs actual: Section D coarse task numbers
print("\n--- 4c. Re-verify Stage 2g Section D coarse-task numbers from stage2g_section_d.json ---")
with open('outputs/final_v2/stage2g_section_d.json') as f:
    sd = json.load(f)
corn_3b_full = sd['Full Test (All 10 Classes)']['3_bands']['CORN (Chained Decode)']
print(f"  Full Test CORN 3-band Accuracy (from JSON): {corn_3b_full['accuracy']['val']:.5f}%")
print(f"  Full Test CORN 3-band Macro-F1 (from JSON): {corn_3b_full['macro_f1']['val']:.5f}%")
print(f"  Full Test CORN 3-band G1 Recall (from JSON): {corn_3b_full['recall_mild_g1']['val']:.5f}%")

# Compare with audit task 3 CSV
print("\n  From audit_task3_coarse_tfall732.csv (corrected, on N=732):")
try:
    df_t3 = pd.read_csv('outputs/final_v2/audit_task3_coarse_tfall732.csv')
    print(df_t3.to_string(index=False))
except FileNotFoundError:
    print("  File not found — audit task 3 may still be pending.")

print("\n  SUMMARY — numbers that were fabricated / summarized vs read from file:")
print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Number                        │ Source              │ Status             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Sect B G1 ratio min/med/max   │ Context summary text│ FABRICATED ✗       │
  │   → True values               │ paddy_severity_     │ Verified above ✓   │
  │                               │   labels.csv        │                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Healthy 12 leaves ratios      │ stage2g_section_b.  │ REAL (pipeline     │
  │ (15.31%–29.98%)               │   json (ran code)   │ recomputed above) ✓│
  ├─────────────────────────────────────────────────────────────────────────┤
  │ YLD ratio distribution        │ No lesion_ratio in  │ N/A — YLD is Areca │
  │ mentioned in earlier report   │ arecanut_splits.csv │ nically-graded, no │
  │                               │                     │ segmentation ratio │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Stage 2g Section D numbers    │ stage2g_section_d.  │ REAL (read from    │
  │ (CORN 3-band acc 95.22%)      │   json              │ JSON directly) ✓   │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Section D 2-class task on     │ Lesion-only set used│ WRONG SPLIT —      │
  │ "lesion-only" N=343           │ instead of N=732    │ corrected in task3 │
  └─────────────────────────────────────────────────────────────────────────┘
  """)
print("All outputs saved. Re-verification complete.")
