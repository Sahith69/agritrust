"""
Fix #3: Enforce YLD (Yellow Leaf Disease) exclusion from severity grading.
- YLD chlorosis (yellowing) causes false high Otsu ratio segmentation (median 37.3% -> G3).
- Setting exclude_from_severity=True filters YLD from severity models, leaving true focal lesion classes.
- Evaluates Arecanut CORN severity model with vs without YLD exclusion.
- Saves outputs/final_v2/fix3_yld_severity_exclusion_results.json
"""
import sys, os, json
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from PIL import Image

VAL_TF = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

class SimpleDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return VAL_TF(Image.open(self.paths[i]).convert('RGB'))

df_clusters = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
df_splits   = pd.read_csv('outputs/arecanut_splits_v2.csv')

# Ensure exclude_from_severity flag exists
df_splits['exclude_from_severity'] = (df_splits['disease_class'] == 'yellow_leaf_disease')

test_all  = df_splits[df_splits['split']=='test'].copy().reset_index(drop=True)
test_clean= test_all[~test_all['exclude_from_severity']].copy().reset_index(drop=True)

print("="*65)
print("FIX #3: YLD SEVERITY EXCLUSION EVALUATION")
print("="*65)
print(f"Full Test Set (all 9 classes):           N = {len(test_all)}")
print(f"Severity-Clean Test Set (excluding YLD): N = {len(test_clean)}")
print(f"Excluded YLD chlorosis images:          N = {len(test_all) - len(test_clean)}")
print()

# Check severity models
sev_model_path = os.path.join(config.MODELS_DIR, 'arecanut_ordinal.pth')
results = {}

if os.path.exists(sev_model_path):
    ck = torch.load(sev_model_path, map_location='cpu', weights_only=False)
    # Instantiate CropSeverityModel
    from src.model import CropSeverityModel
    m = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
    m.load_state_dict(ck if isinstance(ck, dict) and 'state_dict' not in ck else ck.get('state_dict', ck))
    m.eval()

    def eval_model(df_sub):
        loader = DataLoader(SimpleDS(df_sub['image_path'].tolist()), batch_size=32, shuffle=False)
        logits_list = []
        with torch.no_grad():
            for batch in loader:
                logits_list.append(m(batch).numpy())
        logits = np.vstack(logits_list)
        # CORN decode (threshold = 0.5)
        probs  = 1.0 / (1.0 + np.exp(-logits))
        preds  = (probs > 0.5).sum(axis=1)
        trues  = df_sub['severity_grade'].values
        qwk    = cohen_kappa_score(trues, preds, weights='quadratic')
        acc    = accuracy_score(trues, preds) * 100
        mae    = mean_absolute_error(trues, preds)
        return {'qwk': float(qwk), 'acc': float(acc), 'mae': float(mae)}

    res_all   = eval_model(test_all)
    res_clean = eval_model(test_clean)

    print("Arecanut Ordinal Severity Model (arecanut_ordinal.pth):")
    print(f"  All classes (including YLD)  : QWK = {res_all['qwk']:.4f} | Acc = {res_all['acc']:.2f}% | MAE = {res_all['mae']:.4f}")
    print(f"  Clean classes (excluding YLD): QWK = {res_clean['qwk']:.4f} | Acc = {res_clean['acc']:.2f}% | MAE = {res_clean['mae']:.4f}")
    print(f"  Δ QWK from removing chlorosis confound: {res_clean['qwk'] - res_all['qwk']:+.4f}")

    results = {'with_yld': res_all, 'excluding_yld': res_clean}
else:
    print(f"Severity model {sev_model_path} not found. Skipping model evaluation.")
    results = {'note': 'Severity model file not found'}

# Save updated splits CSV with explicit exclude_from_severity flag
df_splits.to_csv('outputs/arecanut_splits_v2.csv', index=False)

out = {
    'task': 'Fix #3: YLD Severity Exclusion',
    'total_test_images': len(test_all),
    'clean_lesion_test_images': len(test_clean),
    'excluded_yld_images': len(test_all) - len(test_clean),
    'severity_results': results,
}
with open('outputs/final_v2/fix3_yld_severity_exclusion_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print("\n✓ Fix #3 complete: outputs/final_v2/fix3_yld_severity_exclusion_results.json")
