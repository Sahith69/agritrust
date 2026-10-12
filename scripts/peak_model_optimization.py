import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error, f1_score
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("PEAK MODEL OPTIMIZATION: LIGHTNING-FAST TENSOR TTA & ENSEMBLE")
print("=" * 80, flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Paddy splits
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
val_df = df_paddy[df_paddy['split'] == 'val'].copy().reset_index(drop=True)
test_df = df_paddy[df_paddy['split'] == 'test'].copy().reset_index(drop=True)

pred_cache = pd.read_csv('outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv')
test_df['comp_le8'] = pred_cache['comp_le8'].values
test_df['has_twin_le8'] = pred_cache['has_twin_le8'].astype(bool).values

classes_7 = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'downy_mildew', 'tungro'
]

test_tf_7 = test_df[(~test_df['has_twin_le8']) & (test_df['disease_class'].isin(classes_7))].copy().reset_index(drop=True)
test_tf_all = test_df[~test_df['has_twin_le8']].copy().reset_index(drop=True)

print(f"Val set size: {len(val_df)}", flush=True)
print(f"Full Test size: {len(test_df)}", flush=True)
print(f"Twin-Free Full Test size: {len(test_tf_all)}", flush=True)
print(f"Twin-Free 7-Class Lesion Test size: {len(test_tf_7)}", flush=True)

# Define models to load
models_config = [
    ('ResNet-18 CORN', 'outputs/models/paddy_ordinal.pth', 'resnet18', True),
    ('ResNet-34 CORN', 'outputs/models/architecture_comparison/paddy_resnet34_ordinal.pth', 'resnet34', True),
    ('MobileNet-V2 CORN', 'outputs/models/architecture_comparison/paddy_mobilenet_v2_ordinal.pth', 'mobilenet_v2', True),
    ('ResNet-18 Softmax', 'outputs/models/paddy_baseline.pth', 'resnet18', False)
]

loaded_models = []
for label, ckpt_path, backbone, is_ord in models_config:
    if os.path.exists(ckpt_path):
        m = CropSeverityModel(backbone=backbone, num_classes=4, ordinal=is_ord, pretrained=False)
        state = torch.load(ckpt_path, map_location='cpu')
        m.load_state_dict(state if 'model_state_dict' not in state else state['model_state_dict'])
        m.to(device).eval()
        loaded_models.append({'label': label, 'model': m, 'backbone': backbone, 'ordinal': is_ord})
        print(f"Loaded {label} from {ckpt_path}", flush=True)

tf_test = get_transforms('test')

class FastPaddyDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
        
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

def get_model_outputs_tensor_tta(model_info, df, use_tta=False):
    model = model_info['model']
    is_ord = model_info['ordinal']
    ds = FastPaddyDataset(df['image_path'].values, tf_test)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not use_tta:
                out = model(batch)
                if is_ord:
                    prob = torch.sigmoid(out).cpu().numpy()
                else:
                    prob = F.softmax(out, dim=1).cpu().numpy()
                all_probs.append(prob)
            else:
                # Fast Tensor TTA: orig, hflip, vflip, hvflip
                b_orig = batch
                b_hflip = torch.flip(batch, dims=[3])
                b_vflip = torch.flip(batch, dims=[2])
                b_hvflip = torch.flip(batch, dims=[2, 3])
                
                b_stacked = torch.cat([b_orig, b_hflip, b_vflip, b_hvflip], dim=0)
                out_stacked = model(b_stacked)
                if is_ord:
                    p_stacked = torch.sigmoid(out_stacked)
                else:
                    p_stacked = F.softmax(out_stacked, dim=1)
                    
                # Split back into 4 TTA versions and average
                p_chunks = torch.chunk(p_stacked, 4, dim=0)
                p_mean = torch.stack(p_chunks, dim=0).mean(dim=0).cpu().numpy()
                all_probs.append(p_mean)
                
    return np.vstack(all_probs)

print("\nExtracting fast tensor predictions...", flush=True)

val_preds = {}
test_preds = {}
test_tf_7_preds = {}

for minfo in loaded_models:
    lbl = minfo['label']
    print(f"Processing {lbl}...", flush=True)
    val_preds[lbl + ' (Standard)'] = get_model_outputs_tensor_tta(minfo, val_df, use_tta=False)
    val_preds[lbl + ' (TTA)'] = get_model_outputs_tensor_tta(minfo, val_df, use_tta=True)
    
    test_preds[lbl + ' (Standard)'] = get_model_outputs_tensor_tta(minfo, test_df, use_tta=False)
    test_preds[lbl + ' (TTA)'] = get_model_outputs_tensor_tta(minfo, test_df, use_tta=True)
    
    test_tf_7_preds[lbl + ' (Standard)'] = get_model_outputs_tensor_tta(minfo, test_tf_7, use_tta=False)
    test_tf_7_preds[lbl + ' (TTA)'] = get_model_outputs_tensor_tta(minfo, test_tf_7, use_tta=True)

# Build Ensembles
corn_labels_std = [m['label'] + ' (Standard)' for m in loaded_models if m['ordinal']]
corn_labels_tta = [m['label'] + ' (TTA)' for m in loaded_models if m['ordinal']]

val_preds['CORN Architecture Ensemble (Standard)'] = np.mean([val_preds[k] for k in corn_labels_std], axis=0)
val_preds['CORN Architecture Ensemble (TTA)'] = np.mean([val_preds[k] for k in corn_labels_tta], axis=0)

test_preds['CORN Architecture Ensemble (Standard)'] = np.mean([test_preds[k] for k in corn_labels_std], axis=0)
test_preds['CORN Architecture Ensemble (TTA)'] = np.mean([test_preds[k] for k in corn_labels_tta], axis=0)

test_tf_7_preds['CORN Architecture Ensemble (Standard)'] = np.mean([test_tf_7_preds[k] for k in corn_labels_std], axis=0)
test_tf_7_preds['CORN Architecture Ensemble (TTA)'] = np.mean([test_tf_7_preds[k] for k in corn_labels_tta], axis=0)

# Fully Vectorized CORN Decode Helper
def decode_corn_fast(probs, thresholds=(0.5, 0.5, 0.5)):
    p0 = np.where(probs[:, 0] > thresholds[0], probs[:, 0], 0.0)
    p1 = np.where(probs[:, 1] > thresholds[1], p0 * probs[:, 1], 0.0)
    p2 = np.where(probs[:, 2] > thresholds[2], p1 * probs[:, 2], 0.0)
    
    pk = np.zeros((len(probs), 4))
    pk[:, 0] = 1.0 - p0
    pk[:, 1] = p0 - p1
    pk[:, 2] = p1 - p2
    pk[:, 3] = p2
    return np.argmax(pk, axis=1)

# Ultra-fast NumPy QWK
def fast_qwk(y_true, y_pred, n_classes=4):
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(n_classes, n_classes)
    hist_true = cm.sum(axis=1)
    hist_pred = cm.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / float(len(y_true))
    i_mat = np.arange(n_classes)
    w_mat = ((i_mat[:, None] - i_mat[None, :]) ** 2) / float((n_classes - 1) ** 2)
    num = np.sum(w_mat * cm)
    den = np.sum(w_mat * E)
    return 1.0 - (num / den) if den > 0 else 0.0

# Calibrate thresholds on Val set
print("\nCalibrating ordinal thresholds on Val set...", flush=True)
best_thresholds = {}
for k, v_probs in val_preds.items():
    if 'Softmax' in k: continue
    y_val = val_df['severity_grade'].values
    
    best_qwk = -1.0
    best_th = (0.5, 0.5, 0.5)
    
    for t1 in [0.4, 0.5, 0.6]:
        for t2 in [0.4, 0.5, 0.6]:
            for t3 in [0.4, 0.5, 0.6]:
                preds_val = decode_corn_fast(v_probs, (t1, t2, t3))
                q = fast_qwk(y_val, preds_val)
                if q > best_qwk:
                    best_qwk = q
                    best_th = (t1, t2, t3)
    best_thresholds[k] = best_th
    print(f"Val Calibrated Thresholds for {k:40s}: ({best_th[0]:.2f}, {best_th[1]:.2f}, {best_th[2]:.2f}) | Val QWK: {best_qwk:.4f}", flush=True)

# Evaluate metrics on Test sets
def eval_all_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    acc = accuracy_score(y_true, y_pred) * 100.0
    w1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100.0
    mae = mean_absolute_error(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro') * 100.0
    return {'qwk': qwk, 'acc': acc, 'w1': w1, 'mae': mae, 'f1_macro': f1_macro}

print("\n" + "=" * 80, flush=True)
print("FINAL EVALUATION ON ZERO-LEAK TWIN-FREE TEST SET (7 LESION CLASSES, N=343)", flush=True)
print("=" * 80, flush=True)

results_summary = []
y_test_tf7 = test_tf_7['severity_grade'].values

for k, t_probs in test_tf_7_preds.items():
    if 'Softmax' in k:
        y_pred = np.argmax(t_probs, axis=1)
        th_used = 'N/A Argmax'
    else:
        th = best_thresholds.get(k, (0.5, 0.5, 0.5))
        y_pred = decode_corn_fast(t_probs, th)
        th_used = f"({th[0]:.2f}, {th[1]:.2f}, {th[2]:.2f})"
        
    m = eval_all_metrics(y_test_tf7, y_pred)
    results_summary.append({
        'Model Configuration': k,
        'Thresholds': th_used,
        'QWK': f"{m['qwk']:.4f}",
        'Exact Acc %': f"{m['acc']:.2f}%",
        'Within-1 %': f"{m['w1']:.2f}%",
        'MAE': f"{m['mae']:.4f}",
        'Macro-F1 %': f"{m['f1_macro']:.2f}%"
    })

df_res = pd.DataFrame(results_summary)
print(df_res.to_string(index=False), flush=True)

df_res.to_csv('outputs/final_v2/peak_model_optimization_results.csv', index=False)
print("\nSaved peak optimization results to outputs/final_v2/peak_model_optimization_results.csv", flush=True)
