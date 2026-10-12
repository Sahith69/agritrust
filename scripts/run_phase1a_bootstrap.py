import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

from src.model import CropSeverityModel
from src.dataset import CropDiseaseDataset, get_transforms
from src.corn_loss import corn_probs_to_grade

start_time = time.time()
os.makedirs('/home/sahith/Projects/Crop/outputs/improvements', exist_ok=True)
device = 'cpu'

# 1. Load predictions for all 6 models
def get_model_preds(model_path, backbone, split_csv, ordinal=True):
    ds = CropDiseaseDataset(split_csv, 'test', get_transforms('test'))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model = CropSeverityModel(backbone=backbone, num_classes=config.NUM_SEVERITY_GRADES, ordinal=ordinal, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in loader:
            imgs, labels = b[0], b[1]
            outs = model(imgs)
            if ordinal:
                preds, _ = corn_probs_to_grade(torch.sigmoid(outs))
            else:
                preds = torch.argmax(torch.softmax(outs, dim=1), dim=1)
            y_pred.extend(preds.numpy())
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)

print("Extracting predictions for all 6 models...")
# Paddy test set
paddy_test_df = pd.read_csv(config.PADDY_SPLIT_CSV)
paddy_test_df = paddy_test_df[paddy_test_df['split'] == 'test'].reset_index(drop=True)

paddy_corn_true, paddy_corn_pred = get_model_preds(
    os.path.join(config.MODELS_DIR, 'paddy_ordinal.pth'), 'resnet18', config.PADDY_SPLIT_CSV, ordinal=True
)
paddy_base_true, paddy_base_pred = get_model_preds(
    os.path.join(config.MODELS_DIR, 'paddy_baseline.pth'), 'resnet18', config.PADDY_SPLIT_CSV, ordinal=False
)

# Arecanut test set
areca_test_df = pd.read_csv(config.ARECANUT_SPLIT_CSV)
areca_test_df = areca_test_df[areca_test_df['split'] == 'test'].reset_index(drop=True)

areca_corn_true, areca_corn_pred = get_model_preds(
    os.path.join(config.MODELS_DIR, 'arecanut_ordinal.pth'), 'resnet18', config.ARECANUT_SPLIT_CSV, ordinal=True
)
areca_base_true, areca_base_pred = get_model_preds(
    os.path.join(config.MODELS_DIR, 'arecanut_baseline.pth'), 'resnet18', config.ARECANUT_SPLIT_CSV, ordinal=False
)
areca_r34_true, areca_r34_pred = get_model_preds(
    os.path.join(config.MODELS_DIR, 'architecture_comparison/arecanut_resnet34_ordinal.pth'), 'resnet34', config.ARECANUT_SPLIT_CSV, ordinal=True
)
areca_mob_true, areca_mob_pred = get_model_preds(
    os.path.join(config.MODELS_DIR, 'architecture_comparison/arecanut_mobilenet_v2_ordinal.pth'), 'mobilenet_v2', config.ARECANUT_SPLIT_CSV, ordinal=True
)

def compute_metrics(y_true, y_pred):
    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    return {'qwk': qwk, 'mae': mae, 'exact_accuracy': acc, 'macro_f1': f1}

# Bootstrap settings
N_BOOTSTRAP = 2000
np.random.seed(42)

# --- A. Arecanut Cluster Bootstrap by parent_id ---
parent_ids = areca_test_df['parent_id'].values
unique_parents = np.unique(parent_ids)
parent_to_indices = {pid: np.where(parent_ids == pid)[0] for pid in unique_parents}

areca_models = {
    'Arecanut ResNet-18 (CORN)': (areca_corn_true, areca_corn_pred),
    'Arecanut ResNet-18 (Softmax)': (areca_base_true, areca_base_pred),
    'Arecanut ResNet-34 (CORN)': (areca_r34_true, areca_r34_pred),
    'Arecanut MobileNetV2 (CORN)': (areca_mob_true, areca_mob_pred),
}

areca_bootstrap_results = {k: {'qwk': [], 'mae': [], 'exact_accuracy': [], 'macro_f1': []} for k in areca_models}

print(f"Running {N_BOOTSTRAP} cluster bootstrap iterations for Arecanut across {len(unique_parents)} parent plant groups...")
for b in range(N_BOOTSTRAP):
    sampled_parents = np.random.choice(unique_parents, size=len(unique_parents), replace=True)
    sample_indices = np.concatenate([parent_to_indices[pid] for pid in sampled_parents])
    
    for mname, (y_t, y_p) in areca_models.items():
        m_t = y_t[sample_indices]
        m_p = y_p[sample_indices]
        res = compute_metrics(m_t, m_p)
        for metric, val in res.items():
            areca_bootstrap_results[mname][metric].append(val)

# --- B. Paddy Bootstrap ---
paddy_models = {
    'Paddy ResNet-18 (CORN)': (paddy_corn_true, paddy_corn_pred),
    'Paddy ResNet-18 (Softmax)': (paddy_base_true, paddy_base_pred),
}
paddy_bootstrap_results = {k: {'qwk': [], 'mae': [], 'exact_accuracy': [], 'macro_f1': []} for k in paddy_models}

n_paddy_test = len(paddy_corn_true)
print(f"Running {N_BOOTSTRAP} bootstrap iterations for Paddy across {n_paddy_test} test samples...")
for b in range(N_BOOTSTRAP):
    sample_indices = np.random.choice(n_paddy_test, size=n_paddy_test, replace=True)
    
    for mname, (y_t, y_p) in paddy_models.items():
        m_t = y_t[sample_indices]
        m_p = y_p[sample_indices]
        res = compute_metrics(m_t, m_p)
        for metric, val in res.items():
            paddy_bootstrap_results[mname][metric].append(val)

# Combine and compute statistics
final_output = {
    "bootstrap_iterations": N_BOOTSTRAP,
    "resampling_strategy": {
        "arecanut": "cluster_resampling_by_parent_plant_group (36 unique parent groups resampled with replacement)",
        "paddy": "standard_resampling_with_replacement (1507 test samples)"
    },
    "models": {}
}

all_bootstraps = {**paddy_bootstrap_results, **areca_bootstrap_results}
all_raw_data = {**paddy_models, **areca_models}

for mname in all_bootstraps:
    y_t, y_p = all_raw_data[mname]
    point_est = compute_metrics(y_t, y_p)
    
    final_output["models"][mname] = {}
    for metric in ['qwk', 'mae', 'exact_accuracy', 'macro_f1']:
        vals = np.array(all_bootstraps[mname][metric])
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        
        final_output["models"][mname][metric] = {
            "point_estimate": point_est[metric],
            "bootstrap_mean": mean_val,
            "bootstrap_std": std_val,
            "ci_95": [ci_lower, ci_upper]
        }

# Compare Arecanut ResNet-34 vs ResNet-18 vs MobileNetV2 QWK differences
r34_qwk = np.array(areca_bootstrap_results['Arecanut ResNet-34 (CORN)']['qwk'])
r18_qwk = np.array(areca_bootstrap_results['Arecanut ResNet-18 (CORN)']['qwk'])
mob_qwk = np.array(areca_bootstrap_results['Arecanut MobileNetV2 (CORN)']['qwk'])

# Individual CI intervals
ci_r34 = final_output["models"]['Arecanut ResNet-34 (CORN)']['qwk']['ci_95']
ci_r18 = final_output["models"]['Arecanut ResNet-18 (CORN)']['qwk']['ci_95']
ci_mob = final_output["models"]['Arecanut MobileNetV2 (CORN)']['qwk']['ci_95']

# Check interval overlap
overlap_r34_r18 = not (ci_r34[0] > ci_r18[1] or ci_r18[0] > ci_r34[1])
overlap_r34_mob = not (ci_r34[0] > ci_mob[1] or ci_mob[0] > ci_r34[1])
overlap_r18_mob = not (ci_r18[0] > ci_mob[1] or ci_mob[0] > ci_r18[1])

# Paired bootstrap differences (delta = M1 - M2 on each resample)
diff_r34_r18 = r34_qwk - r18_qwk
diff_r34_mob = r34_qwk - mob_qwk
diff_r18_mob = r18_qwk - mob_qwk

final_output["backbone_qwk_statistical_distinguishability"] = {
    "individual_ci_overlap": {
        "resnet34_vs_resnet18": {
            "resnet34_ci": ci_r34,
            "resnet18_ci": ci_r18,
            "intervals_overlap": overlap_r34_r18
        },
        "resnet34_vs_mobilenet_v2": {
            "resnet34_ci": ci_r34,
            "mobilenet_v2_ci": ci_mob,
            "intervals_overlap": overlap_r34_mob
        },
        "resnet18_vs_mobilenet_v2": {
            "resnet18_ci": ci_r18,
            "mobilenet_v2_ci": ci_mob,
            "intervals_overlap": overlap_r18_mob
        }
    },
    "paired_bootstrap_differences": {
        "resnet34_minus_resnet18": {
            "mean_delta": float(np.mean(diff_r34_r18)),
            "delta_ci_95": [float(np.percentile(diff_r34_r18, 2.5)), float(np.percentile(diff_r34_r18, 97.5))],
            "statistically_distinguishable_from_zero": bool(np.percentile(diff_r34_r18, 2.5) > 0 or np.percentile(diff_r34_r18, 97.5) < 0)
        },
        "resnet34_minus_mobilenet_v2": {
            "mean_delta": float(np.mean(diff_r34_mob)),
            "delta_ci_95": [float(np.percentile(diff_r34_mob, 2.5)), float(np.percentile(diff_r34_mob, 97.5))],
            "statistically_distinguishable_from_zero": bool(np.percentile(diff_r34_mob, 2.5) > 0 or np.percentile(diff_r34_mob, 97.5) < 0)
        },
        "resnet18_minus_mobilenet_v2": {
            "mean_delta": float(np.mean(diff_r18_mob)),
            "delta_ci_95": [float(np.percentile(diff_r18_mob, 2.5)), float(np.percentile(diff_r18_mob, 97.5))],
            "statistically_distinguishable_from_zero": bool(np.percentile(diff_r18_mob, 2.5) > 0 or np.percentile(diff_r18_mob, 97.5) < 0)
        }
    }
}

out_path = '/home/sahith/Projects/Crop/outputs/improvements/bootstrap_ci_results.json'
with open(out_path, 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"Saved bootstrap CI results to {out_path}")
print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
