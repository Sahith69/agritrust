#!/usr/bin/env python3
"""
paddy_final_numbers.py — Rigorous Evaluation of Deployed Paddy Models

Evaluates the deployed Paddy models exactly as app.py runs them:
- outputs/models/paddy_disease_classifier.pth
- outputs/models/paddy_ordinal.pth (ResNet-18 CORN, chain-rule argmax decode, no tuned thresholds, no TTA, no ensemble)

Evaluated under Setting 2 twin detection (test vs train+val, orig/hflip/vflip, Hamming <= 8):
1. Full Test (N=1507)
2. Lesion-Only Full (N=757)
3. Twin-Free Full (N=732)
4. Lesion-Only Twin-Free (N=343)

Metrics computed:
- Disease: Accuracy, Macro-F1 with 2,000-resample bootstrap 95% CIs
- Severity: QWK, Exact Accuracy, Within-1-Grade Accuracy, MAE with 2,000-resample bootstrap 95% CIs
- Class-Only Baseline: Most frequent train grade per disease class
All numbers printed with explicit N, unrounded.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_ROOT = "/home/sahith/Projects/Hackathon"
sys.path.insert(0, PROJECT_ROOT)
import config
from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade

torch.set_num_threads(8)
DEVICE = torch.device('cpu')
RANDOM_SEED = 42

CLASSES_7 = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'downy_mildew', 'tungro'
]


# ─── BOOTSTRAP RESAMPLING ───────────────────────────────────────────────────
def bootstrap_disease_metrics(y_true, y_pred, n_bootstraps=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_acc = []
    boot_f1 = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        boot_f1.append(f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0))
    
    ci_acc = (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5)))
    ci_f1 = (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5)))
    return ci_acc, ci_f1


def bootstrap_severity_metrics(y_true, y_pred, n_bootstraps=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_qwk = []
    boot_exact = []
    boot_w1 = []
    boot_mae = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        boot_qwk.append(cohen_kappa_score(yt, yp, weights='quadratic'))
        boot_exact.append(np.mean(yt == yp))
        boot_w1.append(np.mean(np.abs(yt - yp) <= 1))
        boot_mae.append(np.mean(np.abs(yt - yp)))
        
    ci_qwk = (float(np.percentile(boot_qwk, 2.5)), float(np.percentile(boot_qwk, 97.5)))
    ci_exact = (float(np.percentile(boot_exact, 2.5)), float(np.percentile(boot_exact, 97.5)))
    ci_w1 = (float(np.percentile(boot_w1, 2.5)), float(np.percentile(boot_w1, 97.5)))
    ci_mae = (float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5)))
    return ci_qwk, ci_exact, ci_w1, ci_mae


def main():
    print("=" * 90)
    print("PADDY DOCTOR FINAL EVALUATION (DEPLOYED MODELS & SETTING 2 TWIN CHECK)")
    print("=" * 90)

    # 1. Load splits and cache
    splits_csv = os.path.join(PROJECT_ROOT, "outputs/paddy_splits.csv")
    df = pd.read_csv(splits_csv)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # Load Setting 2 twin flags
    pred_cache_path = os.path.join(PROJECT_ROOT, "outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv")
    pred_cache = pd.read_csv(pred_cache_path)
    test_df['has_twin_le8'] = pred_cache['has_twin_le8'].astype(bool).values

    # Train mode per disease class for class-only baseline
    mode_per_class = train_df.groupby('disease_class')['severity_grade'].agg(lambda s: int(s.mode()[0])).to_dict()
    print("Class-Only Baseline: Most Frequent Training Severity Grade per Disease Class:")
    for cls in sorted(mode_per_class.keys()):
        print(f"  {cls:<26}: Grade {mode_per_class[cls]}")

    # 2. Define the 4 subsets
    subsets = {
        "Full Test": test_df.copy(),
        "Lesion-Only Full": test_df[test_df['disease_class'].isin(CLASSES_7)].copy().reset_index(drop=True),
        "Twin-Free Full": test_df[~test_df['has_twin_le8']].copy().reset_index(drop=True),
        "Lesion-Only Twin-Free": test_df[(~test_df['has_twin_le8']) & (test_df['disease_class'].isin(CLASSES_7))].copy().reset_index(drop=True),
    }

    print("\n--- Subsets Summary ---")
    for name, sub in subsets.items():
        print(f"  {name:<25}: N = {len(sub)}")

    # 3. Load deployed disease classifier
    dis_model_path = os.path.join(config.MODELS_DIR, "paddy_disease_classifier.pth")
    ck_dis = torch.load(dis_model_path, map_location='cpu', weights_only=True)
    classes = ck_dis['classes']
    idx_to_class = ck_dis.get('idx_to_class', {i: c for i, c in enumerate(classes)})
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    dis_model = models.resnet18(weights=None)
    if 'fc.1.weight' in ck_dis['model_state_dict']:
        dis_model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(512, len(classes)))
    else:
        dis_model.fc = nn.Linear(512, len(classes))
    dis_model.load_state_dict(ck_dis['model_state_dict'])
    dis_model.eval()

    # 4. Load deployed ordinal severity model
    sev_model_path = os.path.join(config.MODELS_DIR, "paddy_ordinal.pth")
    sev_model = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
    ck_sev = torch.load(sev_model_path, map_location='cpu', weights_only=True)
    sd = ck_sev.get('model_state_dict', ck_sev) if isinstance(ck_sev, dict) else ck_sev
    sev_model.load_state_dict(sd)
    sev_model.eval()

    # Image transform (matching app.py lines 554-558, 608-612)
    tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    print(f"\nRunning inference on all {len(test_df)} test images...")
    tensors = torch.stack([tf(Image.open(p).convert('RGB')) for p in test_df['image_path']])

    with torch.no_grad():
        # Disease inference
        dis_logits = dis_model(tensors)
        dis_preds_idx = dis_logits.argmax(dim=1).numpy()
        test_df['dis_pred_cls'] = np.array([idx_to_class[i] for i in dis_preds_idx])

        # Severity inference (chain-rule argmax decode matching app.py line 619)
        sev_logits = sev_model(tensors)
        p_raw = torch.sigmoid(sev_logits)
        # corn_probs_to_grade handles batch tensor
        sev_preds = corn_probs_to_grade(p_raw)[0].numpy()
        test_df['sev_pred_grade'] = sev_preds

    # Class-only baseline predictions
    test_df['base_pred_grade'] = test_df['disease_class'].map(mode_per_class).values

    # 5. Evaluate and report on all 4 subsets
    results_summary = {}
    for name in ["Full Test", "Lesion-Only Full", "Twin-Free Full", "Lesion-Only Twin-Free"]:
        sub = subsets[name]
        # Re-slice predictions
        sub_indices = sub.index
        y_true_dis = test_df.loc[sub_indices, 'disease_class'].values
        y_pred_dis = test_df.loc[sub_indices, 'dis_pred_cls'].values

        y_true_sev = test_df.loc[sub_indices, 'severity_grade'].values
        y_pred_sev = test_df.loc[sub_indices, 'sev_pred_grade'].values
        y_pred_base = test_df.loc[sub_indices, 'base_pred_grade'].values
        n_samples = len(sub)

        # Disease metrics
        acc_point = accuracy_score(y_true_dis, y_pred_dis)
        f1_point = f1_score(y_true_dis, y_pred_dis, average='macro', zero_division=0)
        ci_acc, ci_f1 = bootstrap_disease_metrics(y_true_dis, y_pred_dis)

        # Severity metrics (Deployed ResNet-18 CORN)
        qwk_point = cohen_kappa_score(y_true_sev, y_pred_sev, weights='quadratic')
        exact_point = accuracy_score(y_true_sev, y_pred_sev)
        w1_point = np.mean(np.abs(y_true_sev - y_pred_sev) <= 1)
        mae_point = mean_absolute_error(y_true_sev, y_pred_sev)
        ci_qwk, ci_exact, ci_w1, ci_mae = bootstrap_severity_metrics(y_true_sev, y_pred_sev)

        # Class-only baseline metrics
        base_qwk_point = cohen_kappa_score(y_true_sev, y_pred_base, weights='quadratic')
        base_exact_point = accuracy_score(y_true_sev, y_pred_base)
        base_w1_point = np.mean(np.abs(y_true_sev - y_pred_base) <= 1)
        base_mae_point = mean_absolute_error(y_true_sev, y_pred_base)
        b_ci_qwk, b_ci_exact, b_ci_w1, b_ci_mae = bootstrap_severity_metrics(y_true_sev, y_pred_base)

        results_summary[name] = {
            "N": n_samples,
            "disease": {
                "accuracy": acc_point, "accuracy_ci": ci_acc,
                "macro_f1": f1_point, "macro_f1_ci": ci_f1
            },
            "severity_corn": {
                "qwk": qwk_point, "qwk_ci": ci_qwk,
                "exact_acc": exact_point, "exact_acc_ci": ci_exact,
                "within_1": w1_point, "within_1_ci": ci_w1,
                "mae": mae_point, "mae_ci": ci_mae
            },
            "class_only_baseline": {
                "qwk": base_qwk_point, "qwk_ci": b_ci_qwk,
                "exact_acc": base_exact_point, "exact_acc_ci": b_ci_exact,
                "within_1": base_w1_point, "within_1_ci": b_ci_w1,
                "mae": base_mae_point, "mae_ci": b_ci_mae
            }
        }

        print(f"\n==================== {name.upper()} (N = {n_samples}) ====================")
        print("  [Disease Classification: Deployed ResNet-18 Softmax]")
        print(f"    Accuracy:  {acc_point}  [95% CI: {ci_acc[0]}, {ci_acc[1]}]")
        print(f"    Macro-F1:  {f1_point}  [95% CI: {ci_f1[0]}, {ci_f1[1]}]")
        print("  [Severity Grading: Deployed ResNet-18 CORN (Chain-Rule Argmax, Single Model)]")
        print(f"    QWK:       {qwk_point}  [95% CI: {ci_qwk[0]}, {ci_qwk[1]}]")
        print(f"    Exact Acc: {exact_point}  [95% CI: {ci_exact[0]}, {ci_exact[1]}]")
        print(f"    Within-1:  {w1_point}  [95% CI: {ci_w1[0]}, {ci_w1[1]}]")
        print(f"    MAE:       {mae_point}  [95% CI: {ci_mae[0]}, {ci_mae[1]}]")
        print("  [Class-Only Severity Baseline (Predicts Training Set Mode Grade)]")
        print(f"    QWK:       {base_qwk_point}  [95% CI: {b_ci_qwk[0]}, {b_ci_qwk[1]}]")
        print(f"    Exact Acc: {base_exact_point}  [95% CI: {b_ci_exact[0]}, {b_ci_exact[1]}]")
        print(f"    Within-1:  {base_w1_point}  [95% CI: {b_ci_w1[0]}, {b_ci_w1[1]}]")
        print(f"    MAE:       {base_mae_point}  [95% CI: {b_ci_mae[0]}, {b_ci_mae[1]}]")

    print("\n" + "=" * 90)
    print("PADDY EVALUATION FINISHED.")
    print("=" * 90)


if __name__ == "__main__":
    main()
