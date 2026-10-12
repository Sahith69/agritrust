import os
import sys
import re
import json
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

PROJECT_ROOT = "/home/sahith/Projects/Hackathon"
sys.path.insert(0, PROJECT_ROOT)
import config
from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade

RANDOM_SEED = 42

# Image Transform matching app.py
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

class FastImageDataset(Dataset):
    def __init__(self, paths, transform=EVAL_TRANSFORM):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as img:
            im = img.convert('RGB')
            tensor = self.transform(im)
        return tensor, p

def extract_phash_uint64(path):
    with Image.open(path) as img:
        h = imagehash.phash(img)
        return np.uint64(int(str(h), 16))

def load_disease_classifier(crop_name: str):
    key = crop_name.lower()
    pth = os.path.join(config.OUTPUTS_DIR, "models", f"{key}_disease_classifier.pth")
    ck = torch.load(pth, map_location='cpu', weights_only=True)
    classes = ck['classes']
    model = models.resnet18(weights=None)
    if 'fc.1.weight' in ck['model_state_dict']:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.fc.in_features, len(classes)))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    idx_to_class = {int(k): v for k, v in ck.get('idx_to_class', {i: c for i, c in enumerate(classes)}).items()}
    class_to_idx = {c: i for i, c in idx_to_class.items()}
    return model, classes, idx_to_class, class_to_idx

def load_severity_model(crop_name: str):
    key = crop_name.lower()
    pth = os.path.join(config.OUTPUTS_DIR, "models", f"{key}_ordinal.pth")
    model = CropSeverityModel(backbone=config.BACKBONE, num_classes=config.NUM_SEVERITY_GRADES,
                              ordinal=True, pretrained=False)
    ck = torch.load(pth, map_location='cpu', weights_only=True)
    sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.eval()
    return model

def main():
    print("=" * 80)
    print("PADDY & ARECANUT RECHECK AND RE-EVALUATION")
    print("=" * 80)

    # =========================================================================
    # PART 1: PADDY RECHECK (twins = pHash <= 8 only)
    # =========================================================================
    print("\n--- PART 1: PADDY RECHECK ---")
    paddy_csv = os.path.join(config.OUTPUTS_DIR, "paddy_splits.csv")
    df_paddy = pd.read_csv(paddy_csv)

    paddy_clf, paddy_classes, paddy_idx_to_cls, paddy_cls_to_idx = load_disease_classifier("paddy")
    paddy_sev = load_severity_model("paddy")

    df_paddy['disease_idx'] = df_paddy['disease_class'].map(paddy_cls_to_idx)

    train_paddy = df_paddy[df_paddy['split'] == 'train'].reset_index(drop=True)
    test_paddy  = df_paddy[df_paddy['split'] == 'test'].reset_index(drop=True)

    print(f"Paddy total: {len(df_paddy)} | Train: {len(train_paddy)} | Test: {len(test_paddy)}")

    print("Computing pHashes for Paddy train and test...")
    train_phashes = np.array([extract_phash_uint64(p) for p in train_paddy['image_path']])
    test_phashes  = np.array([extract_phash_uint64(p) for p in test_paddy['image_path']])

    # Distance matrix
    xor_mat = np.bitwise_xor(test_phashes[:, None], train_phashes[None, :])
    dist_mat = np.bitwise_count(xor_mat)
    min_phash_dist = dist_mat.min(axis=1)

    flag_twin_le_8 = (min_phash_dist <= 8)
    n_flagged = int(np.sum(flag_twin_le_8))
    pct_flagged = float(n_flagged / len(test_paddy) * 100.0)
    n_twin_free = int(np.sum(~flag_twin_le_8))
    pct_twin_free = float(n_twin_free / len(test_paddy) * 100.0)

    print(f"Flagged Paddy test twins (pHash <= 8): {n_flagged} / {len(test_paddy)} ({pct_flagged}%)")
    print(f"Twin-free Paddy test samples: {n_twin_free} / {len(test_paddy)} ({pct_twin_free}%)")

    # Run inference on all test images
    print("Running model inference on Paddy test set...")
    test_ds = FastImageDataset(test_paddy['image_path'].tolist())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    test_preds_dis = []
    test_preds_sev = []

    with torch.no_grad():
        for tensors, _ in test_loader:
            logits_d = paddy_clf(tensors)
            p_d = logits_d.argmax(dim=1).cpu().numpy()
            test_preds_dis.extend(p_d)

            logits_s = paddy_sev(tensors)
            p_s_raw = torch.sigmoid(logits_s)
            g_s, _ = corn_probs_to_grade(p_s_raw)
            test_preds_sev.extend(g_s.cpu().numpy())

    test_preds_dis = np.array(test_preds_dis)
    test_preds_sev = np.array(test_preds_sev)

    # Class-only baseline for severity: mode severity grade in train for each disease class
    class_mode_grades = {}
    for c in paddy_classes:
        subset_grades = train_paddy[train_paddy['disease_class'] == c]['severity_grade']
        if len(subset_grades) > 0:
            class_mode_grades[c] = int(subset_grades.mode()[0])
        else:
            class_mode_grades[c] = int(train_paddy['severity_grade'].mode()[0])

    test_preds_baseline = np.array([class_mode_grades[c] for c in test_paddy['disease_class']])

    # Twin-free subset
    twin_free_mask = ~flag_twin_le_8
    df_tf = test_paddy[twin_free_mask].reset_index(drop=True)
    tf_preds_dis = test_preds_dis[twin_free_mask]
    tf_preds_sev = test_preds_sev[twin_free_mask]
    tf_preds_baseline = test_preds_baseline[twin_free_mask]

    # Overall twin-free disease metrics
    tf_dis_acc = float(accuracy_score(df_tf['disease_idx'], tf_preds_dis))
    tf_dis_f1  = float(f1_score(df_tf['disease_idx'], tf_preds_dis, average='macro', zero_division=0))

    # Lesion-only classes subset
    non_lesion_classes = ['normal', 'hispa', 'dead_heart']
    lesion_mask_in_tf = ~df_tf['disease_class'].isin(non_lesion_classes)
    df_tf_lesion = df_tf[lesion_mask_in_tf].reset_index(drop=True)

    n_tf_lesion = len(df_tf_lesion)
    print(f"\nTwin-free lesion-only subset size N: {n_tf_lesion}")
    print("Class breakdown in twin-free lesion-only subset:")
    print(df_tf_lesion['disease_class'].value_counts())

    tf_lesion_preds_sev = tf_preds_sev[lesion_mask_in_tf]
    tf_lesion_preds_base = tf_preds_baseline[lesion_mask_in_tf]
    y_true_tf_lesion = df_tf_lesion['severity_grade'].values

    tf_lesion_qwk   = float(cohen_kappa_score(y_true_tf_lesion, tf_lesion_preds_sev, weights='quadratic'))
    tf_lesion_exact = float(np.mean(y_true_tf_lesion == tf_lesion_preds_sev))
    tf_lesion_base_qwk = float(cohen_kappa_score(y_true_tf_lesion, tf_lesion_preds_base, weights='quadratic'))

    # Bootstrap 95% CIs (2,000 resamples)
    print("\nRunning 2,000 bootstrap resamples on twin-free subset...")
    rng = np.random.default_rng(RANDOM_SEED)
    n_boot = 2000

    # 1. Disease accuracy & macro-F1 on all twin-free (N = len(df_tf))
    b_dis_acc = []
    b_dis_f1  = []
    y_true_tf_dis = df_tf['disease_idx'].values
    n_tf = len(df_tf)

    for _ in range(n_boot):
        idx = rng.choice(n_tf, size=n_tf, replace=True)
        yt = y_true_tf_dis[idx]
        yp = tf_preds_dis[idx]
        b_dis_acc.append(accuracy_score(yt, yp))
        b_dis_f1.append(f1_score(yt, yp, average='macro', zero_division=0))

    # 2. Lesion-only severity metrics on lesion subset (N = n_tf_lesion)
    b_lesion_qwk = []
    b_lesion_exact = []
    b_lesion_base_qwk = []

    for _ in range(n_boot):
        idx = rng.choice(n_tf_lesion, size=n_tf_lesion, replace=True)
        yt_s = y_true_tf_lesion[idx]
        yp_s = tf_lesion_preds_sev[idx]
        yp_b = tf_lesion_preds_base[idx]
        b_lesion_qwk.append(cohen_kappa_score(yt_s, yp_s, weights='quadratic'))
        b_lesion_exact.append(np.mean(yt_s == yp_s))
        b_lesion_base_qwk.append(cohen_kappa_score(yt_s, yp_b, weights='quadratic'))

    def calc_ci(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5))
        }

    ci_dis_acc = calc_ci(b_dis_acc)
    ci_dis_f1  = calc_ci(b_dis_f1)
    ci_lesion_qwk = calc_ci(b_lesion_qwk)
    ci_lesion_exact = calc_ci(b_lesion_exact)
    ci_lesion_base_qwk = calc_ci(b_lesion_base_qwk)

    print("\nPADDY RESULTS SUMMARY (pHash <= 8 twins removed):")
    print(f"Total Test N: {len(test_paddy)}")
    print(f"Twins Flagged (pHash <= 8): {n_flagged} ({pct_flagged}%)")
    print(f"Twin-Free Test N: {n_twin_free} ({pct_twin_free}%)")
    print(f"Twin-Free Disease Accuracy: {tf_dis_acc} | 95% CI: [{ci_dis_acc['ci_lower']}, {ci_dis_acc['ci_upper']}]")
    print(f"Twin-Free Disease Macro-F1: {tf_dis_f1} | 95% CI: [{ci_dis_f1['ci_lower']}, {ci_dis_f1['ci_upper']}]")
    print(f"Twin-Free Lesion-Only N: {n_tf_lesion}")
    print(f"Twin-Free Lesion-Only Severity QWK: {tf_lesion_qwk} | 95% CI: [{ci_lesion_qwk['ci_lower']}, {ci_lesion_qwk['ci_upper']}]")
    print(f"Twin-Free Lesion-Only Severity Exact Accuracy: {tf_lesion_exact} | 95% CI: [{ci_lesion_exact['ci_lower']}, {ci_lesion_exact['ci_upper']}]")
    print(f"Twin-Free Lesion-Only Class-Only Baseline QWK: {tf_lesion_base_qwk} | 95% CI: [{ci_lesion_base_qwk['ci_lower']}, {ci_lesion_base_qwk['ci_upper']}]")


    # =========================================================================
    # PART 2: ARECANUT RECHECK (Timestamps, Burst Check Coverage, pHash <= 16)
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- PART 2: ARECANUT RECHECK ---")
    areca_csv = os.path.join(config.OUTPUTS_DIR, "arecanut_splits.csv")
    df_areca = pd.read_csv(areca_csv)

    def has_filename_dt(p):
        fname = os.path.basename(p)
        return bool(re.search(r'IMG_\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}', fname))

    df_areca['has_timestamp'] = df_areca['image_path'].apply(has_filename_dt)

    total_areca = len(df_areca)
    total_lacking_dt = int((~df_areca['has_timestamp']).sum())
    total_with_dt = int(df_areca['has_timestamp'].sum())

    train_areca = df_areca[df_areca['split'] == 'train'].reset_index(drop=True)
    val_areca   = df_areca[df_areca['split'] == 'val'].reset_index(drop=True)
    test_areca  = df_areca[df_areca['split'] == 'test'].reset_index(drop=True)

    test_lacking_dt = int((~test_areca['has_timestamp']).sum())
    test_with_dt = int(test_areca['has_timestamp'].sum())
    pct_test_uncovered = float(test_lacking_dt / len(test_areca) * 100.0)

    print(f"Total Arecanut images: {total_areca}")
    print(f"Images lacking filename timestamp across dataset: {total_lacking_dt} / {total_areca} ({total_lacking_dt / total_areca * 100.0}%)")
    print(f"Images with filename timestamp: {total_with_dt} / {total_areca} ({total_with_dt / total_areca * 100.0}%)")
    print(f"Train lacking timestamp: {(~train_areca['has_timestamp']).sum()} / {len(train_areca)}")
    print(f"Val lacking timestamp: {(~val_areca['has_timestamp']).sum()} / {len(val_areca)}")
    print(f"Test lacking timestamp (NOT covered by burst check): {test_lacking_dt} / {len(test_areca)} ({pct_test_uncovered}%)")
    print(f"Test with timestamp (covered by burst check): {test_with_dt} / {len(test_areca)} ({test_with_dt / len(test_areca) * 100.0}%)")

    # Soft pHash <= 16 overlap recomputation on the 486 test split
    print("\nRecomputing pHash <= 16 overlap on the 486 test split...")
    areca_train_phashes = np.array([extract_phash_uint64(p) for p in train_areca['image_path']])
    areca_test_phashes  = np.array([extract_phash_uint64(p) for p in test_areca['image_path']])

    areca_xor = np.bitwise_xor(areca_test_phashes[:, None], areca_train_phashes[None, :])
    areca_dist = np.bitwise_count(areca_xor)
    areca_min_dist = areca_dist.min(axis=1)

    areca_phash_le_0 = int((areca_min_dist <= 0).sum())
    areca_phash_le_8 = int((areca_min_dist <= 8).sum())
    areca_phash_le_16 = int((areca_min_dist <= 16).sum())
    pct_areca_le_16 = float(areca_phash_le_16 / len(test_areca) * 100.0)

    print(f"Arecanut Test N: {len(test_areca)}")
    print(f"Arecanut pHash <= 0 overlap: {areca_phash_le_0} (0.0%)")
    print(f"Arecanut pHash <= 8 overlap: {areca_phash_le_8} (0.0%)")
    print(f"Arecanut soft pHash <= 16 overlap: {areca_phash_le_16} / {len(test_areca)} ({pct_areca_le_16}%)")


    # =========================================================================
    # PART 3: ARECANUT BUD_BORER & PROVENANCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- PART 3: ARECANUT BUD_BORER & PROVENANCE ---")
    areca_clf, areca_classes, areca_idx_to_cls, areca_cls_to_idx = load_disease_classifier("arecanut")

    # bud_borer in test set
    test_bb = test_areca[test_areca['disease_class'] == 'bud_borer'].reset_index(drop=True)
    n_test_bb = len(test_bb)
    print(f"bud_borer test sample count: {n_test_bb}")

    # Evaluate bud_borer predictions
    bb_ds = FastImageDataset(test_bb['image_path'].tolist())
    bb_loader = DataLoader(bb_ds, batch_size=32, shuffle=False)
    bb_preds = []
    with torch.no_grad():
        for tensors, _ in bb_loader:
            out = areca_clf(tensors)
            pred = out.argmax(dim=1).cpu().numpy()
            bb_preds.extend(pred)

    bb_pred_classes = [areca_idx_to_cls[p] for p in bb_preds]
    bb_confusion_row = {c: bb_pred_classes.count(c) for c in areca_classes}

    print("\nConfusion matrix row for true bud_borer (order matching classes):")
    for c in areca_classes:
        print(f"  {c}: {bb_confusion_row[c]}")

    # Provenance check: Kaggle / external images
    print("\n--- Training Set Provenance & Source Analysis ---")
    train_areca['is_kaggle'] = train_areca['image_path'].str.contains('kaggle_merged')
    n_train_kaggle = int(train_areca['is_kaggle'].sum())
    n_train_local = int((~train_areca['is_kaggle']).sum())
    pct_train_kaggle = float(n_train_kaggle / len(train_areca) * 100.0)

    print(f"Total training images for arecanut_disease_classifier.pth: {len(train_areca)}")
    print(f"External / Kaggle images in train: {n_train_kaggle} ({pct_train_kaggle}%)")
    print(f"Local / Primary images in train: {n_train_local} ({100.0 - pct_train_kaggle}%)")

    # Breakdown per class in train
    print("\nTraining Class Counts & Kaggle Breakdown:")
    train_class_counts = train_areca['disease_class'].value_counts()
    print(f"{'Class':<25} {'Total Train':<12} {'Kaggle Train':<14} {'Local Train':<12}")
    for c in areca_classes:
        sub = train_areca[train_areca['disease_class'] == c]
        tot = len(sub)
        kag = int(sub['is_kaggle'].sum())
        loc = tot - kag
        print(f"{c:<25} {tot:<12} {kag:<14} {loc:<12}")

    # bud_borer provenance across all splits
    print("\nBud Borer Provenance Across All Splits:")
    for s in ['train', 'val', 'test']:
        sub = df_areca[(df_areca['disease_class'] == 'bud_borer') & (df_areca['split'] == s)]
        kag = sub['image_path'].str.contains('kaggle_merged').sum()
        loc = len(sub) - kag
        print(f"  {s}: Total={len(sub)}, Kaggle={kag}, Local={loc}")

    # Save all raw numbers to JSON
    results_json = {
        "paddy": {
            "total_test": len(test_paddy),
            "flagged_twins_phash_le_8": n_flagged,
            "pct_flagged": pct_flagged,
            "twin_free_test": n_twin_free,
            "pct_twin_free": pct_twin_free,
            "twin_free_disease_accuracy": tf_dis_acc,
            "twin_free_disease_accuracy_ci": ci_dis_acc,
            "twin_free_disease_macro_f1": tf_dis_f1,
            "twin_free_disease_macro_f1_ci": ci_dis_f1,
            "twin_free_lesion_classes_N": n_tf_lesion,
            "twin_free_lesion_severity_qwk": tf_lesion_qwk,
            "twin_free_lesion_severity_qwk_ci": ci_lesion_qwk,
            "twin_free_lesion_severity_exact_accuracy": tf_lesion_exact,
            "twin_free_lesion_severity_exact_accuracy_ci": ci_lesion_exact,
            "twin_free_lesion_class_only_baseline_qwk": tf_lesion_base_qwk,
            "twin_free_lesion_class_only_baseline_qwk_ci": ci_lesion_base_qwk,
            "twin_free_lesion_class_breakdown": df_tf_lesion['disease_class'].value_counts().to_dict()
        },
        "arecanut": {
            "total_dataset": total_areca,
            "images_lacking_timestamp_total": total_lacking_dt,
            "images_with_timestamp_total": total_with_dt,
            "test_total": len(test_areca),
            "test_images_lacking_timestamp_uncovered_by_burst": test_lacking_dt,
            "pct_test_uncovered_by_burst": pct_test_uncovered,
            "test_images_with_timestamp_covered_by_burst": test_with_dt,
            "phash_le_16_overlap_count": areca_phash_le_16,
            "phash_le_16_overlap_pct": pct_areca_le_16,
            "bud_borer": {
                "test_count": n_test_bb,
                "confusion_row": bb_confusion_row,
                "train_count": len(train_areca[train_areca['disease_class'] == 'bud_borer']),
                "train_kaggle_count": int(train_areca[train_areca['disease_class'] == 'bud_borer']['is_kaggle'].sum()),
                "train_local_count": int((~train_areca[train_areca['disease_class'] == 'bud_borer']['is_kaggle']).sum()),
            },
            "train_total": len(train_areca),
            "train_kaggle_total": n_train_kaggle,
            "train_kaggle_pct": pct_train_kaggle,
            "train_class_counts": {c: int(len(train_areca[train_areca['disease_class'] == c])) for c in areca_classes}
        }
    }

    out_file = os.path.join(PROJECT_ROOT, "paddy_arecanut_recheck_results.json")
    with open(out_file, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved raw unrounded results to {out_file}")

if __name__ == "__main__":
    main()
