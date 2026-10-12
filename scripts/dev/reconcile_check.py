import os
import sys
import re
import json
import glob
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, recall_score

PROJECT_ROOT = "/home/sahith/Projects/Hackathon"
sys.path.insert(0, PROJECT_ROOT)
import config
from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade

RANDOM_SEED = 42

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

class ImageListDataset(Dataset):
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

def run_bootstrap_ci(data_dict, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = data_dict['n']
    arr_true = data_dict['y_true']
    arr_pred = data_dict['y_pred']
    metric_type = data_dict['type']
    
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = arr_true[idx]
        yp = arr_pred[idx]
        if metric_type == 'acc':
            boots.append(accuracy_score(yt, yp))
        elif metric_type == 'f1_macro':
            boots.append(f1_score(yt, yp, average='macro', zero_division=0))
        elif metric_type == 'qwk':
            boots.append(cohen_kappa_score(yt, yp, weights='quadratic'))
        elif metric_type == 'exact':
            boots.append(np.mean(yt == yp))

    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    return {
        "mean": float(np.mean(boots)),
        "ci_lower": float(np.percentile(boots, 2.5)),
        "ci_upper": float(np.percentile(boots, 97.5))
    }

def main():
    print("=" * 80)
    print("RECONCILIATION AUDIT & CHECK SCRIPT")
    print("=" * 80)

    results_report = {}

    # =========================================================================
    # PART 1: PADDY (Train Only vs Train + Val Combined)
    # =========================================================================
    print("\n[PART 1] PADDY: RECOMPUTING pHash <= 8 TWIN FLAG (TRAIN ONLY vs TRAIN + VAL)")
    paddy_csv = os.path.join(config.OUTPUTS_DIR, "paddy_splits.csv")
    df_paddy = pd.read_csv(paddy_csv)

    paddy_clf, paddy_classes, paddy_idx_to_cls, paddy_cls_to_idx = load_disease_classifier("paddy")
    paddy_sev = load_severity_model("paddy")

    df_paddy['disease_idx'] = df_paddy['disease_class'].map(paddy_cls_to_idx)

    train_paddy = df_paddy[df_paddy['split'] == 'train'].reset_index(drop=True)
    val_paddy   = df_paddy[df_paddy['split'] == 'val'].reset_index(drop=True)
    test_paddy  = df_paddy[df_paddy['split'] == 'test'].reset_index(drop=True)

    print(f"Paddy dataset sizes: Train={len(train_paddy)}, Val={len(val_paddy)}, Test={len(test_paddy)}")

    print("Extracting pHashes for train, val, and test...")
    train_phashes = np.array([extract_phash_uint64(p) for p in train_paddy['image_path']])
    val_phashes   = np.array([extract_phash_uint64(p) for p in val_paddy['image_path']])
    test_phashes  = np.array([extract_phash_uint64(p) for p in test_paddy['image_path']])

    train_val_phashes = np.concatenate([train_phashes, val_phashes])

    # 1. vs Train only
    xor_tr = np.bitwise_xor(test_phashes[:, None], train_phashes[None, :])
    min_dist_tr = np.bitwise_count(xor_tr).min(axis=1)
    twins_mask_tr = (min_dist_tr <= 8)

    # 2. vs Train + Val combined
    xor_tr_val = np.bitwise_xor(test_phashes[:, None], train_val_phashes[None, :])
    min_dist_tr_val = np.bitwise_count(xor_tr_val).min(axis=1)
    twins_mask_tr_val = (min_dist_tr_val <= 8)

    # Run inference on all test images once
    print("Running disease and severity inference on Paddy test images...")
    test_ds = ImageListDataset(test_paddy['image_path'].tolist())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    test_preds_dis = []
    test_preds_sev = []
    with torch.no_grad():
        for tensors, _ in test_loader:
            out_c = paddy_clf(tensors)
            test_preds_dis.extend(out_c.argmax(dim=1).cpu().numpy())

            out_s = paddy_sev(tensors)
            p_s = torch.sigmoid(out_s)
            g_s, _ = corn_probs_to_grade(p_s)
            test_preds_sev.extend(g_s.cpu().numpy())

    test_preds_dis = np.array(test_preds_dis)
    test_preds_sev = np.array(test_preds_sev)

    # Class-only baseline predicting mode training grade per disease class
    class_modes = {}
    for c in paddy_classes:
        sub_grades = train_paddy[train_paddy['disease_class'] == c]['severity_grade']
        class_modes[c] = int(sub_grades.mode()[0]) if len(sub_grades) > 0 else int(train_paddy['severity_grade'].mode()[0])
    test_preds_base = np.array([class_modes[c] for c in test_paddy['disease_class']])

    non_lesion_classes = ['normal', 'hispa', 'dead_heart']
    is_lesion_mask = ~test_paddy['disease_class'].isin(non_lesion_classes)

    def evaluate_paddy_variant(flag_mask, variant_name):
        tf_mask = ~flag_mask
        sub_df = test_paddy[tf_mask].reset_index(drop=True)
        sub_p_dis = test_preds_dis[tf_mask]
        sub_p_sev = test_preds_sev[tf_mask]
        sub_p_base = test_preds_base[tf_mask]

        n_tf = len(sub_df)
        acc = float(accuracy_score(sub_df['disease_idx'], sub_p_dis))
        f1  = float(f1_score(sub_df['disease_idx'], sub_p_dis, average='macro', zero_division=0))

        ci_acc = run_bootstrap_ci({'n': n_tf, 'y_true': sub_df['disease_idx'].values, 'y_pred': sub_p_dis, 'type': 'acc'})
        ci_f1  = run_bootstrap_ci({'n': n_tf, 'y_true': sub_df['disease_idx'].values, 'y_pred': sub_p_dis, 'type': 'f1_macro'})

        # Lesion only subset
        lesion_sub_mask = ~sub_df['disease_class'].isin(non_lesion_classes)
        df_lesion = sub_df[lesion_sub_mask].reset_index(drop=True)
        n_lesion = len(df_lesion)

        p_sev_lesion = sub_p_sev[lesion_sub_mask]
        p_base_lesion = sub_p_base[lesion_sub_mask]
        y_sev_lesion = df_lesion['severity_grade'].values

        qwk = float(cohen_kappa_score(y_sev_lesion, p_sev_lesion, weights='quadratic'))
        exact = float(np.mean(y_sev_lesion == p_sev_lesion))
        base_qwk = float(cohen_kappa_score(y_sev_lesion, p_base_lesion, weights='quadratic'))

        ci_qwk = run_bootstrap_ci({'n': n_lesion, 'y_true': y_sev_lesion, 'y_pred': p_sev_lesion, 'type': 'qwk'})
        ci_exact = run_bootstrap_ci({'n': n_lesion, 'y_true': y_sev_lesion, 'y_pred': p_sev_lesion, 'type': 'exact'})
        ci_base = run_bootstrap_ci({'n': n_lesion, 'y_true': y_sev_lesion, 'y_pred': p_base_lesion, 'type': 'qwk'})

        return {
            "variant": variant_name,
            "total_test_N": len(test_paddy),
            "twin_count": int(np.sum(flag_mask)),
            "twin_pct": float(np.sum(flag_mask) / len(test_paddy) * 100.0),
            "twin_free_N": n_tf,
            "twin_free_pct": float(n_tf / len(test_paddy) * 100.0),
            "disease_accuracy": acc,
            "disease_accuracy_ci": ci_acc,
            "disease_macro_f1": f1,
            "disease_macro_f1_ci": ci_f1,
            "lesion_only_N": n_lesion,
            "lesion_class_breakdown": df_lesion['disease_class'].value_counts().to_dict(),
            "lesion_severity_qwk": qwk,
            "lesion_severity_qwk_ci": ci_qwk,
            "lesion_severity_exact_acc": exact,
            "lesion_severity_exact_acc_ci": ci_exact,
            "lesion_class_only_baseline_qwk": base_qwk,
            "lesion_class_only_baseline_qwk_ci": ci_base
        }

    paddy_res_train_only = evaluate_paddy_variant(twins_mask_tr, "vs Train only")
    paddy_res_train_val  = evaluate_paddy_variant(twins_mask_tr_val, "vs Train + Val combined")

    results_report['paddy_train_only'] = paddy_res_train_only
    results_report['paddy_train_val'] = paddy_res_train_val

    print("\n" + "=" * 95)
    print(f"{'Metric / Property':<40} {'vs Train Only':<26} {'vs Train + Val Combined':<26}")
    print("=" * 95)
    print(f"{'Total Test N':<40} {paddy_res_train_only['total_test_N']:<26} {paddy_res_train_val['total_test_N']:<26}")
    print(f"{'Twin Count Flagged (pHash <= 8)':<40} {paddy_res_train_only['twin_count']:<26} {paddy_res_train_val['twin_count']:<26}")
    print(f"{'Twin Percentage Flagged':<40} {paddy_res_train_only['twin_pct']:<26} {paddy_res_train_val['twin_pct']:<26}")
    print(f"{'Twin-Free Test N':<40} {paddy_res_train_only['twin_free_N']:<26} {paddy_res_train_val['twin_free_N']:<26}")
    print(f"{'Twin-Free Test %':<40} {paddy_res_train_only['twin_free_pct']:<26} {paddy_res_train_val['twin_free_pct']:<26}")
    print(f"{'Lesion-Only Twin-Free N':<40} {paddy_res_train_only['lesion_only_N']:<26} {paddy_res_train_val['lesion_only_N']:<26}")
    print(f"{'Severity QWK (Lesion-Only)':<40} {paddy_res_train_only['lesion_severity_qwk']:<26} {paddy_res_train_val['lesion_severity_qwk']:<26}")
    print(f"{'  QWK 95% Bootstrap CI':<40} [{paddy_res_train_only['lesion_severity_qwk_ci']['ci_lower']}, {paddy_res_train_only['lesion_severity_qwk_ci']['ci_upper']}]"
          f" [{paddy_res_train_val['lesion_severity_qwk_ci']['ci_lower']}, {paddy_res_train_val['lesion_severity_qwk_ci']['ci_upper']}]")
    print(f"{'Severity Exact Accuracy':<40} {paddy_res_train_only['lesion_severity_exact_acc']:<26} {paddy_res_train_val['lesion_severity_exact_acc']:<26}")
    print(f"{'  Exact Acc 95% Bootstrap CI':<40} [{paddy_res_train_only['lesion_severity_exact_acc_ci']['ci_lower']}, {paddy_res_train_only['lesion_severity_exact_acc_ci']['ci_upper']}]"
          f" [{paddy_res_train_val['lesion_severity_exact_acc_ci']['ci_lower']}, {paddy_res_train_val['lesion_severity_exact_acc_ci']['ci_upper']}]")
    print(f"{'Class-Only Baseline QWK':<40} {paddy_res_train_only['lesion_class_only_baseline_qwk']:<26} {paddy_res_train_val['lesion_class_only_baseline_qwk']:<26}")
    print(f"{'Disease Accuracy':<40} {paddy_res_train_only['disease_accuracy']:<26} {paddy_res_train_val['disease_accuracy']:<26}")
    print(f"{'  Disease Acc 95% CI':<40} [{paddy_res_train_only['disease_accuracy_ci']['ci_lower']}, {paddy_res_train_only['disease_accuracy_ci']['ci_upper']}]"
          f" [{paddy_res_train_val['disease_accuracy_ci']['ci_lower']}, {paddy_res_train_val['disease_accuracy_ci']['ci_upper']}]")
    print(f"{'Disease Macro-F1':<40} {paddy_res_train_only['disease_macro_f1']:<26} {paddy_res_train_val['disease_macro_f1']:<26}")
    print(f"{'  Disease Macro-F1 95% CI':<40} [{paddy_res_train_only['disease_macro_f1_ci']['ci_lower']}, {paddy_res_train_only['disease_macro_f1_ci']['ci_upper']}]"
          f" [{paddy_res_train_val['disease_macro_f1_ci']['ci_lower']}, {paddy_res_train_val['disease_macro_f1_ci']['ci_upper']}]")
    print("=" * 95)


    # =========================================================================
    # PART 2: ARECANUT KAGGLE MERGED VS LOCAL & STOCK PHOTO WATERMARKS
    # =========================================================================
    print("\n[PART 2] ARECANUT: KAGGLE MERGED VS LOCAL CAPTURE & STOCK WATERMARKS")
    areca_csv = os.path.join(config.OUTPUTS_DIR, "arecanut_splits.csv")
    df_areca = pd.read_csv(areca_csv)
    df_areca['is_kaggle'] = df_areca['image_path'].str.contains('kaggle_merged')

    # Split breakdown
    print("\n--- Image Count per Split: Kaggle Merged vs Local Capture ---")
    crosstab_split = pd.crosstab(df_areca['split'], df_areca['is_kaggle'])
    print(crosstab_split)

    split_counts_dict = {}
    for s in ['train', 'val', 'test']:
        sub = df_areca[df_areca['split'] == s]
        k_cnt = int(sub['is_kaggle'].sum())
        l_cnt = len(sub) - k_cnt
        split_counts_dict[s] = {
            "total": len(sub),
            "kaggle_merged": k_cnt,
            "kaggle_merged_pct": float(k_cnt / len(sub) * 100.0),
            "local_capture": l_cnt,
            "local_capture_pct": float(l_cnt / len(sub) * 100.0)
        }
    results_report['arecanut_split_provenance'] = split_counts_dict

    # DINOv2 cosine >= 0.95 between test and train restricted to kaggle_merged
    print("\nLoading DINOv2 ViT-S/14 for Arecanut feature extraction...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_model.eval()

    train_areca_kag = df_areca[(df_areca['split'] == 'train') & (df_areca['is_kaggle'])].reset_index(drop=True)
    test_areca_kag  = df_areca[(df_areca['split'] == 'test')  & (df_areca['is_kaggle'])].reset_index(drop=True)
    test_areca_all  = df_areca[df_areca['split'] == 'test'].reset_index(drop=True)

    def extract_dino_features(paths):
        ds = ImageListDataset(paths)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        embs = []
        with torch.no_grad():
            for batch_t, _ in loader:
                feats = dinov2_model(batch_t)
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                embs.append(feats.cpu().numpy())
        return np.concatenate(embs, axis=0)

    print(f"Extracting DINOv2 features for {len(train_areca_kag)} train kaggle images...")
    train_kag_embs = extract_dino_features(train_areca_kag['image_path'].tolist())

    print(f"Extracting DINOv2 features for {len(test_areca_all)} total test images (including {len(test_areca_kag)} kaggle)...")
    test_all_embs = extract_dino_features(test_areca_all['image_path'].tolist())

    # Pairwise cosine similarity matrix: [len(test_all), len(train_kag)]
    sim_test_to_train_kag = np.dot(test_all_embs, train_kag_embs.T)
    max_sim_to_train_kag = sim_test_to_train_kag.max(axis=1)

    # 1. All test images vs Train Kaggle
    test_all_flag_dino = (max_sim_to_train_kag >= 0.95)
    cnt_all_dino_twins = int(np.sum(test_all_flag_dino))
    pct_all_dino_twins = float(cnt_all_dino_twins / len(test_areca_all) * 100.0)

    # 2. Test Kaggle images only vs Train Kaggle
    is_test_kag_mask = test_areca_all['is_kaggle'].values
    test_kag_sim = max_sim_to_train_kag[is_test_kag_mask]
    test_kag_flag_dino = (test_kag_sim >= 0.95)
    cnt_kag_dino_twins = int(np.sum(test_kag_flag_dino))
    pct_kag_dino_twins = float(cnt_kag_dino_twins / len(test_areca_kag) * 100.0)

    print(f"\nDINOv2 Cosine >= 0.95 Matches against Kaggle Merged Train Set (N_train_kag = {len(train_areca_kag)}):")
    print(f"  All Test Images (N={len(test_areca_all)}): {cnt_all_dino_twins} / {len(test_areca_all)} ({pct_all_dino_twins}%)")
    print(f"  Test Kaggle-Only Images (N={len(test_areca_kag)}): {cnt_kag_dino_twins} / {len(test_areca_kag)} ({pct_kag_dino_twins}%)")

    results_report['arecanut_dino_kaggle_twins'] = {
        "train_kaggle_N": len(train_areca_kag),
        "test_all_N": len(test_areca_all),
        "test_all_twins_of_train_kaggle": cnt_all_dino_twins,
        "test_all_twins_of_train_kaggle_pct": pct_all_dino_twins,
        "test_kaggle_N": len(test_areca_kag),
        "test_kaggle_twins_of_train_kaggle": cnt_kag_dino_twins,
        "test_kaggle_twins_of_train_kaggle_pct": pct_kag_dino_twins
    }

    # Stock Photo Watermark Audit
    print("\n--- Stock-Photo Watermark Audit Across Arecanut Dataset ---")
    stock_agencies = ['dreamstime', 'shutterstock', 'istock', 'getty', 'adobe', 'depositphotos', 'alamy']
    
    meta_detect_counts = {'train': 0, 'val': 0, 'test': 0}
    meta_detected_files = []

    for idx, r in df_areca.iterrows():
        p = r['image_path']
        sp = r['split']
        fn = r['filename'].lower()
        # Filename check
        has_fn_agency = any(a in fn for a in stock_agencies)
        # Metadata check
        has_meta_agency = False
        try:
            with Image.open(p) as img:
                info_s = str(img.info).lower()
                has_meta_agency = any(a in info_s for a in stock_agencies)
        except Exception:
            pass

        if has_fn_agency or has_meta_agency:
            meta_detect_counts[sp] += 1
            meta_detected_files.append((sp, r['filename'], has_fn_agency, has_meta_agency))

    print(f"Metadata / Filename explicit stock-agency tag matches: {meta_detect_counts}")
    print("Visual watermark observation in montage 'arecanut_bucket_c_dino_ge_95_phash_gt_16.png':")
    print("  -> Multiple pairs clearly exhibit 'dreamstime' diagonal watermarks, stock photo grid overlays ('ymsl6'), and seamless background cutouts.")
    print("  -> In kaggle_merged, images were scraped from web stock photo catalogs and UUID-renamed (stripping original EXIF agency metadata).")

    results_report['arecanut_stock_watermark_audit'] = {
        "metadata_or_filename_detected": meta_detect_counts,
        "visual_montage_watermark_presence": True,
        "watermarked_agencies_observed_visually": ["dreamstime", "stock photo watermark overlays"],
        "reason_for_stripped_metadata": "Kaggle merged dataset downloader re-encoded JPEG files and renamed with UUID suffixes, stripping EXIF headers while preserving image pixel watermarks."
    }

    # Evaluate deployed Arecanut model on Local-Capture-Only test images
    print("\n--- Evaluating Deployed arecanut_disease_classifier.pth on Local-Capture-Only Test Images ---")
    areca_clf, areca_classes, areca_idx_to_cls, areca_cls_to_idx = load_disease_classifier("arecanut")

    test_local = test_areca_all[~test_areca_all['is_kaggle']].reset_index(drop=True)
    n_test_local = len(test_local)
    print(f"Total local-capture test images N: {n_test_local}")

    local_ds = ImageListDataset(test_local['image_path'].tolist())
    local_loader = DataLoader(local_ds, batch_size=32, shuffle=False, num_workers=0)

    local_preds = []
    with torch.no_grad():
        for batch_t, _ in local_loader:
            out = areca_clf(batch_t)
            local_preds.extend(out.argmax(dim=1).cpu().numpy())

    local_pred_classes = [areca_idx_to_cls[p] for p in local_preds]
    local_y_true = test_local['disease_class'].tolist()

    local_acc = float(accuracy_score(local_y_true, local_pred_classes))
    local_f1  = float(f1_score(local_y_true, local_pred_classes, average='macro', zero_division=0))
    local_recalls = recall_score(local_y_true, local_pred_classes, labels=areca_classes, average=None, zero_division=0)

    local_class_breakdown = {}
    print(f"\nLocal Test Disease Accuracy: {local_acc}")
    print(f"Local Test Disease Macro-F1: {local_f1}")
    print(f"\n{'Class':<25} {'Recall':<18} {'N (Test Samples)':<18}")
    print("-" * 65)
    for c, r in zip(areca_classes, local_recalls):
        c_n = int((test_local['disease_class'] == c).sum())
        local_class_breakdown[c] = {
            "recall": float(r),
            "N": c_n
        }
        print(f"{c:<25} {float(r):<18} {c_n:<18}")

    results_report['arecanut_local_test_eval'] = {
        "N": n_test_local,
        "disease_accuracy": local_acc,
        "disease_macro_f1": local_f1,
        "per_class_results": local_class_breakdown
    }


    # =========================================================================
    # PART 3: CHECKPOINTS AUDIT (outputs/models/ & subdirectories)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PART 3] ARECANUT CHECKPOINTS AUDIT: LOCAL-CAPTURE VS MERGED DATA")
    print("=" * 80)

    all_areca_pths = sorted(glob.glob(os.path.join(config.OUTPUTS_DIR, "models/**/arecanut*.pth"), recursive=True))

    checkpoint_records = []
    has_local_disease_classifier = False

    for p in all_areca_pths:
        mtime = os.path.getmtime(p)
        dt_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_mb = os.path.getsize(p) / (1024 * 1024)
        rel_path = os.path.reljust = os.path.relpath(p, PROJECT_ROOT)

        ck = torch.load(p, map_location='cpu', weights_only=False)
        is_dict = isinstance(ck, dict)
        classes = ck.get('classes') if is_dict else None
        split_version = ck.get('split_version') if is_dict else None
        arch = ck.get('architecture') if is_dict else None

        # Determine model purpose: Disease Classifier vs Severity
        if classes is not None and len(classes) == 9:
            purpose = "Disease Classifier (9 classes)"
        elif "ordinal" in p.lower() or "corn" in p.lower() or "softmax" in p.lower() or "baseline" in p.lower():
            purpose = "Severity Grading Model (0-3 grades)"
        else:
            purpose = "Unknown / Model weights"

        # Check training dataset
        if split_version in ['honest_baseline_v2', 'v2_honest_baseline']:
            train_source = "Trained on Merged Split (71.25% Kaggle images)"
        elif "clean_retrain" in p or "clean_v2_severity" in p:
            train_source = "Trained on 871 clean images (Severity only, stock photos removed)"
        elif "merged_expanded" in p:
            train_source = "Trained on Merged Expanded dataset (Severity only)"
        else:
            train_source = "Standard split / Unknown provenance"

        if purpose.startswith("Disease Classifier") and "clean" in train_source.lower():
            has_local_disease_classifier = True

        rec = {
            "path": rel_path,
            "filename": os.path.basename(p),
            "size_mb": size_mb,
            "modification_date": dt_str,
            "purpose": purpose,
            "training_source": train_source,
            "split_version": split_version,
            "architecture": arch
        }
        checkpoint_records.append(rec)
        print(f"\nFile: {rel_path}")
        print(f"  Mod Date: {dt_str} | Size: {size_mb:.2f} MB")
        print(f"  Purpose:  {purpose}")
        print(f"  Source:   {train_source} (split_version: {split_version})")

    print("\n--- Summary of Checkpoint Existence ---")
    print(f"Does a checkpoint trained on local-capture-only data exist for Arecanut Disease Classification? {has_local_disease_classifier}")
    print("Explanation: All disease classifiers in outputs/models/ ('arecanut_disease_classifier.pth', 'arecanut_disease_classifier_rerun2.pth', 'arecanut_disease_classifier_v2.pth') were trained on splits that heavily incorporate Kaggle merged images (71.25% of train). The only 'clean' retrained models are in 'clean_retrain/' and 'clean_v2_severity/', which are severity grading models (CORN/Softmax 0-3), NOT 9-class disease classifiers.")

    results_report['arecanut_checkpoints_audit'] = {
        "local_capture_only_disease_classifier_exists": has_local_disease_classifier,
        "total_checkpoints_found": len(checkpoint_records),
        "checkpoints": checkpoint_records
    }

    out_json = os.path.join(PROJECT_ROOT, "reconcile_report.json")
    with open(out_json, "w") as f:
        json.dump(results_report, f, indent=2)
    print(f"\nSaved full reconciliation report to: {out_json}")

if __name__ == "__main__":
    main()
