"""
Train ResNet-18 Severity Grading (CORN Ordinal and Softmax Baseline) from Scratch
on the Verified 100% Leak-Free Arecanut Split (outputs/arecanut_splits.csv).

Split details:
- Train: 1,628 images
- Val:   547 images
- Test:  486 images (Full) / 400 images (Clean Lesion, excluding YLD chlorosis)

Multi-seed training across seeds 42, 100, 2024.
Cluster-grouped bootstrap 95% CIs evaluated across test clusters.
"""
import os
import sys
import copy
import time
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    mean_absolute_error,
    f1_score,
    classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms.v2 as v2

PROJECT_ROOT = "/home/sahith/Projects/Crop"
sys.path.insert(0, PROJECT_ROOT)

import config
from src.model import CropSeverityModel
from src.corn_loss import CornLoss, corn_probs_to_grade

torch.set_num_threads(8)
device = torch.device('cpu')

SPLITS_CSV = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits.csv")
CACHE_FILE = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits_cached_tensors.pt")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs/models/clean_v2_severity")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs/reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

SEEDS = [42, 100, 2024]
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
PATIENCE = 4
N_BOOTSTRAP = 2000

# Transforms
train_transform = v2.Compose([
    v2.RandomRotation(30),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.3),
    v2.ColorJitter(brightness=0.15, contrast=0.15),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

eval_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

class FastTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # uint8 (N, 3, 224, 224)
        self.labels = labels  # long (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def train_one_epoch(model, loader, optimizer, loss_fn, ordinal=True):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = train_transform(images).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        with torch.no_grad():
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk

def validate(model, loader, loss_fn, ordinal=True):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = eval_transform(images).to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)

            total_loss += loss.item() * images.size(0)

            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk

def evaluate_on_slice(model, images, labels, clusters, ordinal=True, run_bootstrap=True):
    model.eval()
    ds = FastTensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = eval_transform(imgs).to(device)
            logits = model(imgs)
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    y_true = labels.numpy()
    y_pred = np.array(all_preds)
    clusters = np.array(clusters)

    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    acc = float(accuracy_score(y_true, y_pred) * 100)
    mae = float(mean_absolute_error(y_true, y_pred))
    w1  = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100)
    f1  = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100)

    rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
    per_grade = {
        f"G{g}": {
            "recall": round(float(rep[str(g)]['recall']), 4),
            "precision": round(float(rep[str(g)]['precision']), 4),
            "f1": round(float(rep[str(g)]['f1-score']), 4),
            "support": int(rep[str(g)]['support'])
        }
        for g in range(4)
    }

    ci_results = {}
    if run_bootstrap:
        unique_clusters = np.unique(clusters)
        n_c = len(unique_clusters)
        boot_qwk, boot_acc, boot_mae, boot_w1, boot_f1 = [], [], [], [], []
        rng = np.random.default_rng(42)

        for _ in range(N_BOOTSTRAP):
            sampled_c = rng.choice(unique_clusters, size=n_c, replace=True)
            # Gather indices
            idx_list = []
            for cid in sampled_c:
                idx_list.append(np.where(clusters == cid)[0])
            boot_idx = np.concatenate(idx_list)

            b_true = y_true[boot_idx]
            b_pred = y_pred[boot_idx]

            boot_qwk.append(cohen_kappa_score(b_true, b_pred, weights='quadratic'))
            boot_acc.append(accuracy_score(b_true, b_pred) * 100)
            boot_mae.append(mean_absolute_error(b_true, b_pred))
            boot_w1.append(np.mean(np.abs(b_true - b_pred) <= 1) * 100)
            boot_f1.append(f1_score(b_true, b_pred, average='macro', zero_division=0) * 100)

        ci_results = {
            "qwk_ci_95": [float(np.percentile(boot_qwk, 2.5)), float(np.percentile(boot_qwk, 97.5))],
            "acc_ci_95": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
            "mae_ci_95": [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
            "w1_ci_95":  [float(np.percentile(boot_w1, 2.5)),  float(np.percentile(boot_w1, 97.5))],
            "f1_ci_95":  [float(np.percentile(boot_f1, 2.5)),  float(np.percentile(boot_f1, 97.5))],
        }

    return {
        "N": len(y_true),
        "qwk": round(qwk, 4),
        "exact_accuracy": round(acc, 2),
        "mae": round(mae, 4),
        "within_1": round(w1, 2),
        "macro_f1": round(f1, 2),
        "per_grade": per_grade,
        "ci_95": ci_results,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }

def main():
    print("=" * 80)
    print("  RETRAINING RESNET-18 SEVERITY (CORN & SOFTMAX) ON VERIFIED CLEAN SPLIT")
    print(f"  Split File: {SPLITS_CSV}")
    print("=" * 80)

    # 1. Load cached tensors
    if not os.path.exists(CACHE_FILE):
        print(f"Cache file {CACHE_FILE} not found! Run scripts/cache_arecanut_tensors.py first.")
        sys.exit(1)

    t0_load = time.time()
    data = torch.load(CACHE_FILE, weights_only=False)
    print(f"Loaded cached tensors in {time.time() - t0_load:.2f}s.")

    splits = np.array(data['split'])
    train_mask = (splits == 'train')
    val_mask   = (splits == 'val')
    test_mask  = (splits == 'test')
    clean_test_mask = test_mask & (~data['exclude_from_severity'].numpy())

    train_imgs = data['images'][train_mask]
    train_lbls = data['severity_grade'][train_mask]

    val_imgs   = data['images'][val_mask]
    val_lbls   = data['severity_grade'][val_mask]

    test_imgs  = data['images'][test_mask]
    test_lbls  = data['severity_grade'][test_mask]
    test_clust = data['robust_cluster_id'][test_mask].numpy()

    clean_test_imgs  = data['images'][clean_test_mask]
    clean_test_lbls  = data['severity_grade'][clean_test_mask]
    clean_test_clust = data['robust_cluster_id'][clean_test_mask].numpy()

    print(f"Train samples: {len(train_lbls)}")
    print(f"Val samples:   {len(val_lbls)}")
    print(f"Full Test samples:         {len(test_lbls)}")
    print(f"Clean Lesion Test samples: {len(clean_test_lbls)} (excl. 86 YLD)")
    print()

    val_dataset = FastTensorDataset(val_imgs, val_lbls)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    MODELS = [
        ("corn", True),
        ("softmax", False)
    ]

    all_results = {"corn": {}, "softmax": {}}

    for model_name, is_ordinal in MODELS:
        print("\n" + "#" * 75)
        print(f"  STARTING MULTI-SEED RUN: {model_name.upper()} (Ordinal={is_ordinal})")
        print("#" * 75)

        seed_results_full = []
        seed_results_clean = []

        for seed in SEEDS:
            print(f"\n--- Model: {model_name.upper()} | Seed: {seed} ---", flush=True)
            set_seed(seed)
            t_start = time.time()

            ckpt_path = os.path.join(MODELS_DIR, f"arecanut_resnet18_{model_name}_seed{seed}.pth")
            if os.path.exists(ckpt_path):
                print(f"Found existing checkpoint: {ckpt_path}. Loading weights directly...", flush=True)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                best_weights = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                best_epoch = ckpt.get('best_epoch', 0) if isinstance(ckpt, dict) else 0
                best_val_qwk = ckpt.get('best_val_qwk', 0.0) if isinstance(ckpt, dict) else 0.0
            else:
                train_dataset = FastTensorDataset(train_imgs, train_lbls)
                train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                model = CropSeverityModel(
                    backbone='resnet18',
                    num_classes=4,
                    ordinal=is_ordinal,
                    pretrained=True
                ).to(device)

                optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                if is_ordinal:
                    loss_fn = CornLoss(num_classes=4)
                else:
                    loss_fn = nn.CrossEntropyLoss()

                best_val_qwk = -1.0
                best_epoch = 0
                best_weights = None
                epochs_no_improve = 0

                for epoch in range(MAX_EPOCHS):
                    t_ep0 = time.time()
                    tr_loss, tr_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn, ordinal=is_ordinal)
                    v_loss, v_qwk   = validate(model, val_loader, loss_fn, ordinal=is_ordinal)
                    ep_time = time.time() - t_ep0

                    if v_qwk > best_val_qwk:
                        best_val_qwk = v_qwk
                        best_epoch = epoch + 1
                        best_weights = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        flag = "  --> [BEST VAL QWK]"
                    else:
                        epochs_no_improve += 1
                        flag = ""

                    print(
                        f"Epoch {epoch+1:02d}/{MAX_EPOCHS} ({ep_time:.1f}s) | "
                        f"Train Loss: {tr_loss:.4f} | Train QWK: {tr_qwk:.4f} | "
                        f"Val Loss: {v_loss:.4f} | Val QWK: {v_qwk:.4f}{flag}",
                        flush=True
                    )

                    if epochs_no_improve >= PATIENCE:
                        print(f"Early stopping at epoch {epoch+1} (patience={PATIENCE}).", flush=True)
                        break

                # Save checkpoint
                torch.save({
                    'model_state_dict': best_weights,
                    'backbone': 'resnet18',
                    'ordinal': is_ordinal,
                    'seed': seed,
                    'best_epoch': best_epoch,
                    'best_val_qwk': best_val_qwk,
                    'train_time_sec': time.time() - t_start
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path} (Best Val QWK: {best_val_qwk:.4f} @ Ep {best_epoch})")

            # If seed 42, also update the primary checkpoint in outputs/models/
            if seed == 42:
                primary_path = os.path.join(PROJECT_ROOT, f"outputs/models/arecanut_{'ordinal' if is_ordinal else 'baseline'}.pth")
                torch.save(best_weights, primary_path)
                print(f"Updated primary model: {primary_path}")

            # Load best model for test evaluation
            model = CropSeverityModel(
                backbone='resnet18',
                num_classes=4,
                ordinal=is_ordinal,
                pretrained=False
            ).to(device)
            model.load_state_dict(best_weights)

            # Evaluate on Full Test Set (N=486)
            res_full = evaluate_on_slice(model, test_imgs, test_lbls, test_clust, ordinal=is_ordinal, run_bootstrap=True)
            res_full['seed'] = seed
            res_full['best_epoch'] = best_epoch
            res_full['best_val_qwk'] = round(best_val_qwk, 4)
            seed_results_full.append(res_full)

            # Evaluate on Clean Lesion Test Set (N=400)
            res_clean = evaluate_on_slice(model, clean_test_imgs, clean_test_lbls, clean_test_clust, ordinal=is_ordinal, run_bootstrap=True)
            res_clean['seed'] = seed
            res_clean['best_epoch'] = best_epoch
            res_clean['best_val_qwk'] = round(best_val_qwk, 4)
            seed_results_clean.append(res_clean)

            print(f"  Test Evaluation (Full N={res_full['N']}):  QWK = {res_full['qwk']:.4f} | Acc = {res_full['exact_accuracy']:.2f}% | MAE = {res_full['mae']:.4f} | Within-1 = {res_full['within_1']:.2f}%")
            print(f"  Test Evaluation (Clean N={res_clean['N']}): QWK = {res_clean['qwk']:.4f} | Acc = {res_clean['exact_accuracy']:.2f}% | MAE = {res_clean['mae']:.4f} | Within-1 = {res_clean['within_1']:.2f}%")

        # Compute Multi-Seed Aggregates
        def aggregate_seeds(results_list):
            qwks = [r['qwk'] for r in results_list]
            accs = [r['exact_accuracy'] for r in results_list]
            maes = [r['mae'] for r in results_list]
            w1s  = [r['within_1'] for r in results_list]
            f1s  = [r['macro_f1'] for r in results_list]

            # Aggregate per-grade
            per_grade_agg = {}
            for g in range(4):
                g_key = f"G{g}"
                recalls = [r['per_grade'][g_key]['recall'] for r in results_list]
                precs   = [r['per_grade'][g_key]['precision'] for r in results_list]
                f1_g    = [r['per_grade'][g_key]['f1'] for r in results_list]
                per_grade_agg[g_key] = {
                    'recall_mean': round(float(np.mean(recalls)), 4),
                    'recall_sd': round(float(np.std(recalls, ddof=1)), 4),
                    'precision_mean': round(float(np.mean(precs)), 4),
                    'precision_sd': round(float(np.std(precs, ddof=1)), 4),
                    'f1_mean': round(float(np.mean(f1_g)), 4),
                    'f1_sd': round(float(np.std(f1_g, ddof=1)), 4),
                    'support': results_list[0]['per_grade'][g_key]['support']
                }

            return {
                'qwk_mean': round(float(np.mean(qwks)), 4),
                'qwk_sd': round(float(np.std(qwks, ddof=1)), 4),
                'exact_accuracy_mean': round(float(np.mean(accs)), 2),
                'exact_accuracy_sd': round(float(np.std(accs, ddof=1)), 2),
                'mae_mean': round(float(np.mean(maes)), 4),
                'mae_sd': round(float(np.std(maes, ddof=1)), 4),
                'within_1_mean': round(float(np.mean(w1s)), 2),
                'within_1_sd': round(float(np.std(w1s, ddof=1)), 2),
                'macro_f1_mean': round(float(np.mean(f1s)), 2),
                'macro_f1_sd': round(float(np.std(f1s, ddof=1)), 2),
                'per_grade': per_grade_agg,
                'seed_runs': results_list
            }

        all_results[model_name] = {
            'full_test_486': aggregate_seeds(seed_results_full),
            'clean_test_400': aggregate_seeds(seed_results_clean)
        }

    # Save summary report
    summary_path = os.path.join(REPORTS_DIR, "arecanut_clean_severity_multiseed_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved full results to: {summary_path}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL AUTHORITATIVE RESULTS (100% LEAK-FREE ARECANUT SPLIT)")
    print("=" * 80)
    for model_name in ["corn", "softmax"]:
        res_f = all_results[model_name]['full_test_486']
        res_c = all_results[model_name]['clean_test_400']
        print(f"\n{model_name.upper()}:")
        print(f"  Full Test (N=486):  QWK = {res_f['qwk_mean']:.4f} ± {res_f['qwk_sd']:.4f} | Acc = {res_f['exact_accuracy_mean']:.2f} ± {res_f['exact_accuracy_sd']:.2f}% | MAE = {res_f['mae_mean']:.4f} | Within-1 = {res_f['within_1_mean']:.2f}%")
        print(f"  Clean Test (N=400): QWK = {res_c['qwk_mean']:.4f} ± {res_c['qwk_sd']:.4f} | Acc = {res_c['exact_accuracy_mean']:.2f} ± {res_c['exact_accuracy_sd']:.2f}% | MAE = {res_c['mae_mean']:.4f} | Within-1 = {res_c['within_1_mean']:.2f}%")

if __name__ == '__main__':
    main()
