"""
Task A: Class-weighted retrain for Arecanut Grade 2/3 imbalance.
Applies inverse-class-frequency weighting via WeightedRandomSampler in DataLoader.
Trains ResNet-18 with both CORN ordinal loss and Softmax loss.
Saves models to outputs/improvements/ and metrics to outputs/improvements/class_weighted_arecanut_results.json.
"""
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

import os
import time
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score
)

from src.dataset import CropDiseaseDataset, get_transforms
from src.model import CropSeverityModel
from src.corn_loss import CornLoss, corn_probs_to_grade
from src.ordinal_ce_loss import OrdinalCrossEntropyLoss

num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)
NUM_WORKERS = 4
device = torch.device('cpu')

IMPROVEMENTS_DIR = os.path.join(config.OUTPUTS_DIR, "improvements")
os.makedirs(IMPROVEMENTS_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def get_weighted_dataloaders(split_csv, batch_size=16):
    train_ds = CropDiseaseDataset(split_csv, 'train', get_transforms('train'))
    val_ds = CropDiseaseDataset(split_csv, 'val', get_transforms('val'))
    test_ds = CropDiseaseDataset(split_csv, 'test', get_transforms('test'))

    # Extract train severity grades
    train_labels = np.array([int(train_ds.df.iloc[i]['severity_grade']) for i in range(len(train_ds))])
    class_counts = np.bincount(train_labels, minlength=config.NUM_SEVERITY_GRADES)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader, class_counts


def train_one_epoch(model, loader, optimizer, loss_fn, ordinal=True):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _, _ in loader:
        images = images.to(device)
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
        for images, targets, _, _ in loader:
            images = images.to(device)
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


def train_and_eval(model_name, ordinal=True, seed=42):
    print(f"\n=======================================================")
    print(f"Starting Weighted Training: {model_name} (Ordinal={ordinal})")
    print(f"=======================================================")
    set_seed(seed)
    start_t = time.time()

    train_loader, val_loader, test_loader, class_counts = get_weighted_dataloaders(
        config.ARECANUT_SPLIT_CSV,
        batch_size=config.BATCH_SIZE
    )

    model = CropSeverityModel(
        backbone='resnet18',
        num_classes=config.NUM_SEVERITY_GRADES,
        ordinal=ordinal,
        pretrained=True
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    if ordinal:
        loss_fn = CornLoss(num_classes=config.NUM_SEVERITY_GRADES)
    else:
        loss_fn = OrdinalCrossEntropyLoss(num_classes=config.NUM_SEVERITY_GRADES)

    best_qwk = -1.0
    best_model_state = None
    epochs_no_improve = 0
    best_epoch = 0

    history = {'train_loss': [], 'train_qwk': [], 'val_loss': [], 'val_qwk': []}

    for epoch in range(config.NUM_EPOCHS):
        ep_t0 = time.time()
        train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn, ordinal)
        val_loss, val_qwk = validate(model, val_loader, loss_fn, ordinal)
        ep_t1 = time.time()

        history['train_loss'].append(train_loss)
        history['train_qwk'].append(train_qwk)
        history['val_loss'].append(val_loss)
        history['val_qwk'].append(val_qwk)

        is_best = val_qwk > best_qwk
        if is_best:
            best_qwk = val_qwk
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_mark = " *BEST*"
        else:
            epochs_no_improve += 1
            best_mark = ""

        print(f"Epoch {epoch+1:2d}/{config.NUM_EPOCHS} ({ep_t1-ep_t0:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, QWK: {train_qwk:.4f} | "
              f"Val Loss: {val_loss:.4f}, QWK: {val_qwk:.4f}{best_mark}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (patience={config.EARLY_STOPPING_PATIENCE})")
            break

    total_training_time = time.time() - start_t
    print(f"Completed in {total_training_time:.1f}s. Best Val QWK: {best_qwk:.4f} at epoch {best_epoch}.")

    # Save checkpoint to outputs/improvements/
    ckpt_filename = f"arecanut_weighted_{'ordinal' if ordinal else 'baseline'}.pth"
    ckpt_path = os.path.join(IMPROVEMENTS_DIR, ckpt_filename)
    torch.save(best_model_state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Evaluate on held-out test set
    model.load_state_dict(best_model_state)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets, _, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            y_true.extend(targets.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    test_mae = float(mean_absolute_error(y_true, y_pred))
    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1_macro = float(f1_score(y_true, y_pred, average='macro'))
    test_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    report_dict = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)

    per_grade = {}
    for g in range(config.NUM_SEVERITY_GRADES):
        str_g = str(g)
        per_grade[f"Grade_{g}"] = {
            "precision": float(report_dict[str_g]["precision"]),
            "recall": float(report_dict[str_g]["recall"]),
            "f1": float(report_dict[str_g]["f1-score"]),
            "support": int(report_dict[str_g]["support"])
        }

    return {
        "model_name": model_name,
        "ordinal": ordinal,
        "runtime_seconds": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "best_val_qwk": round(best_qwk, 4),
        "test_metrics": {
            "qwk": round(test_qwk, 4),
            "mae": round(test_mae, 4),
            "exact_accuracy": round(test_acc, 4),
            "macro_f1": round(test_f1_macro, 4)
        },
        "confusion_matrix": test_cm,
        "per_grade_metrics": per_grade,
        "checkpoint_path": ckpt_path
    }


def evaluate_existing_baseline(path, ordinal=True):
    test_ds = CropDiseaseDataset(config.ARECANUT_SPLIT_CSV, 'test', get_transforms('test'))
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=ordinal, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device)
            logits = model(images)
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            y_true.extend(targets.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    test_mae = float(mean_absolute_error(y_true, y_pred))
    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1_macro = float(f1_score(y_true, y_pred, average='macro'))
    test_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    report_dict = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)

    per_grade = {}
    for g in range(config.NUM_SEVERITY_GRADES):
        str_g = str(g)
        per_grade[f"Grade_{g}"] = {
            "precision": float(report_dict[str_g]["precision"]),
            "recall": float(report_dict[str_g]["recall"]),
            "f1": float(report_dict[str_g]["f1-score"]),
            "support": int(report_dict[str_g]["support"])
        }

    return {
        "qwk": round(test_qwk, 4),
        "mae": round(test_mae, 4),
        "exact_accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1_macro, 4),
        "confusion_matrix": test_cm,
        "per_grade_metrics": per_grade
    }


def main():
    # 1. Check training-set class distribution
    train_df = pd.read_csv(config.ARECANUT_SPLIT_CSV)
    train_df = train_df[train_df['split'] == 'train'].reset_index(drop=True)
    counts = train_df['severity_grade'].value_counts().sort_index().to_dict()

    print("=== Arecanut Training-Set Class Distribution ===")
    total = len(train_df)
    for g, cnt in counts.items():
        print(f"Grade {g}: {cnt} samples ({cnt/total*100:.2f}%)")
    print(f"Total: {total} samples\n")

    # 2. Get baseline unweighted metrics
    print("Extracting baseline unweighted metrics from existing checkpoints...")
    orig_corn = evaluate_existing_baseline(os.path.join(config.MODELS_DIR, 'arecanut_ordinal.pth'), ordinal=True)
    orig_base = evaluate_existing_baseline(os.path.join(config.MODELS_DIR, 'arecanut_baseline.pth'), ordinal=False)

    # 3. Retrain CORN with WeightedRandomSampler
    weighted_corn = train_and_eval("Arecanut ResNet-18 (CORN Weighted)", ordinal=True, seed=42)

    # 4. Retrain Softmax with WeightedRandomSampler
    weighted_base = train_and_eval("Arecanut ResNet-18 (Softmax Weighted)", ordinal=False, seed=42)

    # 5. Compile side-by-side results
    results = {
        "training_set_class_distribution": {
            f"Grade {g}": {"count": int(cnt), "percentage": round(cnt/total*100, 2)}
            for g, cnt in counts.items()
        },
        "total_train_samples": total,
        "sampling_method": "WeightedRandomSampler (inverse class frequency: w_k = 1.0 / count_k)",
        "hyperparameters": {
            "backbone": "resnet18",
            "batch_size": 16,
            "learning_rate": 1e-4,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "early_stopping_patience": 5,
            "random_seed": 42
        },
        "comparison_corn": {
            "original_unweighted": orig_corn,
            "class_weighted_retrain": weighted_corn
        },
        "comparison_softmax": {
            "original_unweighted": orig_base,
            "class_weighted_retrain": weighted_base
        }
    }

    out_file = os.path.join(IMPROVEMENTS_DIR, "class_weighted_arecanut_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=======================================================")
    print(f"Results successfully saved to {out_file}")
    print(f"=======================================================")

if __name__ == '__main__':
    main()
