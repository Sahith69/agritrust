"""
scripts/dev/train_and_eval_apple_fgvc.py - Fine-tune Apple Classifier on FGVC Orchard + Lab Data,
Refit OOD Guard, and Execute Comprehensive Re-evaluation

Stage 3 Steps 3d, 3e, 3f:
- Architecture: ResNet-18, 4 outputs: ['apple_scab', 'black_rot', 'cedar_apple_rust', 'healthy']
- Training data: PlantVillage train (2,219 images) + fgvc_train (1,037 images, oversampled 2x so FGVC >= 40%)
- Augmentations: Strong augmentation on orchard photos (random resized crop, jitter, blur, rotation)
- Early layers frozen (conv1, bn1, layer1, layer2); train layer3, layer4, fc
- Time-box: 15 epochs, seed 42
- Saves checkpoint to outputs/apple_fgvc/apple_disease_fgvc_v1.pth (never touches deployed model)
- OOD Guard: Re-fits Ledoit-Wolf Mahalanobis parameters on train features into
  outputs/apple_fgvc/ood_params_apple_fgvc.pt, calibrating 95% retention threshold on fgvc_val
- Re-evaluates on frozen fgvc_test (guarded + raw argmax), computes bootstrap CIs,
  confusion matrices, delta vs baseline, and McNemar test p-value
- Evaluates black_rot retention on PlantVillage test set
- Provides final deployment recommendation
"""

import os
import sys
import time
import json
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import chi2, binomtest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

sys.path.insert(0, os.path.abspath("."))
import config

# ── Seed Initialization
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

OUT_DIR = "outputs/apple_fgvc"
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUT_DIR, "apple_disease_fgvc_v1.pth")
OOD_PARAMS_PATH = os.path.join(OUT_DIR, "ood_params_apple_fgvc.pt")
DEPLOYED_MODEL_PATH = "outputs/models/apple_disease_classifier.pth"

CLASSES = ["apple_scab", "black_rot", "cedar_apple_rust", "healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── Dataset Definition
class DualAppleDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], orchard_tf=None, lab_tf=None):
        self.samples = samples
        self.orchard_tf = orchard_tf
        self.lab_tf = lab_tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        p = item["path"]
        lbl = item["label"]
        is_orchard = item["is_orchard"]

        with Image.open(p) as img:
            rgb = img.convert("RGB")

        if is_orchard and self.orchard_tf:
            tensor = self.orchard_tf(rgb)
        elif self.lab_tf:
            tensor = self.lab_tf(rgb)
        else:
            raise ValueError("No transform specified")

        return tensor, lbl


# ── Transform Pipelines
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

orchard_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

lab_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])


# ── Build Training & Validation Datasets
def build_datasets():
    # 1. PlantVillage apple train (lab)
    pv_df = pd.read_csv("outputs/apple_splits.csv")
    pv_train = pv_df[pv_df["split"] == "train"].copy()

    lab_samples = []
    for _, r in pv_train.iterrows():
        lab_samples.append({
            "path": r["image_path"],
            "label": CLASS_TO_IDX[r["disease_class"]],
            "is_orchard": False
        })

    # 2. FGVC train (orchard)
    fgvc_df = pd.read_csv("outputs/apple_fgvc/fgvc_splits.csv")
    fgvc_train = fgvc_df[fgvc_df["split"] == "train"].copy()

    orchard_samples = []
    for _, r in fgvc_train.iterrows():
        orchard_samples.append({
            "path": r["image_path"],
            "label": CLASS_TO_IDX[r["disease_class"]],
            "is_orchard": True
        })

    # Oversample FGVC 2x: 1037 * 2 = 2074 orchard samples.
    # Total: 2219 lab + 2074 orchard = 4293 samples. Orchard share = 48.3% (>= 40%).
    train_samples = lab_samples + orchard_samples + orchard_samples
    random.shuffle(train_samples)

    print(f"Dataset Composition for Training:")
    print(f"  PlantVillage lab train: {len(lab_samples)} images")
    print(f"  FGVC orchard train (raw): {len(orchard_samples)} images")
    print(f"  FGVC orchard train (oversampled 2x): {len(orchard_samples) * 2} images")
    print(f"  Total train pool per epoch: {len(train_samples)} images")
    print(f"  FGVC orchard share: {len(orchard_samples)*2 / len(train_samples) * 100:.1f}% (Required: >= 40%)")

    # Validation: FGVC val
    fgvc_val = fgvc_df[fgvc_df["split"] == "val"].copy()
    val_samples = []
    for _, r in fgvc_val.iterrows():
        val_samples.append({
            "path": r["image_path"],
            "label": CLASS_TO_IDX[r["disease_class"]],
            "is_orchard": True
        })
    print(f"  Validation pool (FGVC val): {len(val_samples)} images\n")

    train_ds = DualAppleDataset(train_samples, orchard_tf=orchard_train_transform, lab_tf=lab_train_transform)
    val_ds = DualAppleDataset(val_samples, orchard_tf=eval_transform, lab_tf=eval_transform)

    return train_ds, val_ds, lab_samples + orchard_samples, fgvc_df


# ── Model Construction with Frozen Early Layers
def build_model():
    print("Constructing ResNet-18 initialized from deployed apple classifier weights...")
    deployed_ck = torch.load(DEPLOYED_MODEL_PATH, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 4)

    # Load weights
    sd = deployed_ck["model_state_dict"]
    model.load_state_dict(sd)

    # Freeze early layers: conv1, bn1, layer1, layer2
    for name, p in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Frozen parameters:    {n_frozen:,} (conv1, bn1, layer1, layer2)")
    print(f"  Trainable parameters: {n_trainable:,} (layer3, layer4, fc)\n")
    return model


# ── Training Loop
def train_model():
    train_ds, val_ds, unique_train_samples, fgvc_df = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)

    n_epochs = 15
    print("="*80)
    print(f"STARTING FINE-TUNING: 15 EPOCHS (TIME-BOX: 90 MIN)")
    print("="*80)
    start_time = time.time()

    best_val_recall = 0.0
    history = []

    for epoch in range(1, n_epochs + 1):
        t_epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches

        # Evaluate on FGVC val
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                out_v = model(x_v)
                preds = torch.argmax(out_v, dim=1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(y_v.numpy().tolist())

        val_acc = accuracy_score(val_targets, val_preds)

        # Scab + Rust recall on val
        scab_idx = CLASS_TO_IDX["apple_scab"]
        rust_idx = CLASS_TO_IDX["cedar_apple_rust"]

        val_t_np = np.array(val_targets)
        val_p_np = np.array(val_preds)

        scab_recall = np.sum((val_t_np == scab_idx) & (val_p_np == scab_idx)) / np.sum(val_t_np == scab_idx)
        rust_recall = np.sum((val_t_np == rust_idx) & (val_p_np == rust_idx)) / np.sum(val_t_np == rust_idx)
        scab_rust_recall = (scab_recall + rust_recall) / 2.0

        epoch_time = time.time() - t_epoch_start
        total_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "val_scab_recall": scab_recall,
            "val_rust_recall": rust_recall,
            "val_scab_rust_recall": scab_rust_recall,
            "epoch_time_s": epoch_time,
            "total_elapsed_s": total_time
        })

        print(
            f"Epoch {epoch:02d}/15 | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc*100:5.2f}% | "
            f"Val Scab+Rust Rec: {scab_rust_recall*100:5.2f}% (Scab: {scab_recall*100:.1f}%, Rust: {rust_recall*100:.1f}%) | "
            f"Epoch: {epoch_time:4.1f}s | "
            f"Elapsed: {total_time/60:4.1f}m"
        )

    # Save model weights to outputs/apple_fgvc/apple_disease_fgvc_v1.pth
    ck = {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "idx_to_class": {i: c for i, c in enumerate(CLASSES)}
    }
    torch.save(ck, CHECKPOINT_PATH)
    print(f"\nSaved fine-tuned checkpoint to {CHECKPOINT_PATH}")
    print(f"Total training duration: {time.time() - start_time:.1f}s ({(time.time() - start_time)/60:.2f} min)\n")

    return model, unique_train_samples, fgvc_df, history


# ── Step 3e: OOD Guard on the New Model
def fit_new_ood_guard(model, unique_train_samples, fgvc_df):
    print("="*80)
    print("STEP 3e: REFITTING MAHALANOBIS OOD GUARD FOR NEW MODEL")
    print("="*80)

    # Extract 512-d penultimate features on unique train samples
    feature_extractor = models.resnet18(weights=None)
    feature_extractor.fc = nn.Identity()
    # Load backbone weights
    ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in ck["model_state_dict"].items() if not k.startswith("fc.")}
    feature_extractor.load_state_dict(sd, strict=False)
    feature_extractor.eval()

    train_ds = DualAppleDataset(unique_train_samples, orchard_tf=eval_transform, lab_tf=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    features_list = []
    labels_list = []
    t0 = time.time()
    with torch.no_grad():
        for x, y in train_loader:
            f = feature_extractor(x)
            features_list.append(f.numpy())
            labels_list.append(y.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"Extracted features for {len(features)} unique train images in {time.time() - t0:.2f}s.")

    # 1. Class means
    num_classes = len(CLASSES)
    means = np.zeros((num_classes, 512), dtype=np.float32)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            means[c] = features[mask].mean(axis=0)

    # 2. Residuals & Ledoit-Wolf precision
    residuals = np.zeros_like(features)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            residuals[mask] = features[mask] - means[c]

    lw = LedoitWolf(assume_centered=True)
    lw.fit(residuals)
    precision = lw.precision_.astype(np.float32)

    # 3. Calibrate on fgvc_val at 95% retention
    fgvc_val_df = fgvc_df[fgvc_df["split"] == "val"]
    val_samples = [{"path": r["image_path"], "label": CLASS_TO_IDX[r["disease_class"]], "is_orchard": True} for _, r in fgvc_val_df.iterrows()]
    val_ds = DualAppleDataset(val_samples, orchard_tf=eval_transform, lab_tf=eval_transform)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    val_feats_list = []
    with torch.no_grad():
        for x, _ in val_loader:
            val_feats_list.append(feature_extractor(x).numpy())
    val_feats = np.concatenate(val_feats_list, axis=0)

    # Mahalanobis scores on val
    def calc_scores(feats, m_arr, prec):
        n = feats.shape[0]
        c_count = m_arr.shape[0]
        dists = np.zeros((n, c_count), dtype=np.float32)
        for c in range(c_count):
            diff = feats - m_arr[c]
            term = np.sum((diff @ prec) * diff, axis=1)
            dists[:, c] = np.sqrt(np.maximum(term, 0.0))
        return np.min(dists, axis=1)

    val_scores = calc_scores(val_feats, means, precision)
    threshold = float(np.percentile(val_scores, 95.0))
    val_retention = float((val_scores <= threshold).mean())

    print(f"Calibrated OOD parameters:")
    print(f"  Shrinkage coefficient: {lw.shrinkage_:.4f}")
    print(f"  Val score 95th percentile threshold: {threshold:.4f}")
    print(f"  Val retention: {val_retention*100:.1f}% ({int(val_retention*len(val_scores))}/{len(val_scores)})")

    ood_dict = {
        "means": torch.tensor(means),
        "precision": torch.tensor(precision),
        "threshold": threshold,
        "classes": CLASSES
    }
    torch.save(ood_dict, OOD_PARAMS_PATH)
    print(f"Saved new OOD guard parameters to {OOD_PARAMS_PATH}\n")

    return feature_extractor, means, precision, threshold


# ── Step 3f: Re-evaluate on Frozen fgvc_test
def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_func(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def reevaluate_fgvc_test(model, feature_extractor, means, precision, threshold, fgvc_df):
    print("="*80)
    print("STEP 3f: RE-EVALUATION OF NEW MODEL ON FROZEN FGVC TEST")
    print("="*80)

    test_df = fgvc_df[fgvc_df["split"] == "test"].copy().reset_index(drop=True)
    n_test = len(test_df)
    test_classes = ["apple_scab", "cedar_apple_rust", "healthy"]

    # Load baseline results for comparison
    with open(os.path.join(OUT_DIR, "baseline_eval_results.json")) as f:
        baseline_res = json.load(f)

    # Feature extraction & Mahalanobis distance
    test_samples = [{"path": r["image_path"], "label": CLASS_TO_IDX[r["disease_class"]], "is_orchard": True} for _, r in test_df.iterrows()]
    test_ds = DualAppleDataset(test_samples, orchard_tf=eval_transform, lab_tf=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    feature_extractor.eval()

    all_raw_preds = []
    all_raw_confs = []
    all_probs = []
    all_scores = []

    with torch.no_grad():
        for x, _ in test_loader:
            f = feature_extractor(x).numpy()
            # OOD distance
            c_count = means.shape[0]
            dists = np.zeros((len(f), c_count), dtype=np.float32)
            for c in range(c_count):
                diff = f - means[c]
                dists[:, c] = np.sqrt(np.maximum(np.sum((diff @ precision) * diff, axis=1), 0.0))
            scores = np.min(dists, axis=1)
            all_scores.extend(scores.tolist())

            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            all_raw_preds.extend(preds.tolist())
            all_raw_confs.extend(confs.tolist())
            all_probs.extend(probs.tolist())

    res_list = []
    for idx, r in test_df.iterrows():
        raw_p_idx = all_raw_preds[idx]
        raw_p_cls = CLASSES[raw_p_idx]
        sc = all_scores[idx]
        is_ood = bool(sc > threshold)
        conf = all_raw_confs[idx]

        if is_ood:
            g_status = "rejected"
            g_pred = "rejected"
        elif conf < 0.70:
            g_status = "abstained"
            g_pred = "abstained"
        else:
            g_status = "diagnosed"
            g_pred = raw_p_cls

        res_list.append({
            "image_id": r["image_id"],
            "filename": os.path.basename(r["image_path"]),
            "true_class": r["disease_class"],
            "raw_pred": raw_p_cls,
            "raw_conf": conf,
            "ood_score": sc,
            "ood_threshold": threshold,
            "is_ood": is_ood,
            "guarded_status": g_status,
            "guarded_pred": g_pred
        })

    res_df = pd.DataFrame(res_list)

    # Metrics
    n_rejected = (res_df["guarded_status"] == "rejected").sum()
    n_abstained = (res_df["guarded_status"] == "abstained").sum()
    n_diagnosed = (res_df["guarded_status"] == "diagnosed").sum()

    share_rejected = n_rejected / n_test
    share_abstained = n_abstained / n_test

    diagnosed_df = res_df[res_df["guarded_status"] == "diagnosed"]
    acc_diagnosed = accuracy_score(diagnosed_df["true_class"], diagnosed_df["guarded_pred"]) if len(diagnosed_df) > 0 else 0.0

    y_true = np.array([CLASS_TO_IDX[c] for c in res_df["true_class"]])
    y_pred_guarded = np.array([CLASS_TO_IDX.get(c, -1) for c in res_df["guarded_pred"]])

    acc_guarded_all = (y_true == y_pred_guarded).mean()
    acc_g_ci = bootstrap_ci(y_true, y_pred_guarded, lambda yt, yp: (yt == yp).mean())

    def calc_macro_f1_3(yt, yp):
        f1s = []
        for c_idx in [CLASS_TO_IDX[c] for c in test_classes]:
            tp = np.sum((yt == c_idx) & (yp == c_idx))
            fp = np.sum((yt != c_idx) & (yp == c_idx))
            fn = np.sum((yt == c_idx) & (yp != c_idx))
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s))

    macro_f1_guarded = calc_macro_f1_3(y_true, y_pred_guarded)
    macro_f1_g_ci = bootstrap_ci(y_true, y_pred_guarded, calc_macro_f1_3)

    recalls_all = {}
    recalls_diag = {}
    for c in test_classes:
        c_sub = res_df[res_df["true_class"] == c]
        recalls_all[c] = (c_sub["guarded_pred"] == c).sum() / len(c_sub)
        c_diag = c_sub[c_sub["guarded_status"] == "diagnosed"]
        recalls_diag[c] = (c_diag["guarded_pred"] == c).sum() / len(c_diag) if len(c_diag) > 0 else 0.0

    # Guarded Confusion matrix
    all_cols = CLASSES + ["abstained", "rejected"]
    cm_guarded = pd.DataFrame(0, index=test_classes, columns=all_cols)
    for _, r in res_df.iterrows():
        cm_guarded.loc[r["true_class"], r["guarded_pred"]] += 1

    # Confidence
    if len(diagnosed_df) > 0:
        correct_mask = (diagnosed_df["true_class"] == diagnosed_df["guarded_pred"])
        mean_conf_correct = float(diagnosed_df.loc[correct_mask, "raw_conf"].mean()) if correct_mask.sum() > 0 else 0.0
        mean_conf_wrong = float(diagnosed_df.loc[~correct_mask, "raw_conf"].mean()) if (~correct_mask).sum() > 0 else 0.0
    else:
        mean_conf_correct, mean_conf_wrong = 0.0, 0.0

    # Raw Argmax
    y_pred_raw = np.array([CLASS_TO_IDX[c] for c in res_df["raw_pred"]])
    acc_raw = accuracy_score(y_true, y_pred_raw)
    acc_raw_ci = bootstrap_ci(y_true, y_pred_raw, accuracy_score)
    macro_f1_raw = calc_macro_f1_3(y_true, y_pred_raw)
    macro_f1_raw_ci = bootstrap_ci(y_true, y_pred_raw, calc_macro_f1_3)

    recalls_raw = {}
    for c in test_classes:
        c_sub = res_df[res_df["true_class"] == c]
        recalls_raw[c] = (c_sub["raw_pred"] == c).sum() / len(c_sub)

    cm_raw = pd.DataFrame(0, index=test_classes, columns=CLASSES)
    for _, r in res_df.iterrows():
        cm_raw.loc[r["true_class"], r["raw_pred"]] += 1

    # ── McNemar Test vs Baseline (Raw predictions on FGVC test)
    # Load baseline raw predictions from baseline_eval_results.json
    base_examples_df = pd.DataFrame(baseline_res["examples_12"]) # or load full results if saved
    # Reconstruct baseline raw predictions
    base_raw_acc = baseline_res["raw_argmax"]["accuracy"]

    # Compute contingency table
    # We load the full baseline dataframe
    deployed_ck = torch.load(DEPLOYED_MODEL_PATH, map_location="cpu", weights_only=True)
    m_base = models.resnet18(weights=None)
    m_base.fc = nn.Linear(512, 4)
    m_base.load_state_dict(deployed_ck["model_state_dict"])
    m_base.eval()

    base_raw_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            l = m_base(x)
            p = torch.argmax(l, dim=1).numpy()
            base_raw_preds.extend(p.tolist())
    base_raw_preds = np.array(base_raw_preds)

    base_correct = (base_raw_preds == y_true)
    new_correct = (y_pred_raw == y_true)

    # b: base correct, new wrong; c: base wrong, new correct
    b = int(np.sum(base_correct & (~new_correct)))
    c = int(np.sum((~base_correct) & new_correct))
    a = int(np.sum(base_correct & new_correct))
    d = int(np.sum((~base_correct) & (~new_correct)))

    if (b + c) > 0:
        # Continuity-corrected chi-square
        chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = float(chi2.sf(chi2_stat, df=1))
    else:
        chi2_stat = 0.0
        p_val = 1.0

    # ── Evaluate Black Rot Retention on PlantVillage Apple Test
    print("\nEvaluating black_rot retention on PlantVillage Apple Test...")
    pv_df = pd.read_csv("outputs/apple_splits.csv")
    pv_test = pv_df[pv_df["split"] == "test"].copy()
    pv_rot = pv_test[pv_test["disease_class"] == "black_rot"]

    rot_samples = [{"path": r["image_path"], "label": CLASS_TO_IDX["black_rot"], "is_orchard": False} for _, r in pv_rot.iterrows()]
    rot_ds = DualAppleDataset(rot_samples, orchard_tf=eval_transform, lab_tf=eval_transform)
    rot_loader = DataLoader(rot_ds, batch_size=32, shuffle=False)

    # Deployed model on black rot
    base_rot_preds = []
    new_rot_preds = []
    with torch.no_grad():
        for x, _ in rot_loader:
            base_rot_preds.extend(torch.argmax(m_base(x), dim=1).numpy().tolist())
            new_rot_preds.extend(torch.argmax(model(x), dim=1).numpy().tolist())

    rot_idx = CLASS_TO_IDX["black_rot"]
    base_rot_recall = (np.array(base_rot_preds) == rot_idx).mean()
    new_rot_recall = (np.array(new_rot_preds) == rot_idx).mean()
    print(f"  PlantVillage black_rot test count: {len(pv_rot)}")
    print(f"  Baseline deployed model black_rot recall: {base_rot_recall*100:.2f}% ({int(base_rot_recall*len(pv_rot))}/{len(pv_rot)})")
    print(f"  New fine-tuned model black_rot recall:    {new_rot_recall*100:.2f}% ({int(new_rot_recall*len(pv_rot))}/{len(pv_rot)})")

    # 12 Example predictions
    examples_12 = []
    for c in test_classes:
        sub = res_df[res_df["true_class"] == c].head(4)
        for _, r in sub.iterrows():
            examples_12.append({
                "filename": r["filename"],
                "true_class": r["true_class"],
                "raw_pred": r["raw_pred"],
                "raw_conf": f"{r['raw_conf']:.4f}",
                "ood_score": f"{r['ood_score']:.2f}",
                "ood_thresh": f"{r['ood_threshold']:.2f}",
                "guarded_pred": r["guarded_pred"],
                "status": r["guarded_status"]
            })

    # Recommendation
    if acc_raw > base_raw_acc and new_rot_recall >= 0.90:
        rec_str = (
            f"Deploy recommendation: Deploy the fine-tuned model (apple_disease_fgvc_v1.pth); it elevates raw FGVC orchard "
            f"accuracy from {base_raw_acc*100:.1f}% to {acc_raw*100:.1f}% (p={p_val:.2e}, McNemar) while fully preserving "
            f"laboratory black rot diagnostic capability ({new_rot_recall*100:.1f}% recall)."
        )
    else:
        rec_str = (
            f"Deploy recommendation: Retain the lab model; fine-tuning did not meet target retention or safety criteria."
        )

    out_data = {
        "n_test": n_test,
        "baseline_raw_acc": base_raw_acc,
        "new_raw_acc": acc_raw,
        "acc_raw_ci_95": acc_raw_ci,
        "new_macro_f1_raw": macro_f1_raw,
        "macro_f1_raw_ci_95": macro_f1_raw_ci,
        "recalls_raw": recalls_raw,
        "guarded": {
            "n_rejected": int(n_rejected),
            "share_rejected": float(share_rejected),
            "n_abstained": int(n_abstained),
            "share_abstained": float(share_abstained),
            "n_diagnosed": int(n_diagnosed),
            "acc_diagnosed": float(acc_diagnosed),
            "acc_guarded_all": float(acc_guarded_all),
            "acc_g_ci": acc_g_ci,
            "macro_f1_guarded": float(macro_f1_guarded),
            "macro_f1_g_ci": macro_f1_g_ci,
            "recalls_all": recalls_all,
            "recalls_diagnosed": recalls_diag,
            "mean_conf_correct": mean_conf_correct,
            "mean_conf_wrong": mean_conf_wrong,
            "confusion_matrix": cm_guarded.to_dict()
        },
        "raw_confusion_matrix": cm_raw.to_dict(),
        "mcnemar": {
            "table": {"a": a, "b": b, "c": c, "d": d},
            "chi2": float(chi2_stat),
            "p_value": float(p_val)
        },
        "black_rot_retention": {
            "test_count": len(pv_rot),
            "baseline_recall": float(base_rot_recall),
            "new_recall": float(new_rot_recall)
        },
        "examples_12": examples_12,
        "recommendation": rec_str
    }

    out_eval_path = os.path.join(OUT_DIR, "final_eval_results.json")
    with open(out_eval_path, "w") as f:
        json.dump(out_data, f, indent=2)

    # ── Print Full Formatted Report
    print("\n" + "="*80)
    print("STAGE 3f FINAL COMPARATIVE EVALUATION REPORT")
    print("="*80)
    print(f"Total FGVC test images: {n_test}")
    print(f"\n[Comparison Table: Baseline Deployed vs Fine-Tuned Model]")
    print(f"{'Metric':<35} | {'Baseline Deployed':<20} | {'Fine-Tuned (New)':<20} | {'Delta':<15}")
    print("-" * 95)
    print(f"{'Raw Argmax Accuracy':<35} | {base_raw_acc*100:19.2f}% | {acc_raw*100:19.2f}% | {(acc_raw-base_raw_acc)*100:+14.2f}%")
    print(f"{'Raw Macro-F1 (3 classes)':<35} | {baseline_res['raw_argmax']['macro_f1']:19.4f} | {macro_f1_raw:19.4f} | {macro_f1_raw-baseline_res['raw_argmax']['macro_f1']:+14.4f}")
    print(f"{'  - Scab Raw Recall':<35} | {baseline_res['raw_argmax']['recalls']['apple_scab']*100:19.2f}% | {recalls_raw['apple_scab']*100:19.2f}% | {(recalls_raw['apple_scab']-baseline_res['raw_argmax']['recalls']['apple_scab'])*100:+14.2f}%")
    print(f"{'  - Rust Raw Recall':<35} | {baseline_res['raw_argmax']['recalls']['cedar_apple_rust']*100:19.2f}% | {recalls_raw['cedar_apple_rust']*100:19.2f}% | {(recalls_raw['cedar_apple_rust']-baseline_res['raw_argmax']['recalls']['cedar_apple_rust'])*100:+14.2f}%")
    print(f"{'  - Healthy Raw Recall':<35} | {baseline_res['raw_argmax']['recalls']['healthy']*100:19.2f}% | {recalls_raw['healthy']*100:19.2f}% | {(recalls_raw['healthy']-baseline_res['raw_argmax']['recalls']['healthy'])*100:+14.2f}%")
    print(f"{'Guarded OOD Rejection Share':<35} | {baseline_res['guarded']['share_rejected']*100:19.2f}% | {share_rejected*100:19.2f}% | {(share_rejected-baseline_res['guarded']['share_rejected'])*100:+14.2f}%")
    print(f"{'Guarded Abstain Share (<0.70)':<35} | {baseline_res['guarded']['share_abstained']*100:19.2f}% | {share_abstained*100:19.2f}% | {(share_abstained-baseline_res['guarded']['share_abstained'])*100:+14.2f}%")
    print(f"{'Guarded Diagnosed Accuracy':<35} | {baseline_res['guarded']['accuracy_diagnosed']*100:19.2f}% | {acc_diagnosed*100:19.2f}% | {(acc_diagnosed-baseline_res['guarded']['accuracy_diagnosed'])*100:+14.2f}%")
    print(f"{'Guarded Accuracy (All Images)':<35} | {baseline_res['guarded']['accuracy_all']*100:19.2f}% | {acc_guarded_all*100:19.2f}% | {(acc_guarded_all-baseline_res['guarded']['accuracy_all'])*100:+14.2f}%")
    print(f"{'Guarded Macro-F1 (All Images)':<35} | {baseline_res['guarded']['macro_f1_all']:19.4f} | {macro_f1_guarded:19.4f} | {macro_f1_guarded-baseline_res['guarded']['macro_f1_all']:+14.4f}")
    print(f"{'PlantVillage Black Rot Recall':<35} | {base_rot_recall*100:19.2f}% | {new_rot_recall*100:19.2f}% | {(new_rot_recall-base_rot_recall)*100:+14.2f}%")

    print(f"\nMcNemar Test Result:")
    print(f"  Contingency Table: Base Correct & New Correct = {a}, Base Correct & New Wrong = {b}, Base Wrong & New Correct = {c}, Both Wrong = {d}")
    print(f"  Chi-Square Statistic: {chi2_stat:.4f} | p-value: {p_val:.4e}")

    print(f"\nGuarded Confusion Matrix (New Model):")
    print(cm_guarded)

    print(f"\nRaw Confusion Matrix (New Model):")
    print(cm_raw)

    print(f"\n{rec_str}\n")
    print("="*80)


if __name__ == "__main__":
    trained_model, unique_train_samples, fgvc_df, hist = train_model()
    feat_ext, m, p, th = fit_new_ood_guard(trained_model, unique_train_samples, fgvc_df)
    reevaluate_fgvc_test(trained_model, feat_ext, m, p, th, fgvc_df)
