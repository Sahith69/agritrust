#!/usr/bin/env python3
"""
scripts/dev/predeploy_eval.py
==============================
Full pre-deployment evaluation for the new apple model (apple_disease_fgvc_v1.pth).
New files only.  Does NOT modify ood_params.pt; swaps apple OOD in memory.

Tasks:
  1. PlantVillage apple test: deployed vs new model, accuracy/F1/recall, 95% CIs,
     confusion matrices; new model retention under new OOD guard; demo image scores.
  2. FGVC test: 95% CIs for raw acc/F1/per-class recall; reconcile guarded confusion
     matrix with 97.19%; which epoch was saved and why.
  3. Cross-crop guard re-run with new apple OOD: false block on PV test, fgvc_test,
     6 wrong-crop combos, 60 non-crop, fgvc_test images in paddy/arecanut mode.
  4. Abstain calibration on (fgvc_val + PV val) then fgvc_test and PV test at 0.70.
  5. Distribution check: DINOv2 nearest-train cosine for fgvc_test; pHash<=16;
     accuracy/guarded accuracy for near-/far-from-train subsets.
  6. Write outputs/apple_fgvc/predeploy_report.md.
"""

import os, sys, time, random, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from ood_guard import (
    get_feature_extractor, get_image_transform, compute_mahalanobis_scores, load_ood_params
)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
APPLE_V1_PTH   = os.path.join(PROJECT_ROOT, "outputs", "apple_fgvc", "apple_disease_fgvc_v1.pth")
APPLE_OOD_NEW  = os.path.join(PROJECT_ROOT, "outputs", "apple_fgvc", "ood_params_apple_fgvc.pt")
FGVC_SPLITS    = os.path.join(PROJECT_ROOT, "outputs", "apple_fgvc", "fgvc_splits.csv")
FGVC_FEATURES  = os.path.join(PROJECT_ROOT, "outputs", "apple_fgvc", "fgvc_features.npz")
PV_FEATURES    = os.path.join(PROJECT_ROOT, "outputs", "apple_fgvc", "plantvillage_train_features.npz")
APPLE_SPLITS   = os.path.join(PROJECT_ROOT, "outputs", "apple_splits.csv")
SAMPLES_DIR    = os.path.join(PROJECT_ROOT, "tests", "samples")
ABSTAIN_TAU    = 0.70
CROPS          = ["paddy", "arecanut", "apple"]
IMG_SIZE       = config.IMG_SIZE
IMG_MEAN       = config.IMAGENET_MEAN
IMG_STD        = config.IMAGENET_STD
OUT_DIR        = os.path.join(PROJECT_ROOT, "outputs", "apple_fgvc")

# ── helpers ───────────────────────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

def load_classifier(pth_path):
    """Load a ResNet-18 disease classifier, return (model, classes)."""
    ck = torch.load(pth_path, map_location="cpu", weights_only=True)
    classes = ck["classes"]
    model = models.resnet18(weights=None)
    if "fc.1.weight" in ck["model_state_dict"]:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.fc.in_features, len(classes)))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, classes, ck.get("best_epoch"), ck.get("selection_metric")

def predict_batch(model, paths, batch_size=64):
    """Returns (preds_idx, confidences, all_probs) arrays."""
    pred_idxs, confs, all_probs = [], [], []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        tensors = []
        for p in batch:
            with Image.open(p) as im:
                tensors.append(TF(im.convert("RGB")))
        t = torch.stack(tensors)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1).numpy()
        pred_idxs.extend(np.argmax(probs, axis=1).tolist())
        confs.extend(probs.max(axis=1).tolist())
        all_probs.extend(probs.tolist())
    return np.array(pred_idxs), np.array(confs), np.array(all_probs)

def extract_features_classifier(model_no_fc, paths, batch_size=64):
    """Extract 512-d penultimate features using a backbone with fc=Identity."""
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        tensors = []
        for p in batch:
            with Image.open(p) as im:
                tensors.append(TF(im.convert("RGB")))
        t = torch.stack(tensors)
        with torch.no_grad():
            f = model_no_fc(t).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)

def bootstrap_ci(metric_fn, y_true, y_pred, n=2000, seed=42, **kw):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        scores.append(metric_fn(y_true[idx], y_pred[idx], **kw))
    lo, hi = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    return np.mean(scores), lo, hi

def accuracy_fn(yt, yp): return np.mean(yt == yp)

def macro_f1_fn(yt, yp, classes=None):
    from collections import defaultdict
    classes = classes or sorted(set(yt.tolist()))
    recalls, precisions = [], []
    for c in classes:
        tp = np.sum((yt==c)&(yp==c)); fp = np.sum((yt!=c)&(yp==c)); fn = np.sum((yt==c)&(yp!=c))
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        f = 2*p*r/(p+r) if (p+r)>0 else 0.0
        recalls.append(r); precisions.append(p)
    return np.mean([2*(p*r/(p+r)) if (p+r)>0 else 0 for p,r in zip(precisions,recalls)])

def per_class_recall(yt, yp, c):
    mask = (yt==c); return np.mean(yp[mask]==c) if mask.sum()>0 else 0.0

def confusion_matrix_str(yt, yp, labels, extra_cols=None, extra_data=None):
    """Print confusion matrix as markdown table."""
    label_names = [str(l) for l in labels]
    col_headers = label_names + (list(extra_cols) if extra_cols else [])
    lines = ["| True \\ Pred | " + " | ".join(col_headers) + " |"]
    lines.append("|" + "---|"*(len(col_headers)+1))
    for li, l in enumerate(labels):
        row = [str(np.sum((yt==l)&(yp==c))) for c in labels]
        if extra_data:
            row += [str(extra_data[li][j]) for j in range(len(extra_cols))]
        lines.append(f"| {label_names[li]} | " + " | ".join(row) + " |")
    return "\n".join(lines)

def ood_score_single(feat, means, precision):
    return float(compute_mahalanobis_scores(feat[np.newaxis], means, precision)[0])

def extract_feat_single(backbone, path):
    with Image.open(path) as im:
        t = TF(im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return backbone(t).cpu().numpy()  # (1,512)

# ── load everything ───────────────────────────────────────────────────────────
print("="*70)
print("Loading deployed apple model (apple_disease_classifier.pth)...")
deployed_pth = os.path.join(PROJECT_ROOT, "outputs", "models", "apple_disease_classifier.pth")
dep_model, dep_classes, _, _ = load_classifier(deployed_pth)
print(f"  Classes: {dep_classes}")

print("\nLoading new apple model (apple_disease_fgvc_v1.pth)...")
new_ck = torch.load(APPLE_V1_PTH, map_location="cpu", weights_only=True)
new_classes = new_ck["classes"]
best_epoch = new_ck.get("best_epoch", "N/A")
selection_metric = new_ck.get("selection_metric", "N/A")
new_model_full, new_classes, _, _ = load_classifier(APPLE_V1_PTH)
print(f"  Classes: {new_classes}  best_epoch={best_epoch}  metric={selection_metric}")

print("\nLoading new apple OOD guard (ood_params_apple_fgvc.pt)...")
new_ood_raw = torch.load(APPLE_OOD_NEW, map_location="cpu", weights_only=False)
new_ood_means     = new_ood_raw["means"].numpy()
new_ood_precision = new_ood_raw["precision"].numpy()
new_ood_threshold = float(new_ood_raw["threshold"])
print(f"  Threshold: {new_ood_threshold:.4f}")

print("\nLoading deployed OOD params (in-memory only, NOT overwriting)...")
ood_params_all = load_ood_params()  # loaded to memory, never overwritten
dep_ood_means     = ood_params_all["apple"]["means"].numpy()
dep_ood_precision = ood_params_all["apple"]["precision"].numpy()
dep_ood_threshold = float(ood_params_all["apple"]["threshold"])
print(f"  Deployed apple OOD threshold: {dep_ood_threshold:.4f}")

# Build backbone models for feature extraction
print("\nBuilding backbone feature extractors...")
dep_backbone, _ = get_feature_extractor("apple")   # deployed apple backbone

new_backbone = models.resnet18(weights=None)
ck_new = torch.load(APPLE_V1_PTH, map_location="cpu", weights_only=True)
nc = new_classes
if "fc.1.weight" in ck_new["model_state_dict"]:
    new_backbone.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(new_backbone.fc.in_features, len(nc)))
else:
    new_backbone.fc = nn.Linear(new_backbone.fc.in_features, len(nc))
new_backbone.load_state_dict(ck_new["model_state_dict"])
new_backbone.fc = nn.Identity()
new_backbone.eval()

# ── load splits ───────────────────────────────────────────────────────────────
pv_df = pd.read_csv(APPLE_SPLITS)
pv_test = pv_df[pv_df["split"]=="test"].copy().reset_index(drop=True)
pv_val  = pv_df[pv_df["split"]=="val"].copy().reset_index(drop=True)
pv_train= pv_df[pv_df["split"]=="train"].copy().reset_index(drop=True)

fgvc_df   = pd.read_csv(FGVC_SPLITS)
fgvc_test = fgvc_df[fgvc_df["split"]=="fgvc_test"].copy().reset_index(drop=True)
fgvc_val  = fgvc_df[fgvc_df["split"]=="fgvc_val"].copy().reset_index(drop=True)
fgvc_train= fgvc_df[fgvc_df["split"]=="fgvc_train"].copy().reset_index(drop=True)

# remap fgvc class names for new_classes
CLASS_MAP_FGVC = {"scab": "apple_scab", "rust": "cedar_apple_rust", "healthy": "healthy"}
fgvc_test["cls_mapped"]  = fgvc_test["label"].map(CLASS_MAP_FGVC)
fgvc_val["cls_mapped"]   = fgvc_val["label"].map(CLASS_MAP_FGVC)
fgvc_train["cls_mapped"] = fgvc_train["label"].map(CLASS_MAP_FGVC)

print(f"\nPV test:   {len(pv_test)} images  ({pv_test['disease_class'].value_counts().to_dict()})")
print(f"PV val:    {len(pv_val)} images")
print(f"FGVC test: {len(fgvc_test)} images ({fgvc_test['cls_mapped'].value_counts().to_dict()})")
print(f"FGVC val:  {len(fgvc_val)} images")

# ── check image paths ─────────────────────────────────────────────────────────
for col, df in [("image_path", pv_test), ("image_path", fgvc_test)]:
    missing = [p for p in df[col] if not os.path.exists(p)]
    if missing:
        print(f"WARNING: {len(missing)} missing paths, e.g. {missing[:2]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PlantVillage apple test – deployed vs new, OOD retention, demo
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1: PlantVillage Apple Test Split")
print("="*70)

pv_test_paths   = pv_test["image_path"].tolist()
pv_test_labels  = pv_test["disease_class"].tolist()

# PV label -> index for dep model
dep_cls2idx = {c:i for i,c in enumerate(dep_classes)}
new_cls2idx = {c:i for i,c in enumerate(new_classes)}

# --- 1a. Deployed model on PV test ---
print("\nRunning deployed model on PV test...")
t0 = time.time()
dep_pred_idx, dep_conf, dep_probs = predict_batch(dep_model, pv_test_paths)
dep_pred_cls = [dep_classes[i] for i in dep_pred_idx]
print(f"  Done in {time.time()-t0:.1f}s")

# OOD scores using deployed OOD (backbone=dep_backbone, params=dep_ood)
print("Extracting deployed backbone features for PV test OOD scores...")
t0 = time.time()
dep_feats_pv = extract_features_classifier(dep_backbone, pv_test_paths)
dep_ood_scores_pv = compute_mahalanobis_scores(dep_feats_pv, dep_ood_means, dep_ood_precision)
print(f"  Done in {time.time()-t0:.1f}s")

# --- 1b. New model on PV test ---
print("Running new model on PV test...")
t0 = time.time()
new_pred_idx, new_conf, new_probs = predict_batch(new_model_full, pv_test_paths)
new_pred_cls = [new_classes[i] for i in new_pred_idx]
print(f"  Done in {time.time()-t0:.1f}s")

# OOD scores using NEW OOD (backbone=new_backbone)
print("Extracting new backbone features for PV test OOD scores...")
t0 = time.time()
new_feats_pv = extract_features_classifier(new_backbone, pv_test_paths)
new_ood_scores_pv = compute_mahalanobis_scores(new_feats_pv, new_ood_means, new_ood_precision)
print(f"  Done in {time.time()-t0:.1f}s")

# Retention & rejection under new OOD on PV test
new_retained_pv = new_ood_scores_pv <= new_ood_threshold
new_rejected_pv = ~new_retained_pv

print(f"\nNew OOD guard on PV test:")
print(f"  Retention: {new_retained_pv.sum()}/{len(new_retained_pv)} ({100*new_retained_pv.mean():.2f}%)")
print(f"  Rejection: {new_rejected_pv.sum()}/{len(new_rejected_pv)} ({100*new_rejected_pv.mean():.2f}%)")
per_class_retention = {}
for cls in dep_classes:
    mask = np.array(pv_test_labels)==cls
    r = new_retained_pv[mask]
    per_class_retention[cls] = (int(r.sum()), int(mask.sum()), 100*r.mean())
    print(f"  {cls}: {r.sum()}/{mask.sum()} retained ({100*r.mean():.1f}%)")

# Compute metrics
pv_labels_np = np.array([dep_cls2idx.get(l, -1) for l in pv_test_labels])
dep_pred_np  = np.array(dep_pred_idx)
new_pred_pv_np = np.array([new_cls2idx.get(c,-1) for c in new_pred_cls])

dep_acc  = np.mean(dep_pred_np == pv_labels_np)
new_acc  = np.mean(new_pred_pv_np == pv_labels_np)

print(f"\nDeployed model accuracy on PV test: {dep_acc*100:.2f}%")
print(f"New model accuracy on PV test:      {new_acc*100:.2f}%")
print(f"Accuracy drop:                      {(dep_acc - new_acc)*100:+.2f} pp")

# Bootstrap CIs
def bs_acc(yt, yp): return accuracy_fn(yt, yp)
dep_acc_ci   = bootstrap_ci(bs_acc, pv_labels_np, dep_pred_np)
new_acc_ci   = bootstrap_ci(bs_acc, pv_labels_np, new_pred_pv_np)

pv_class_list = sorted(set(dep_cls2idx.keys()))
dep_f1_ci = bootstrap_ci(lambda yt,yp: macro_f1_fn(yt,yp,list(range(len(dep_classes)))), pv_labels_np, dep_pred_np)
new_f1_ci = bootstrap_ci(lambda yt,yp: macro_f1_fn(yt,yp,list(range(len(new_classes)))), pv_labels_np, new_pred_pv_np)

print(f"\nDeployed: acc={dep_acc_ci[0]*100:.2f}% [{dep_acc_ci[1]*100:.2f},{dep_acc_ci[2]*100:.2f}]  F1={dep_f1_ci[0]:.4f} [{dep_f1_ci[1]:.4f},{dep_f1_ci[2]:.4f}]")
print(f"New:      acc={new_acc_ci[0]*100:.2f}% [{new_acc_ci[1]*100:.2f},{new_acc_ci[2]*100:.2f}]  F1={new_f1_ci[0]:.4f} [{new_f1_ci[1]:.4f},{new_f1_ci[2]:.4f}]")

print(f"\nPer-class recall (PV test):")
pv_recall_table = []
for i,cls in enumerate(dep_classes):
    mask = pv_labels_np==i
    N_c  = mask.sum()
    dep_r= np.mean(dep_pred_np[mask]==i) if N_c>0 else 0
    new_r= np.mean(new_pred_pv_np[mask]==i) if N_c>0 else 0
    # CI for new
    new_r_ci = bootstrap_ci(per_class_recall, pv_labels_np, new_pred_pv_np, c=i)
    dep_r_ci = bootstrap_ci(per_class_recall, pv_labels_np, dep_pred_np, c=i)
    pv_recall_table.append((cls, N_c, dep_r, dep_r_ci, new_r, new_r_ci))
    print(f"  {cls:25s} N={N_c:4d}  dep={dep_r*100:5.1f}% [{dep_r_ci[1]*100:.1f},{dep_r_ci[2]*100:.1f}]  new={new_r*100:5.1f}% [{new_r_ci[1]*100:.1f},{new_r_ci[2]*100:.1f}]  delta={new_r*100-dep_r*100:+.1f}pp")

# Confusion matrices
print(f"\nDeployed model CONFUSION MATRIX (PV test):")
print(confusion_matrix_str(pv_labels_np, dep_pred_np, list(range(len(dep_classes)))))
print("  Row labels: " + ", ".join([f"{i}={c}" for i,c in enumerate(dep_classes)]))

print(f"\nNew model CONFUSION MATRIX (PV test, raw):")
print(confusion_matrix_str(pv_labels_np, new_pred_pv_np, list(range(len(new_classes)))))
print("  Row/col labels: " + ", ".join([f"{i}={c}" for i,c in enumerate(new_classes)]))

# New model guarded on PV test (OOD + abstain)
new_guard_pred_pv = new_pred_pv_np.copy()
new_guard_status_pv = []  # "diagnosed", "abstained", "rejected"
for i in range(len(pv_test_paths)):
    if new_rejected_pv[i]:
        new_guard_status_pv.append("rejected")
    elif new_conf[i] < ABSTAIN_TAU:
        new_guard_status_pv.append("abstained")
    else:
        new_guard_status_pv.append("diagnosed")

diagnosed_mask_pv = np.array(new_guard_status_pv) == "diagnosed"
abstained_mask_pv = np.array(new_guard_status_pv) == "abstained"
rejected_mask_pv  = np.array(new_guard_status_pv) == "rejected"
print(f"\nNew model GUARDED on PV test:")
print(f"  Rejected:  {rejected_mask_pv.sum()} ({100*rejected_mask_pv.mean():.2f}%)")
print(f"  Abstained: {abstained_mask_pv.sum()} ({100*abstained_mask_pv.mean():.2f}%)")
print(f"  Diagnosed: {diagnosed_mask_pv.sum()} ({100*diagnosed_mask_pv.mean():.2f}%)")
if diagnosed_mask_pv.sum() > 0:
    ga = np.mean(new_pred_pv_np[diagnosed_mask_pv] == pv_labels_np[diagnosed_mask_pv])
    print(f"  Accuracy on diagnosed: {ga*100:.2f}%")
    print(f"  Accuracy on ALL: {np.mean(new_pred_pv_np[diagnosed_mask_pv]==pv_labels_np[diagnosed_mask_pv])*diagnosed_mask_pv.sum()/len(pv_test_paths)*100:.2f}% (partial)")
ga_all_pv = np.sum(new_pred_pv_np[diagnosed_mask_pv] == pv_labels_np[diagnosed_mask_pv]) / len(pv_test_paths)
print(f"  Guarded accuracy (ALL, rejections=miss): {ga_all_pv*100:.2f}%")

# ── demo images ───────────────────────────────────────────────────────────────
print(f"\n--- Demo image OOD scores and confidence ---")
apple_samples = sorted([os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR)
                        if "apple" in f.lower() and f.lower().endswith((".jpg",".jpeg",".png"))])
print(f"Found {len(apple_samples)} apple sample(s): {[os.path.basename(p) for p in apple_samples]}")
demo_rows = []
for p in apple_samples:
    bn = os.path.basename(p)
    # deployed backbone + OOD
    dep_f = extract_feat_single(dep_backbone, p)
    dep_score = ood_score_single(dep_f, dep_ood_means, dep_ood_precision)
    dep_is_ood = dep_score > dep_ood_threshold
    # new backbone + OOD
    new_f = extract_feat_single(new_backbone, p)
    new_score = ood_score_single(new_f, new_ood_means, new_ood_precision)
    new_is_ood = new_score > new_ood_threshold
    # classifier confidence
    with torch.no_grad():
        t = TF(Image.open(p).convert("RGB")).unsqueeze(0)
        dep_logits = dep_model(t)
        dep_prob = torch.softmax(dep_logits,1)[0].numpy()
        new_logits = new_model_full(t)
        new_prob = torch.softmax(new_logits,1)[0].numpy()
    dep_pred_s = dep_classes[np.argmax(dep_prob)]
    new_pred_s = new_classes[np.argmax(new_prob)]
    dep_conf_s = float(dep_prob.max())
    new_conf_s = float(new_prob.max())
    print(f"\n  {bn}:")
    print(f"    Deployed OOD: score={dep_score:.4f}, threshold={dep_ood_threshold:.4f}, OOD={dep_is_ood}")
    print(f"    New OOD:      score={new_score:.4f}, threshold={new_ood_threshold:.4f}, OOD={new_is_ood}")
    print(f"    Deployed pred: {dep_pred_s} ({dep_conf_s*100:.1f}%)")
    print(f"    New pred:      {new_pred_s} ({new_conf_s*100:.1f}%)")
    demo_rows.append(dict(file=bn,
        dep_ood_score=dep_score, dep_ood_thr=dep_ood_threshold, dep_is_ood=dep_is_ood,
        new_ood_score=new_score, new_ood_thr=new_ood_threshold, new_is_ood=new_is_ood,
        dep_pred=dep_pred_s, dep_conf=dep_conf_s, new_pred=new_pred_s, new_conf=new_conf_s))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FGVC test – CIs, reconcile confusion, epoch info
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 2: FGVC Test — Bootstrap CIs and Reconciliation")
print("="*70)

fgvc_test_paths  = fgvc_test["image_path"].tolist()
fgvc_test_labels = fgvc_test["cls_mapped"].tolist()

# New model on fgvc_test
print(f"\nRunning new model on fgvc_test ({len(fgvc_test_paths)} images)...")
t0 = time.time()
fgvc_pred_idx, fgvc_conf, fgvc_probs = predict_batch(new_model_full, fgvc_test_paths)
fgvc_pred_cls = [new_classes[i] for i in fgvc_pred_idx]
print(f"  Done in {time.time()-t0:.1f}s")

# OOD scores for fgvc_test with new guard
print("Extracting new backbone features for fgvc_test...")
t0 = time.time()
new_feats_fgvc = extract_features_classifier(new_backbone, fgvc_test_paths)
new_ood_scores_fgvc = compute_mahalanobis_scores(new_feats_fgvc, new_ood_means, new_ood_precision)
print(f"  Done in {time.time()-t0:.1f}s")

fgvc_retained = new_ood_scores_fgvc <= new_ood_threshold
fgvc_rejected  = ~fgvc_retained

fgvc_guard_status = []
for i in range(len(fgvc_test_paths)):
    if fgvc_rejected[i]:
        fgvc_guard_status.append("rejected")
    elif fgvc_conf[i] < ABSTAIN_TAU:
        fgvc_guard_status.append("abstained")
    else:
        fgvc_guard_status.append("diagnosed")

fgvc_guard_status_np = np.array(fgvc_guard_status)
fgvc_diag  = fgvc_guard_status_np == "diagnosed"
fgvc_abst  = fgvc_guard_status_np == "abstained"
fgvc_rej   = fgvc_guard_status_np == "rejected"

# Integer labels
fgvc_cls2idx = {c:i for i,c in enumerate(new_classes)}
fgvc_labels_np = np.array([fgvc_cls2idx.get(l,-1) for l in fgvc_test_labels])
fgvc_pred_np   = np.array(fgvc_pred_idx)

# Raw metrics
fgvc_raw_acc  = np.mean(fgvc_pred_np == fgvc_labels_np)
fgvc_raw_f1   = macro_f1_fn(fgvc_labels_np, fgvc_pred_np, list(range(len(new_classes))))

print(f"\nRaw accuracy on fgvc_test: {fgvc_raw_acc*100:.2f}%")
print(f"Raw macro-F1:              {fgvc_raw_f1:.4f}")

# Bootstrap CIs
fgvc_acc_ci = bootstrap_ci(bs_acc, fgvc_labels_np, fgvc_pred_np)
fgvc_f1_ci  = bootstrap_ci(lambda yt,yp: macro_f1_fn(yt,yp,list(range(len(new_classes)))), fgvc_labels_np, fgvc_pred_np)
print(f"  Raw acc CI (95%): [{fgvc_acc_ci[1]*100:.2f}%, {fgvc_acc_ci[2]*100:.2f}%]")
print(f"  Raw F1 CI (95%):  [{fgvc_f1_ci[1]:.4f}, {fgvc_f1_ci[2]:.4f}]")

print(f"\nPer-class recall on fgvc_test (raw):")
fgvc_recall_table = []
for i,cls in enumerate(new_classes):
    mask = fgvc_labels_np==i
    N_c  = mask.sum()
    if N_c == 0:
        continue
    r  = np.mean(fgvc_pred_np[mask]==i)
    ci = bootstrap_ci(per_class_recall, fgvc_labels_np, fgvc_pred_np, c=i)
    fgvc_recall_table.append((cls, N_c, r, ci))
    print(f"  {cls:25s} N={N_c:4d}  recall={r*100:.2f}% [{ci[1]*100:.2f}%, {ci[2]*100:.2f}%]")

# Guarded confusion matrix reconciliation
n_diagnosed = int(fgvc_diag.sum())
n_correct_diag = int(np.sum(fgvc_pred_np[fgvc_diag] == fgvc_labels_np[fgvc_diag]))
n_total = len(fgvc_test_paths)

print(f"\nGuarded pipeline summary on fgvc_test:")
print(f"  Rejected:  {fgvc_rej.sum()} ({100*fgvc_rej.mean():.2f}%)")
print(f"  Abstained: {fgvc_abst.sum()} ({100*fgvc_abst.mean():.2f}%)")
print(f"  Diagnosed: {n_diagnosed} ({100*n_diagnosed/n_total:.2f}%)")
print(f"  Correct among diagnosed: {n_correct_diag} ({100*n_correct_diag/n_diagnosed:.2f}% of diagnosed)")
print(f"  Guarded accuracy (ALL): {n_correct_diag/n_total*100:.2f}%")
print(f"\nRECONCILIATION: 97.19% of {n_diagnosed} diagnosed = {0.9719*n_diagnosed:.1f} correct")
print(f"  Actual: {n_correct_diag}/{n_diagnosed} = {100*n_correct_diag/n_diagnosed:.2f}%")
print(f"  Reported 'guarded accuracy (ALL)' in Stage 3 = 89.88% = {int(round(0.8988*346))}/346 correct")
print(f"  Actual: {n_correct_diag}/{n_total} = {100*n_correct_diag/n_total:.2f}%")

print(f"\nEpoch saved: {best_epoch}")
print(f"Selection metric: {selection_metric}")

# Guarded CM with abstained/rejected columns
print(f"\nNew model GUARDED CONFUSION MATRIX (fgvc_test):")
diag_pred = fgvc_pred_np[fgvc_diag]
diag_true = fgvc_labels_np[fgvc_diag]
abst_true  = fgvc_labels_np[fgvc_abst]
rej_true   = fgvc_labels_np[fgvc_rej]
labels_fgvc_active = [i for i,cls in enumerate(new_classes) if cls != "black_rot" or (fgvc_labels_np==i).sum()>0]
print("| True \\ Pred | " + " | ".join(new_classes) + " | abstained | rejected |")
print("|" + "---|"*(len(new_classes)+3))
for i,cls in enumerate(new_classes):
    row_mask_all = fgvc_labels_np == i
    if row_mask_all.sum() == 0 and cls == "black_rot":
        continue
    row = [str(np.sum((fgvc_labels_np==i)&fgvc_diag&(fgvc_pred_np==j))) for j in range(len(new_classes))]
    abst_n = int(np.sum(fgvc_abst & (fgvc_labels_np==i)))
    rej_n  = int(np.sum(fgvc_rej  & (fgvc_labels_np==i)))
    print(f"| {cls} | " + " | ".join(row) + f" | {abst_n} | {rej_n} |")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Cross-crop guard re-run with new apple OOD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 3: Cross-Crop Guard Re-Evaluation (new apple OOD in-memory)")
print("="*70)

# Build in-memory ood_params with new apple OOD swapped in
params_new_apple = {}
for c in CROPS:
    if c == "apple":
        params_new_apple[c] = {
            "means":     torch.tensor(new_ood_means),
            "precision": torch.tensor(new_ood_precision),
            "threshold": new_ood_threshold,
        }
    else:
        params_new_apple[c] = ood_params_all[c]

# Build feature extractor models dict (new backbone for apple, deployed for others)
feat_models = {}
for c in CROPS:
    if c == "apple":
        feat_models[c] = new_backbone
    else:
        m, _ = get_feature_extractor(c)
        feat_models[c] = m

def compute_all_ratios(image_paths, batch_size=64):
    """Returns dict crop -> ratio array using new apple OOD in-memory."""
    N = len(image_paths)
    ratios = {c: np.zeros(N, dtype=np.float32) for c in CROPS}
    means_d  = {c: params_new_apple[c]["means"].numpy() for c in CROPS}
    precs_d  = {c: params_new_apple[c]["precision"].numpy() for c in CROPS}
    thresh_d = {c: float(params_new_apple[c]["threshold"]) for c in CROPS}
    for i in range(0, N, batch_size):
        batch = image_paths[i:i+batch_size]
        tensors = []
        for p in batch:
            with Image.open(p) as im:
                tensors.append(TF(im.convert("RGB")))
        bt = torch.stack(tensors)
        for c in CROPS:
            with torch.no_grad():
                feats = feat_models[c](bt).cpu().numpy()
            scores = compute_mahalanobis_scores(feats, means_d[c], precs_d[c])
            ratios[c][i:i+len(batch)] = scores / thresh_d[c]
    return ratios

MARGIN = 0.25

def guard_decision(ratios, selected):
    sel_r = ratios[selected]
    others = [c for c in CROPS if c != selected]
    other_r = [ratios[o] for o in others]
    min_other = np.minimum(other_r[0], other_r[1])
    all_above1 = (sel_r>1.0)&(other_r[0]>1.0)&(other_r[1]>1.0)
    old_ood = sel_r > 1.0
    cond1 = (sel_r>1.0) & (min_other<1.0)
    cond2 = (sel_r - min_other) >= MARGIN
    cc = (cond1|cond2) & (~all_above1)
    total = old_ood | cc
    return old_ood, cc, total

# Load paddy/arecanut/apple test splits
split_dfs = {}
for c in CROPS:
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", f"{c}_splits.csv"))
    split_dfs[c] = df[df["split"]=="test"].copy().reset_index(drop=True)
    print(f"  Loaded {c} test: {len(split_dfs[c])} images")

# Cache file for new-guard ratios (separate from old one)
cache_new = os.path.join(OUT_DIR, "cross_crop_eval_new_apple.npz")

# Sample 200 from each crop test
sample_200 = {}
for c in CROPS:
    paths = split_dfs[c]["image_path"].tolist()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(paths), size=min(200, len(paths)), replace=False)
    sample_200[c] = [paths[i] for i in idx]

# FGVC test images (200) for apple mode + other modes
fgvc_test_paths_all = fgvc_test["image_path"].tolist()
rng2 = np.random.default_rng(42)
fgvc_200_idx = rng2.choice(len(fgvc_test_paths_all), size=200, replace=False)
fgvc_200 = [fgvc_test_paths_all[i] for i in fgvc_200_idx]

# Non-crop disk images
non_crop_cands = []
for pat in ["/usr/share/icons/hicolor/128x128/apps/*.png",
            "/usr/share/icons/hicolor/256x256/apps/*.png",
            "/usr/share/pixmaps/*.png",
            "/usr/share/wallpapers/**/*.jpg",
            "/usr/share/backgrounds/**/*.jpg"]:
    for p in sorted(glob.glob(pat, recursive=True)):
        if os.path.isfile(p):
            try:
                with Image.open(p) as im:
                    if im.size[0]>=64 and im.size[1]>=64:
                        non_crop_cands.append(p)
            except: pass
rng3 = np.random.default_rng(42)
if len(non_crop_cands) >= 60:
    non_crop_60 = [non_crop_cands[i] for i in rng3.choice(len(non_crop_cands), 60, replace=False)]
else:
    non_crop_60 = non_crop_cands
    print(f"WARNING: only {len(non_crop_60)} non-crop images found")
print(f"Non-crop images available: {len(non_crop_60)}")

if os.path.exists(cache_new):
    print(f"\nLoading cached ratios from {cache_new}...")
    cd = np.load(cache_new)
    full_ratios   = {c: {cc: cd[f"full_{c}_{cc}"] for cc in CROPS} for c in CROPS}
    s200_ratios   = {c: {cc: cd[f"s200_{c}_{cc}"] for cc in CROPS} for c in CROPS}
    nc_ratios     = {cc: cd[f"nc_{cc}"] for cc in CROPS}
    fgvc_r_all    = {cc: cd[f"fgvc_{cc}"] for cc in CROPS}
    fgvc200_r_all = {cc: cd[f"fgvc200_{cc}"] for cc in CROPS}
else:
    print("\nExtracting ratios for Section 3 (this takes ~20 min)...")
    full_ratios = {}
    for c in CROPS:
        paths = split_dfs[c]["image_path"].tolist()
        print(f"  Full test {c} ({len(paths)} imgs)...")
        t0 = time.time()
        r = compute_all_ratios(paths)
        full_ratios[c] = r
        print(f"    Done {time.time()-t0:.1f}s")

    s200_ratios = {}
    for c in CROPS:
        print(f"  Sample200 {c}...")
        t0 = time.time()
        r = compute_all_ratios(sample_200[c])
        s200_ratios[c] = r
        print(f"    Done {time.time()-t0:.1f}s")

    print("  Non-crop 60 images...")
    t0 = time.time()
    nc_ratios = compute_all_ratios(non_crop_60)
    print(f"    Done {time.time()-t0:.1f}s")

    print(f"  FGVC test full ({len(fgvc_test_paths_all)} imgs)...")
    t0 = time.time()
    fgvc_r_all = compute_all_ratios(fgvc_test_paths_all)
    print(f"    Done {time.time()-t0:.1f}s")

    print(f"  FGVC 200 sample...")
    t0 = time.time()
    fgvc200_r_all = compute_all_ratios(fgvc_200)
    print(f"    Done {time.time()-t0:.1f}s")

    save_dict = {}
    for c in CROPS:
        for cc in CROPS:
            save_dict[f"full_{c}_{cc}"] = full_ratios[c][cc]
            save_dict[f"s200_{c}_{cc}"] = s200_ratios[c][cc]
    for cc in CROPS:
        save_dict[f"nc_{cc}"] = nc_ratios[cc]
        save_dict[f"fgvc_{cc}"] = fgvc_r_all[cc]
        save_dict[f"fgvc200_{cc}"] = fgvc200_r_all[cc]
    np.savez_compressed(cache_new, **save_dict)
    print(f"Saved cache to {cache_new}")

# --- 3A. Correct-crop false block rate ---
print("\n--- 3A. Correct-Crop False Block Rate (PV test sets + fgvc_test) ---")
fb_results = {}
for c in CROPS:
    old_b, cc_b, tot_b = guard_decision(full_ratios[c], c)
    N = len(tot_b)
    fb_results[c] = dict(N=N, old_b=old_b.sum(), cc_b=cc_b.sum(), tot_b=tot_b.sum(),
                         old_r=old_b.mean(), cc_r=cc_b.mean(), tot_r=tot_b.mean())
    print(f"  {c.capitalize()} PV test (N={N}):  OOD-only={old_b.sum()}/{N}({old_b.mean()*100:.4f}%)  CC={cc_b.sum()}/{N}({cc_b.mean()*100:.4f}%)  TOTAL={tot_b.sum()}/{N}({tot_b.mean()*100:.4f}%)")

# Apple fgvc_test false block rate (apple mode)
old_b_fgvc, cc_b_fgvc, tot_b_fgvc = guard_decision(fgvc_r_all, "apple")
N_fgvc = len(fgvc_test_paths_all)
print(f"\n  Apple FGVC test (N={N_fgvc}): OOD-only={old_b_fgvc.sum()}/{N_fgvc}({old_b_fgvc.mean()*100:.4f}%)  CC={cc_b_fgvc.sum()}/{N_fgvc}({cc_b_fgvc.mean()*100:.4f}%)  TOTAL={tot_b_fgvc.sum()}/{N_fgvc}({tot_b_fgvc.mean()*100:.4f}%)")
apple_fgvc_fb_pass = tot_b_fgvc.mean()*100 <= 5.0

# --- 3B. 6 wrong-crop combos ---
print("\n--- 3B. Wrong-Crop Combos (200 images each) ---")
wrong_results = {}
for true_c in CROPS:
    for tested_c in CROPS:
        if true_c == tested_c: continue
        old_b, cc_b, tot_b = guard_decision(s200_ratios[true_c], tested_c)
        N = len(tot_b)
        wrong_results[(true_c,tested_c)] = (tot_b.sum(), N, tot_b.mean(), old_b.mean(), cc_b.mean())
        print(f"  True {true_c.capitalize()} in {tested_c.capitalize()} mode: OOD={old_b.sum()/N*100:.4f}%  CC={cc_b.sum()/N*100:.4f}%  TOTAL={tot_b.sum()}/{N} ({tot_b.mean()*100:.4f}%)")

# --- 3C. Non-crop in all 3 modes ---
print("\n--- 3C. Non-Crop Disk Images (60) in all 3 modes ---")
nc_results = {}
for mode in CROPS:
    old_b, cc_b, tot_b = guard_decision(nc_ratios, mode)
    N = len(tot_b)
    nc_results[mode] = (tot_b.sum(), N, tot_b.mean(), old_b.mean(), cc_b.mean())
    print(f"  Non-crop in {mode.capitalize()} mode: OOD={old_b.sum()/N*100:.4f}%  CC={cc_b.sum()/N*100:.4f}%  TOTAL={tot_b.sum()}/{N} ({tot_b.mean()*100:.4f}%)")

# --- 3D. FGVC orchard (200) in paddy and arecanut mode ---
print("\n--- 3D. Apple Orchard (FGVC 200) in Paddy and Arecanut Mode ---")
fgvc_wrong = {}
for mode in ["paddy", "arecanut"]:
    old_b, cc_b, tot_b = guard_decision(fgvc200_r_all, mode)
    N = len(tot_b)
    fgvc_wrong[mode] = (tot_b.sum(), N, tot_b.mean(), old_b.mean(), cc_b.mean())
    print(f"  FGVC Apple 200 in {mode.capitalize()} mode: OOD={old_b.sum()/N*100:.4f}%  CC={cc_b.sum()/N*100:.4f}%  TOTAL={tot_b.sum()}/{N} ({tot_b.mean()*100:.4f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Abstain calibration
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 4: Abstain Calibration (fgvc_val + PV val)")
print("="*70)

# Run new model on fgvc_val and PV val
fgvc_val_paths  = fgvc_val["image_path"].tolist()
fgvc_val_labels = fgvc_val["cls_mapped"].tolist()
pv_val_paths    = pv_val["image_path"].tolist()
pv_val_labels   = pv_val["disease_class"].tolist()

print(f"Running new model on fgvc_val ({len(fgvc_val_paths)})...")
t0 = time.time()
fv_pred_idx, fv_conf, fv_probs = predict_batch(new_model_full, fgvc_val_paths)
fv_pred_cls = [new_classes[i] for i in fv_pred_idx]
# OOD on fgvc_val
fv_feats = extract_features_classifier(new_backbone, fgvc_val_paths)
fv_ood   = compute_mahalanobis_scores(fv_feats, new_ood_means, new_ood_precision)
fv_ood_pass = fv_ood <= new_ood_threshold
print(f"  Done in {time.time()-t0:.1f}s")

print(f"Running new model on PV val ({len(pv_val_paths)})...")
t0 = time.time()
pv_v_pred_idx, pv_v_conf, pv_v_probs = predict_batch(new_model_full, pv_val_paths)
pv_v_feats = extract_features_classifier(new_backbone, pv_val_paths)
pv_v_ood   = compute_mahalanobis_scores(pv_v_feats, new_ood_means, new_ood_precision)
pv_v_ood_pass = pv_v_ood <= new_ood_threshold
print(f"  Done in {time.time()-t0:.1f}s")

# Combine val sets
fv_labels_np  = np.array([fgvc_cls2idx.get(l,-1) for l in fgvc_val_labels])
pv_v_labels_np= np.array([new_cls2idx.get(l,-1) for l in pv_val_labels])
# Only keep classes in new model
combined_conf   = np.concatenate([fv_conf,   pv_v_conf])
combined_pred   = np.concatenate([np.array(fv_pred_idx), np.array(pv_v_pred_idx)])
combined_labels = np.concatenate([fv_labels_np, pv_v_labels_np])
combined_ood_pass= np.concatenate([fv_ood_pass, pv_v_ood_pass])

# Coverage vs accuracy for thresholds 0.4 to 0.9
print(f"\nAbstain calibration table (combined fgvc_val + PV val, N={len(combined_conf)}):")
print(f"  (Only OOD-passed images counted for coverage/accuracy)")
ood_passed_mask = combined_ood_pass & (combined_labels >= 0)
N_ood_pass = ood_passed_mask.sum()
print(f"  OOD-passed: {N_ood_pass}/{len(combined_conf)}")
print(f"\n  {'Threshold':>10} {'Coverage':>12} {'Accuracy':>12} {'N_diagnosed':>12} {'N_abstained':>12}")
abstain_table = []
for tau in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    mask_diag = ood_passed_mask & (combined_conf >= tau)
    mask_abst = ood_passed_mask & (combined_conf < tau)
    N_diag = mask_diag.sum()
    N_abst = mask_abst.sum()
    cov  = N_diag / N_ood_pass if N_ood_pass > 0 else 0
    acc  = np.mean(combined_pred[mask_diag] == combined_labels[mask_diag]) if N_diag>0 else 0
    print(f"  {tau:10.2f} {cov*100:11.1f}% {acc*100:11.1f}% {N_diag:12d} {N_abst:12d}")
    abstain_table.append(dict(tau=tau,coverage=cov,accuracy=acc,N_diag=N_diag,N_abst=N_abst))

# At tau=0.70 on fgvc_test and PV test
print(f"\nAt tau=0.70 on fgvc_test:")
fgvc_t_diag = fgvc_diag  # already computed above
fgvc_t_corr = np.sum(fgvc_pred_np[fgvc_t_diag] == fgvc_labels_np[fgvc_t_diag])
print(f"  Diagnosed: {fgvc_t_diag.sum()}/{len(fgvc_test_paths_all)}  Accuracy on diagnosed: {100*fgvc_t_corr/fgvc_t_diag.sum():.2f}%  Guarded all: {100*fgvc_t_corr/len(fgvc_test_paths_all):.2f}%")

print(f"At tau=0.70 on PV test:")
pv_diag_mask = diagnosed_mask_pv
pv_t_corr = np.sum(new_pred_pv_np[pv_diag_mask] == pv_labels_np[pv_diag_mask])
print(f"  Diagnosed: {pv_diag_mask.sum()}/{len(pv_test_paths)}  Accuracy on diagnosed: {100*pv_t_corr/pv_diag_mask.sum() if pv_diag_mask.sum()>0 else 0:.2f}%  Guarded all: {100*pv_t_corr/len(pv_test_paths):.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Distribution check (DINOv2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 5: Distribution Check (DINOv2 + pHash)")
print("="*70)

# Load precomputed fgvc features file
print("Loading fgvc_features.npz...")
fgvc_feat_data = np.load(FGVC_FEATURES, allow_pickle=True)
print("  Keys:", list(fgvc_feat_data.files))

# We need DINOv2 embeddings for fgvc_test, fgvc_train and PV train
# From stage 3b, fgvc_features.npz has: image_ids, dinov2_features, phashes, labels, split
fgvc_all_splits   = fgvc_feat_data.get("splits",   fgvc_feat_data.get("split", None))
fgvc_all_dino     = fgvc_feat_data.get("dinov2",   fgvc_feat_data.get("dino_features", None))
fgvc_all_phashes  = fgvc_feat_data.get("phashes",  fgvc_feat_data.get("phash", None))
fgvc_all_ids      = fgvc_feat_data.get("image_ids",fgvc_feat_data.get("ids", None))

# Try different key names
for key in fgvc_feat_data.files:
    arr = fgvc_feat_data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# Load PV train features
print("Loading plantvillage_train_features.npz...")
pv_feat_data = np.load(PV_FEATURES, allow_pickle=True)
for key in pv_feat_data.files:
    arr = pv_feat_data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# Extract test/train indices from fgvc_features.npz
# We'll try to match with fgvc_splits.csv by image_id or path
fgvc_splits_df = pd.read_csv(FGVC_SPLITS)
fgvc_test_ids  = set(fgvc_splits_df[fgvc_splits_df["split"]=="fgvc_test"]["image_id"].tolist())
fgvc_train_ids = set(fgvc_splits_df[fgvc_splits_df["split"]=="fgvc_train"]["image_id"].tolist())

# Build index maps
if fgvc_all_ids is not None:
    ids_list = [str(x) for x in fgvc_all_ids]
    test_idxs  = [i for i,x in enumerate(ids_list) if x in fgvc_test_ids]
    train_idxs = [i for i,x in enumerate(ids_list) if x in fgvc_train_ids]
    print(f"  fgvc test idx count:  {len(test_idxs)}")
    print(f"  fgvc train idx count: {len(train_idxs)}")
else:
    # Fallback: use split labels if available
    if fgvc_all_splits is not None:
        test_idxs  = [i for i,s in enumerate(fgvc_all_splits) if str(s) in ["fgvc_test","test"]]
        train_idxs = [i for i,s in enumerate(fgvc_all_splits) if str(s) in ["fgvc_train","train"]]
        print(f"  fgvc test idx count:  {len(test_idxs)}")
        print(f"  fgvc train idx count: {len(train_idxs)}")
    else:
        test_idxs = train_idxs = None

# Determine DINOv2 feature key
dino_key = None
for k in fgvc_feat_data.files:
    if "dino" in k.lower() or "embed" in k.lower() or (fgvc_feat_data[k].ndim==2 and fgvc_feat_data[k].shape[1]==384):
        dino_key = k
        break

pv_dino_key = None
for k in pv_feat_data.files:
    if "dino" in k.lower() or "embed" in k.lower() or (pv_feat_data[k].ndim==2 and pv_feat_data[k].shape[1]==384):
        pv_dino_key = k
        break

if dino_key and pv_dino_key and test_idxs and train_idxs:
    fgvc_dino_all = fgvc_feat_data[dino_key].astype(np.float32)
    pv_dino       = pv_feat_data[pv_dino_key].astype(np.float32)

    # L2 normalize
    def l2norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    fgvc_dino_all = l2norm(fgvc_dino_all)
    pv_dino       = l2norm(pv_dino)

    test_dino  = fgvc_dino_all[test_idxs]
    train_dino = fgvc_dino_all[train_idxs]

    print(f"\nDINOv2 embedding shapes: test={test_dino.shape}, fgvc_train={train_dino.shape}, pv_train={pv_dino.shape}")

    # Nearest fgvc_train cosine for each fgvc_test image (batched)
    print("Computing nearest-fgvc_train cosines...")
    t0 = time.time()
    nn_fgvc_cos = np.zeros(len(test_dino), dtype=np.float32)
    bs = 50
    for i in range(0, len(test_dino), bs):
        sims = test_dino[i:i+bs] @ train_dino.T  # (bs, N_train)
        nn_fgvc_cos[i:i+bs] = sims.max(axis=1)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Nearest PV train cosine for each fgvc_test image
    print("Computing nearest-PV_train cosines...")
    t0 = time.time()
    nn_pv_cos = np.zeros(len(test_dino), dtype=np.float32)
    for i in range(0, len(test_dino), bs):
        sims = test_dino[i:i+bs] @ pv_dino.T
        nn_pv_cos[i:i+bs] = sims.max(axis=1)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Percentiles
    print(f"\nNearest fgvc_train DINOv2 cosine (fgvc_test, N={len(nn_fgvc_cos)}):")
    for p in [10, 50, 90, 99]:
        print(f"  p{p}: {np.percentile(nn_fgvc_cos, p):.4f}")

    print(f"\nNearest PV_train DINOv2 cosine (fgvc_test, N={len(nn_pv_cos)}):")
    for p in [10, 50, 90, 99]:
        print(f"  p{p}: {np.percentile(nn_pv_cos, p):.4f}")

    # pHash <= 16 to any training image
    phash_key = None
    for k in fgvc_feat_data.files:
        if "phash" in k.lower() or "hash" in k.lower():
            phash_key = k
            break
    pv_phash_key = None
    for k in pv_feat_data.files:
        if "phash" in k.lower() or "hash" in k.lower():
            pv_phash_key = k
            break

    if phash_key:
        fgvc_phashes_all = fgvc_feat_data[phash_key]
        test_phashes  = fgvc_phashes_all[test_idxs]
        train_phashes = fgvc_phashes_all[train_idxs]

        def hamming_dist(h1, h2):
            return int(bin(int(h1)^int(h2)).count('1'))

        n_close_phash = 0
        for ph_t in test_phashes:
            for ph_tr in train_phashes:
                if hamming_dist(ph_t, ph_tr) <= 16:
                    n_close_phash += 1
                    break
        if pv_phash_key:
            pv_phashes = pv_feat_data[pv_phash_key]
            for i, ph_t in enumerate(test_phashes):
                already = False
                for ph_tr in train_phashes:
                    if hamming_dist(ph_t, ph_tr) <= 16:
                        already = True; break
                if not already:
                    for ph_pv in pv_phashes:
                        if hamming_dist(ph_t, ph_pv) <= 16:
                            n_close_phash += 1; break
        print(f"\nfgvc_test images with pHash<=16 to ANY training image: {n_close_phash}/{len(test_phashes)}")
    else:
        print("\n(pHash data not found in fgvc_features.npz — skipping pHash check)")
        n_close_phash = "N/A"

    # Accuracy on subset with nearest-train cosine < 0.90
    LOW_COS = 0.90
    far_mask = nn_fgvc_cos < LOW_COS
    near_mask = ~far_mask

    # Align with fgvc_test (need labels for test_idxs)
    # fgvc_labels_np corresponds to fgvc_test ordered as fgvc_test df
    # test_idxs correspond to fgvc_feat_data rows
    # If lengths match
    if len(test_idxs) == len(fgvc_test_paths_all):
        far_labels  = fgvc_labels_np[far_mask]
        far_pred    = fgvc_pred_np[far_mask]
        near_labels = fgvc_labels_np[near_mask]
        near_pred   = fgvc_pred_np[near_mask]

        # Guarded predictions for far subset
        far_rej  = fgvc_rej[far_mask]
        far_abst = fgvc_abst[far_mask]
        far_diag = fgvc_diag[far_mask]

        raw_acc_far = np.mean(far_pred == far_labels) if len(far_labels)>0 else 0
        gua_acc_far = np.sum(far_pred[far_diag] == far_labels[far_diag])/len(far_labels) if len(far_labels)>0 else 0

        print(f"\nSubset: nearest fgvc_train DINOv2 cosine < {LOW_COS} (N={far_mask.sum()}):")
        print(f"  Raw accuracy: {raw_acc_far*100:.2f}%")
        print(f"  Guarded accuracy (ALL): {gua_acc_far*100:.2f}%")
        print(f"  Rejected: {far_rej.sum()}/{far_mask.sum()}")
        for i,cls in enumerate(new_classes):
            m2 = far_labels==i
            if m2.sum()==0: continue
            r = np.mean(far_pred[m2]==i)
            print(f"    {cls}: N={m2.sum()}  raw_recall={r*100:.1f}%")
    else:
        print(f"WARNING: test_idxs length {len(test_idxs)} != fgvc_test length {len(fgvc_test_paths_all)}, skipping subset analysis")
        far_mask = None; n_close_phash = "N/A"
        raw_acc_far = gua_acc_far = None

else:
    print("DINOv2 embeddings not available in saved NPZ files — skipping Section 5 quantitative analysis")
    nn_fgvc_cos = nn_pv_cos = None
    n_close_phash = "N/A"
    raw_acc_far = gua_acc_far = None
    far_mask = None

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Write predeploy_report.md
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 6: Writing predeploy_report.md")
print("="*70)

# Compute deploy condition checks
pv_acc_drop_pp = (dep_acc - new_acc)*100
pv_retention_pct = 100*new_retained_pv.mean()
demo_pass = all(not r["new_is_ood"] for r in demo_rows) if demo_rows else False
nc_apple_block_pct = nc_results["apple"][2]*100
fb_apple_pv_pct    = fb_results["apple"]["tot_r"]*100
fb_apple_fgvc_pct  = tot_b_fgvc.mean()*100
# Max cross-crop false block across all crops and modes (correct crop only)
max_cc_fb = max([
    fb_results["paddy"]["tot_r"]*100,
    fb_results["arecanut"]["tot_r"]*100,
    fb_apple_pv_pct,
    fb_apple_fgvc_pct,
])

COND_CHECKS = [
    ("PV accuracy drop ≤ 1 pp",    pv_acc_drop_pp <= 1.0,      f"{pv_acc_drop_pp:+.2f} pp"),
    ("PV retention ≥ 93%",          pv_retention_pct >= 93.0,   f"{pv_retention_pct:.2f}%"),
    ("Demo apple images pass guard", demo_pass,                  f"{'all pass' if demo_pass else 'FAIL'}"),
    ("Non-crop rejection in Apple mode ≥ 95%", nc_apple_block_pct >= 95.0, f"{nc_apple_block_pct:.2f}%"),
    ("Apple false block ≤ 5% (PV test)",  fb_apple_pv_pct <= 5.0,   f"{fb_apple_pv_pct:.4f}%"),
    ("Apple false block ≤ 5% (FGVC test)",fb_apple_fgvc_pct <= 5.0, f"{fb_apple_fgvc_pct:.4f}%"),
    ("No cross-crop false block > 5% for any crop", max_cc_fb <= 5.0, f"max={max_cc_fb:.4f}%"),
]

md_lines = [
    "# Pre-Deployment Report: apple_disease_fgvc_v1.pth",
    "",
    f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
    f"> New model: `outputs/apple_fgvc/apple_disease_fgvc_v1.pth`  ",
    f"> New OOD:   `outputs/apple_fgvc/ood_params_apple_fgvc.pt` (threshold={new_ood_threshold:.4f})  ",
    f"> Deployed model: `outputs/models/apple_disease_classifier.pth` (OOD threshold={dep_ood_threshold:.4f})  ",
    f"> **ood_params.pt was NOT overwritten; all OOD comparisons use in-memory swap.**",
    "",
    "---",
    "",
    "## 1. PlantVillage Apple Test Split",
    "",
    f"### 1a. Accuracy and Macro-F1",
    "",
    "| Model | Accuracy | 95% CI | Macro-F1 | 95% CI |",
    "|---|---|---|---|---|",
    f"| Deployed | {dep_acc*100:.2f}% | [{dep_acc_ci[1]*100:.2f}%, {dep_acc_ci[2]*100:.2f}%] | {dep_f1_ci[0]:.4f} | [{dep_f1_ci[1]:.4f}, {dep_f1_ci[2]:.4f}] |",
    f"| New (fgvc_v1) | {new_acc*100:.2f}% | [{new_acc_ci[1]*100:.2f}%, {new_acc_ci[2]*100:.2f}%] | {new_f1_ci[0]:.4f} | [{new_f1_ci[1]:.4f}, {new_f1_ci[2]:.4f}] |",
    f"| **Δ** | **{pv_acc_drop_pp:+.2f} pp** | — | **{new_f1_ci[0]-dep_f1_ci[0]:+.4f}** | — |",
    "",
    "### 1b. Per-Class Recall (PV test)",
    "",
    "| Class | N | Deployed Recall | 95% CI | New Recall | 95% CI | Δ |",
    "|---|---|---|---|---|---|---|",
]
for cls, N_c, dep_r, dep_r_ci, new_r, new_r_ci in pv_recall_table:
    md_lines.append(f"| {cls} | {N_c} | {dep_r*100:.1f}% | [{dep_r_ci[1]*100:.1f}%, {dep_r_ci[2]*100:.1f}%] | {new_r*100:.1f}% | [{new_r_ci[1]*100:.1f}%, {new_r_ci[2]*100:.1f}%] | {new_r*100-dep_r*100:+.1f} pp |")

md_lines += [
    "",
    "### 1c. New Model OOD Retention on PV Test",
    "",
    f"| Metric | Value |",
    f"|---|---|",
    f"| Threshold (new OOD) | {new_ood_threshold:.4f} |",
    f"| Retention | {new_retained_pv.sum()}/{len(new_retained_pv)} ({pv_retention_pct:.2f}%) |",
    f"| Rejection | {new_rejected_pv.sum()}/{len(new_rejected_pv)} ({100*new_rejected_pv.mean():.2f}%) |",
]
for cls, (kept, tot, pct) in per_class_retention.items():
    md_lines.append(f"| — {cls} retained | {kept}/{tot} ({pct:.1f}%) |")

md_lines += [
    "",
    "### 1d. Deployed Model Confusion Matrix (PV test)",
    "",
    "| True \\ Pred | " + " | ".join(dep_classes) + " |",
    "|" + "---|"*(len(dep_classes)+1),
]
for i, cls in enumerate(dep_classes):
    row = [str(int(np.sum((pv_labels_np==i)&(dep_pred_np==j)))) for j in range(len(dep_classes))]
    md_lines.append(f"| {cls} | " + " | ".join(row) + " |")

md_lines += [
    "",
    "### 1e. New Model Confusion Matrix (PV test, raw)",
    "",
    "| True \\ Pred | " + " | ".join(new_classes) + " |",
    "|" + "---|"*(len(new_classes)+1),
]
for i, cls in enumerate(new_classes):
    if (pv_labels_np==i).sum() == 0 and cls == "black_rot":
        continue
    row = [str(int(np.sum((pv_labels_np==i)&(new_pred_pv_np==j)))) for j in range(len(new_classes))]
    md_lines.append(f"| {cls} | " + " | ".join(row) + " |")

md_lines += [
    "",
    "### 1f. Guarded Pipeline on PV Test (new model + new OOD + tau=0.70)",
    "",
    f"| Status | N | % |",
    f"|---|---|---|",
    f"| Rejected (OOD) | {rejected_mask_pv.sum()} | {100*rejected_mask_pv.mean():.2f}% |",
    f"| Abstained (conf<0.70) | {abstained_mask_pv.sum()} | {100*abstained_mask_pv.mean():.2f}% |",
    f"| Diagnosed | {diagnosed_mask_pv.sum()} | {100*diagnosed_mask_pv.mean():.2f}% |",
    f"| Accuracy on diagnosed | — | {100*pv_t_corr/diagnosed_mask_pv.sum() if diagnosed_mask_pv.sum()>0 else 0:.2f}% |",
    f"| Guarded accuracy (ALL) | — | {ga_all_pv*100:.2f}% |",
    "",
    "### 1g. Demo Image Scores",
    "",
    "| File | Dep OOD Score | Dep Threshold | Dep OOD? | New OOD Score | New Threshold | New OOD? | Dep Pred | New Pred |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in demo_rows:
    md_lines.append(f"| {r['file']} | {r['dep_ood_score']:.4f} | {r['dep_ood_thr']:.4f} | {r['dep_is_ood']} | {r['new_ood_score']:.4f} | {r['new_ood_thr']:.4f} | {r['new_is_ood']} | {r['dep_pred']} ({r['dep_conf']*100:.1f}%) | {r['new_pred']} ({r['new_conf']*100:.1f}%) |")

md_lines += [
    "",
    "---",
    "",
    "## 2. FGVC Test — Bootstrap CIs and Reconciliation",
    "",
    "### 2a. Raw Metrics (new model)",
    "",
    f"| Metric | Value | 95% CI |",
    f"|---|---|---|",
    f"| Raw Accuracy | {fgvc_acc_ci[0]*100:.2f}% | [{fgvc_acc_ci[1]*100:.2f}%, {fgvc_acc_ci[2]*100:.2f}%] |",
    f"| Raw Macro-F1 | {fgvc_f1_ci[0]:.4f} | [{fgvc_f1_ci[1]:.4f}, {fgvc_f1_ci[2]:.4f}] |",
]
for cls, N_c, r, ci in fgvc_recall_table:
    md_lines.append(f"| Raw Recall — {cls} (N={N_c}) | {r*100:.2f}% | [{ci[1]*100:.2f}%, {ci[2]*100:.2f}%] |")

md_lines += [
    "",
    "### 2b. Guarded Confusion Matrix (fgvc_test)",
    "",
    "| True \\ Pred | " + " | ".join(new_classes) + " | abstained | rejected |",
    "|" + "---|"*(len(new_classes)+3),
]
for i,cls in enumerate(new_classes):
    if (fgvc_labels_np==i).sum()==0: continue
    row = [str(int(np.sum((fgvc_labels_np==i)&fgvc_diag&(fgvc_pred_np==j)))) for j in range(len(new_classes))]
    abst_n = int(np.sum(fgvc_abst & (fgvc_labels_np==i)))
    rej_n  = int(np.sum(fgvc_rej  & (fgvc_labels_np==i)))
    md_lines.append(f"| {cls} | " + " | ".join(row) + f" | {abst_n} | {rej_n} |")

md_lines += [
    "",
    "### 2c. Reconciliation",
    "",
    f"| Metric | Value |",
    f"|---|---|",
    f"| Total fgvc_test images | {n_total} |",
    f"| Rejected | {fgvc_rej.sum()} ({100*fgvc_rej.mean():.2f}%) |",
    f"| Abstained | {fgvc_abst.sum()} ({100*fgvc_abst.mean():.2f}%) |",
    f"| Diagnosed | {n_diagnosed} ({100*n_diagnosed/n_total:.2f}%) |",
    f"| Correct among diagnosed | {n_correct_diag} ({100*n_correct_diag/n_diagnosed:.2f}%) |",
    f"| Guarded accuracy (ALL) | {100*n_correct_diag/n_total:.2f}% |",
    f"| Epoch saved | {best_epoch} |",
    f"| Selection metric | {selection_metric} |",
    "",
    "> The Stage 3 report stated 97.19% accuracy on 320 diagnosed = ~311 correct, "
    f"giving 311/346 = 89.88% guarded all. "
    f"This run: {n_correct_diag}/{n_diagnosed} = {100*n_correct_diag/n_diagnosed:.2f}% "
    f"on diagnosed, {n_correct_diag}/{n_total} = {100*n_correct_diag/n_total:.2f}% guarded all.",
    "",
    "---",
    "",
    "## 3. Cross-Crop Guard Re-Evaluation (new apple OOD, margin=0.25)",
    "",
    "### 3a. Correct-Crop False Block Rate",
    "",
    "| Crop | N | OOD-only | Cross-Crop | Total | ≤5%? |",
    "|---|---|---|---|---|---|",
    f"| Paddy (PV test) | {fb_results['paddy']['N']} | {fb_results['paddy']['old_r']*100:.4f}% | {fb_results['paddy']['cc_r']*100:.4f}% | {fb_results['paddy']['tot_r']*100:.4f}% | {'✅' if fb_results['paddy']['tot_r']*100<=5 else '❌'} |",
    f"| Arecanut (PV test) | {fb_results['arecanut']['N']} | {fb_results['arecanut']['old_r']*100:.4f}% | {fb_results['arecanut']['cc_r']*100:.4f}% | {fb_results['arecanut']['tot_r']*100:.4f}% | {'✅' if fb_results['arecanut']['tot_r']*100<=5 else '❌'} |",
    f"| Apple (PV test) | {fb_results['apple']['N']} | {fb_results['apple']['old_r']*100:.4f}% | {fb_results['apple']['cc_r']*100:.4f}% | {fb_results['apple']['tot_r']*100:.4f}% | {'✅' if fb_results['apple']['tot_r']*100<=5 else '❌'} |",
    f"| Apple (FGVC test) | {N_fgvc} | {old_b_fgvc.mean()*100:.4f}% | {cc_b_fgvc.mean()*100:.4f}% | {tot_b_fgvc.mean()*100:.4f}% | {'✅' if tot_b_fgvc.mean()*100<=5 else '❌'} |",
    "",
    "### 3b. Wrong-Crop Combinations (200 images each)",
    "",
    "| True Crop | Tested Mode | Old OOD-only | Cross-Crop | Total Block |",
    "|---|---|---|---|---|",
]
for (tc,mc), (tot_n, N, tot_r, old_r, cc_r) in wrong_results.items():
    md_lines.append(f"| {tc.capitalize()} | {mc.capitalize()} | {old_r*100:.4f}% | {cc_r*100:.4f}% | {tot_n}/{N} ({tot_r*100:.4f}%) |")

md_lines += [
    "",
    "### 3c. Non-Crop Disk Images (60) in All 3 Modes",
    "",
    "| Mode | Old OOD-only | Cross-Crop | Total Block |",
    "|---|---|---|---|",
]
for mode, (tot_n, N, tot_r, old_r, cc_r) in nc_results.items():
    md_lines.append(f"| {mode.capitalize()} | {old_r*100:.4f}% | {cc_r*100:.4f}% | {tot_n}/{N} ({tot_r*100:.4f}%) |")

md_lines += [
    "",
    "### 3d. Apple Orchard (FGVC 200) in Paddy and Arecanut Mode",
    "",
    "| Mode | Old OOD-only | Cross-Crop | Total Block |",
    "|---|---|---|---|",
]
for mode, (tot_n, N, tot_r, old_r, cc_r) in fgvc_wrong.items():
    md_lines.append(f"| {mode.capitalize()} | {old_r*100:.4f}% | {cc_r*100:.4f}% | {tot_n}/{N} ({tot_r*100:.4f}%) |")

md_lines += [
    "",
    "---",
    "",
    "## 4. Abstain Calibration",
    "",
    "### 4a. Coverage vs Accuracy (fgvc_val + PV val, OOD-passed only)",
    "",
    f"Combined N={len(combined_conf)}, OOD-passed={N_ood_pass}",
    "",
    "| Threshold (τ) | Coverage | Accuracy | N_diagnosed | N_abstained |",
    "|---|---|---|---|---|",
]
for row in abstain_table:
    md_lines.append(f"| {row['tau']:.2f} | {row['coverage']*100:.1f}% | {row['accuracy']*100:.1f}% | {row['N_diag']} | {row['N_abst']} |")

md_lines += [
    "",
    "### 4b. Results at τ=0.70",
    "",
    "| Split | Diagnosed | Acc on diagnosed | Guarded acc (ALL) |",
    "|---|---|---|---|",
    f"| fgvc_test | {fgvc_t_diag.sum()}/{len(fgvc_test_paths_all)} | {100*fgvc_t_corr/fgvc_t_diag.sum():.2f}% | {100*fgvc_t_corr/len(fgvc_test_paths_all):.2f}% |",
    f"| PV test | {pv_diag_mask.sum()}/{len(pv_test_paths)} | {100*pv_t_corr/pv_diag_mask.sum() if pv_diag_mask.sum()>0 else 0:.2f}% | {100*pv_t_corr/len(pv_test_paths):.2f}% |",
    "",
    "---",
    "",
    "## 5. Distribution Check",
    "",
]

if nn_fgvc_cos is not None:
    md_lines += [
        "### 5a. Nearest-Train DINOv2 Cosine (fgvc_test)",
        "",
        "| Percentile | Nearest fgvc_train | Nearest PV_train |",
        "|---|---|---|",
    ]
    for p in [10,50,90,99]:
        md_lines.append(f"| p{p} | {np.percentile(nn_fgvc_cos, p):.4f} | {np.percentile(nn_pv_cos, p):.4f} |")
    md_lines += [
        "",
        f"pHash ≤ 16 to any training image: {n_close_phash}/{len(test_idxs) if test_idxs else 'N/A'}",
        "",
        "### 5b. Accuracy on Subset: nearest fgvc_train cosine < 0.90",
        "",
    ]
    if far_mask is not None:
        md_lines += [
            f"N = {far_mask.sum()}",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Raw accuracy | {raw_acc_far*100:.2f}% |",
            f"| Guarded accuracy (ALL) | {gua_acc_far*100:.2f}% |",
        ]
else:
    md_lines.append("DINOv2 embeddings not available in saved NPZ — Section 5 skipped.")

md_lines += [
    "",
    "---",
    "",
    "## 6. Deploy Conditions Summary",
    "",
    "| Condition | Threshold | Actual | PASS/FAIL |",
    "|---|---|---|---|",
]
for cond, passed, actual in COND_CHECKS:
    status = "✅ PASS" if passed else "❌ FAIL"
    md_lines.append(f"| {cond} | — | {actual} | {status} |")

all_pass = all(p for _, p, _ in COND_CHECKS)
md_lines += [
    "",
    f"> **Overall verdict: {'✅ ALL CONDITIONS PASS — SAFE TO DEPLOY' if all_pass else '❌ ONE OR MORE CONDITIONS FAILED — DO NOT DEPLOY'}**",
    "",
    "> `apple_disease_fgvc_v1.pth` + `ood_params_apple_fgvc.pt` were not moved to the production location by this script.",
    "> The deployed model `outputs/models/apple_disease_classifier.pth` was NOT modified.",
    "> `ood_params.pt` was NOT overwritten.",
]

report_path = os.path.join(OUT_DIR, "predeploy_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nReport written to {report_path}")

print("\n" + "="*70)
print("DONE. All sections complete.")
print("="*70)
print(f"\n{'='*60}")
print("DEPLOY CONDITIONS SUMMARY")
print(f"{'='*60}")
for cond, passed, actual in COND_CHECKS:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {cond}: {actual}")
all_pass = all(p for _, p, _ in COND_CHECKS)
print(f"\n  Overall: {'✅ ALL PASS — SAFE TO DEPLOY' if all_pass else '❌ FAIL — DO NOT DEPLOY'}")
