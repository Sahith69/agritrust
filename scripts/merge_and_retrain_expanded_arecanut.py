import os
import sys
import re
import io
import time
import json
import copy
import random
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import zipfile
import scipy.fftpack
import scipy.ndimage
from scipy.sparse import csgraph, csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    classification_report
)

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import CornLoss, corn_probs_to_grade
from src.ordinal_ce_loss import OrdinalCrossEntropyLoss
from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    compute_ratio
)
from src.severity_grader import ratio_to_grade

torch.set_num_threads(8)
device = torch.device('cpu')

PROJECT_ROOT = "/home/sahith/Projects/Crop"
CLEAN_CSV = os.path.join(PROJECT_ROOT, "outputs/improvements/arecanut_splits_clean_871.csv")
KAGGLE_ZIP = os.path.join(PROJECT_ROOT, "data/external/arecanut_kaggle_tejpaviraj/archive.zip")
MERGED_IMG_DIR = os.path.join(PROJECT_ROOT, "data/arecanut/dataset/kaggle_merged")
OUTPUT_SPLIT_CSV = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits.csv")
IMPROVEMENTS_DIR = os.path.join(PROJECT_ROOT, "outputs/improvements")
MERGED_MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs/models/merged_expanded")
KAGGLE_EXTRACTED_CSV = os.path.join(IMPROVEMENTS_DIR, "kaggle_extracted_1896.csv")
KAGGLE_EXTRACTED_NPZ = os.path.join(IMPROVEMENTS_DIR, "kaggle_extracted_1896_phashes.npz")

os.makedirs(MERGED_IMG_DIR, exist_ok=True)
os.makedirs(IMPROVEMENTS_DIR, exist_ok=True)
os.makedirs(MERGED_MODELS_DIR, exist_ok=True)

print("=" * 75)
print("  MERGE KAGGLE TEJPAVIRAJ EXPANDED DATASET & RE-RUN PIPELINE")
print("=" * 75)


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: LOAD CLEAN 871 DATASET & EXTRACT CONSERVATIVE KAGGLE SAMPLES
# ══════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading clean 871-image dataset...")
df_clean = pd.read_csv(CLEAN_CSV)
print(f"Loaded {len(df_clean)} clean images.")

def compute_phash_64(image_l):
    resized = image_l.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

# Reference pHashes for clean 871
clean_hashes_orig = []
for p in df_clean['image_path']:
    img_l = Image.open(p).convert('L')
    clean_hashes_orig.append(compute_phash_64(img_l))

H_clean_orig = np.array(clean_hashes_orig, dtype=bool)

CLASS_MAP = {
    'healthy_leaf': 'healthy_leaf',
    'healthy_nut': 'healthy_nut',
    'healthy_trunk': 'healthy_trunk',
    'healthy_foot': 'healthy_foot',
    'mahali_koleroga': 'mahali_koleroga',
    'stem_bleeding': 'stem_bleeding',
    'bud borer': 'bud_borer',
    'stem cracking': 'stem_cracking',
    'yellow leaf disease': 'yellow_leaf_disease'
}

def extract_class(path):
    parts = path.split('/')
    folder = parts[-2].lower()
    if folder in CLASS_MAP:
        return CLASS_MAP[folder]
    fn = parts[-1].lower()
    for k, v in CLASS_MAP.items():
        if fn.startswith(k) or fn.startswith(v):
            return v
    return 'unknown'

def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

def extract_timestamp(fn):
    m = re.search(r'IMG_(\d{8})_(\d{6})', fn)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None

if not (os.path.exists(KAGGLE_EXTRACTED_CSV) and os.path.exists(KAGGLE_EXTRACTED_NPZ)):
    print("\nCached Kaggle dataset not found. Running scripts/prepare_kaggle_merged_cache.py...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts/prepare_kaggle_merged_cache.py")], check=True)

print(f"\n[1b/5] Loading clean Kaggle dataset from cache ({KAGGLE_EXTRACTED_CSV})...")
df_extracted = pd.read_csv(KAGGLE_EXTRACTED_CSV)
npz_data = np.load(KAGGLE_EXTRACTED_NPZ)
extracted_records = []
for idx, r in df_extracted.iterrows():
    extracted_records.append({
        'image_path': r['image_path'],
        'filename': r['filename'],
        'disease_class': r['disease_class'],
        'severity_grade': int(r['severity_grade']),
        'parent_id': str(r['parent_id']),
        'timestamp': str(r['timestamp']) if pd.notna(r['timestamp']) and str(r['timestamp']) != '' else None,
        'h_orig': npz_data['h_orig'][idx],
        'h_hflip': npz_data['h_hflip'][idx],
        'h_vflip': npz_data['h_vflip'][idx]
    })
print(f"Loaded {len(extracted_records)} clean new Kaggle images.")
print("\nNewly Extracted Images Breakdown:")
print(pd.crosstab(df_extracted['disease_class'], df_extracted['severity_grade'], margins=True))


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: UNIFIED DEDUPLICATION & 120s INTERLEAVED BURST CLUSTERING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  PHASE 2: UNIFIED DEDUPLICATION GRAPH & 120s BURST CLUSTERING")
print("=" * 75)

# Prepare existing clean records
existing_records = []
for idx, r in df_clean.iterrows():
    fn = r['filename']
    ts = extract_timestamp(fn)
    # Load hashes
    im_l = Image.open(r['image_path']).convert('L')
    ho = compute_phash_64(im_l)
    hh = compute_phash_64(ImageOps.mirror(im_l))
    hv = compute_phash_64(ImageOps.flip(im_l))
    existing_records.append({
        'image_path': r['image_path'],
        'filename': fn,
        'disease_class': r['disease_class'],
        'severity_grade': int(r['severity_grade']),
        'parent_id': r['parent_id'],
        'timestamp': ts,
        'h_orig': ho,
        'h_hflip': hh,
        'h_vflip': hv
    })

combined_records = existing_records + extracted_records
df_combined = pd.DataFrame(combined_records)
M = len(df_combined)
print(f"Total Combined Dataset: {M} images (871 existing + {len(extracted_records)} new).")

# Build unified multi-criteria adjacency graph
adj = np.zeros((M, M), dtype=bool)

# 1. Base ID edge
base_ids = df_combined['parent_id'].values
for i in range(M):
    for j in range(i + 1, M):
        if base_ids[i] == base_ids[j]:
            adj[i, j] = True
            adj[j, i] = True

# 2. Camera Burst & Interleaved Proximity Edge (<= 120.0 seconds)
# Connects any two images of the same class captured within 120s of each other!
# This strictly unifies the 19 interleaved photos with their existing neighbors.
timestamps = df_combined['timestamp'].values
classes = df_combined['disease_class'].values
interleaved_burst_edges = 0

for i in range(M):
    ts_i = timestamps[i]
    if not isinstance(ts_i, str) or pd.isna(ts_i) or not ts_i:
        continue
    d_i = ts_i[:8]
    sec_i = int(ts_i[9:11])*3600 + int(ts_i[11:13])*60 + int(ts_i[13:15])
    
    for j in range(i + 1, M):
        ts_j = timestamps[j]
        if not isinstance(ts_j, str) or pd.isna(ts_j) or not ts_j:
            continue
        d_j = ts_j[:8]
        if d_i == d_j and classes[i] == classes[j]:
            sec_j = int(ts_j[9:11])*3600 + int(ts_j[11:13])*60 + int(ts_j[13:15])
            if abs(sec_i - sec_j) <= 120.0:
                if not adj[i, j]:
                    interleaved_burst_edges += 1
                adj[i, j] = True
                adj[j, i] = True

print(f"Added {interleaved_burst_edges} burst & interleaved proximity edges (<=120s window).")

# 3. Multi-orientation Perceptual Hash (Hamming <= 5 bits, same class)
H_all_orig = np.array([r['h_orig'] for r in combined_records], dtype=bool)
H_all_hflip = np.array([r['h_hflip'] for r in combined_records], dtype=bool)
H_all_vflip = np.array([r['h_vflip'] for r in combined_records], dtype=bool)

dist_o = (H_all_orig[:, None, :] != H_all_orig[None, :, :]).sum(axis=-1)
dist_h = (H_all_orig[:, None, :] != H_all_hflip[None, :, :]).sum(axis=-1)
dist_v = (H_all_orig[:, None, :] != H_all_vflip[None, :, :]).sum(axis=-1)
min_dist_all = np.minimum(dist_o, np.minimum(dist_h, dist_v))

phash_edges = 0
for i in range(M):
    for j in range(i + 1, M):
        if classes[i] == classes[j] and min_dist_all[i, j] <= 5:
            if not adj[i, j]:
                phash_edges += 1
            adj[i, j] = True
            adj[j, i] = True

print(f"Added {phash_edges} perceptual near-duplicate edges (Hamming <= 5).")

# Connected components
n_components, labels = csgraph.connected_components(csr_matrix(adj), directed=False)
df_combined['cluster_id'] = [f"cluster_{l:04d}" for l in labels]
print(f"Unified {M} images into {n_components} independent physical clusters.")

# StratifiedGroupKFold Partitioning
sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
folds = np.zeros(M, dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df_combined, df_combined['disease_class'], groups=df_combined['cluster_id'])):
    folds[val_idx] = fold_idx

split_arr = np.array(['train'] * M, dtype=object)
split_arr[folds == 0] = 'test'
split_arr[folds == 1] = 'val'
df_combined['split'] = split_arr

print("\n--- Merged Dataset Split Breakdown ---")
print(df_combined['split'].value_counts())
print("\nGrade Distribution across Splits:")
print(pd.crosstab(df_combined['severity_grade'], df_combined['split'], margins=True))
print("\nDisease Class Distribution across Splits:")
print(pd.crosstab(df_combined['disease_class'], df_combined['split'], margins=True))

# Strict Leakage Verification
train_clusters = set(df_combined[df_combined['split'] == 'train']['cluster_id'])
val_clusters = set(df_combined[df_combined['split'] == 'val']['cluster_id'])
test_clusters = set(df_combined[df_combined['split'] == 'test']['cluster_id'])

c_leak_tv = len(train_clusters & val_clusters)
c_leak_tt = len(train_clusters & test_clusters)
c_leak_vt = len(val_clusters & test_clusters)

splits = df_combined['split'].values
cross_split_phash = 0
cross_split_time_leak = 0

for i in range(M):
    for j in range(i + 1, M):
        if splits[i] != splits[j]:
            if min_dist_all[i, j] <= 5:
                cross_split_phash += 1
            ts_i = timestamps[i]
            ts_j = timestamps[j]
            if isinstance(ts_i, str) and isinstance(ts_j, str) and not pd.isna(ts_i) and not pd.isna(ts_j) and ts_i and ts_j and classes[i] == classes[j]:
                if ts_i[:8] == ts_j[:8]:
                    s_i = int(ts_i[9:11])*3600 + int(ts_i[11:13])*60 + int(ts_i[13:15])
                    s_j = int(ts_j[9:11])*3600 + int(ts_j[11:13])*60 + int(ts_j[13:15])
                    if abs(s_i - s_j) <= 120.0:
                        cross_split_time_leak += 1

print("\n--- Mathematical Leakage Verification ---")
print(f"  Train & Val cluster overlap: {c_leak_tv}")
print(f"  Train & Test cluster overlap: {c_leak_tt}")
print(f"  Val & Test cluster overlap: {c_leak_vt}")
print(f"  Cross-split pHash pairs (Hamming <= 5): {cross_split_phash}")
print(f"  Cross-split camera proximity pairs (<= 120s): {cross_split_time_leak}")
assert c_leak_tt == 0 and c_leak_tv == 0 and c_leak_vt == 0, "Cluster leakage detected!"
assert cross_split_phash == 0, "pHash leakage detected!"
assert cross_split_time_leak == 0, "Temporal proximity leakage detected!"
print(">>> VERIFICATION PASSED: EXACTLY 0% LEAKAGE CONFIRMED! <<<")

# Save combined splits
out_cols = ['image_path', 'filename', 'disease_class', 'severity_grade', 'parent_id', 'split', 'cluster_id']
df_combined[out_cols].to_csv(OUTPUT_SPLIT_CSV, index=False)
merged_csv_copy = os.path.join(IMPROVEMENTS_DIR, "arecanut_splits_merged_expanded.csv")
df_combined[out_cols].to_csv(merged_csv_copy, index=False)
print(f"Saved merged split to {OUTPUT_SPLIT_CSV} and {merged_csv_copy}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: RETRAIN CORN & SOFTMAX ACROSS 3 SEEDS FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  PHASE 3: FULL RETRAINING FROM SCRATCH ON MERGED EXPANDED DATASET")
print("=" * 75)

class FastCachedDataset(Dataset):
    def __init__(self, df, split):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.labels = [int(g) for g in self.df['severity_grade']]
        print(f"Caching {len(self.df)} {split} images into RAM...", flush=True)
        t0 = time.time()
        
        if split == 'train':
            self.images = [
                Image.open(p).convert('RGB').resize((config.IMG_SIZE, config.IMG_SIZE))
                for p in self.df['image_path']
            ]
            self.transform = transforms.Compose([
                transforms.RandomRotation(config.AUGMENTATION['rotation_degrees']),
                transforms.RandomHorizontalFlip(config.AUGMENTATION['horizontal_flip_p']),
                transforms.RandomVerticalFlip(config.AUGMENTATION['vertical_flip_p']),
                transforms.ColorJitter(
                    brightness=config.AUGMENTATION['brightness_jitter'], 
                    contrast=config.AUGMENTATION['contrast_jitter']
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
            ])
            self.tensors = None
        else:
            eval_tf = get_transforms(split)
            self.images = None
            self.transform = None
            self.tensors = torch.stack([
                eval_tf(Image.open(p).convert('RGB'))
                for p in self.df['image_path']
            ])
        print(f"Cached {split} in {time.time()-t0:.2f}s.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.split == 'train':
            img = self.images[idx]
            img = self.transform(img)
            return img, self.labels[idx]
        else:
            return self.tensors[idx], self.labels[idx]


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
        for images, targets in loader:
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


def evaluate_model_on_test(model, loader, ordinal=True):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(logits, dim=1)

            y_true.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    mae = float(mean_absolute_error(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)

    per_grade = {
        f"G{g}": {
            "p": round(float(rep[str(g)]['precision']), 4),
            "r": round(float(rep[str(g)]['recall']), 4),
            "f1": round(float(rep[str(g)]['f1-score']), 4),
            "support": int(rep[str(g)]['support'])
        }
        for g in range(4)
    }

    return {
        "num_test_samples": len(y_true),
        "qwk": round(qwk, 4),
        "mae": round(mae, 4),
        "exact_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_grade": per_grade
    }


# Initialize datasets
train_dataset = FastCachedDataset(df_combined, 'train')
val_dataset = FastCachedDataset(df_combined, 'val')
test_dataset = FastCachedDataset(df_combined, 'test')

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Verify zero test images in train dataset
train_paths = set(train_dataset.df['image_path'])
test_paths = set(test_dataset.df['image_path'])
assert len(train_paths & test_paths) == 0, "Test image found in train!"
print(f"\nVerification: {len(test_paths)} test images, exactly 0 present in train ({len(train_paths)} images).")

SEEDS = [42, 100, 2024]
MODELS_TO_TRAIN = [
    ("corn", True),
    ("softmax", False)
]

all_results = {"corn": {}, "softmax": {}}

for model_name, is_ordinal in MODELS_TO_TRAIN:
    print("\n" + "#" * 75)
    print(f"  STARTING MULTI-SEED RETRAINING: {model_name.upper()} (Ordinal={is_ordinal})")
    print("#" * 75)

    for seed in SEEDS:
        print(f"\n>>> Model: {model_name.upper()} | Seed: {seed} <<<", flush=True)
        set_seed(seed)
        t_start = time.time()

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

        model = CropSeverityModel(
            backbone='resnet18',
            num_classes=config.NUM_SEVERITY_GRADES,
            ordinal=is_ordinal,
            pretrained=True
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        if is_ordinal:
            loss_fn = CornLoss(num_classes=config.NUM_SEVERITY_GRADES)
        else:
            loss_fn = OrdinalCrossEntropyLoss(num_classes=config.NUM_SEVERITY_GRADES)

        best_val_qwk = -1.0
        best_epoch = 0
        best_weights = None
        epochs_no_improve = 0

        for epoch in range(config.NUM_EPOCHS):
            t_ep0 = time.time()
            train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn, ordinal=is_ordinal)
            val_loss, val_qwk = validate(model, val_loader, loss_fn, ordinal=is_ordinal)
            ep_time = time.time() - t_ep0

            improved = val_qwk > best_val_qwk
            if improved:
                best_val_qwk = val_qwk
                best_epoch = epoch + 1
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                flag = "  --> New Best Val QWK!"
            else:
                epochs_no_improve += 1
                flag = ""

            print(
                f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} ({ep_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}{flag}",
                flush=True
            )

            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1} (patience={config.EARLY_STOPPING_PATIENCE}).", flush=True)
                break

        # Save merged checkpoint
        ckpt_filename = f"arecanut_merged_{model_name}_seed{seed}.pth"
        ckpt_path = os.path.join(MERGED_MODELS_DIR, ckpt_filename)
        torch.save(best_weights, ckpt_path)
        print(f"Saved best model (Val QWK: {best_val_qwk:.4f} @ Epoch {best_epoch}) to: {ckpt_path}", flush=True)

        # If seed 42, also update the primary checkpoint in outputs/models/
        if seed == 42:
            primary_path = os.path.join(config.MODELS_DIR, f"arecanut_{'ordinal' if is_ordinal else 'baseline'}.pth")
            torch.save(best_weights, primary_path)
            print(f"Updated primary production model at: {primary_path}", flush=True)

        # Load best weights for test set evaluation
        model.load_state_dict(best_weights)
        test_eval = evaluate_model_on_test(model, test_loader, ordinal=is_ordinal)
        test_eval['best_epoch'] = best_epoch
        test_eval['best_val_qwk'] = round(float(best_val_qwk), 4)
        test_eval['train_time_sec'] = round(time.time() - t_start, 2)

        print(
            f"Test Evaluation (Seed {seed}): QWK={test_eval['qwk']:.4f} | "
            f"MAE={test_eval['mae']:.4f} | Acc={test_eval['exact_accuracy']*100:.2f}% | "
            f"Macro-F1={test_eval['macro_f1']:.4f}",
            flush=True
        )

        all_results[model_name][f"seed_{seed}"] = test_eval

# Compute 3-seed means and standard deviations
summary_results = copy.deepcopy(all_results)
for model_name in ["corn", "softmax"]:
    seeds_data = [all_results[model_name][f"seed_{s}"] for s in SEEDS]
    qwks = [d['qwk'] for d in seeds_data]
    maes = [d['mae'] for d in seeds_data]
    accs = [d['exact_accuracy'] for d in seeds_data]
    f1s = [d['macro_f1'] for d in seeds_data]

    g_metrics = {}
    for g in range(4):
        g_key = f"G{g}"
        g_ps = [d['per_grade'][g_key]['p'] for d in seeds_data]
        g_rs = [d['per_grade'][g_key]['r'] for d in seeds_data]
        g_f1s = [d['per_grade'][g_key]['f1'] for d in seeds_data]
        g_metrics[g_key] = {
            "p_mean": round(float(np.mean(g_ps)), 4),
            "p_sd": round(float(np.std(g_ps)), 4),
            "r_mean": round(float(np.mean(g_rs)), 4),
            "r_sd": round(float(np.std(g_rs)), 4),
            "f1_mean": round(float(np.mean(g_f1s)), 4),
            "f1_sd": round(float(np.std(g_f1s)), 4),
            "support": seeds_data[0]['per_grade'][g_key]['support']
        }

    summary_results[model_name]["mean_sd"] = {
        "qwk_mean": round(float(np.mean(qwks)), 4),
        "qwk_sd": round(float(np.std(qwks)), 4),
        "mae_mean": round(float(np.mean(maes)), 4),
        "mae_sd": round(float(np.std(maes)), 4),
        "exact_accuracy_mean": round(float(np.mean(accs)), 4),
        "exact_accuracy_sd": round(float(np.std(accs)), 4),
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_sd": round(float(np.std(f1s)), 4),
        "per_grade": g_metrics
    }

# Save final results JSON
final_json_path = os.path.join(IMPROVEMENTS_DIR, "merged_expanded_retrain_results.json")
with open(final_json_path, 'w') as f:
    json.dump(summary_results, f, indent=2)

print("\n" + "=" * 75)
print(f"RETRAINING & EVALUATION COMPLETE! Saved to {final_json_path}")
print("=" * 75)
