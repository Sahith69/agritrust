import os
import sys
import re
import json
import time
import copy
import random
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import scipy.fftpack
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

# Set CPU threads
torch.set_num_threads(4)
device = torch.device('cpu')

PROJECT_ROOT = "/home/sahith/Projects/Crop"
BACKUP_SPLIT_CSV = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits_LEAKY_BACKUP.csv")
OUTPUT_SPLIT_CSV = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits.csv")
IMPROVEMENTS_DIR = os.path.join(PROJECT_ROOT, "outputs/improvements")
CLEAN_MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs/models/clean_retrain")
os.makedirs(IMPROVEMENTS_DIR, exist_ok=True)
os.makedirs(CLEAN_MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: REMOVE STOCK PHOTOS & RE-RUN LEAK-FREE GRAPH DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════
print("=" * 75)
print("  STEP 1: REMOVE CONTAMINATED STOCK PHOTOS & BUILD LEAK-FREE SPLIT")
print("=" * 75)

raw_df = pd.read_csv(BACKUP_SPLIT_CSV)
print(f"Loaded raw dataset from backup: {len(raw_df)} images.")

# 1. Identify all 24 stock / studio images
stock_candidates = []
for idx, r in raw_df.iterrows():
    p = r['image_path']
    if not os.path.exists(p):
        continue
    im = Image.open(p)
    arr = np.array(im)
    h, w = arr.shape[:2]
    borders = np.concatenate([
        arr[:10, :, :].reshape(-1, 3),
        arr[-10:, :, :].reshape(-1, 3),
        arr[:, :10, :].reshape(-1, 3),
        arr[:, -10:, :].reshape(-1, 3)
    ], axis=0)
    border_mean = borders.mean(axis=0)
    border_std = borders.std(axis=0)
    is_white_bg = bool(np.all(border_mean > 235) and np.all(border_std < 20))
    is_black_bg = bool(np.all(border_mean < 20) and np.all(border_std < 15))
    info = im.info
    info_str = str(info).lower()
    has_agency_meta = any(k in info_str for k in ['dreamstime', 'shutterstock', 'istock', 'getty', 'adobe', 'depositphotos', 'alamy'])
    if is_white_bg or is_black_bg or has_agency_meta:
        stock_candidates.append(r['image_path'])

print(f"Identified {len(stock_candidates)} contaminated stock/studio images.")
assert len(stock_candidates) == 24, f"Expected 24 stock images, got {len(stock_candidates)}"

# Save list of 24 removed images
removed_json_path = os.path.join(IMPROVEMENTS_DIR, "removed_stock_images.json")
with open(removed_json_path, 'w') as f:
    json.dump(stock_candidates, f, indent=2)
print(f"Saved removed stock image paths to: {removed_json_path}")

# Filter dataset to 871 clean images
clean_df = raw_df[~raw_df['image_path'].isin(stock_candidates)].copy().reset_index(drop=True)
print(f"Clean dataset size: {len(clean_df)} images.")
assert len(clean_df) == 871, f"Expected 871 clean images, got {len(clean_df)}"

# 2. Extract base IDs and camera timestamps
def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

def extract_timestamp(fn):
    m = re.search(r'IMG_(\d{8})_(\d{6})', fn)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except Exception:
            return None
    return None

clean_df['base_id'] = clean_df['filename'].apply(extract_base_id)
clean_df['timestamp'] = clean_df['filename'].apply(extract_timestamp)

# 3. Compute 3x 64-bit DCT pHashes for clean set
def compute_phash_64(image_l):
    resized = image_l.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

print("Computing 3x pHashes (orig, hflip, vflip) for clean set...")
t_hash0 = time.time()
hashes_orig = []
hashes_hflip = []
hashes_vflip = []

for p in clean_df['image_path']:
    img = Image.open(p).convert('L')
    hashes_orig.append(compute_phash_64(img))
    hashes_hflip.append(compute_phash_64(ImageOps.mirror(img)))
    hashes_vflip.append(compute_phash_64(ImageOps.flip(img)))

H_orig = np.array(hashes_orig)
H_hflip = np.array(hashes_hflip)
H_vflip = np.array(hashes_vflip)
print(f"pHashes computed in {time.time()-t_hash0:.2f}s.")

# Pairwise minimum distance
dist_orig = (H_orig[:, None, :] != H_orig[None, :, :]).sum(axis=-1)
dist_hflip = (H_orig[:, None, :] != H_hflip[None, :, :]).sum(axis=-1)
dist_vflip = (H_orig[:, None, :] != H_vflip[None, :, :]).sum(axis=-1)
min_dist = np.minimum(dist_orig, np.minimum(dist_hflip, dist_vflip))

# 4. Multi-criteria adjacency graph
N = len(clean_df)
adj = np.zeros((N, N), dtype=bool)

# Edge 1: Identical base ID
base_ids = clean_df['base_id'].values
for i in range(N):
    for j in range(i + 1, N):
        if base_ids[i] == base_ids[j]:
            adj[i, j] = True
            adj[j, i] = True

# Edge 2: Camera burst shots (same class, <= 10.0s window)
timestamps = clean_df['timestamp'].values
classes = clean_df['disease_class'].values
burst_edges = 0
for i in range(N):
    if pd.isna(timestamps[i]):
        continue
    for j in range(i + 1, N):
        if pd.isna(timestamps[j]):
            continue
        if classes[i] == classes[j]:
            diff_sec = abs(float((timestamps[i] - timestamps[j]) / np.timedelta64(1, 's')))
            if diff_sec <= 10.0:
                if not adj[i, j]:
                    burst_edges += 1
                adj[i, j] = True
                adj[j, i] = True

# Edge 3: Perceptual near-duplicates (Hamming dist <= 5, same class)
phash_edges = 0
for i in range(N):
    for j in range(i + 1, N):
        if classes[i] == classes[j] and min_dist[i, j] <= 5:
            if not adj[i, j]:
                phash_edges += 1
            adj[i, j] = True
            adj[j, i] = True

print(f"Graph constructed: {burst_edges} burst edges (<=10s), {phash_edges} pHash edges (<=5 bits).")

# Connected components
n_components, labels = csgraph.connected_components(csr_matrix(adj), directed=False)
clean_df['cluster_id'] = [f"cluster_{l:04d}" for l in labels]
print(f"Unified 871 clean images into {n_components} independent physical clusters.")

# 5. StratifiedGroupKFold partition
sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
folds = np.zeros(len(clean_df), dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(clean_df, clean_df['disease_class'], groups=clean_df['cluster_id'])):
    folds[val_idx] = fold_idx

split_arr = np.array(['train'] * len(clean_df), dtype=object)
split_arr[folds == 0] = 'test'
split_arr[folds == 1] = 'val'
clean_df['split'] = split_arr

print("\n--- Clean 871 Split Summary ---")
print(clean_df['split'].value_counts())
print("\nGrade distribution across splits:")
print(pd.crosstab(clean_df['severity_grade'], clean_df['split']))
print("\nClass distribution across splits:")
print(pd.crosstab(clean_df['disease_class'], clean_df['split']))

# 6. Leakage Verification
train_clusters = set(clean_df[clean_df['split'] == 'train']['cluster_id'])
val_clusters = set(clean_df[clean_df['split'] == 'val']['cluster_id'])
test_clusters = set(clean_df[clean_df['split'] == 'test']['cluster_id'])

c_leak_tv = len(train_clusters & val_clusters)
c_leak_tt = len(train_clusters & test_clusters)
c_leak_vt = len(val_clusters & test_clusters)

clean_splits = clean_df['split'].values
cross_split_phash = 0
for i in range(N):
    for j in range(i + 1, N):
        if clean_splits[i] != clean_splits[j] and min_dist[i, j] <= 5:
            cross_split_phash += 1

print(f"\nVerification Results:")
print(f"  Train & Val cluster overlap: {c_leak_tv}")
print(f"  Train & Test cluster overlap: {c_leak_tt}")
print(f"  Val & Test cluster overlap: {c_leak_vt}")
print(f"  Cross-split pHash pairs (Hamming <= 5): {cross_split_phash}")
assert c_leak_tt == 0 and c_leak_tv == 0 and c_leak_vt == 0 and cross_split_phash == 0, "Leakage detected!"
print(">>> VERIFICATION PASSED: EXACTLY 0% LEAKAGE CONFIRMED! <<<")

# Save cleaned split CSVs
out_cols = ['image_path', 'filename', 'disease_class', 'severity_grade', 'parent_id', 'split', 'cluster_id']
clean_df[out_cols].to_csv(OUTPUT_SPLIT_CSV, index=False)
clean_csv_copy = os.path.join(IMPROVEMENTS_DIR, "arecanut_splits_clean_871.csv")
clean_df[out_cols].to_csv(clean_csv_copy, index=False)
print(f"Saved clean split to {OUTPUT_SPLIT_CSV} and {clean_csv_copy}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: FULL RETRAIN FROM SCRATCH (CORN & SOFTMAX ACROSS 3 SEEDS)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  STEP 2: FULL RETRAINING FROM SCRATCH (CORN & SOFTMAX, SEEDS 42, 100, 2024)")
print("=" * 75)

# Fast RAM-cached Dataset
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
train_dataset = FastCachedDataset(clean_df, 'train')
val_dataset = FastCachedDataset(clean_df, 'val')
test_dataset = FastCachedDataset(clean_df, 'test')

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

        # Save clean checkpoint
        ckpt_filename = f"arecanut_resnet18_{model_name}_seed{seed}.pth"
        ckpt_path = os.path.join(CLEAN_MODELS_DIR, ckpt_filename)
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

# Save final comprehensive results
final_json_path = os.path.join(IMPROVEMENTS_DIR, "clean_leakfree_retrain_results.json")
with open(final_json_path, 'w') as f:
    json.dump(summary_results, f, indent=2)

print("\n" + "=" * 75)
print(f"RETRAINING & EVALUATION COMPLETE! Saved to {final_json_path}")
print("=" * 75)
