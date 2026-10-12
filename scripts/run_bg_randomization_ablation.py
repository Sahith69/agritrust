"""
Ablation Study: Background-Randomization Training for Arecanut ResNet-18 CORN.
Tests whether replacing background pixels with random solid colors during training
mitigates shortcut learning and improves diseased-tissue pointing-game accuracy.
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
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    mean_absolute_error,
    f1_score
)

from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import CornLoss, corn_probs_to_grade
from src.gradcam import GradCAM, overlay_heatmap
from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    detect_dead_heart,
    detect_hispa
)

torch.set_num_threads(4)
device = torch.device('cpu')

IMPROVEMENTS_DIR = os.path.join(config.OUTPUTS_DIR, "improvements")
EXAMPLES_DIR = os.path.join(IMPROVEMENTS_DIR, "bg_randomization_examples")
GALLERY_DIR = os.path.join(IMPROVEMENTS_DIR, "bg_randomization_gallery")
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)


class GradCAMPlusPlus:
    """Grad-CAM++ formulation for higher-order gradient visualization."""
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        target_layer = None
        if hasattr(self.model, 'backbone'):
            for name, module in self.model.backbone.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break
        else:
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break

        if target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found in model.")

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            if output.shape[1] == config.NUM_ORDINAL_TASKS:
                probs = torch.sigmoid(output)
                pred_grade, _ = corn_probs_to_grade(probs)
                target_class = pred_grade.item()
            else:
                target_class = output.argmax(dim=1).item()

        if output.shape[1] == config.NUM_ORDINAL_TASKS:
            if target_class == 0:
                target = -output[0, 0]
            elif target_class == config.NUM_ORDINAL_TASKS:
                target = output[0, target_class - 1]
            else:
                target = output[0, target_class - 1]
        else:
            target = output[0, target_class]

        target.backward()

        grads = self.gradients.detach()
        activations = self.activations.detach()

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-7

        ai_j = grads_power_2 / (2.0 * grads_power_2 + sum_activations * grads_power_3 + eps)
        ai_j = torch.where(grads != 0, ai_j, torch.zeros_like(ai_j))

        weights = torch.maximum(grads, torch.zeros_like(grads)) * ai_j
        weights = torch.sum(weights, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


class BgRandomizedDataset(Dataset):
    """
    Dataset that applies training-time Background Randomization:
    1. Preloads raw images and pre-resizes to 224x224.
    2. Applies geometric & color augmentations (rotation, flip, jitter).
    3. Segments organ mask M_organ using adaptive HSV/Lab thresholding.
    4. Replaces all pixels OUTSIDE M_organ with a random uniform solid RGB color.
    5. Leaves all pixels INSIDE M_organ (including lesions) untouched.
    6. Normalizes with standard ImageNet statistics.
    For validation and test: No background randomization, standard real images.
    """
    def __init__(self, split_csv, split, randomize_bg=False):
        self.df = pd.read_csv(split_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.split = split
        self.randomize_bg = randomize_bg
        self.labels = [int(g) for g in self.df['severity_grade']]

        print(f"Preloading {len(self.df)} {split} images (randomize_bg={randomize_bg})...", flush=True)
        t0 = time.time()
        if split == 'train':
            self.images = [
                Image.open(p).convert('RGB').resize((config.IMG_SIZE, config.IMG_SIZE))
                for p in self.df['image_path']
            ]
            self.geom_transform = transforms.Compose([
                transforms.RandomRotation(config.AUGMENTATION['rotation_degrees']),
                transforms.RandomHorizontalFlip(config.AUGMENTATION['horizontal_flip_p']),
                transforms.RandomVerticalFlip(config.AUGMENTATION['vertical_flip_p']),
                transforms.ColorJitter(
                    brightness=config.AUGMENTATION['brightness_jitter'],
                    contrast=config.AUGMENTATION['contrast_jitter']
                )
            ])
            self.norm_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
            ])
            self.tensors = None
        else:
            # Deterministic eval transforms
            eval_tf = get_transforms(split)
            self.images = None
            self.tensors = torch.stack([
                eval_tf(Image.open(p).convert('RGB'))
                for p in self.df['image_path']
            ])
        print(f"Preloaded in {time.time()-t0:.2f}s.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.split == 'train':
            pil_img = self.images[idx]
            aug_pil = self.geom_transform(pil_img)
            aug_np = np.array(aug_pil)

            if self.randomize_bg:
                # 1. Compute organ mask on the augmented 224x224 array
                organ_mask = segment_leaf_adaptive(aug_np)
                # 2. Pick random solid RGB color
                rand_color = np.random.randint(0, 256, 3, dtype=np.uint8)
                # 3. Replace background pixels only
                rand_np = aug_np.copy()
                rand_np[~organ_mask] = rand_color
                final_pil = Image.fromarray(rand_np)
            else:
                final_pil = aug_pil

            tensor = self.norm_transform(final_pil)
            return tensor, self.labels[idx], 0, ""
        else:
            return self.tensors[idx], self.labels[idx], 0, ""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def train_one_epoch(model, loader, optimizer, loss_fn):
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
            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk


def validate(model, loader, loss_fn):
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

            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk


def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            y_true.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    mae = float(mean_absolute_error(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)

    per_grade = {}
    for g in range(4):
        str_g = str(g)
        per_grade[f"Grade_{g}"] = {
            "precision": round(float(report[str_g]["precision"]), 4),
            "recall": round(float(report[str_g]["recall"]), 4),
            "f1": round(float(report[str_g]["f1-score"]), 4),
            "support": int(report[str_g]["support"])
        }

    return {
        "qwk": round(qwk, 4),
        "mae": round(mae, 4),
        "exact_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "per_grade": per_grade
    }


def get_symptom_mask(img_np, disease_class):
    try:
        leaf_mask = segment_leaf_adaptive(img_np)
        if disease_class == 'dead_heart':
            mask = detect_dead_heart(img_np, leaf_mask)
        elif disease_class == 'hispa':
            mask = detect_hispa(img_np, leaf_mask)
        elif disease_class in config.PADDY_GROUP_A or disease_class in config.ARECANUT_GROUP_A:
            mask = detect_lesions_group_a(img_np, leaf_mask)
        elif disease_class == 'normal' or disease_class in [config.ARECANUT_CLASSES[c] for c in config.ARECANUT_HEALTHY_CLASSES]:
            mask = leaf_mask
        else:
            mask = detect_lesions_group_a(img_np, leaf_mask)
        return mask
    except Exception:
        return None


def run_pointing_game_comparison(orig_model_path, new_model_path, split_csv):
    """
    Evaluates both original model and new background-randomized model
    on the exact same 124 test images using Grad-CAM and Grad-CAM++.
    """
    print(f"\n=======================================================", flush=True)
    print(f"  RUNNING POINTING-GAME COMPARISON (ARECANUT TEST SET)", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Load models
    model_orig = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False).to(device)
    model_orig.load_state_dict(torch.load(orig_model_path, map_location=device, weights_only=True))
    model_orig.eval()

    model_new = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False).to(device)
    model_new.load_state_dict(torch.load(new_model_path, map_location=device, weights_only=True))
    model_new.eval()

    gc_orig = GradCAM(model_orig, target_layer_name='layer4')
    gcpp_orig = GradCAMPlusPlus(model_orig, target_layer_name='layer4')

    gc_new = GradCAM(model_new, target_layer_name='layer4')
    gcpp_new = GradCAMPlusPlus(model_new, target_layer_name='layer4')

    transform = get_transforms('test')
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    total_overall = 0
    total_diseased = 0

    hits_orig_gc, hits_orig_gcpp = 0, 0
    hits_new_gc, hits_new_gcpp = 0, 0

    per_class = {}
    comparison_cases = []

    for idx, row in test_df.iterrows():
        p = row['image_path']
        if not os.path.exists(p):
            continue
        true_g = int(row['severity_grade'])
        cls_name = row['disease_class']

        image = Image.open(p).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predictions
        with torch.no_grad():
            out_orig = model_orig(input_tensor)
            pred_orig_t, _ = corn_probs_to_grade(torch.sigmoid(out_orig))
            pred_orig = int(pred_orig_t.item())

            out_new = model_new(input_tensor)
            pred_new_t, _ = corn_probs_to_grade(torch.sigmoid(out_new))
            pred_new = int(pred_new_t.item())

        # Generate CAMs
        cam_orig_gc = gc_orig.generate(input_tensor, target_class=pred_orig)
        cam_orig_gcpp = gcpp_orig.generate(input_tensor, target_class=pred_orig)

        cam_new_gc = gc_new.generate(input_tensor, target_class=pred_new)
        cam_new_gcpp = gcpp_new.generate(input_tensor, target_class=pred_new)

        # Ground truth symptom mask
        w, h = image.size
        scale = 800 / max(w, h) if max(w, h) > 800 else 1.0
        img_for_mask = image.resize((int(w * scale), int(h * scale))) if scale < 1.0 else image
        symptom_mask = get_symptom_mask(np.array(img_for_mask), cls_name)
        if symptom_mask is None:
            continue

        mask_224 = np.array(Image.fromarray((symptom_mask * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)) > 0

        # Pointing evaluation
        def check_hit(cam):
            max_idx = np.argmax(cam)
            r, c = np.unravel_index(max_idx, cam.shape)
            return bool(mask_224[r, c]), (r, c)

        hit_orig_gc, peak_orig_gc = check_hit(cam_orig_gc)
        hit_orig_gcpp, peak_orig_gcpp = check_hit(cam_orig_gcpp)

        hit_new_gc, peak_new_gc = check_hit(cam_new_gc)
        hit_new_gcpp, peak_new_gcpp = check_hit(cam_new_gcpp)

        is_diseased = (true_g > 0 and cls_name not in ['normal', 'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'])

        total_overall += 1
        if hit_orig_gc: hits_orig_gc += 1
        if hit_orig_gcpp: hits_orig_gcpp += 1
        if hit_new_gc: hits_new_gc += 1
        if hit_new_gcpp: hits_new_gcpp += 1

        if is_diseased:
            total_diseased += 1

        if cls_name not in per_class:
            per_class[cls_name] = {
                'total': 0, 'diseased_total': 0,
                'orig_gc_hits': 0, 'orig_gcpp_hits': 0,
                'new_gc_hits': 0, 'new_gcpp_hits': 0,
                'diseased_orig_gc_hits': 0, 'diseased_orig_gcpp_hits': 0,
                'diseased_new_gc_hits': 0, 'diseased_new_gcpp_hits': 0
            }

        per_class[cls_name]['total'] += 1
        if hit_orig_gc: per_class[cls_name]['orig_gc_hits'] += 1
        if hit_orig_gcpp: per_class[cls_name]['orig_gcpp_hits'] += 1
        if hit_new_gc: per_class[cls_name]['new_gc_hits'] += 1
        if hit_new_gcpp: per_class[cls_name]['new_gcpp_hits'] += 1

        if is_diseased:
            per_class[cls_name]['diseased_total'] += 1
            if hit_orig_gc: per_class[cls_name]['diseased_orig_gc_hits'] += 1
            if hit_orig_gcpp: per_class[cls_name]['diseased_orig_gcpp_hits'] += 1
            if hit_new_gc: per_class[cls_name]['diseased_new_gc_hits'] += 1
            if hit_new_gcpp: per_class[cls_name]['diseased_new_gcpp_hits'] += 1

        # Keep candidates for visual gallery
        diff_score = abs(int(hit_new_gc) - int(hit_orig_gc)) + abs(int(hit_new_gcpp) - int(hit_orig_gcpp))
        comparison_cases.append({
            'image': image,
            'mask_224': mask_224,
            'cam_orig_gc': cam_orig_gc,
            'cam_new_gc': cam_new_gc,
            'hit_orig_gc': hit_orig_gc,
            'hit_new_gc': hit_new_gc,
            'peak_orig_gc': peak_orig_gc,
            'peak_new_gc': peak_new_gc,
            'disease_class': cls_name,
            'true_grade': true_g,
            'pred_orig': pred_orig,
            'pred_new': pred_new,
            'diff_score': diff_score,
            'is_diseased': is_diseased,
            'path': p
        })

    # Summary rates
    def calc_rate(hits, tot):
        return round(hits / tot * 100.0, 2) if tot > 0 else 0.0

    diseased_orig_gc_total = sum(per_class[c]['diseased_orig_gc_hits'] for c in per_class)
    diseased_orig_gcpp_total = sum(per_class[c]['diseased_orig_gcpp_hits'] for c in per_class)
    diseased_new_gc_total = sum(per_class[c]['diseased_new_gc_hits'] for c in per_class)
    diseased_new_gcpp_total = sum(per_class[c]['diseased_new_gcpp_hits'] for c in per_class)

    summary = {
        "total_test_samples": total_overall,
        "total_diseased_samples": total_diseased,
        "overall": {
            "orig_gradcam_pct": calc_rate(hits_orig_gc, total_overall),
            "orig_gradcam_plusplus_pct": calc_rate(hits_orig_gcpp, total_overall),
            "new_gradcam_pct": calc_rate(hits_new_gc, total_overall),
            "new_gradcam_plusplus_pct": calc_rate(hits_new_gcpp, total_overall)
        },
        "diseased": {
            "orig_gradcam_pct": calc_rate(diseased_orig_gc_total, total_diseased),
            "orig_gradcam_plusplus_pct": calc_rate(diseased_orig_gcpp_total, total_diseased),
            "new_gradcam_pct": calc_rate(diseased_new_gc_total, total_diseased),
            "new_gradcam_plusplus_pct": calc_rate(diseased_new_gcpp_total, total_diseased)
        },
        "per_class": {}
    }

    for c, s in per_class.items():
        summary["per_class"][c] = {
            "total_samples": s['total'],
            "diseased_samples": s['diseased_total'],
            "orig_gradcam_pct": calc_rate(s['orig_gc_hits'], s['total']),
            "new_gradcam_pct": calc_rate(s['new_gc_hits'], s['total']),
            "diseased_orig_gradcam_pct": calc_rate(s['diseased_orig_gc_hits'], s['diseased_total']) if s['diseased_total'] > 0 else None,
            "diseased_new_gradcam_pct": calc_rate(s['diseased_new_gc_hits'], s['diseased_total']) if s['diseased_total'] > 0 else None
        }

    # 4. Save 4-6 gallery figures
    # Prioritize cases where attention changed (hit_new != hit_orig) or diseased tissue
    comparison_cases.sort(key=lambda x: (x['diff_score'], int(x['is_diseased'])), reverse=True)
    gallery_files = []

    for i, item in enumerate(comparison_cases[:6]):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

        # Panel 1: Original
        axes[0].imshow(item['image'].resize((224, 224)))
        axes[0].set_title(f"Original Test Image\n({item['disease_class']})", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Symptom Mask
        img_arr = np.array(item['image'].resize((224, 224)))
        overlay_mask = img_arr.copy()
        overlay_mask[item['mask_224']] = [239, 68, 68]
        axes[1].imshow(overlay_mask)
        axes[1].set_title(f"Ground-Truth Mask\n(True Grade: {item['true_grade']})", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # Panel 3: Baseline CAM
        ov_orig = overlay_heatmap(item['image'].resize((224, 224)), item['cam_orig_gc'], alpha=0.5)
        axes[2].imshow(ov_orig)
        stat_orig = "HIT (+)" if item['hit_orig_gc'] else "MISS (-)"
        col_orig = "green" if item['hit_orig_gc'] else "red"
        axes[2].plot(item['peak_orig_gc'][1], item['peak_orig_gc'][0], 'w*', markersize=12, markeredgecolor='black')
        axes[2].set_title(f"Baseline Model CAM: {stat_orig}\n(Pred Grade: {item['pred_orig']})", fontsize=11, fontweight='bold', color=col_orig)
        axes[2].axis('off')

        # Panel 4: Background-Randomized Model CAM
        ov_new = overlay_heatmap(item['image'].resize((224, 224)), item['cam_new_gc'], alpha=0.5)
        axes[3].imshow(ov_new)
        stat_new = "HIT (+)" if item['hit_new_gc'] else "MISS (-)"
        col_new = "green" if item['hit_new_gc'] else "red"
        axes[3].plot(item['peak_new_gc'][1], item['peak_new_gc'][0], 'w*', markersize=12, markeredgecolor='black')
        axes[3].set_title(f"Bg-Randomized Model CAM: {stat_new}\n(Pred Grade: {item['pred_new']})", fontsize=11, fontweight='bold', color=col_new)
        axes[3].axis('off')

        title_info = f"Arecanut {item['disease_class']} | True G{item['true_grade']} | Baseline Pred: G{item['pred_orig']} | Bg-Rand Pred: G{item['pred_new']}"
        plt.suptitle(title_info, fontsize=12, y=0.98)
        plt.tight_layout()

        out_fname = f"arecanut_bg_comp_{item['disease_class']}_trueG{item['true_grade']}_ex{i+1}.png"
        out_fpath = os.path.join(GALLERY_DIR, out_fname)
        fig.savefig(out_fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gallery_files.append(out_fpath)
        print(f"Saved gallery comparison: {out_fpath}")

    summary["gallery_images"] = gallery_files
    return summary


def main():
    print("=" * 60)
    print("  STEP 2: RETRAINING ARECANUT RESNET-18 CORN ACROSS 3 SEEDS")
    print("  (Training-time Background Randomization Enabled)")
    print("=" * 60)

    # 1. Preload datasets
    # Training: with background randomization
    train_ds = BgRandomizedDataset(config.ARECANUT_SPLIT_CSV, 'train', randomize_bg=True)
    # Validation & Test: clean real backgrounds
    val_ds = BgRandomizedDataset(config.ARECANUT_SPLIT_CSV, 'val', randomize_bg=False)
    test_ds = BgRandomizedDataset(config.ARECANUT_SPLIT_CSV, 'test', randomize_bg=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    PATIENCE = 5
    MAX_EPOCHS = config.NUM_EPOCHS
    SEEDS = [42, 100, 2024]

    training_results = {}

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---", flush=True)
        set_seed(seed)
        t_seed = time.time()

        model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=True, pretrained=True).to(device)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        loss_fn = CornLoss(num_classes=config.NUM_SEVERITY_GRADES)

        best_val_qwk = -1.0
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(MAX_EPOCHS):
            ep_t0 = time.time()
            train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_qwk = validate(model, val_loader, loss_fn)
            ep_time = time.time() - ep_t0

            if val_qwk > best_val_qwk:
                best_val_qwk = val_qwk
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                mark = " *BEST*"
            else:
                no_improve += 1
                mark = ""

            print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} ({ep_time:.1f}s) | Train QWK: {train_qwk:.4f} | Val QWK: {val_qwk:.4f}{mark}", flush=True)

            if no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1} (patience={PATIENCE})", flush=True)
                break

        # Save checkpoint
        ckpt_path = os.path.join(IMPROVEMENTS_DIR, f"arecanut_resnet18_corn_bgrand_seed{seed}.pth")
        torch.save(best_state, ckpt_path)

        # Evaluate on test set
        model.load_state_dict(best_state)
        eval_metrics = evaluate_model(model, test_loader)
        seed_time = time.time() - t_seed

        print(f"Seed {seed} finished ({seed_time:.1f}s): Best Val QWK: {best_val_qwk:.4f} @ Ep {best_epoch} | "
              f"Test QWK: {eval_metrics['qwk']:.4f} | Exact: {eval_metrics['exact_accuracy']:.4f} | "
              f"G2 F1: {eval_metrics['per_grade']['Grade_2']['f1']:.4f} | G3 F1: {eval_metrics['per_grade']['Grade_3']['f1']:.4f}", flush=True)

        training_results[str(seed)] = {
            "seed": seed,
            "epochs_run": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_qwk": round(best_val_qwk, 4),
            "runtime_seconds": round(seed_time, 2),
            "checkpoint_path": ckpt_path,
            **eval_metrics
        }

    # Compute 3-seed Mean and SD
    qwks = [training_results[str(s)]["qwk"] for s in SEEDS]
    mf1s = [training_results[str(s)]["macro_f1"] for s in SEEDS]
    maes = [training_results[str(s)]["mae"] for s in SEEDS]
    accs = [training_results[str(s)]["exact_accuracy"] for s in SEEDS]
    g2_f1s = [training_results[str(s)]["per_grade"]["Grade_2"]["f1"] for s in SEEDS]
    g3_f1s = [training_results[str(s)]["per_grade"]["Grade_3"]["f1"] for s in SEEDS]

    training_summary = {
        "qwk_mean": round(float(np.mean(qwks)), 4),
        "qwk_std": round(float(np.std(qwks, ddof=1)), 4),
        "macro_f1_mean": round(float(np.mean(mf1s)), 4),
        "macro_f1_std": round(float(np.std(mf1s, ddof=1)), 4),
        "mae_mean": round(float(np.mean(maes)), 4),
        "mae_std": round(float(np.std(maes, ddof=1)), 4),
        "exact_accuracy_mean": round(float(np.mean(accs)), 4),
        "exact_accuracy_std": round(float(np.std(accs, ddof=1)), 4),
        "grade_2_f1_mean": round(float(np.mean(g2_f1s)), 4),
        "grade_2_f1_std": round(float(np.std(g2_f1s, ddof=1)), 4),
        "grade_3_f1_mean": round(float(np.mean(g3_f1s)), 4),
        "grade_3_f1_std": round(float(np.std(g3_f1s, ddof=1)), 4)
    }

    # Reference original unweighted metrics (3 seeds)
    unweighted_ref = {
        "qwk_mean": 0.8552,
        "qwk_std": 0.0733,
        "macro_f1_mean": 0.8073,
        "macro_f1_std": 0.0567,
        "seed42": {"qwk": 0.9177, "macro_f1": 0.8698, "exact_acc": 0.9032, "g2_f1": 0.8235, "g3_f1": 0.7805},
        "seed100": {"qwk": 0.7745, "macro_f1": 0.7594, "exact_acc": 0.7984, "g2_f1": 0.6923, "g3_f1": 0.7111},
        "seed2024": {"qwk": 0.8734, "macro_f1": 0.7927, "exact_acc": 0.8468, "g2_f1": 0.6857, "g3_f1": 0.6667}
    }

    train_out_file = os.path.join(IMPROVEMENTS_DIR, "bg_randomization_training_results.json")
    with open(train_out_file, 'w') as f:
        json.dump({
            "experiment": "Arecanut ResNet-18 CORN with Training-Time Background Randomization",
            "hyperparameters": {
                "backbone": "resnet18",
                "loss": "CORN",
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "weight_decay": config.WEIGHT_DECAY,
                "patience": PATIENCE,
                "max_epochs": MAX_EPOCHS,
                "seeds": SEEDS
            },
            "per_seed_results": training_results,
            "summary_mean_std": training_summary,
            "original_unweighted_reference": unweighted_ref
        }, f, indent=2)
    print(f"\nSaved training results to {train_out_file}")

    # 2. Re-run Pointing Game Protocol on Seed 42 Checkpoint
    orig_ckpt = os.path.join(config.MODELS_DIR, "arecanut_ordinal.pth")
    new_ckpt = training_results["42"]["checkpoint_path"]
    pointing_game_summary = run_pointing_game_comparison(orig_ckpt, new_ckpt, config.ARECANUT_SPLIT_CSV)

    pointing_out_file = os.path.join(IMPROVEMENTS_DIR, "bg_randomization_pointing_game.json")
    with open(pointing_out_file, 'w') as f:
        json.dump({
            "experiment": "Pointing-Game Comparison: Baseline vs Background-Randomized Model",
            "models_evaluated": {
                "original_baseline": orig_ckpt,
                "background_randomized": new_ckpt
            },
            **pointing_game_summary
        }, f, indent=2)
    print(f"Saved pointing game results to {pointing_out_file}")

    # 3. Determine Verdict
    # Compare pointing game accuracy on diseased tissue
    orig_pg = pointing_game_summary["diseased"]["orig_gradcam_pct"]
    new_pg = pointing_game_summary["diseased"]["new_gradcam_pct"]
    new_qwk_mean = training_summary["qwk_mean"]
    orig_qwk_mean = unweighted_ref["qwk_mean"]

    print("\n" + "=" * 60)
    print(f"POINTING GAME (DISEASED TISSUE): Original = {orig_pg}% | New = {new_pg}%")
    print(f"QWK (3-SEED MEAN): Original = {orig_qwk_mean} ± {unweighted_ref['qwk_std']} | New = {new_qwk_mean} ± {training_summary['qwk_std']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
