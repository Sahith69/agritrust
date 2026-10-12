"""
Phase 4 Threshold Sensitivity: recompute per-class Grad-CAM IoU at
threshold=0.3 and threshold=0.5 on our actual paddy test data.
This gives a real, within-data comparison rather than citing literature ranges.

Runs on the paddy ordinal model only (20 samples/class, same seed as full eval).
"""
import sys; sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from src.model import CropSeverityModel
from src.gradcam import GradCAM
from src.dataset import get_transforms
from src.segmentation import (
    segment_leaf_adaptive, detect_lesions_group_a,
    detect_dead_heart, detect_hispa
)

def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def get_symptom_mask(img_np, disease_class):
    try:
        leaf_mask = segment_leaf_adaptive(img_np)
        if disease_class == 'dead_heart':
            return detect_dead_heart(img_np, leaf_mask)
        elif disease_class == 'hispa':
            return detect_hispa(img_np, leaf_mask)
        elif disease_class in config.PADDY_GROUP_A or disease_class in config.ARECANUT_GROUP_A:
            return detect_lesions_group_a(img_np, leaf_mask)
        elif disease_class == 'normal':
            return leaf_mask
        else:
            return detect_lesions_group_a(img_np, leaf_mask)
    except Exception:
        return None

def run_threshold_comparison(split_csv, crop_name, samples_per_class=20):
    device = torch.device('cpu')
    model = CropSeverityModel(
        backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES,
        ordinal=True, pretrained=False
    )
    model_path = os.path.join(config.MODELS_DIR, f'{crop_name}_ordinal.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    transform = get_transforms('test')
    gradcam = GradCAM(model, config.GRADCAM_TARGET_LAYER)

    df = pd.read_csv(split_csv)
    df_test = df[df['split'] == 'test'].copy()

    THRESHOLDS = [0.3, 0.5]
    results = {}  # cls -> {thresh -> [iou values]}

    for cls_name in sorted(df_test['disease_class'].unique()):
        cls_df = df_test[df_test['disease_class'] == cls_name]
        sample_df = cls_df.sample(
            min(samples_per_class, len(cls_df)),
            random_state=config.RANDOM_SEED
        )
        per_thresh = {t: [] for t in THRESHOLDS}

        for _, row in sample_df.iterrows():
            img_path = row['image_path']
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert('RGB')
            img_np = np.array(image)

            input_tensor = transform(image).unsqueeze(0)
            heatmap = gradcam.generate(input_tensor)  # (224,224), float in [0,1]

            symptom_mask = get_symptom_mask(img_np, cls_name)
            if symptom_mask is None:
                continue

            # Resize mask to heatmap size
            if heatmap.shape != symptom_mask.shape:
                m_img = Image.fromarray((symptom_mask * 255).astype(np.uint8))
                m_img = m_img.resize(heatmap.shape[::-1], Image.NEAREST)
                symptom_mask = np.array(m_img) > 0

            for thresh in THRESHOLDS:
                cam_binary = heatmap > thresh
                per_thresh[thresh].append(iou(cam_binary, symptom_mask))

        results[cls_name] = per_thresh

    return results

def print_table(crop_name, results):
    THRESHOLDS = [0.3, 0.5]
    print(f"\n{'='*72}")
    print(f"  {crop_name.upper()} — Per-class IoU at threshold=0.3 vs 0.5")
    print(f"{'='*72}")
    header = f"{'Class':<35} {'N':>4}  {'Mean@0.3':>9} {'Mean@0.5':>9}  {'Δ (0.3−0.5)':>12}  {'Max@0.3':>8} {'Max@0.5':>8}"
    print(header)
    print("-" * 72)

    all_means = {t: [] for t in THRESHOLDS}
    rows_for_json = []

    for cls_name, per_thresh in sorted(results.items()):
        vals_03 = per_thresh[0.3]
        vals_05 = per_thresh[0.5]
        if not vals_03:
            continue
        n = len(vals_03)
        m03 = np.mean(vals_03)
        m05 = np.mean(vals_05) if vals_05 else float('nan')
        delta = m03 - m05
        mx03 = np.max(vals_03)
        mx05 = np.max(vals_05) if vals_05 else float('nan')
        all_means[0.3].append(m03)
        all_means[0.5].append(m05)
        print(f"{cls_name:<35} {n:>4}  {m03:>9.4f} {m05:>9.4f}  {delta:>+12.4f}  {mx03:>8.4f} {mx05:>8.4f}")
        rows_for_json.append({
            'class': cls_name, 'n': n,
            'mean_03': round(m03, 4), 'mean_05': round(m05, 4),
            'delta': round(delta, 4),
            'std_03': round(np.std(vals_03), 4),
            'std_05': round(np.std(vals_05), 4) if vals_05 else None,
            'min_03': round(np.min(vals_03), 4),
            'min_05': round(np.min(vals_05), 4) if vals_05 else None,
            'max_03': round(mx03, 4),
            'max_05': round(mx05, 4),
        })

    print("-" * 72)
    ov03 = np.mean(all_means[0.3])
    ov05 = np.mean(all_means[0.5])
    print(f"{'OVERALL MEAN':<35} {'':>4}  {ov03:>9.4f} {ov05:>9.4f}  {ov03-ov05:>+12.4f}")
    print(f"\n  Threshold 0.3 lifts overall mean IoU by {(ov03-ov05)*100:+.1f} percentage points")
    print(f"  = {(ov03/ov05 - 1)*100:+.1f}% relative improvement over threshold 0.5")
    return rows_for_json

def main():
    print("Running threshold sensitivity comparison (20 samples/class)...")
    print("This measures how much of our low IoU is threshold artifact vs genuine misalignment.\n")

    paddy_results = run_threshold_comparison(config.PADDY_SPLIT_CSV, 'paddy')
    paddy_rows = print_table('Paddy', paddy_results)

    areca_results = run_threshold_comparison(config.ARECANUT_SPLIT_CSV, 'arecanut')
    areca_rows = print_table('Arecanut', areca_results)

    # Save to JSON for the report
    out = {'paddy': paddy_rows, 'arecanut': areca_rows}
    out_path = os.path.join(config.REPORTS_DIR, 'phase4_threshold_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
