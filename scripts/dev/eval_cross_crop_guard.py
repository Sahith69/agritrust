#!/usr/bin/env python3
"""
scripts/dev/eval_cross_crop_guard.py — Evaluates Cross-Crop Guard & OOD Guard.
Measures:
1. 200 images of each crop run in each wrong mode (2 wrong modes per crop).
2. 60 non-crop disk images in all 3 modes.
3. Every correct-crop test image (false block rate, per crop).
Compares Old OOD-only vs Cross-Crop Guard at margin=0.25 (and tightened margin if needed).
"""

import os
import sys
import glob
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from ood_guard import (
    load_ood_params,
    get_feature_extractor,
    get_image_transform,
    compute_mahalanobis_scores
)

# Deterministic seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

CROPS = ["paddy", "arecanut", "apple"]
DISPLAY_NAMES = {"paddy": "Paddy", "arecanut": "Arecanut", "apple": "Apple"}

def load_all_models_and_params():
    params = load_ood_params()
    models = {}
    for c in CROPS:
        m, _ = get_feature_extractor(c)
        models[c] = m
    return models, params

def extract_features_batch(model, img_tensors):
    with torch.no_grad():
        feats = model(img_tensors).cpu().numpy()
    return feats

def compute_all_ratios_for_paths(image_paths, models, params, batch_size=64):
    tf = get_image_transform()
    N = len(image_paths)
    all_scores = {c: np.zeros(N, dtype=np.float32) for c in CROPS}
    all_ratios = {c: np.zeros(N, dtype=np.float32) for c in CROPS}

    means = {c: params[c]["means"].numpy() for c in CROPS}
    precs = {c: params[c]["precision"].numpy() for c in CROPS}
    thresh = {c: params[c]["threshold"] for c in CROPS}

    for i in range(0, N, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            with Image.open(p) as img:
                tensors.append(tf(img.convert("RGB")))
        batch_tensor = torch.stack(tensors, dim=0)

        for c in CROPS:
            feats = extract_features_batch(models[c], batch_tensor)
            scores = compute_mahalanobis_scores(feats, means[c], precs[c])
            all_scores[c][i:i + len(batch_paths)] = scores
            all_ratios[c][i:i + len(batch_paths)] = scores / thresh[c]

    return all_scores, all_ratios

def evaluate_guard_decision(ratios_dict, selected_crop, margin=0.25):
    """
    Given ratios for all crops for a set of images:
    Returns:
      old_ood_blocked: array of bool
      cross_crop_blocked: array of bool
      total_blocked: array of bool (blocked by either cross-crop or old OOD)
      cross_crop_suggested: list of suggested crops
    """
    sel_r = ratios_dict[selected_crop]
    other_crops = [c for c in CROPS if c != selected_crop]

    other_ratios = [ratios_dict[oc] for oc in other_crops]
    # min ratio among other crops
    min_other_r = np.minimum(other_ratios[0], other_ratios[1])
    # which other crop is min
    best_other = np.where(other_ratios[0] < other_ratios[1], other_crops[0], other_crops[1])

    # Old OOD rule: ratio > 1.0 on selected crop
    old_ood_blocked = (sel_r > 1.0)

    # Cross-crop rule:
    # 1. selected ratio > 1.0 AND another crop ratio < 1.0
    cond1 = (sel_r > 1.0) & (min_other_r < 1.0)
    # 2. another crop ratio is below selected crop ratio by at least margin
    cond2 = (sel_r - min_other_r >= margin)

    # Note: keep existing OOD refusal for ratio above 1 for all crops
    # If ratio > 1 for ALL crops, it's generic OOD, not cross-crop!
    all_above_1 = (sel_r > 1.0) & (ratios_dict[other_crops[0]] > 1.0) & (ratios_dict[other_crops[1]] > 1.0)

    # Cross-crop block specifically:
    # Triggered when (cond1 or cond2) AND NOT all_above_1 (or even if all_above_1, total blocked is true)
    cross_crop_blocked = (cond1 | cond2) & (~all_above_1)

    # Total blocked = cross-crop blocked OR standard OOD blocked
    total_blocked = cross_crop_blocked | old_ood_blocked

    return old_ood_blocked, cross_crop_blocked, total_blocked, best_other

def main():
    print("Loading models and OOD parameters...")
    models, params = load_all_models_and_params()

    # 1. Load test splits
    split_dfs = {}
    for c in CROPS:
        csv_p = os.path.join(config.OUTPUTS_DIR, f"{c}_splits.csv")
        df = pd.read_csv(csv_p)
        split_dfs[c] = df[df["split"] == "test"].copy()
        print(f"Loaded {c.capitalize()} test split: {len(split_dfs[c])} images")

    # Sample 200 images for each crop
    sample_200 = {}
    for c in CROPS:
        paths = split_dfs[c]["image_path"].tolist()
        random.seed(42)
        sample_200[c] = random.sample(paths, 200)

    # Sample 60 non-crop images
    non_crop_candidates = []
    for pat in [
        "/usr/share/icons/hicolor/128x128/apps/*.png",
        "/usr/share/icons/hicolor/256x256/apps/*.png",
        "/usr/share/pixmaps/*.png",
        "/usr/share/wallpapers/**/*.jpg",
        "/usr/share/backgrounds/**/*.jpg",
    ]:
        for p in sorted(glob.glob(pat, recursive=True)):
            if os.path.isfile(p):
                try:
                    with Image.open(p) as im:
                        if im.size[0] >= 64 and im.size[1] >= 64:
                            non_crop_candidates.append(p)
                except Exception:
                    pass
    random.seed(42)
    non_crop_60 = sorted(random.sample(non_crop_candidates, 60))
    print(f"Sampled 60 non-crop images from system disk.")

    cache_path = os.path.join(config.OUTPUTS_DIR, "cross_crop_eval_cache.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached ratios from {cache_path}...")
        cdata = np.load(cache_path)
        full_ratios = {c: cdata[f"full_{c}"] for c in CROPS}
        sample_200_ratios = {c: cdata[f"sample200_{c}"] for c in CROPS}
        non_crop_ratios = {c: cdata[f"noncrop_{c}"] for c in CROPS}
    else:
        # Extract all ratios
        print("\n--- Extracting ratios for Correct-Crop Full Test Sets ---")
        full_ratios = {}
        for c in CROPS:
            t0 = time.time()
            paths = split_dfs[c]["image_path"].tolist()
            _, r = compute_all_ratios_for_paths(paths, models, params)
            full_ratios[c] = r
            print(f"  Extracted {c} test set ({len(paths)} imgs) in {time.time()-t0:.2f}s")

        print("\n--- Extracting ratios for 200-image evaluation subsets ---")
        sample_200_ratios = {}
        for c in CROPS:
            t0 = time.time()
            _, r = compute_all_ratios_for_paths(sample_200[c], models, params)
            sample_200_ratios[c] = r
            print(f"  Extracted {c} 200-sample in {time.time()-t0:.2f}s")

        print("\n--- Extracting ratios for 60 non-crop images ---")
        t0 = time.time()
        _, non_crop_ratios = compute_all_ratios_for_paths(non_crop_60, models, params)
        print(f"  Extracted 60 non-crop in {time.time()-t0:.2f}s")

        save_dict = {}
        for c in CROPS:
            save_dict[f"full_{c}"] = full_ratios[c]
            save_dict[f"sample200_{c}"] = sample_200_ratios[c]
            save_dict[f"noncrop_{c}"] = non_crop_ratios[c]
        np.savez_compressed(cache_path, **save_dict)
        print(f"Saved evaluation cache to {cache_path}")

    # Evaluate for margin = 0.25 and tightened margin = 0.30
    for margin in [0.25, 0.30]:
        print(f"\n========================================================")
        print(f"EVALUATION AT MARGIN = {margin}")
        print(f"========================================================")

        # A. False block rate on correct-crop test images
        print("\n--- 1. CORRECT-CROP TEST IMAGES (False Block Rate) ---")
        false_block_rates = {}
        for c in CROPS:
            old_b, cc_b, tot_b, _ = evaluate_guard_decision(full_ratios[c], c, margin=margin)
            N = len(tot_b)
            old_rate = np.mean(old_b)
            cc_rate = np.mean(cc_b)
            tot_rate = np.mean(tot_b)
            # False block specifically from cross-crop or total:
            false_block_rates[c] = (np.sum(tot_b), N, tot_rate, np.sum(old_b), old_rate, np.sum(cc_b), cc_rate)
            print(f"  {DISPLAY_NAMES[c]} ({N} test images):")
            print(f"    - Old OOD-only block rate: {np.sum(old_b)}/{N} ({old_rate * 100:.4f}%)")
            print(f"    - Cross-crop refusal rate: {np.sum(cc_b)}/{N} ({cc_rate * 100:.4f}%)")
            print(f"    - Total blocked rate:     {np.sum(tot_b)}/{N} ({tot_rate * 100:.4f}%)")

        # B. 200 images of each crop run in each wrong mode (2 wrong modes per crop)
        print("\n--- 2. WRONG-MODE EVALUATION (200 images per crop x 2 wrong modes) ---")
        wrong_mode_results = {}
        for true_c in CROPS:
            for tested_c in CROPS:
                if true_c == tested_c:
                    continue
                old_b, cc_b, tot_b, suggested = evaluate_guard_decision(sample_200_ratios[true_c], tested_c, margin=margin)
                N = len(tot_b)
                old_rate = np.mean(old_b)
                tot_rate = np.mean(tot_b)
                cc_rate = np.mean(cc_b)
                wrong_mode_results[(true_c, tested_c)] = (np.sum(tot_b), N, tot_rate, np.sum(old_b), old_rate, np.sum(cc_b), cc_rate)
                print(f"  True {DISPLAY_NAMES[true_c]} (200 imgs) run in {DISPLAY_NAMES[tested_c]} mode:")
                print(f"    - Old OOD block rate:       {np.sum(old_b)}/{N} ({old_rate * 100:.4f}%)")
                print(f"    - Cross-crop block rate:    {np.sum(cc_b)}/{N} ({cc_rate * 100:.4f}%)")
                print(f"    - Total block rate:         {np.sum(tot_b)}/{N} ({tot_rate * 100:.4f}%)")

        # C. 60 non-crop disk images in all 3 modes
        print("\n--- 3. NON-CROP DISK IMAGES (60 images in all 3 modes) ---")
        non_crop_results = {}
        for mode_c in CROPS:
            old_b, cc_b, tot_b, _ = evaluate_guard_decision(non_crop_ratios, mode_c, margin=margin)
            N = len(tot_b)
            old_rate = np.mean(old_b)
            tot_rate = np.mean(tot_b)
            cc_rate = np.mean(cc_b)
            non_crop_results[mode_c] = (np.sum(tot_b), N, tot_rate, np.sum(old_b), old_rate, np.sum(cc_b), cc_rate)
            print(f"  Non-Crop (60 imgs) run in {DISPLAY_NAMES[mode_c]} mode:")
            print(f"    - Old OOD block rate:       {np.sum(old_b)}/{N} ({old_rate * 100:.4f}%)")
            print(f"    - Cross-crop block rate:    {np.sum(cc_b)}/{N} ({cc_rate * 100:.4f}%)")
            print(f"    - Total block rate:         {np.sum(tot_b)}/{N} ({tot_rate * 100:.4f}%)")

if __name__ == "__main__":
    main()
