import os
import sys
import glob
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.covariance import LedoitWolf

import torch
import torch.nn as nn
from torchvision import models, transforms

import config

OOD_PARAMS_FILE = os.path.join(config.OUTPUTS_DIR, "ood_params.pt")

def get_feature_extractor(crop_name: str):
    key = crop_name.lower()
    pth = os.path.join(config.OUTPUTS_DIR, "models", f"{key}_disease_classifier.pth")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Model checkpoint not found: {pth}")

    ck = torch.load(pth, map_location="cpu", weights_only=True)
    classes = ck["classes"]
    model = models.resnet18(weights=None)
    if "fc.1.weight" in ck["model_state_dict"]:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.fc.in_features, len(classes)))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(ck["model_state_dict"])
    model.fc = nn.Identity()
    model.eval()
    return model, classes

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

def extract_features_from_paths(model, image_paths, batch_size: int = 64):
    tf = get_image_transform()
    features_list = []
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            with Image.open(p) as img:
                tensors.append(tf(img.convert("RGB")))
        batch_tensor = torch.stack(tensors, dim=0)
        with torch.no_grad():
            feat = model(batch_tensor).cpu().numpy()
        features_list.append(feat)

    if len(features_list) == 0:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(features_list, axis=0)

def compute_mahalanobis_scores(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """
    Computes min-class Mahalanobis distance for each feature vector.
    features: (N, 512)
    means: (C, 512)
    precision: (512, 512)
    returns: (N,) minimum Mahalanobis distance to any class mean
    """
    N, D = features.shape
    C = means.shape[0]
    class_dists = np.zeros((N, C), dtype=np.float32)

    for c in range(C):
        diff = features - means[c]  # (N, 512)
        # (diff @ precision) * diff
        term = np.sum((diff @ precision) * diff, axis=1)
        term = np.maximum(term, 0.0)  # numerical safety
        class_dists[:, c] = np.sqrt(term)

    min_dists = np.min(class_dists, axis=1)
    return min_dists

def fit_crop_ood(crop_name: str, cache_dir: str = "outputs"):
    print(f"\n========================================================")
    print(f"Fitting Mahalanobis OOD Guard for {crop_name.upper()}...")
    print(f"========================================================")
    model, classes = get_feature_extractor(crop_name)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    csv_path = os.path.join(config.OUTPUTS_DIR, f"{crop_name.lower()}_splits.csv")
    df = pd.read_csv(csv_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    cache_file = os.path.join(cache_dir, f"ood_features_{crop_name.lower()}.npz")
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        data = np.load(cache_file)
        train_features = data["train_features"]
        train_labels = data["train_labels"]
        val_features = data["val_features"]
        test_features = data["test_features"]
    else:
        print(f"Extracting 512-d penultimate features for {len(train_df)} TRAIN images...")
        t0 = time.time()
        train_features = extract_features_from_paths(model, train_df["image_path"].tolist())
        train_labels = np.array([class_to_idx[c] for c in train_df["disease_class"].tolist()], dtype=np.int64)
        print(f"  Extracted TRAIN in {time.time() - t0:.2f}s")

        print(f"Extracting features for {len(val_df)} VAL images...")
        t0 = time.time()
        val_features = extract_features_from_paths(model, val_df["image_path"].tolist())
        print(f"  Extracted VAL in {time.time() - t0:.2f}s")

        print(f"Extracting features for {len(test_df)} TEST images...")
        t0 = time.time()
        test_features = extract_features_from_paths(model, test_df["image_path"].tolist())
        print(f"  Extracted TEST in {time.time() - t0:.2f}s")

        np.savez_compressed(
            cache_file,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            test_features=test_features
        )
        print(f"Saved feature cache to {cache_file}")

    # 1. Class-conditional means for TRAIN images only
    num_classes = len(classes)
    means = np.zeros((num_classes, 512), dtype=np.float32)
    for c in range(num_classes):
        mask = (train_labels == c)
        if np.sum(mask) == 0:
            raise ValueError(f"Class {classes[c]} has 0 training images!")
        means[c] = np.mean(train_features[mask], axis=0)

    # 2. Shared covariance with Ledoit-Wolf shrinkage on centered residuals
    residuals = np.zeros_like(train_features)
    for c in range(num_classes):
        mask = (train_labels == c)
        residuals[mask] = train_features[mask] - means[c]

    print("Fitting Ledoit-Wolf shrinkage covariance on centered TRAIN residuals...")
    lw = LedoitWolf(assume_centered=True).fit(residuals)
    precision = lw.precision_.astype(np.float32)
    shrinkage = lw.shrinkage_
    print(f"  Ledoit-Wolf shrinkage factor: {shrinkage}")

    # 3. Compute OOD score on VAL set to determine threshold (95% pass)
    val_scores = compute_mahalanobis_scores(val_features, means, precision)
    # Threshold such that 95% of VAL images pass (score <= threshold)
    threshold = float(np.percentile(val_scores, 95.0))
    val_passed = (val_scores <= threshold)
    val_pass_fraction = float(np.mean(val_passed))
    print(f"VAL score distribution: min={np.min(val_scores):.4f}, median={np.median(val_scores):.4f}, 95th={threshold:.4f}, max={np.max(val_scores):.4f}")
    print(f"OOD Threshold set to {threshold:.6f} -> VAL pass rate: {np.sum(val_passed)} / {len(val_scores)} ({val_pass_fraction * 100:.2f}%)")

    # In-domain test scores
    test_scores = compute_mahalanobis_scores(test_features, means, precision)

    return {
        "classes": classes,
        "means": torch.tensor(means, dtype=torch.float32),
        "precision": torch.tensor(precision, dtype=torch.float32),
        "threshold": threshold,
        "val_pass_rate": val_pass_fraction,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "shrinkage": shrinkage
    }

def fit_all_crops_and_save():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    all_params = {}
    for crop in ["paddy", "arecanut", "apple"]:
        all_params[crop] = fit_crop_ood(crop)

    torch.save(all_params, OOD_PARAMS_FILE)
    print(f"\nAll OOD guard parameters successfully saved to {OOD_PARAMS_FILE}")
    return all_params

# Global cache for inference
_OOD_MODELS = {}
_OOD_PARAMS = None

def load_ood_params():
    global _OOD_PARAMS
    if _OOD_PARAMS is None:
        if not os.path.exists(OOD_PARAMS_FILE):
            print(f"OOD params file not found at {OOD_PARAMS_FILE}, fitting now...")
            _OOD_PARAMS = fit_all_crops_and_save()
        else:
            _OOD_PARAMS = torch.load(OOD_PARAMS_FILE, map_location="cpu", weights_only=False)
    return _OOD_PARAMS

def check_ood(image: Image.Image, crop_name: str):
    """
    Inference entrypoint for app.py.
    Returns:
      is_ood (bool): True if score > threshold (image is rejected as OOD)
      ood_score (float): minimum Mahalanobis distance
      threshold (float): calibrated 95% VAL threshold
    """
    key = crop_name.lower()
    params = load_ood_params()
    if key not in params:
        raise ValueError(f"Unknown crop: {crop_name}")

    crop_data = params[key]
    means = crop_data["means"].numpy()
    precision = crop_data["precision"].numpy()
    threshold = crop_data["threshold"]

    global _OOD_MODELS
    if key not in _OOD_MODELS:
        model, _ = get_feature_extractor(crop_name)
        _OOD_MODELS[key] = model
    model = _OOD_MODELS[key]

    tf = get_image_transform()
    t = tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = model(t).cpu().numpy()  # (1, 512)

    score = float(compute_mahalanobis_scores(feat, means, precision)[0])
    is_ood = bool(score > threshold)
    return is_ood, score, threshold
