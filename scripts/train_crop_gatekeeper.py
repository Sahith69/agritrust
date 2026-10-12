"""
Train a 3-way Crop Gatekeeper & OOD Detection Model:
- Class 0: Paddy (Paddy rice leaves / diseases)
- Class 1: Arecanut (Arecanut palm leaves, trunks, nuts)
- Class 2: Other / Non-Crop (Everyday objects, textures, backgrounds, non-plants)
"""
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

import os
import glob
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

torch.set_num_threads(4)
device = torch.device('cpu')

MODELS_DIR = os.path.join(config.OUTPUTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
GATEKEEPER_PATH = os.path.join(MODELS_DIR, "crop_gatekeeper.pth")


def get_gatekeeper_dataset():
    # 1. Paddy images
    p_df = pd.read_csv(config.PADDY_SPLIT_CSV)
    p_train = p_df[p_df['split'] == 'train']['image_path'].tolist()
    random.seed(42)
    random.shuffle(p_train)
    paddy_samples = p_train[:600]

    # 2. Arecanut images
    a_df = pd.read_csv(config.ARECANUT_SPLIT_CSV)
    a_train = a_df[a_df['split'] == 'train']['image_path'].tolist()
    random.shuffle(a_train)
    # If fewer than 600, repeat slightly to balance
    arecanut_samples = (a_train * 2)[:600]

    # 3. Non-crop images from system
    non_crop_candidates = []
    # Search system wallpapers, icons, photos
    for pat in [
        '/home/sahith/.cache/wallpaper-thumbs/*.jpg',
        '/home/sahith/Projects/c/*.png',
        '/usr/share/kmix/pics/*.png',
        '/usr/share/icons/hicolor/128x128/apps/*.png',
        '/usr/share/icons/hicolor/256x256/apps/*.png',
        '/usr/share/wallpapers/**/*.jpg',
        '/usr/share/backgrounds/**/*.jpg',
        '/usr/share/pixmaps/*.png'
    ]:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p):
                try:
                    with Image.open(p) as im:
                        if im.size[0] >= 64 and im.size[1] >= 64:
                            non_crop_candidates.append(p)
                except Exception:
                    pass

    random.shuffle(non_crop_candidates)
    non_crop_samples = non_crop_candidates[:600]
    
    # If not enough system images, synthesize diverse non-crop patterns
    while len(non_crop_samples) < 600:
        non_crop_samples.extend(non_crop_candidates[:600 - len(non_crop_samples)])

    print(f"Dataset compiled: Paddy={len(paddy_samples)}, Arecanut={len(arecanut_samples)}, Non-Crop={len(non_crop_samples)}")

    data = []
    for p in paddy_samples:
        data.append((p, 0))  # 0: Paddy
    for p in arecanut_samples:
        data.append((p, 1))  # 1: Arecanut
    for p in non_crop_samples:
        data.append((p, 2))  # 2: Non-Crop

    random.shuffle(data)
    # 80/20 train/val split
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]


class SimpleImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Fallback blank image if file read error
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def train_gatekeeper():
    print("=" * 60)
    print("  TRAINING CROP GATEKEEPER & OOD DETECTOR")
    print("=" * 60)
    t0 = time.time()

    train_data, val_data = get_gatekeeper_dataset()

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    train_loader = DataLoader(SimpleImageDataset(train_data, train_tf), batch_size=32, shuffle=True)
    val_loader = DataLoader(SimpleImageDataset(val_data, val_tf), batch_size=32, shuffle=False)

    # Use pretrained ResNet-18 with 3 output classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    best_acc = 0.0
    best_weights = None

    for epoch in range(4):
        ep_t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/4 ({time.time()-ep_t0:.1f}s) | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict()

    # Save model and metadata
    save_dict = {
        'model_state_dict': best_weights,
        'classes': ['Paddy', 'Arecanut', 'Non-Crop / Other'],
        'best_val_acc': best_acc,
        'timestamp': time.time()
    }
    torch.save(save_dict, GATEKEEPER_PATH)
    print(f"\nGatekeeper training complete in {time.time()-t0:.1f}s! Best Val Acc: {best_acc*100:.2f}%")
    print(f"Saved model to {GATEKEEPER_PATH}")


if __name__ == '__main__':
    train_gatekeeper()
