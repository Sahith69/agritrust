import os
import sys
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score, f1_score

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms

torch.set_num_threads(8)
device = torch.device('cpu')

PROJECT_ROOT = "/home/sahith/Projects/Crop"
SPLITS_CSV = os.path.join(PROJECT_ROOT, "outputs/improvements/arecanut_splits_merged_expanded.csv")
SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs/models/merged_expanded/arecanut_merged_softmax_seed42_patience15.pth")

class FastCachedDataset(Dataset):
    def __init__(self, df, split):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.labels = [int(g) for g in self.df['severity_grade']]
        
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

def train_one_epoch(model, loader, optimizer, loss_fn):
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
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return total_loss / len(loader.dataset), compute_qwk(all_targets, all_preds)

def validate(model, loader, loss_fn):
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
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return total_loss / len(loader.dataset), compute_qwk(all_targets, all_preds)

def evaluate_on_test(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    acc = accuracy_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    diff = np.abs(all_targets - all_preds)
    w1 = np.mean(diff <= 1) * 100
    cat = np.mean(diff >= 2) * 100
    return qwk, acc, mae, f1, w1, cat

def main():
    print("=" * 75)
    print("  RERUN SOFTMAX SEED 42 WITH PATIENCE = 15")
    print("=" * 75)
    
    df_splits = pd.read_csv(SPLITS_CSV)
    train_dataset = FastCachedDataset(df_splits, 'train')
    val_dataset = FastCachedDataset(df_splits, 'val')
    test_dataset = FastCachedDataset(df_splits, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    set_seed(42)
    model = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_val_qwk = -1.0
    best_epoch = -1
    best_weights = None
    epochs_no_improve = 0
    PATIENCE = 15
    MAX_EPOCHS = 30
    
    print(f"\nStarting training (Max Epochs: {MAX_EPOCHS}, Patience: {PATIENCE})...", flush=True)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_qwk = validate(model, val_loader, loss_fn)
        dur = time.time() - t0
        
        improved = ""
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_epoch = epoch
            import copy
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            improved = " --> New Best Val QWK!"
        else:
            epochs_no_improve += 1
            
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} ({dur:.1f}s) | Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f} (Patience: {epochs_no_improve}/{PATIENCE}){improved}", flush=True)
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch} (patience={PATIENCE}).", flush=True)
            break
            
    print(f"\nBest Validation QWK: {best_val_qwk:.4f} at Epoch {best_epoch}.", flush=True)
    torch.save(best_weights, SAVE_PATH)
    model.load_state_dict(best_weights)
    
    qwk, acc, mae, f1, w1, cat = evaluate_on_test(model, test_loader)
    print(f"\nTest Evaluation (Patience 15): QWK = {qwk:.4f} | Acc = {acc*100:.2f}% | MAE = {mae:.4f} | Macro-F1 = {f1:.4f} | Within-1 = {w1:.2f}% | Catastrophic = {cat:.2f}%", flush=True)

if __name__ == '__main__':
    main()
