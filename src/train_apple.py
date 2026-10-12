import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score

from src.model import CropSeverityModel
from src.corn_loss import CornLoss, corn_probs_to_grade

CLASSES = ["apple_scab", "black_rot", "cedar_apple_rust", "healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class AppleDataset(Dataset):
    def __init__(self, split_csv, split, transform=None):
        df_all = pd.read_csv(split_csv)
        self.df = df_all[df_all['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        disease_label = CLASS_TO_IDX[row['disease_class']]
        severity_label = int(row['severity_grade'])
        
        if self.transform:
            img = self.transform(img)
            
        return img, disease_label, severity_label

def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])

# ─── 1. TRAIN APPLE DISEASE CLASSIFIER ───────────────────────────────────────
def train_apple_disease_classifier(split_csv, epochs=10):
    print("=" * 70)
    print("TRAINING APPLE DISEASE CLASSIFIER (ResNet-18)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_ds = AppleDataset(split_csv, 'train', get_transforms('train'))
    val_ds   = AppleDataset(split_csv, 'val', get_transforms('val'))
    test_ds  = AppleDataset(split_csv, 'test', get_transforms('test'))
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        
        for imgs, d_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, d_labels = imgs.to(device), d_labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, d_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_targets.extend(d_labels.cpu().numpy())
            
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for imgs, d_labels, _ in val_loader:
                imgs, d_labels = imgs.to(device), d_labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, d_labels)
                val_loss += loss.item() * imgs.size(0)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(d_labels.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            
    # Save Model Checkpoint
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    save_path = os.path.join(config.MODELS_DIR, "apple_disease_classifier.pth")
    checkpoint = {
        'model_state_dict': best_state,
        'classes': CLASSES,
        'idx_to_class': {i: c for i, c in enumerate(CLASSES)}
    }
    torch.save(checkpoint, save_path)
    print(f"\nSaved Apple Disease Classifier Checkpoint to: {save_path} (Best Val Acc: {best_acc*100:.2f}%)")
    
    # Test Evaluation
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, d_labels, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_targets.extend(d_labels.numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    print(f"Test Accuracy: {test_acc*100:.2f}%\n")


# ─── 2. TRAIN APPLE ORDINAL SEVERITY GRADER (CORN LOSS) ─────────────────────
def train_apple_severity_grader(split_csv, epochs=10):
    print("=" * 70)
    print("TRAINING APPLE ORDINAL SEVERITY GRADER (ResNet-18 + CORN Loss)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = AppleDataset(split_csv, 'train', get_transforms('train'))
    val_ds   = AppleDataset(split_csv, 'val', get_transforms('val'))
    test_ds  = AppleDataset(split_csv, 'test', get_transforms('test'))
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=True)
    model = model.to(device)
    
    loss_fn = CornLoss(num_classes=config.NUM_SEVERITY_GRADES)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_qwk = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_preds, train_targets = [], []
        
        for imgs, _, s_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, s_labels = imgs.to(device), s_labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, s_labels)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(s_labels.cpu().numpy())
                
        train_qwk = cohen_kappa_score(train_targets, train_preds, weights='quadratic')
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, _, s_labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(s_labels.numpy())
                
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')
        print(f"Epoch {epoch:02d} | Train QWK: {train_qwk:.4f} | Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_state = model.state_dict()
            
    # Save Checkpoint
    save_path = os.path.join(config.MODELS_DIR, "apple_ordinal.pth")
    checkpoint = {'model_state_dict': best_state}
    torch.save(checkpoint, save_path)
    print(f"\nSaved Apple Severity Grader Checkpoint to: {save_path} (Best Val QWK: {best_qwk:.4f})")
    
    # Test Evaluation
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, _, s_labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(s_labels.numpy())
            
    test_qwk = cohen_kappa_score(test_targets, test_preds, weights='quadratic')
    print(f"Test QWK: {test_qwk:.4f}\n")

if __name__ == "__main__":
    split_csv = os.path.join(config.OUTPUTS_DIR, "apple_splits.csv")
    train_apple_disease_classifier(split_csv, epochs=5)
    train_apple_severity_grader(split_csv, epochs=5)
