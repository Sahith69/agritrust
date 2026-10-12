"""
Train Disease Classifier for Paddy (10 classes) and Arecanut (9 classes).
No overlap with severity grader — this is a separate classification head.
Saves: outputs/models/paddy_disease_classifier.pth
        outputs/models/arecanut_disease_classifier.pth
"""
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cpu')
IMG_SIZE    = config.IMG_SIZE
BATCH_SIZE  = 24
LR          = 1e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 20
PATIENCE    = 5
SEED        = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── DATASET ─────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


class DiseaseDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.c2i = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.c2i[row['disease_class']]
        return img, label


# ─── MODEL ───────────────────────────────────────────────────────────────────
def build_classifier(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # Unfreeze last two blocks
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ─── TRAIN LOOP ──────────────────────────────────────────────────────────────
def train(crop_name, df_path, model_save_path, meta_save_path):
    df = pd.read_csv(df_path)
    classes = sorted(df['disease_class'].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(f"\n{'='*60}")
    print(f"  Training {crop_name} Disease Classifier")
    print(f"  Classes ({len(classes)}): {classes}")
    print(f"{'='*60}")

    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}")

    # Class weights for imbalance
    counts = train_df['disease_class'].value_counts()
    max_count = counts.max()
    weights_list = [max_count / counts[c] for c in classes]
    class_weights = torch.tensor(weights_list, dtype=torch.float).to(DEVICE)

    train_loader = DataLoader(
        DiseaseDataset(train_df, class_to_idx, TRAIN_TF),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False
    )
    val_loader = DataLoader(
        DiseaseDataset(val_df, class_to_idx, VAL_TF),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False
    )

    model     = build_classifier(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)
        scheduler.step()
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += len(labels)
        val_acc = 100 * v_correct / v_total

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Acc: {train_acc:.1f}%  "
              f"Val Acc: {val_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # Save
    model.load_state_dict(best_state)
    torch.save({
        'model_state_dict': best_state,
        'classes': classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'best_val_acc': best_val_acc,
        'crop': crop_name,
    }, model_save_path)

    with open(meta_save_path, 'w') as f:
        json.dump({'classes': classes, 'class_to_idx': class_to_idx,
                   'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
                   'best_val_acc': best_val_acc, 'crop': crop_name}, f, indent=2)

    print(f"\n  ✓ Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  ✓ Saved to: {model_save_path}")
    return best_val_acc


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("\nSTARTING DISEASE CLASSIFIER TRAINING")
    print("No training of severity models. No paper edits. Disease classifiers only.")

    paddy_acc = train(
        crop_name      = 'Paddy',
        df_path        = os.path.join(config.OUTPUTS_DIR, 'paddy_splits.csv'),
        model_save_path= os.path.join(config.MODELS_DIR, 'paddy_disease_classifier.pth'),
        meta_save_path = os.path.join(config.MODELS_DIR, 'paddy_disease_classifier_meta.json'),
    )

    areca_acc = train(
        crop_name      = 'Arecanut',
        df_path        = os.path.join(config.OUTPUTS_DIR, 'arecanut_splits.csv'),
        model_save_path= os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier.pth'),
        meta_save_path = os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier_meta.json'),
    )

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Paddy  Disease Classifier Val Acc : {paddy_acc:.2f}%")
    print(f"  Arecanut Disease Classifier Val Acc: {areca_acc:.2f}%")
    print(f"{'='*60}")
