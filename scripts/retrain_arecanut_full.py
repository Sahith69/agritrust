"""
Retrain Arecanut disease classifier for full 20 epochs (no early stopping trick).
Uses class-weighted CE + CosineAnnealingLR.
Saves: outputs/models/arecanut_disease_classifier.pth  (overwrites)
Reports per-epoch val accuracy AND compares against disease-class-mode baseline.
"""
import sys, os, json, time
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image

DEVICE      = torch.device('cpu')
IMG_SIZE    = config.IMG_SIZE
BATCH_SIZE  = 24
LR          = 1e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 20
PATIENCE    = 7          # wider patience so we don't stop early at epoch 1
SEED        = 42
torch.manual_seed(SEED); np.random.seed(SEED)

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
    def __init__(self, df, c2i, transform):
        self.df = df.reset_index(drop=True); self.c2i = c2i; self.T = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['image_path']).convert('RGB')
        return self.T(img), self.c2i[row['disease_class']]

df = pd.read_csv(os.path.join(config.OUTPUTS_DIR, 'arecanut_splits.csv'))
classes    = sorted(df['disease_class'].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train_df = df[df['split']=='train']
val_df   = df[df['split']=='val']
test_df  = df[df['split']=='test']

print(f"Arecanut classes ({len(classes)}): {classes}")
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print(f"Patience = {PATIENCE} (real early stop, not forced at epoch 1)")

# Class weights
counts = train_df['disease_class'].value_counts()
max_c  = counts.max()
cw     = torch.tensor([max_c / counts[c] for c in classes], dtype=torch.float).to(DEVICE)

train_loader = DataLoader(DiseaseDataset(train_df, class_to_idx, TRAIN_TF),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(DiseaseDataset(val_df,   class_to_idx, VAL_TF),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(DiseaseDataset(test_df,  class_to_idx, VAL_TF),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False
for p in model.layer3.parameters(): p.requires_grad = True
for p in model.layer4.parameters(): p.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

epoch_log = []
best_val_acc = 0.0; best_state = None; no_improve = 0

print("\n{'='*55}")
print("  EPOCH-BY-EPOCH TRAINING LOG")
print(f"{'='*55}")
print(f"  {'Epoch':>5}  {'Train Acc':>10}  {'Val Acc':>10}  {'Time':>8}  {'Best?':>5}")
print(f"  {'-'*50}")

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    correct, total, tot_loss = 0, 0, 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        correct += (logits.argmax(1)==labels).sum().item()
        total   += len(labels)
        tot_loss+= loss.item() * len(labels)
    scheduler.step()
    train_acc = 100 * correct / total

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            vc += (model(imgs).argmax(1)==labels).sum().item()
            vt += len(labels)
    val_acc = 100 * vc / vt
    elapsed = time.time() - t0

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve   = 0
        mark = '*'
    else:
        no_improve += 1
        mark = ''

    epoch_log.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc})
    print(f"  {epoch:>5}  {train_acc:>9.2f}%  {val_acc:>9.2f}%  {elapsed:>6.1f}s  {mark:>5}", flush=True)

    if no_improve >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
        break

# Test evaluation
model.load_state_dict(best_state)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        all_preds.extend(model(imgs).argmax(1).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

pred_cls  = [idx_to_class[i] for i in all_preds]
true_cls  = [idx_to_class[i] for i in all_labels]
test_acc  = accuracy_score(true_cls, pred_cls) * 100
test_f1   = f1_score(true_cls, pred_cls, average='macro', zero_division=0) * 100

# Disease-class-mode baseline on test set
mode_cls = train_df['disease_class'].value_counts().idxmax()
mode_preds = [mode_cls] * len(test_df)
mode_acc = accuracy_score(true_cls, mode_preds) * 100
mode_f1  = f1_score(true_cls, mode_preds, average='macro', zero_division=0) * 100

print(f"\n{'='*55}")
print(f"  FINAL RESULTS (Best Val Acc = {best_val_acc:.2f}%)")
print(f"{'='*55}")
print(f"  Mode baseline ('{mode_cls}') : Test Acc = {mode_acc:.2f}%  Macro-F1 = {mode_f1:.2f}%")
print(f"  Disease Classifier            : Test Acc = {test_acc:.2f}%  Macro-F1 = {test_f1:.2f}%")
print(f"  Δ Acc = {test_acc - mode_acc:+.2f}%")

# Save
torch.save({
    'model_state_dict': best_state,
    'classes': classes, 'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class, 'best_val_acc': best_val_acc, 'crop': 'Arecanut',
}, os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier.pth'))

out = {
    'epoch_log': epoch_log,
    'best_val_acc': best_val_acc,
    'test_acc': test_acc, 'test_f1': test_f1,
    'mode_baseline_class': mode_cls,
    'mode_baseline_acc': mode_acc, 'mode_baseline_f1': mode_f1,
}
with open('outputs/final_v2/arecanut_retrain_full_log.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to outputs/final_v2/arecanut_retrain_full_log.json")
