"""
Train the honest Arecanut baseline model on the 1,628-image training set (outputs/arecanut_splits.csv).
- Split: 1,628 train / 547 val / 486 test (outputs/arecanut_splits.csv, 0 cluster leak)
- Architecture: ResNet-18 (layer4 + FC unfrozen, Dropout 0.4)
- Epochs: 5 epochs (takes ~100s on CPU)
- Saves: outputs/models/arecanut_disease_classifier.pth
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PIL import Image

DEVICE      = torch.device('cpu')
IMG_SIZE    = config.IMG_SIZE
BATCH_SIZE  = 24
LR          = 3e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 5
SEED        = 42
torch.manual_seed(SEED); np.random.seed(SEED)

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        self.df = df.reset_index(drop=True)
        self.c2i = c2i
        self.T = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['image_path']).convert('RGB')
        return self.T(img), self.c2i[row['disease_class']]

df = pd.read_csv('outputs/arecanut_splits.csv')
classes      = sorted(df['disease_class'].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train_df = df[df['split']=='train']
val_df   = df[df['split']=='val']
test_df  = df[df['split']=='test']

print(f"Loaded honest split outputs/arecanut_splits.csv — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Class weights
counts = train_df['disease_class'].value_counts()
max_c  = counts.max()
cw     = torch.tensor([max_c / counts.get(c, 1) for c in classes], dtype=torch.float).to(DEVICE)

train_loader = DataLoader(DiseaseDataset(train_df, class_to_idx, TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(DiseaseDataset(val_df,   class_to_idx, VAL_TF),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(DiseaseDataset(test_df,  class_to_idx, VAL_TF),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in base.parameters(): p.requires_grad = False
for p in base.layer4.parameters(): p.requires_grad = True

n_features = base.fc.in_features
base.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(n_features, len(classes)))
base.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_acc = 0.0
best_state   = None

print(f"\n{'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>10}  {'Time':>7}  {'Best?':>5}")
print(f"  {'-'*50}")

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    base.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(base(imgs), labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct += (base(imgs).argmax(1)==labels).sum().item()
            total   += len(labels)
    train_acc = 100 * correct / total

    base.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            vc += (base(imgs).argmax(1)==labels).sum().item()
            vt += len(labels)
    val_acc = 100 * vc / vt
    elapsed = time.time() - t0

    is_best = val_acc > best_val_acc
    mark    = '*' if is_best else ''
    if is_best:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in base.state_dict().items()}

    print(f"{epoch:>6}  {train_acc:>9.2f}%  {val_acc:>9.2f}%  {elapsed:>5.1f}s  {mark:>5}", flush=True)

base.load_state_dict(best_state)
base.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        all_preds.extend(base(imgs.to(DEVICE)).argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

pred_cls = [idx_to_class[i] for i in all_preds]
true_cls = [idx_to_class[i] for i in all_labels]
test_acc = accuracy_score(true_cls, pred_cls) * 100
test_f1  = f1_score(true_cls, pred_cls, average='macro', zero_division=0) * 100

print(f"\nRESULTS ON HONEST BASELINE TEST SET (N={len(test_df)}):")
print(f"  Test Accuracy: {test_acc:.2f}%")
print(f"  Macro-F1:      {test_f1:.2f}%")
print(f"  bud_borer Recall: {(np.array(pred_cls)[np.array(true_cls)=='bud_borer']=='bud_borer').mean()*100:.1f}%")

# Save honest checkpoint as official primary file
save_dict = {
    'model_state_dict': best_state,
    'classes': classes, 'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_f1': test_f1,
    'crop': 'Arecanut',
    'split_version': 'honest_baseline_v2',
    'architecture': 'resnet18_honest_baseline',
}

out_path = os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier.pth')
torch.save(save_dict, out_path)
print(f"\n✓ Saved honest baseline model checkpoint: {out_path}")
