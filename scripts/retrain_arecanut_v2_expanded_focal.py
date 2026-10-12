"""
Retrain Arecanut disease classifier on the expanded leak-free dataset (outputs/arecanut_splits_v2_expanded.csv).
- Training set: 10,531 clean images (1,628 original + 8,903 leak-filtered Kaggle images).
- Val set: 547 images (100% leak-free, identical to v2).
- Test set: 486 images (100% leak-free, identical to v2).
- Uses WeightedRandomSampler + Focal Loss (gamma=2.0) + CosineAnnealingLR.
- Saves: outputs/models/arecanut_disease_classifier_v2_expanded.pth
         outputs/final_v2/arecanut_retrain_v2_expanded_log.json
         outputs/final_v2/arecanut_v2_expanded_confusion_matrix.csv
"""
import sys, os, json, time
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PIL import Image

DEVICE      = torch.device('cpu')
IMG_SIZE    = config.IMG_SIZE
BATCH_SIZE  = 32
LR          = 3e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 25
PATIENCE    = 8
SEED        = 42
torch.manual_seed(SEED); np.random.seed(SEED)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

# Augmentations
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
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

# Load expanded leak-free split
df = pd.read_csv('outputs/arecanut_splits_v2_expanded.csv')
classes      = sorted(df['disease_class'].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train_df = df[df['split']=='train']
val_df   = df[df['split']=='val']
test_df  = df[df['split']=='test']

# Compute sampler weights for balanced training batches
train_targets = [class_to_idx[c] for c in train_df['disease_class']]
class_counts  = np.bincount(train_targets)
class_weights = 1.0 / class_counts
sample_weights = np.array([class_weights[t] for t in train_targets])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

print(f"Expanded Dataset — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("Train per-class counts:")
for c in classes:
    print(f"  {c:<30}: {len(train_df[train_df['disease_class']==c])}")

train_loader = DataLoader(DiseaseDataset(train_df, class_to_idx, TRAIN_TF),
                          batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(DiseaseDataset(val_df,   class_to_idx, VAL_TF),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(DiseaseDataset(test_df,  class_to_idx, VAL_TF),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model: ResNet-18, layer4 + FC unfrozen
base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in base.parameters():
    p.requires_grad = False
for p in base.layer4.parameters():
    p.requires_grad = True

n_features = base.fc.in_features
base.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(n_features, len(classes))
)
base.to(DEVICE)

alpha_weights = torch.tensor(class_weights / class_weights.sum() * len(classes), dtype=torch.float).to(DEVICE)
criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Training loop
epoch_log = []
best_val_f1 = 0.0
best_state  = None
no_improve  = 0

print(f"\n{'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>10}  {'Val Macro-F1':>14}  {'Time':>7}  {'Best?':>5}")
print(f"  {'-'*65}")

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    base.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = base(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct += (logits.argmax(1)==labels).sum().item()
            total   += len(labels)
    scheduler.step()
    train_acc = 100 * correct / total

    base.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            val_preds.extend(base(imgs).argmax(1).cpu().numpy())
            val_targets.extend(labels.numpy())

    val_acc = accuracy_score(val_targets, val_preds) * 100
    val_f1  = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100
    elapsed = time.time() - t0

    is_best = val_f1 > best_val_f1
    mark    = '*' if is_best else ''
    if is_best:
        best_val_f1 = val_f1
        best_state  = {k: v.clone() for k, v in base.state_dict().items()}
        no_improve  = 0
    else:
        no_improve += 1

    epoch_log.append({'epoch': epoch, 'train_acc': round(train_acc,4), 'val_acc': round(val_acc,4), 'val_f1': round(val_f1,4)})
    print(f"{epoch:>6}  {train_acc:>9.2f}%  {val_acc:>9.2f}%  {val_f1:>13.2f}%  {elapsed:>5.1f}s  {mark:>5}", flush=True)

    if no_improve >= PATIENCE:
        print(f"\n  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs).")
        break

# Test evaluation
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

cm = confusion_matrix(true_cls, pred_cls, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

print(f"\n{'='*65}")
print(f"  RESULTS (Best Val Macro-F1 = {best_val_f1:.2f}% | Split = v2_expanded no-leak)")
print(f"{'='*65}")
print(f"  Classifier (Expanded Train): Test Acc = {test_acc:.2f}%  Macro-F1 = {test_f1:.2f}%")
print()
print("  Per-class Recall / Precision:")
for cls in classes:
    mt = np.array(true_cls)==cls
    mp = np.array(pred_cls)==cls
    tp = (mt & mp).sum(); fn = (mt & ~mp).sum(); fp = (~mt & mp).sum()
    rec  = tp/(tp+fn+1e-9)*100
    prec = tp/(tp+fp+1e-9)*100
    print(f"  {cls:<30}  N={mt.sum():>3}  Rec={rec:.1f}%  Prec={prec:.1f}%")

print(f"\n  9×9 Confusion Matrix:")
header = f"  {'TRUE\\PRED':<25}" + "".join(f"{c[:8]:>9}" for c in classes)
print(header)
print("  " + "-"*(25 + 9*len(classes)))
for i, cls in enumerate(classes):
    row = f"  {cls:<25}" + "".join(f"{cm[i,j]:>9}" for j in range(len(classes)))
    print(row)

# Save expanded model and v2 model
torch.save({
    'model_state_dict': best_state,
    'classes': classes, 'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'best_val_f1': best_val_f1,
    'crop': 'Arecanut',
    'split_version': 'v2_expanded',
    'architecture': 'resnet18_focal_expanded',
}, os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier_v2.pth'))

cm_df.to_csv('outputs/final_v2/arecanut_v2_expanded_confusion_matrix.csv')

out = {
    'split': 'v2_expanded', 'architecture': 'resnet18_focal_expanded',
    'epoch_log': epoch_log, 'best_val_f1': best_val_f1,
    'test_acc': test_acc, 'test_f1': test_f1,
}
with open('outputs/final_v2/arecanut_retrain_v2_expanded_log.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n✓ Saved: outputs/models/arecanut_disease_classifier_v2.pth")
print(f"✓ Saved: outputs/final_v2/arecanut_retrain_v2_expanded_log.json")
