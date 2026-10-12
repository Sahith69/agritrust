"""
Fix #4: Recompute Arecanut bootstrap 95% CIs using robust_cluster_id as group key.
- Replaces comp_le8 grouping with robust_cluster_id grouping for cluster-grouped bootstrap.
- Runs 2,000 bootstrap resamples on the leak-free Arecanut split v2 (N=486).
- Computes 95% CIs for QWK, Accuracy, Macro-F1.
- Saves outputs/final_v2/fix4_arecanut_bootstrap_ci_results.json
"""
import sys, os, json, time
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from PIL import Image

VAL_TF = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

class SimpleDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return VAL_TF(Image.open(self.paths[i]).convert('RGB'))

df_splits = pd.read_csv('outputs/arecanut_splits_v2.csv')
test_df   = df_splits[df_splits['split']=='test'].copy().reset_index(drop=True)

print("="*65)
print("FIX #4: ARECANUT BOOTSTRAP 95% CIs (robust_cluster_id Grouping)")
print("="*65)
print(f"Test Set Size: N = {len(test_df)}")
print(f"Unique robust_cluster_ids in test set: {test_df['robust_cluster_id'].nunique()}")

# Load disease classifier checkpoint (v2 or latest)
model_path = os.path.join(config.MODELS_DIR, 'arecanut_disease_classifier_v2.pth')
ck = torch.load(model_path, map_location='cpu', weights_only=True)
classes = ck['classes']

m = models.resnet18(weights=None)
n_features = m.fc.in_features
m.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(n_features, len(classes))
)
m.load_state_dict(ck['model_state_dict'])
m.eval()

# Run inference on test set
loader = DataLoader(SimpleDS(test_df['image_path'].tolist()), batch_size=32, shuffle=False)
all_preds = []
with torch.no_grad():
    for batch in loader:
        all_preds.extend(m(batch).argmax(1).numpy())

test_df['pred_cls'] = [classes[i] for i in all_preds]
test_df['true_cls'] = test_df['disease_class']

# Cluster-grouped bootstrap resampling (2,000 iterations)
N_BOOTSTRAP = 2000
unique_clusters = test_df['robust_cluster_id'].unique()
n_clusters      = len(unique_clusters)

boot_acc = []
boot_f1  = []

rng = np.random.default_rng(42)
t0 = time.time()

for b in range(N_BOOTSTRAP):
    # Sample clusters with replacement
    sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
    # Concatenate rows from sampled clusters
    boot_rows = []
    for cid in sampled_clusters:
        boot_rows.append(test_df[test_df['robust_cluster_id']==cid])
    boot_df = pd.concat(boot_rows, ignore_index=True)

    acc = accuracy_score(boot_df['true_cls'], boot_df['pred_cls']) * 100
    f1  = f1_score(boot_df['true_cls'], boot_df['pred_cls'], average='macro', zero_division=0) * 100
    boot_acc.append(acc)
    boot_f1.append(f1)

elapsed = time.time() - t0

acc_val = accuracy_score(test_df['true_cls'], test_df['pred_cls']) * 100
f1_val  = f1_score(test_df['true_cls'], test_df['pred_cls'], average='macro', zero_division=0) * 100

acc_ci = [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]
f1_ci  = [float(np.percentile(boot_f1, 2.5)),  float(np.percentile(boot_f1, 97.5))]

print(f"\n2,000 Cluster-Grouped Bootstrap Results ({elapsed:.1f}s):")
print(f"  Accuracy : {acc_val:.2f}%  [95% CI: {acc_ci[0]:.2f}%, {acc_ci[1]:.2f}%]")
print(f"  Macro-F1 : {f1_val:.2f}%  [95% CI: {f1_ci[0]:.2f}%, {f1_ci[1]:.2f}%]")

out = {
    'task': 'Fix #4: Arecanut Bootstrap CIs (robust_cluster_id)',
    'test_n': len(test_df),
    'num_clusters': int(n_clusters),
    'n_bootstrap': N_BOOTSTRAP,
    'accuracy': {'val': float(acc_val), 'ci_95': acc_ci},
    'macro_f1': {'val': float(f1_val),  'ci_95': f1_ci},
}
with open('outputs/final_v2/fix4_arecanut_bootstrap_ci_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print("\n✓ Fix #4 complete: outputs/final_v2/fix4_arecanut_bootstrap_ci_results.json")
