import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch

PROJECT_ROOT = "/home/sahith/Projects/Crop"
SPLITS_CSV = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits.csv")
CACHE_FILE = os.path.join(PROJECT_ROOT, "outputs/arecanut_splits_cached_tensors.pt")

def load_img(p):
    im = Image.open(p).convert('RGB').resize((224, 224))
    return torch.from_numpy(np.array(im)).permute(2, 0, 1) # (3, 224, 224) uint8

def main():
    print(f"Loading splits from {SPLITS_CSV}...")
    df = pd.read_csv(SPLITS_CSV)
    print(f"Total rows: {len(df)}")
    
    t0 = time.time()
    print("Pre-caching 224x224 RGB images in parallel...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        tensors = list(ex.map(load_img, df['image_path']))
    stacked_tensors = torch.stack(tensors)
    dt = time.time() - t0
    print(f"Stacked tensor shape: {stacked_tensors.shape} ({dt:.2f}s)")
    
    cache_dict = {
        'images': stacked_tensors, # uint8 (2661, 3, 224, 224)
        'image_path': df['image_path'].tolist(),
        'disease_class': df['disease_class'].tolist(),
        'severity_grade': torch.tensor(df['severity_grade'].values, dtype=torch.long),
        'split': df['split'].tolist(),
        'robust_cluster_id': torch.tensor(df['robust_cluster_id'].values, dtype=torch.long),
        'exclude_from_severity': torch.tensor(df['exclude_from_severity'].values, dtype=torch.bool),
    }
    
    print(f"Saving cache to {CACHE_FILE}...")
    torch.save(cache_dict, CACHE_FILE)
    print(f"Successfully saved cache! Size: {os.path.getsize(CACHE_FILE) / (1024*1024):.1f} MB")

if __name__ == '__main__':
    main()
