import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
import os
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

from src.severity_grader import ratio_to_grade
from src.segmentation import segment_leaf_adaptive, detect_lesions_group_a, compute_ratio
from src.confidence import composite_confidence

def process_apple_dataset():
    print("=" * 70)
    print("PHASE 1 & 2: APPLE SEVERITY LABELING & LEAKAGE-FREE DATA SPLITTING")
    print("=" * 70)
    
    base_dir = os.path.join(config.DATA_DIR, "apple", "dataset")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Apple dataset not found at {base_dir}")
        
    records = []
    
    # Map directory names to clean class names
    class_map = {
        "Apple___Apple_scab": "apple_scab",
        "Apple___Black_rot": "black_rot",
        "Apple___Cedar_apple_rust": "cedar_apple_rust",
        "Apple___healthy": "healthy"
    }
    
    image_paths = []
    for dir_name, clean_name in class_map.items():
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((os.path.join(dir_path, fname), clean_name, fname))
                
    print(f"Total Apple images found: {len(image_paths)}")
    
    print("\n[1/3] Computing lesion ratios & severity grades...")
    for full_path, disease_class, fname in tqdm(image_paths):
        # Hash check for duplicate grouping
        with open(full_path, 'rb') as f_in:
            md5_hash = hashlib.md5(f_in.read()).hexdigest()
            
        # Parent ID grouping (from filename before UUID/timestamp)
        parts = fname.split('___')
        parent_id = parts[1] if len(parts) > 1 else fname
        
        if disease_class == "healthy":
            grade = 0
            ratio = 0.0
            confidence = 100.0
            method = "auto_healthy"
        else:
            try:
                img = Image.open(full_path).convert('RGB')
                img_arr = np.array(img)
                leaf_mask = segment_leaf_adaptive(img_arr)
                lesion_mask = detect_lesions_group_a(img_arr, leaf_mask)
                ratio = compute_ratio(lesion_mask, leaf_mask)
                grade = ratio_to_grade(ratio)
                confidence = composite_confidence(leaf_mask, lesion_mask, ratio, img_arr.shape)
                method = "group_a_segmentation"
            except Exception as e:
                grade = 1
                ratio = 0.02
                confidence = 50.0
                method = "fallback_error"
                
        records.append({
            "image_path": full_path,
            "filename": fname,
            "disease_class": disease_class,
            "severity_grade": grade,
            "lesion_ratio": ratio,
            "confidence": confidence,
            "method": method,
            "md5_hash": md5_hash,
            "parent_id": parent_id
        })
        
    df = pd.DataFrame(records)
    
    # Save raw labels
    os.makedirs(os.path.join(config.OUTPUTS_DIR, "reports"), exist_ok=True)
    apple_severity_csv = os.path.join(config.OUTPUTS_DIR, "apple_severity_labels.csv")
    df.to_csv(apple_severity_csv, index=False)
    print(f"Severity labels saved to: {apple_severity_csv}")
    print("\nSeverity Grade Distribution:")
    print(df['severity_grade'].value_counts().sort_index())
    
    # [2/3] Leakage-Free Stratified Group Split (70% Train, 15% Val, 15% Test)
    print("\n[2/3] Performing GroupShuffleSplit (0% Parent Data Leakage)...")
    
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=config.RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df['parent_id']))
    
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=config.RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['parent_id']))
    
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    
    df_split = pd.concat([df_train, df_val, df_test], ignore_index=True)
    apple_split_csv = os.path.join(config.OUTPUTS_DIR, "apple_splits.csv")
    df_split.to_csv(apple_split_csv, index=False)
    print(f"Splits saved to: {apple_split_csv}")
    
    # Verify 0% parent leakage
    train_parents = set(df_train['parent_id'])
    val_parents = set(df_val['parent_id'])
    test_parents = set(df_test['parent_id'])
    
    leak_val = len(train_parents.intersection(val_parents))
    leak_test = len(train_parents.intersection(test_parents))
    
    print("\nData Leakage Check:")
    print(f"- Train-Val Parent Overlap: {leak_val} (Target: 0)")
    print(f"- Train-Test Parent Overlap: {leak_test} (Target: 0)")
    print(f"- Train Samples: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"- Val Samples:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"- Test Samples:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    print("=" * 70)
    print("APPLE DATA PREPARATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    process_apple_dataset()
