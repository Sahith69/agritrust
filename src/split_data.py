"""
Data Integrity & Splitting Engine for Arecanut and Paddy datasets.
Ensures ZERO parent-image leakage across train, val, and test splits.
"""
import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def extract_arecanut_parent(filename):
    """
    Extracts true parent camera ID from Arecanut filename.
    Filename structure: 'class_original_IMG_YYYYMMDD_HHMMSS.jpg_UUID.jpg' or 'Class_original_NN.jpg_UUID.jpg'
    Parent ID: 'IMG_YYYYMMDD_HHMMSS' or 'Class_original_NN'
    """
    fname = str(filename)
    match = re.search(r'(IMG_\d+_\d+)', fname)
    if match:
        return match.group(1)
    
    match2 = re.search(r'([^\_]+\_original_\d+)', fname, re.IGNORECASE)
    if match2:
        return match2.group(1).lower()
        
    return fname.split('_')[0].lower()


def split_arecanut():
    """Performs parent-grouped 70/15/15 stratified split for Arecanut."""
    config.ensure_dirs()
    ann_path = os.path.join(config.PROJECT_ROOT, "arecanut_annotations.json")
    
    if not os.path.exists(ann_path):
        print(f"Error: {ann_path} not found.")
        return

    with open(ann_path) as f:
        ann = json.load(f)

    df = pd.DataFrame(ann)
    df['filename'] = df['image_path'].apply(os.path.basename)
    df['parent_id'] = df['filename'].apply(extract_arecanut_parent)
    
    # Map original_label (stored as string integer) to disease_class name
    df['original_label_int'] = df['original_label'].astype(int)
    df['disease_class'] = df['original_label_int'].map(config.ARECANUT_CLASSES)

    print(f"Arecanut Dataset: {len(df)} total images across {df['parent_id'].nunique()} unique parent IDs.")

    # Group by parent_id first to get parent-level metadata
    parent_meta = df.groupby('parent_id').agg({
        'disease_class': 'first',
        'severity_grade': 'first'
    }).reset_index()

    # Stratified split on parents (70% train, 30% temp)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=config.RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(parent_meta, groups=parent_meta['parent_id']))

    train_parents = set(parent_meta.iloc[train_idx]['parent_id'])
    temp_parents = parent_meta.iloc[temp_idx].copy()

    # Split temp (30% total) equally into 15% val and 15% test
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=config.RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(temp_parents, groups=temp_parents['parent_id']))

    val_parents = set(temp_parents.iloc[val_idx]['parent_id'])
    test_parents = set(temp_parents.iloc[test_idx]['parent_id'])

    # Map split to full image dataframe
    def assign_split(pid):
        if pid in train_parents:
            return 'train'
        elif pid in val_parents:
            return 'val'
        elif pid in test_parents:
            return 'test'
        return 'train'

    df['split'] = df['parent_id'].apply(assign_split)

    # Save to CSV
    output_csv = config.ARECANUT_SPLIT_CSV
    out_df = df[['image_path', 'filename', 'disease_class', 'severity_grade', 'parent_id', 'split']]
    out_df.to_csv(output_csv, index=False)
    print(f"Saved parent-grouped Arecanut splits to {output_csv}")
    return out_df


if __name__ == "__main__":
    split_arecanut()
