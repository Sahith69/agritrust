import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from src.gradcam import GradCAM, overlay_heatmap

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_normalized_overlap(gradcam_heatmap, lesion_mask, threshold=0.5):
    gradcam_binary = gradcam_heatmap > threshold
    return compute_iou(gradcam_binary, lesion_mask > 0)

def validate_explainability(model, test_csv, segmentation_fn, transform, device, num_samples=50):
    df = pd.read_csv(test_csv)
    # Ensure it's the test split
    if 'split' in df.columns:
        df = df[df['split'] == 'test']
        
    # Filter to Group A disease classes only (segmentation-based severity)
    if 'disease_class' in df.columns:
        paddy_group_a = set(config.PADDY_GROUP_A)
        arecanut_group_a = config.ARECANUT_GROUP_A
        all_group_a = paddy_group_a | arecanut_group_a
        # Map arecanut numeric labels if needed
        arecanut_names = set(config.ARECANUT_CLASSES.values())
        df_filtered = df[df['disease_class'].isin(all_group_a | arecanut_names)]
        if len(df_filtered) > 0:
            df = df_filtered
            
    if len(df) == 0:
        print("No valid Group A samples found in test split for IoU validation.")
        return []
        
    df = df.sample(min(num_samples, len(df)), random_state=config.RANDOM_SEED)
    
    gradcam = GradCAM(model, config.GRADCAM_TARGET_LAYER)
    results = []
    
    output_dir = os.path.join(config.GRADCAM_DIR, "validation_gallery")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        img_path = row.get('image_path', row.get('path', None))
        if img_path is None:
            continue
            
        if not os.path.isabs(img_path):
            if 'paddy' in img_path.lower():
                img_path = os.path.join(config.PADDY_TRAIN_DIR, img_path)
            else:
                img_path = os.path.join(config.ARECANUT_DIR, img_path)
                
        if not os.path.exists(img_path):
            continue
            
        image = Image.open(img_path).convert('RGB')
        img_np = np.array(image)
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        heatmap = gradcam.generate(input_tensor)
        
        try:
            lesion_mask = segmentation_fn(img_np)
        except Exception:
            continue
            
        if lesion_mask is None:
            continue
            
        if heatmap.shape != lesion_mask.shape:
            lesion_mask_img = Image.fromarray((lesion_mask * 255).astype(np.uint8))
            lesion_mask_img = lesion_mask_img.resize(heatmap.shape[::-1], Image.NEAREST)
            lesion_mask = np.array(lesion_mask_img) > 0
            
        iou = compute_normalized_overlap(heatmap, lesion_mask, threshold=0.5)
        
        results.append({
            'image_path': img_path,
            'iou': iou
        })
        
        overlay = overlay_heatmap(image.resize(heatmap.shape[::-1]), heatmap)
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image.resize(heatmap.shape[::-1]))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(overlay)
        axes[1].set_title('Grad-CAM Overlay')
        axes[1].axis('off')
        
        axes[2].imshow(lesion_mask, cmap='gray')
        axes[2].set_title('Lesion Mask')
        axes[2].axis('off')
        
        overlap = np.logical_and(heatmap > 0.5, lesion_mask)
        axes[3].imshow(overlap, cmap='gray')
        axes[3].set_title(f'Overlap (IoU: {iou:.3f})')
        axes[3].axis('off')
        
        base_name = os.path.basename(img_path)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"val_{base_name}"))
        plt.close()
        
    return results
