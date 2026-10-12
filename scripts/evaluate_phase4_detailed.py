"""
Detailed Phase 4 Evaluation & Explainability Analysis Script
Computes QWK, MAE, Exact Accuracy, Within-1-Grade Accuracy for all 4 models.
Computes per-class Grad-CAM vs symptom/lesion mask IoU scores.
"""
import sys; sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score

from src.model import CropSeverityModel
from src.dataset import CropDiseaseDataset, get_transforms
from src.corn_loss import corn_probs_to_grade
from src.gradcam import GradCAM
from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    detect_dead_heart,
    detect_hispa
)
from src.iou_validation import compute_normalized_overlap


def load_model(crop_name, model_type, device):
    ordinal = (model_type == 'ordinal')
    model_path = os.path.join(config.MODELS_DIR, f"{crop_name}_{model_type}.pth")
    model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=ordinal, pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: {model_path} not found!")
    model.to(device)
    model.eval()
    return model


def evaluate_metrics(model, test_loader, device, ordinal=True):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if ordinal:
                sig_probs = torch.sigmoid(outputs)
                preds, _ = corn_probs_to_grade(sig_probs)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = np.mean(np.abs(y_true - y_pred))
    exact_acc = accuracy_score(y_true, y_pred)
    within_1_acc = np.mean(np.abs(y_true - y_pred) <= 1)
    
    return {
        'qwk': qwk,
        'mae': mae,
        'exact_acc': exact_acc,
        'within_1_acc': within_1_acc,
        'count': len(y_true)
    }


def get_symptom_mask(img_np, disease_class):
    """Generates appropriate segmentation/symptom mask depending on disease class."""
    try:
        leaf_mask = segment_leaf_adaptive(img_np)
        if disease_class == 'dead_heart':
            mask = detect_dead_heart(img_np, leaf_mask)
        elif disease_class == 'hispa':
            mask = detect_hispa(img_np, leaf_mask)
        elif disease_class in config.PADDY_GROUP_A or disease_class in config.ARECANUT_GROUP_A:
            mask = detect_lesions_group_a(img_np, leaf_mask)
        elif disease_class == 'normal' or disease_class in [config.ARECANUT_CLASSES[c] for c in config.ARECANUT_HEALTHY_CLASSES]:
            mask = leaf_mask  # Healthy reference
        else:
            mask = detect_lesions_group_a(img_np, leaf_mask)
        return mask
    except Exception as e:
        return None


def run_per_class_iou(model, split_csv, crop_name, device, samples_per_class=15):
    df = pd.read_csv(split_csv)
    df_test = df[df['split'] == 'test'].copy()
    
    transform = get_transforms('test')
    gradcam = GradCAM(model, config.GRADCAM_TARGET_LAYER)
    
    class_ious = {}
    
    for cls_name in df_test['disease_class'].unique():
        cls_df = df_test[df_test['disease_class'] == cls_name]
        sample_df = cls_df.sample(min(samples_per_class, len(cls_df)), random_state=config.RANDOM_SEED)
        
        ious = []
        for idx, row in sample_df.iterrows():
            img_path = row['image_path']
            if not os.path.exists(img_path):
                continue
                
            image = Image.open(img_path).convert('RGB')
            img_np = np.array(image)
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            heatmap = gradcam.generate(input_tensor)
            
            symptom_mask = get_symptom_mask(img_np, cls_name)
            if symptom_mask is None:
                continue
                
            if heatmap.shape != symptom_mask.shape:
                mask_img = Image.fromarray((symptom_mask * 255).astype(np.uint8))
                mask_img = mask_img.resize(heatmap.shape[::-1], Image.NEAREST)
                symptom_mask = np.array(mask_img) > 0
                
            iou = compute_normalized_overlap(heatmap, symptom_mask, threshold=0.5)
            ious.append(iou)
            
        if len(ious) > 0:
            class_ious[cls_name] = {
                'mean_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'min_iou': np.min(ious),
                'max_iou': np.max(ious),
                'count': len(ious)
            }
            
    return class_ious


def main():
    device = torch.device('cpu')
    print("Executing Phase 4 Evaluation across all 4 models...\n")
    
    # Evaluate Paddy Models
    paddy_test_dataset = CropDiseaseDataset(config.PADDY_SPLIT_CSV, split='test', transform=get_transforms('test'))
    paddy_test_loader = DataLoader(paddy_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    paddy_ord_model = load_model('paddy', 'ordinal', device)
    paddy_base_model = load_model('paddy', 'baseline', device)
    
    paddy_ord_metrics = evaluate_metrics(paddy_ord_model, paddy_test_loader, device, ordinal=True)
    paddy_base_metrics = evaluate_metrics(paddy_base_model, paddy_test_loader, device, ordinal=False)
    
    # Evaluate Arecanut Models
    areca_test_dataset = CropDiseaseDataset(config.ARECANUT_SPLIT_CSV, split='test', transform=get_transforms('test'))
    areca_test_loader = DataLoader(areca_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    areca_ord_model = load_model('arecanut', 'ordinal', device)
    areca_base_model = load_model('arecanut', 'baseline', device)
    
    areca_ord_metrics = evaluate_metrics(areca_ord_model, areca_test_loader, device, ordinal=True)
    areca_base_metrics = evaluate_metrics(areca_base_model, areca_test_loader, device, ordinal=False)
    
    # Run Per-Class Grad-CAM / IoU
    print("Running per-class Grad-CAM / IoU validation on Paddy test set...")
    paddy_ious = run_per_class_iou(paddy_ord_model, config.PADDY_SPLIT_CSV, 'paddy', device, samples_per_class=20)
    
    print("Running per-class Grad-CAM / IoU validation on Arecanut test set...")
    areca_ious = run_per_class_iou(areca_ord_model, config.ARECANUT_SPLIT_CSV, 'arecanut', device, samples_per_class=20)
    
    # Save output summary to JSON / Markdown report for printing
    report_data = {
        'paddy_ord': paddy_ord_metrics,
        'paddy_base': paddy_base_metrics,
        'areca_ord': areca_ord_metrics,
        'areca_base': areca_base_metrics,
        'paddy_ious': paddy_ious,
        'areca_ious': areca_ious
    }
    import json
    with open(os.path.join(config.REPORTS_DIR, "phase4_evaluation_summary.json"), "w") as f:
        json.dump(report_data, f, indent=2)
        
    print("\nPhase 4 Evaluation Complete!")

if __name__ == '__main__':
    main()
