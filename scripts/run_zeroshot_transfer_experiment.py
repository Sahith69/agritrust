"""
Zero-Shot Cross-Crop Transfer Evaluation Script
Evaluates:
1. Paddy Ordinal Model (paddy_ordinal.pth) -> Zero-Shot on Arecanut Test Set (n=124)
2. Arecanut Ordinal Model (arecanut_ordinal.pth) -> Zero-Shot on Paddy Test Set (n=1507)

Additive experiment only - does not modify checkpoints, training logic, or datasets.
"""
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score

from src.model import CropSeverityModel
from src.dataset import CropDiseaseDataset, get_transforms
from src.corn_loss import corn_probs_to_grade


def load_checkpoint_model(model_filename, device):
    model_path = os.path.join(config.MODELS_DIR, model_filename)
    model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=True, pretrained=False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_dataset(model, csv_path, split='test', batch_size=32, device='cpu'):
    ds = CropDiseaseDataset(csv_path, split, get_transforms('test'))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            sig_probs = torch.sigmoid(outputs)
            preds, _ = corn_probs_to_grade(sig_probs)
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = np.mean(np.abs(y_true - y_pred))
    exact_acc = accuracy_score(y_true, y_pred)
    within_1_acc = np.mean(np.abs(y_true - y_pred) <= 1)
    
    return {
        'n_samples': len(y_true),
        'qwk': float(qwk),
        'mae': float(mae),
        'exact_acc': float(exact_acc),
        'within_1_acc': float(within_1_acc),
        'confusion_matrix': pd.crosstab(
            pd.Series(y_true, name='True Grade'), 
            pd.Series(y_pred, name='Pred Grade')
        ).to_dict()
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    arecanut_csv = config.ARECANUT_SPLIT_CSV
    paddy_csv = config.PADDY_SPLIT_CSV
    
    # 1. Load Paddy model (trained ONLY on paddy)
    print("\n--- Loading paddy_ordinal.pth ---")
    paddy_model = load_checkpoint_model('paddy_ordinal.pth', device)
    
    # 2. Load Arecanut model (trained ONLY on arecanut)
    print("--- Loading arecanut_ordinal.pth ---")
    arecanut_model = load_checkpoint_model('arecanut_ordinal.pth', device)
    
    # Evaluate 4 pairings:
    # A. Paddy Model -> Arecanut Test (Zero-Shot Transfer)
    print("\n1. Paddy Model -> Arecanut Test Set (Zero-Shot Cross-Crop Transfer)")
    paddy_on_arecanut = evaluate_dataset(paddy_model, arecanut_csv, split='test', device=device)
    
    # B. Arecanut Model -> Paddy Test (Zero-Shot Transfer)
    print("2. Arecanut Model -> Paddy Test Set (Zero-Shot Cross-Crop Transfer)")
    arecanut_on_paddy = evaluate_dataset(arecanut_model, paddy_csv, split='test', device=device)
    
    # C. Native In-Domain benchmarks for direct comparison
    print("3. In-Domain Benchmark: Paddy Model -> Paddy Test Set")
    paddy_native = evaluate_dataset(paddy_model, paddy_csv, split='test', device=device)
    
    print("4. In-Domain Benchmark: Arecanut Model -> Arecanut Test Set")
    arecanut_native = evaluate_dataset(arecanut_model, arecanut_csv, split='test', device=device)
    
    results = {
        'zero_shot_paddy_on_arecanut': paddy_on_arecanut,
        'zero_shot_arecanut_on_paddy': arecanut_on_paddy,
        'native_paddy_on_paddy': paddy_native,
        'native_arecanut_on_arecanut': arecanut_native
    }
    
    reports_dir = config.REPORTS_DIR
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, 'zeroshot_transfer_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved detailed JSON results to: {out_path}\n")
    
    # Print formatted markdown table
    print("| Model Trained On | Test Set Evaluated On | Evaluation Type | N | QWK | MAE | Exact Acc | Within-1 Acc |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| Paddy (`paddy_ordinal.pth`) | Arecanut (`test`) | **Zero-Shot Cross-Crop** | {paddy_on_arecanut['n_samples']} | **{paddy_on_arecanut['qwk']:.4f}** | {paddy_on_arecanut['mae']:.4f} | {paddy_on_arecanut['exact_acc']*100:.2f}% | {paddy_on_arecanut['within_1_acc']*100:.2f}% |")
    print(f"| Arecanut (`arecanut_ordinal.pth`) | Paddy (`test`) | **Zero-Shot Cross-Crop** | {arecanut_on_paddy['n_samples']} | **{arecanut_on_paddy['qwk']:.4f}** | {arecanut_on_paddy['mae']:.4f} | {arecanut_on_paddy['exact_acc']*100:.2f}% | {arecanut_on_paddy['within_1_acc']*100:.2f}% |")
    print(f"| Paddy (`paddy_ordinal.pth`) | Paddy (`test`) | Native In-Domain | {paddy_native['n_samples']} | {paddy_native['qwk']:.4f} | {paddy_native['mae']:.4f} | {paddy_native['exact_acc']*100:.2f}% | {paddy_native['within_1_acc']*100:.2f}% |")
    print(f"| Arecanut (`arecanut_ordinal.pth`) | Arecanut (`test`) | Native In-Domain | {arecanut_native['n_samples']} | {arecanut_native['qwk']:.4f} | {arecanut_native['mae']:.4f} | {arecanut_native['exact_acc']*100:.2f}% | {arecanut_native['within_1_acc']*100:.2f}% |")


if __name__ == '__main__':
    main()
