"""
Script to compute McNemar's Test comparing CORN Ordinal vs. Softmax Baseline
for both Paddy (n=1507) and Arecanut (n=124) test sets.
"""
import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import torch
torch.set_num_threads(2)
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import chi2, binomtest

from src.model import CropSeverityModel
from src.dataset import CropDiseaseDataset, get_transforms
from src.corn_loss import corn_probs_to_grade


def load_model(crop_name, model_type, device):
    ordinal = (model_type == 'ordinal')
    model_path = os.path.join(config.MODELS_DIR, f"{crop_name}_{model_type}.pth")
    model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=ordinal, pretrained=False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def get_predictions(model, split_csv, ordinal=True, device='cpu'):
    ds = CropDiseaseDataset(split_csv, 'test', get_transforms('test'))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
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
            
    return np.array(y_true), np.array(y_pred)


def compute_mcnemar(b, c):
    """
    Computes McNemar's test for discordant pairs b and c.
    b: Model A correct, Model B wrong
    c: Model B correct, Model A wrong
    """
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0, "Zero discordant pairs"
    
    if n_disc < 25:
        # Exact Binomial Test
        res = binomtest(min(b, c), n_disc, 0.5, alternative='two-sided')
        p_val = res.pvalue
        stat = float(min(b, c))
        method = "Exact Binomial"
    else:
        # Chi-Squared with Edwards' Continuity Correction
        stat = ((abs(b - c) - 1.0) ** 2) / float(n_disc)
        p_val = 1.0 - chi2.cdf(stat, df=1)
        method = "Chi-Squared (with continuity correction)"
        
    return float(stat), float(p_val), method


def run_mcnemar_crop(crop_name, split_csv, device='cpu'):
    print(f"\n==========================================", flush=True)
    print(f" McNemar's Test: {crop_name.upper()}", flush=True)
    print(f"==========================================", flush=True)
    
    print(f"Loading {crop_name} models...", flush=True)
    ordinal_model = load_model(crop_name, 'ordinal', device)
    baseline_model = load_model(crop_name, 'baseline', device)
    
    print(f"Evaluating {crop_name} ordinal model...", flush=True)
    y_true_ord, y_pred_ord = get_predictions(ordinal_model, split_csv, ordinal=True, device=device)
    print(f"Evaluating {crop_name} baseline model...", flush=True)
    y_true_base, y_pred_base = get_predictions(baseline_model, split_csv, ordinal=False, device=device)
    
    assert np.array_equal(y_true_ord, y_true_base), "Ground truth labels mismatch between runs!"
    
    correct_ord = (y_pred_ord == y_true_ord)
    correct_base = (y_pred_base == y_true_base)
    
    a = int(np.sum(correct_ord & correct_base))
    b = int(np.sum(correct_ord & (~correct_base)))
    c = int(np.sum((~correct_ord) & correct_base))
    d = int(np.sum((~correct_ord) & (~correct_base)))
    
    stat, pvalue, method = compute_mcnemar(b, c)
    is_significant = pvalue < 0.05
    
    print(f"Sample Size (N): {len(y_true_ord)}", flush=True)
    print(f"CORN Ordinal Exact Acc:  {np.mean(correct_ord)*100:.2f}% ({np.sum(correct_ord)}/{len(y_true_ord)})", flush=True)
    print(f"Baseline Softmax Acc:    {np.mean(correct_base)*100:.2f}% ({np.sum(correct_base)}/{len(y_true_ord)})", flush=True)
    print(f"\nContingency Table:", flush=True)
    print(f"                        Baseline Correct | Baseline Incorrect", flush=True)
    print(f"  Ordinal Correct    :        {a:5d}      |       {b:5d}", flush=True)
    print(f"  Ordinal Incorrect  :        {c:5d}      |       {d:5d}", flush=True)
    print(f"\nDiscordant Pairs: b (CORN right, Base wrong) = {b}, c (Base right, CORN wrong) = {c}", flush=True)
    print(f"Method Used: {method}", flush=True)
    print(f"Test Statistic: {stat:.4f}", flush=True)
    print(f"p-value: {pvalue:.6f}", flush=True)
    print(f"Statistically Significant (alpha=0.05): {is_significant}", flush=True)
    
    return {
        'crop': crop_name,
        'n_samples': len(y_true_ord),
        'ordinal_acc': float(np.mean(correct_ord)),
        'baseline_acc': float(np.mean(correct_base)),
        'contingency_table': {'a': a, 'b': b, 'c': c, 'd': d},
        'method': method,
        'statistic': stat,
        'pvalue': pvalue,
        'is_significant': is_significant
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    paddy_res = run_mcnemar_crop('paddy', config.PADDY_SPLIT_CSV, device=device)
    arecanut_res = run_mcnemar_crop('arecanut', config.ARECANUT_SPLIT_CSV, device=device)
    
    output_data = {
        'paddy': paddy_res,
        'arecanut': arecanut_res
    }
    
    out_file = os.path.join(config.REPORTS_DIR, 'mcnemar_test_results.json')
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {out_file}", flush=True)


if __name__ == '__main__':
    main()
