import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import os
import pandas as pd
from tqdm import tqdm
from src.corn_loss import corn_probs_to_grade

def evaluate_model(model, test_loader, device, ordinal=True):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch[0], batch[1]
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if ordinal:
                sig_probs = torch.sigmoid(outputs)
                preds, grade_probs = corn_probs_to_grade(sig_probs)
                y_probs.extend(grade_probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_probs.extend(probs.cpu().numpy())
                
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def compute_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    acc = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(config.NUM_SEVERITY_GRADES), zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_SEVERITY_GRADES))
    
    return {
        'qwk': qwk,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    plt.style.use('dark_background')
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_pct = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1%})'
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_training_history(history, save_path):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'train_loss' in history and 'val_loss' in history:
        ax1.plot(history['train_loss'], label='Train')
        ax1.plot(history['val_loss'], label='Val')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
    if 'train_qwk' in history and 'val_qwk' in history:
        ax2.plot(history['train_qwk'], label='Train')
        ax2.plot(history['val_qwk'], label='Val')
        ax2.set_title('QWK')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def error_analysis(model, test_loader, device, ordinal, top_n=20):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Error Analysis")):
            images, labels = batch[0], batch[1]
            images = images.to(device)
            outputs = model(images)
            
            if ordinal:
                sig_probs = torch.sigmoid(outputs)
                preds_t, _ = corn_probs_to_grade(sig_probs)
                preds = preds_t.cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                
            labels_np = labels.cpu().numpy()
            
            for i in range(len(preds)):
                diff = abs(int(preds[i]) - int(labels_np[i]))
                if diff > 0:
                    errors.append({
                        'batch_idx': batch_idx,
                        'item_idx': i,
                        'true': int(labels_np[i]),
                        'pred': int(preds[i]),
                        'diff': diff
                    })
                    
    errors = sorted(errors, key=lambda x: x['diff'], reverse=True)[:top_n]
    return errors

def generate_full_report(crop_name, ordinal_model, baseline_model, test_loader, iou_scores, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Evaluating {crop_name} models...")
    y_true_ord, y_pred_ord, _ = evaluate_model(ordinal_model, test_loader, device, ordinal=True)
    y_true_base, y_pred_base, _ = evaluate_model(baseline_model, test_loader, device, ordinal=False)
    
    metrics_ord = compute_metrics(y_true_ord, y_pred_ord)
    metrics_base = compute_metrics(y_true_base, y_pred_base)
    
    classes = [f"Grade {i}" for i in range(config.NUM_SEVERITY_GRADES)]
    plot_confusion_matrix(y_true_ord, y_pred_ord, classes, 
                          f'{crop_name.capitalize()} Ordinal Model CM', 
                          os.path.join(save_dir, f'{crop_name}_ordinal_cm.png'))
    plot_confusion_matrix(y_true_base, y_pred_base, classes, 
                          f'{crop_name.capitalize()} Baseline Model CM', 
                          os.path.join(save_dir, f'{crop_name}_baseline_cm.png'))
                          
    report = [
        f"# Evaluation Report: {crop_name.capitalize()}",
        "",
        "## Overall Metrics",
        "| Metric | Ordinal (CORN) | Baseline (Softmax) |",
        "|--------|----------------|--------------------|",
        f"| QWK    | {metrics_ord['qwk']:.4f} | {metrics_base['qwk']:.4f} |",
        f"| Acc    | {metrics_ord['accuracy']:.4f} | {metrics_base['accuracy']:.4f} |",
        "",
        "## Per-class Metrics (Ordinal)",
        "| Grade | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|"
    ]
    
    for i in range(config.NUM_SEVERITY_GRADES):
        report.append(f"| {i} | {metrics_ord['precision'][i]:.4f} | {metrics_ord['recall'][i]:.4f} | {metrics_ord['f1'][i]:.4f} |")
        
    if iou_scores:
        avg_iou = np.mean([s['iou'] for s in iou_scores])
        report.extend([
            "",
            "## XAI Validation (Grad-CAM vs Segmentation)",
            f"Average IoU: {avg_iou:.4f}"
        ])
        
    report_path = os.path.join(save_dir, f'{crop_name}_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {report_path}")
