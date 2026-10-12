"""
Multi-Seed Arecanut Architecture Comparison Script
Trains ResNet-18, ResNet-34, and MobileNetV2 with CORN ordinal head on Arecanut across 3 random seeds (42, 100, 2024).
Evaluates all seeds on the held-out Arecanut test set (N=124).
"""
import sys; sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import copy
import json
import random
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score

# Optimize PyTorch CPU performance
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)
NUM_WORKERS = 2

from src.dataset import CropDiseaseDataset, get_transforms, get_dataloaders
from src.model import CropSeverityModel
from src.corn_loss import CornLoss, corn_probs_to_grade

ARCH_MODELS_DIR = os.path.join(config.MODELS_DIR, "architecture_comparison")
os.makedirs(ARCH_MODELS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets, _, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, targets)
            
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits)
            preds, _ = corn_probs_to_grade(probs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk


def train_model_seed(crop_name, split_csv, backbone, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 75)
    print(f"TRAINING: Crop={crop_name.upper()} | Backbone={backbone} | Seed={seed} | Head=CORN")
    print("=" * 75)
    
    train_loader, val_loader, test_loader = get_dataloaders(split_csv, config.BATCH_SIZE, num_workers=NUM_WORKERS)
    
    model = CropSeverityModel(backbone=backbone, num_classes=config.NUM_SEVERITY_GRADES, ordinal=True, pretrained=True)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = CornLoss(num_classes=config.NUM_SEVERITY_GRADES)
    
    best_qwk = -1.0
    best_model_state = None
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_qwk': [], 'val_loss': [], 'val_qwk': []}
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_qwk = validate(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_qwk'].append(train_qwk)
        history['val_loss'].append(val_loss)
        history['val_qwk'].append(val_qwk)
        
        log_msg = f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}"
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            log_msg += "  --> New best QWK!"
        else:
            epochs_no_improve += 1
            
        print(log_msg, flush=True)
        
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1} (patience={config.EARLY_STOPPING_PATIENCE})", flush=True)
            break
            
    # Save model in subfolder
    if seed == 42 and backbone != 'resnet18':
        save_filename = f"{crop_name}_{backbone}_ordinal.pth"
    else:
        save_filename = f"{crop_name}_{backbone}_ordinal_seed{seed}.pth"
        
    save_path = os.path.join(ARCH_MODELS_DIR, save_filename)
    torch.save(best_model_state, save_path)
    print(f"Saved model to: {save_path} (Best Val QWK: {best_qwk:.4f})", flush=True)
    
    return save_path, best_qwk


def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
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
        'qwk': float(qwk),
        'mae': float(mae),
        'exact_acc': float(exact_acc),
        'within_1_acc': float(within_1_acc),
        'count': int(len(y_true))
    }


def load_model_checkpoint(crop_name, backbone, seed, device):
    model = CropSeverityModel(backbone=backbone, num_classes=config.NUM_SEVERITY_GRADES, ordinal=True, pretrained=False)
    if backbone == 'resnet18' and seed == 42:
        ckpt_path = os.path.join(config.MODELS_DIR, f"{crop_name}_ordinal.pth")
    elif seed == 42:
        ckpt_path = os.path.join(ARCH_MODELS_DIR, f"{crop_name}_{backbone}_ordinal.pth")
    else:
        ckpt_path = os.path.join(ARCH_MODELS_DIR, f"{crop_name}_{backbone}_ordinal_seed{seed}.pth")
            
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    model.to(device)
    model.eval()
    return model


def main():
    print("=" * 75)
    print("ARECANUT MULTI-SEED ARCHITECTURE COMPARISON (3 SEEDS: 42, 100, 2024)")
    print("Backbones: ResNet-18, ResNet-34, MobileNetV2")
    print("=" * 75)
    
    seeds = [42, 100, 2024]
    backbones = ['resnet18', 'resnet34', 'mobilenet_v2']
    
    # 1. Train missing seed checkpoints
    for backbone in backbones:
        for seed in seeds:
            if backbone == 'resnet18' and seed == 42:
                ckpt_path = os.path.join(config.MODELS_DIR, f"arecanut_ordinal.pth")
                if os.path.exists(ckpt_path):
                    print(f"\n[SKIP] Arecanut {backbone} seed 42 checkpoint already exists at {ckpt_path}")
                    continue
            elif seed == 42:
                ckpt_path = os.path.join(ARCH_MODELS_DIR, f"arecanut_{backbone}_ordinal.pth")
                if os.path.exists(ckpt_path):
                    print(f"\n[SKIP] Arecanut {backbone} seed 42 checkpoint already exists at {ckpt_path}")
                    continue
            else:
                ckpt_path = os.path.join(ARCH_MODELS_DIR, f"arecanut_{backbone}_ordinal_seed{seed}.pth")
                if os.path.exists(ckpt_path):
                    print(f"\n[SKIP] Arecanut {backbone} seed {seed} checkpoint already exists at {ckpt_path}")
                    continue
                    
            train_model_seed('arecanut', config.ARECANUT_SPLIT_CSV, backbone, seed)
            
    # 2. Evaluate all seeds on held-out test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = CropDiseaseDataset(config.ARECANUT_SPLIT_CSV, split='test', transform=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    results = {}
    
    # Evaluate multi-seed ResNet-18, ResNet-34 & MobileNetV2
    for backbone in backbones:
        results[backbone] = {}
        for seed in seeds:
            print(f"Evaluating Arecanut | {backbone} | Seed {seed}...")
            model = load_model_checkpoint('arecanut', backbone, seed, device)
            metrics = evaluate_on_test_set(model, test_loader, device)
            results[backbone][seed] = metrics
            
    # Save results to JSON
    json_path = os.path.join(config.REPORTS_DIR, "arecanut_multiseed_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved multi-seed results to {json_path}")
    
    # Print Multi-Seed Summary Table
    print("\n" + "=" * 90)
    print("              ARECANUT MULTI-SEED TEST SET PERFORMANCE SUMMARY (N=124)")
    print("=" * 90)
    print(f"{'Backbone':<15} | {'Seed':<6} | {'QWK':<8} | {'MAE':<8} | {'Exact Acc':<10} | {'Within-1 Acc':<12}")
    print("-" * 90)
    
    for backbone in backbones:
        qwks = [results[backbone][s]['qwk'] for s in seeds]
        maes = [results[backbone][s]['mae'] for s in seeds]
        exacts = [results[backbone][s]['exact_acc'] for s in seeds]
        within1s = [results[backbone][s]['within_1_acc'] for s in seeds]
        
        if backbone == 'resnet18':
            b_name = "ResNet-18"
        elif backbone == 'resnet34':
            b_name = "ResNet-34"
        else:
            b_name = "MobileNetV2"
            
        for s in seeds:
            m = results[backbone][s]
            print(f"{b_name:<15} | {s:<6} | {m['qwk']:.4f}   | {m['mae']:.4f}   | {m['exact_acc']*100:.2f}%     | {m['within_1_acc']*100:.2f}%")
            
        print(f"--> {b_name} QWK Range: [{min(qwks):.4f} - {max(qwks):.4f}] (Mean ± Std: {np.mean(qwks):.4f} ± {np.std(qwks):.4f})")
        print(f"--> {b_name} Exact Acc Range: [{min(exacts)*100:.2f}% - {max(exacts)*100:.2f}%] (Mean ± Std: {np.mean(exacts)*100:.2f}% ± {np.std(exacts)*100:.2f}%)")
        print("-" * 90)


if __name__ == '__main__':
    main()
