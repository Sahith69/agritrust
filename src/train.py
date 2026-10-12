import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import torch
from torch.optim import AdamW
from sklearn.metrics import cohen_kappa_score
import numpy as np
from tqdm import tqdm
import os
import copy

from src.dataset import get_dataloaders
from src.model import CropSeverityModel
from src.corn_loss import CornLoss, corn_probs_to_grade
from src.ordinal_ce_loss import OrdinalCrossEntropyLoss

def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def train_one_epoch(model, loader, optimizer, loss_fn, device, ordinal=True):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets, _, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        with torch.no_grad():
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(logits, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk

def validate(model, loader, loss_fn, device, ordinal=True):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _, _ in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, targets)
            
            total_loss += loss.item() * images.size(0)
            
            if ordinal:
                probs = torch.sigmoid(logits)
                preds, _ = corn_probs_to_grade(probs)
            else:
                preds = torch.argmax(logits, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = total_loss / len(loader.dataset)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    return epoch_loss, epoch_qwk

def train_model(crop_name, split_csv, ordinal=True, backbone='resnet18'):
    device = torch.device('cpu')
    print(f"Training {crop_name} (Ordinal: {ordinal}) on {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(split_csv, config.BATCH_SIZE, num_workers=0)
    
    model = CropSeverityModel(backbone=backbone, num_classes=config.NUM_SEVERITY_GRADES, ordinal=ordinal)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    if ordinal:
        loss_fn = CornLoss(num_classes=config.NUM_SEVERITY_GRADES)
    else:
        loss_fn = OrdinalCrossEntropyLoss(num_classes=config.NUM_SEVERITY_GRADES)
        
    best_qwk = -1.0
    best_model_state = None
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_qwk': [], 'val_loss': [], 'val_qwk': []}
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, loss_fn, device, ordinal)
        val_loss, val_qwk = validate(model, val_loader, loss_fn, device, ordinal)
        
        history['train_loss'].append(train_loss)
        history['train_qwk'].append(train_qwk)
        history['val_loss'].append(val_loss)
        history['val_qwk'].append(val_qwk)
        
        print(f"Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("  New best QWK!")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Save best model
    config.ensure_dirs()
    model_type = "ordinal" if ordinal else "baseline"
    save_path = os.path.join(config.MODELS_DIR, f"{crop_name}_{model_type}.pth")
    torch.save(best_model_state, save_path)
    print(f"Saved best model to {save_path} with Val QWK: {best_qwk:.4f}")
    
    history['best_val_qwk'] = best_qwk
    return history
