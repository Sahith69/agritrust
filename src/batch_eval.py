import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

from src.inference import load_model
from src.corn_loss import corn_probs_to_grade

class EvalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['severity_grade'])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def run_batch_evaluation(crop_name, model_type='ordinal', split='test'):
    device = torch.device('cpu')
    ordinal = (model_type == 'ordinal')
    model_filename = f"{crop_name}_{model_type}.pth"
    model_path = os.path.join(config.MODELS_DIR, model_filename)
    
    model = load_model(model_path, crop_name, ordinal)
    
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    split_csv = config.PADDY_SPLIT_CSV if crop_name == 'paddy' else config.ARECANUT_SPLIT_CSV
    if not os.path.exists(split_csv):
        print(f"Error: Split CSV {split_csv} not found.")
        return
        
    df = pd.read_csv(split_csv)
    eval_df = df[df['split'] == split]
    
    dataset = EvalDataset(eval_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_paths = []
    all_confidences = []
    
    print(f"Running batch evaluation on {len(eval_df)} images ({crop_name}, {model_type}, {split} split)...")
    
    for images, labels, paths in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            
            if ordinal:
                sig_probs = torch.sigmoid(logits)
                pred_grades, grade_probs = corn_probs_to_grade(sig_probs)
                preds = pred_grades.cpu().numpy()
                confs = grade_probs.max(dim=1).values.cpu().numpy()
            else:
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                confs = np.max(probs, axis=1)
                
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
        all_paths.extend(paths)
        all_confidences.extend(confs)
        
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    acc = accuracy_score(all_targets, all_preds)
    
    print("\n" + "="*40)
    print(f"Results for {crop_name} - {model_type}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    
    results_df = pd.DataFrame({
        'image_path': all_paths,
        'true_grade': all_targets,
        'pred_grade': all_preds,
        'confidence': all_confidences
    })
    results_csv = os.path.join(config.REPORTS_DIR, f"{crop_name}_{model_type}_{split}_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(config.NUM_SEVERITY_GRADES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f"Pred {i}" for i in range(config.NUM_SEVERITY_GRADES)],
                yticklabels=[f"True {i}" for i in range(config.NUM_SEVERITY_GRADES)])
    plt.title(f'Confusion Matrix: {crop_name} ({model_type})')
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    
    cm_plot_path = os.path.join(config.REPORTS_DIR, f"{crop_name}_{model_type}_{split}_cm.png")
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation")
    parser.add_argument("--crop", default="paddy", choices=["paddy", "arecanut"], help="Crop name")
    parser.add_argument("--model", default="ordinal", choices=["ordinal", "baseline"], help="Model type")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_batch_evaluation(args.crop, args.model, args.split)
