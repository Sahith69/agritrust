import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image

from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.corn_loss import corn_probs_to_grade
from src.gradcam import GradCAM, overlay_heatmap
from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    detect_dead_heart,
    detect_hispa
)

torch.set_num_threads(4)
device = 'cpu'
gallery_dir = '/home/sahith/Projects/Crop/outputs/improvements/gradcam_failure_gallery'
os.makedirs(gallery_dir, exist_ok=True)

def get_symptom_mask(img_np, disease_class):
    try:
        leaf_mask = segment_leaf_adaptive(img_np)
        if disease_class == 'dead_heart':
            mask = detect_dead_heart(img_np, leaf_mask)
        elif disease_class == 'hispa':
            mask = detect_hispa(img_np, leaf_mask)
        elif disease_class in config.PADDY_GROUP_A or disease_class in config.ARECANUT_GROUP_A:
            mask = detect_lesions_group_a(img_np, leaf_mask)
        elif disease_class == 'normal' or disease_class in [config.ARECANUT_CLASSES[c] for c in config.ARECANUT_HEALTHY_CLASSES]:
            mask = leaf_mask
        else:
            mask = detect_lesions_group_a(img_np, leaf_mask)
        return mask
    except Exception:
        return None

def evaluate_crop_pointing_game(crop_name, model_path, split_csv, max_eval=300):
    print(f"\n--- Evaluating Pointing Game for {crop_name.upper()} ---")
    t0 = time.time()
    model = CropSeverityModel(backbone='resnet18', num_classes=config.NUM_SEVERITY_GRADES, ordinal=True, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    gradcam = GradCAM(model, target_layer_name='layer4')
    transform = get_transforms('test')
    
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    eval_df = test_df if len(test_df) <= max_eval else test_df.sample(max_eval, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    hits_diseased = 0
    total_diseased = 0
    hits_overall = 0
    total_overall = 0
    
    per_class_stats = {}
    saved_hits = 0
    saved_misses = 0
    saved_gallery_files = []
    
    for idx, row in eval_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
            
        true_grade = int(row['severity_grade'])
        disease_cls = row['disease_class']
        
        try:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            
            # Predict
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.sigmoid(out)
                pred_grade_t, _ = corn_probs_to_grade(probs)
                pred_grade = int(pred_grade_t.item())
                
            # Grad-CAM heatmap (224, 224)
            cam = gradcam.generate(input_tensor, target_class=pred_grade)
            
            # For symptom mask, downsample large images to max 800px to ensure fast & robust segmentation
            if max(w, h) > 800:
                scale = 800 / max(w, h)
                img_for_mask = image.resize((int(w * scale), int(h * scale)))
            else:
                img_for_mask = image
                
            symptom_mask = get_symptom_mask(np.array(img_for_mask), disease_cls)
            if symptom_mask is None:
                continue
                
            # Resize mask to (224, 224) to match CAM grid directly
            mask_pil = Image.fromarray((symptom_mask * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
            symptom_mask_224 = np.array(mask_pil) > 0
            
            # Highest activation pixel on 224x224 grid
            max_idx = np.argmax(cam)
            r_max, c_max = np.unravel_index(max_idx, cam.shape)
            
            is_hit = bool(symptom_mask_224[r_max, c_max])
            
            # Tally stats
            total_overall += 1
            if is_hit:
                hits_overall += 1
                
            is_diseased = (true_grade > 0 and disease_cls not in ['normal', 'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'])
            if is_diseased:
                total_diseased += 1
                if is_hit:
                    hits_diseased += 1
                    
            if disease_cls not in per_class_stats:
                per_class_stats[disease_cls] = {'hits': 0, 'total': 0, 'diseased_hits': 0, 'diseased_total': 0}
            per_class_stats[disease_cls]['total'] += 1
            if is_hit:
                per_class_stats[disease_cls]['hits'] += 1
            if is_diseased:
                per_class_stats[disease_cls]['diseased_total'] += 1
                if is_hit:
                    per_class_stats[disease_cls]['diseased_hits'] += 1
                    
            # Save up to 3 hits for gallery
            if is_hit and true_grade == pred_grade and is_diseased and saved_hits < 3:
                saved_hits += 1
                overlay = overlay_heatmap(image.resize((400, 400)), cam, alpha=0.5)
                fname = f"{crop_name}_hit_{disease_cls}_trueG{true_grade}_predG{pred_grade}_ex{saved_hits}.png"
                fpath = os.path.join(gallery_dir, fname)
                overlay.save(fpath)
                saved_gallery_files.append({
                    "type": "hit",
                    "file": fpath,
                    "details": {"disease_cls": disease_cls, "true_grade": true_grade, "pred_grade": pred_grade}
                })
                
            # Save up to 3 misses / shortcuts for gallery
            if not is_hit and abs(true_grade - pred_grade) <= 1 and is_diseased and saved_misses < 3:
                saved_misses += 1
                overlay = overlay_heatmap(image.resize((400, 400)), cam, alpha=0.5)
                fname = f"{crop_name}_shortcut_{disease_cls}_trueG{true_grade}_predG{pred_grade}_ex{saved_misses}.png"
                fpath = os.path.join(gallery_dir, fname)
                overlay.save(fpath)
                saved_gallery_files.append({
                    "type": "shortcut_miss",
                    "file": fpath,
                    "details": {"disease_cls": disease_cls, "true_grade": true_grade, "pred_grade": pred_grade}
                })
                
        except Exception as e:
            continue
            
    diseased_acc = (hits_diseased / total_diseased * 100.0) if total_diseased > 0 else 0.0
    overall_acc = (hits_overall / total_overall * 100.0) if total_overall > 0 else 0.0
    
    print(f"[{crop_name}] Diseased Pointing Game Accuracy: {diseased_acc:.2f}% ({hits_diseased}/{total_diseased})")
    print(f"[{crop_name}] Overall Pointing Game Accuracy: {overall_acc:.2f}% ({hits_overall}/{total_overall})")
    print(f"[{crop_name}] Elapsed time: {time.time() - t0:.2f}s")
    
    class_summary = {}
    for c, stats in per_class_stats.items():
        class_summary[c] = {
            "total_samples": stats['total'],
            "pointing_accuracy_pct": round(stats['hits'] / stats['total'] * 100.0, 2) if stats['total'] > 0 else 0.0,
            "diseased_samples": stats['diseased_total'],
            "diseased_pointing_accuracy_pct": round(stats['diseased_hits'] / stats['diseased_total'] * 100.0, 2) if stats['diseased_total'] > 0 else None
        }
        
    return {
        "crop": crop_name,
        "model": "ResNet-18 (CORN)",
        "pointing_game_accuracy_diseased_pct": round(diseased_acc, 2),
        "pointing_game_accuracy_overall_pct": round(overall_acc, 2),
        "total_evaluated_samples": total_overall,
        "total_diseased_samples": total_diseased,
        "hits_diseased": hits_diseased,
        "per_class_pointing_accuracy": class_summary,
        "gallery_examples": saved_gallery_files
    }

paddy_results = evaluate_crop_pointing_game(
    'paddy',
    os.path.join(config.MODELS_DIR, 'paddy_ordinal.pth'),
    config.PADDY_SPLIT_CSV,
    max_eval=300
)

arecanut_results = evaluate_crop_pointing_game(
    'arecanut',
    os.path.join(config.MODELS_DIR, 'arecanut_ordinal.pth'),
    config.ARECANUT_SPLIT_CSV,
    max_eval=124
)

final_data = {
    "definition": "Pointing Game Accuracy measures the percentage of test images where the single maximum activation pixel in the Grad-CAM saliency map falls directly inside the ground-truth/symptom lesion mask.",
    "paddy": paddy_results,
    "arecanut": arecanut_results
}

out_file = '/home/sahith/Projects/Crop/outputs/improvements/gradcam_pointing_game.json'
with open(out_file, 'w') as f:
    json.dump(final_data, f, indent=2)

print(f"\nSaved pointing game results to {out_file}")
