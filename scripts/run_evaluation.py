"""
Phase 4+5 Runner: Explainability (Grad-CAM + IoU) and Evaluation.
Loads trained models, runs evaluation, generates Grad-CAM, computes IoU, produces reports.
"""
import sys; sys.path.insert(0, '/home/sahith/Projects/Crop'); import config
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

from src.model import CropSeverityModel
from src.evaluate import generate_full_report, error_analysis
from src.iou_validation import validate_explainability
from src.segmentation import segment_leaf_adaptive, detect_lesions_group_a
from src.dataset import CropDiseaseDataset, get_transforms


def segmentation_fn(img_np):
    """Wrapper for Phase 1 segmentation to match IoU validation interface."""
    leaf_mask, _ = segment_leaf_adaptive(img_np)
    lesion_mask = detect_lesions_group_a(img_np, leaf_mask)
    return lesion_mask


def load_trained_model(crop_name, model_type, device):
    """Load a trained CropSeverityModel checkpoint."""
    ordinal = (model_type == 'ordinal')
    model_path = os.path.join(config.MODELS_DIR, f"{crop_name}_{model_type}.pth")

    model = CropSeverityModel(
        backbone='resnet18',
        num_classes=config.NUM_SEVERITY_GRADES,
        ordinal=ordinal,
        pretrained=False
    )

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"  Loaded model: {model_path}")
    else:
        print(f"  Warning: {model_path} not found. Using untrained model.")

    model.to(device)
    model.eval()
    return model


def run_evaluation_for_crop(crop_name, split_csv):
    device = torch.device('cpu')
    config.ensure_dirs()

    transform = get_transforms('test')

    # Build test DataLoader
    test_dataset = CropDiseaseDataset(split_csv, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"  Test set: {len(test_dataset)} images")

    # Load models
    ordinal_model = load_trained_model(crop_name, 'ordinal', device)
    baseline_model = load_trained_model(crop_name, 'baseline', device)

    # Grad-CAM + IoU validation
    print(f"\n  Running Grad-CAM XAI validation ({config.GRADCAM_NUM_SAMPLES} samples)...")
    iou_scores = validate_explainability(
        model=ordinal_model,
        test_csv=split_csv,
        segmentation_fn=segmentation_fn,
        transform=transform,
        device=device,
        num_samples=config.GRADCAM_NUM_SAMPLES
    )

    # Full report
    print(f"\n  Generating evaluation report...")
    generate_full_report(
        crop_name=crop_name,
        ordinal_model=ordinal_model,
        baseline_model=baseline_model,
        test_loader=test_loader,
        iou_scores=iou_scores,
        device=device,
        save_dir=config.REPORTS_DIR
    )

    # Error analysis
    print(f"\n  Error analysis (ordinal model)...")
    top_errors = error_analysis(ordinal_model, test_loader, device, ordinal=True, top_n=10)
    if top_errors:
        print("  Top misclassifications:")
        for err in top_errors:
            print(f"    Batch {err['batch_idx']}, Item {err['item_idx']}: "
                  f"True={err['true']}, Pred={err['pred']}, Gap={err['diff']}")


def main():
    print("=" * 60)
    print("  EVALUATION & EXPLAINABILITY PIPELINE")
    print("=" * 60)

    if os.path.exists(config.ARECANUT_SPLIT_CSV):
        print("\n" + "-" * 40)
        print("  Evaluating: ARECANUT")
        print("-" * 40)
        run_evaluation_for_crop("arecanut", config.ARECANUT_SPLIT_CSV)

    if os.path.exists(config.PADDY_SPLIT_CSV):
        print("\n" + "-" * 40)
        print("  Evaluating: PADDY")
        print("-" * 40)
        run_evaluation_for_crop("paddy", config.PADDY_SPLIT_CSV)

    if not os.path.exists(config.ARECANUT_SPLIT_CSV) and not os.path.exists(config.PADDY_SPLIT_CSV):
        print("No split CSVs found. Run Phase 2 first.")

    print("\n✅ Evaluation complete. Reports saved to:", config.REPORTS_DIR)


if __name__ == '__main__':
    main()
