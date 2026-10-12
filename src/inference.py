import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import CropSeverityModel
from src.gradcam import GradCAM, overlay_heatmap
from src.corn_loss import corn_probs_to_grade


def load_model(model_path, crop_name, ordinal=True):
    """Load a saved model checkpoint."""
    device = torch.device('cpu')
    model = CropSeverityModel(
        backbone='resnet18',
        num_classes=config.NUM_SEVERITY_GRADES,
        ordinal=ordinal,
        pretrained=False
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: Model path {model_path} does not exist. Using untrained model.")
    model.to(device)
    model.eval()
    return model


def predict_single(image_path, model, transform, device, ordinal=True):
    """
    Run inference on a single image.
    Returns:
        predicted_grade (int): 0-3
        per_grade_probabilities (np.array): 4 values
        gradcam_overlay (PIL.Image): Grad-CAM overlay on original
    """
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)

        if ordinal:
            probs = torch.sigmoid(logits)
            pred_grade, grade_probs = corn_probs_to_grade(probs[0])
            predicted_grade = pred_grade.item()
            per_grade_probabilities = grade_probs.cpu().numpy()
        else:
            probs = F.softmax(logits, dim=1)
            per_grade_probabilities = probs[0].cpu().numpy()
            predicted_grade = int(np.argmax(per_grade_probabilities))

    # Grad-CAM (requires gradients, so outside no_grad)
    try:
        gradcam = GradCAM(model, config.GRADCAM_TARGET_LAYER)
        input_for_cam = transform(image).unsqueeze(0).to(device)
        input_for_cam.requires_grad_(True)
        heatmap = gradcam.generate(input_for_cam, target_class=predicted_grade)
        gradcam_overlay = overlay_heatmap(image, heatmap, alpha=0.5)
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        gradcam_overlay = image

    return predicted_grade, per_grade_probabilities, gradcam_overlay


def run_inference(image_path, crop_name='paddy', model_type='ordinal', output_path=None):
    """End-to-end inference: loads model, predicts, generates overlay."""
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

    print(f"Running inference on: {image_path}")
    print(f"Crop: {crop_name}, Model Type: {model_type}")

    pred_grade, probs, overlay = predict_single(image_path, model, transform, device, ordinal)

    grade_names = ["Healthy", "Mild", "Moderate", "Severe"]
    print("-" * 40)
    print(f"Predicted Grade: {pred_grade} ({grade_names[pred_grade]})")
    print(f"Per-grade probabilities:")
    for i, p in enumerate(probs):
        marker = " ◄" if i == pred_grade else ""
        print(f"  Grade {i} ({grade_names[i]}): {p:.4f}{marker}")
    print("-" * 40)

    if output_path:
        overlay.save(output_path)
        print(f"Saved Grad-CAM overlay to {output_path}")

    return pred_grade, probs, overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image inference for crop disease severity")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--crop", default="paddy", choices=["paddy", "arecanut"], help="Crop name")
    parser.add_argument("--model", default="ordinal", choices=["ordinal", "baseline"], help="Model type")
    parser.add_argument("--output", default="output.png", help="Path to save Grad-CAM overlay")
    args = parser.parse_args()

    run_inference(args.image, args.crop, args.model, args.output)
