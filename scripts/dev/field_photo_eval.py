import os
import sys
import glob
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

import config
import ood_guard
from app import get_disease_classifier, ABSTAIN_THRESHOLDS

def evaluate_crop_field_photos(crop_name: str, photo_dir: str):
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(photo_dir, ext)))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No image files found in {photo_dir}")
        return None

    print(f"\n" + "=" * 80)
    print(f"EVALUATING FIELD PHOTOS FOR {crop_name.upper()} (N = {len(image_paths)})")
    print(f"Directory: {photo_dir}")
    print("=" * 80)

    model, classes, _ = get_disease_classifier(crop_name)
    abstain_tau = ABSTAIN_THRESHOLDS.get(crop_name, 0.70)

    tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    results = []
    for p in image_paths:
        fname = os.path.basename(p)
        try:
            with Image.open(p) as raw_img:
                img = raw_img.convert("RGB")
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            continue

        is_ood, ood_score, ood_thresh = ood_guard.check_ood(img, crop_name)

        if is_ood:
            res_item = {
                "filename": fname,
                "status": "OOD_REJECTED",
                "ood_score": float(ood_score),
                "ood_threshold": float(ood_thresh),
                "pred_cls": None,
                "confidence": None,
            }
            results.append(res_item)
            print(f"  {fname:30s} -> OOD REJECTED (score={ood_score:.4f} > {ood_thresh:.4f})")
            continue

        # In-domain -> run classifier
        t = tf(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_cls = classes[pred_idx]
            conf = float(probs[pred_idx])

        if conf < abstain_tau:
            res_item = {
                "filename": fname,
                "status": "ABSTAINED",
                "ood_score": float(ood_score),
                "ood_threshold": float(ood_thresh),
                "pred_cls": pred_cls,
                "confidence": conf,
            }
            print(f"  {fname:30s} -> ABSTAINED (conf={conf:.4f} < {abstain_tau:.2f}) [raw: {pred_cls}]")
        else:
            res_item = {
                "filename": fname,
                "status": "DIAGNOSED",
                "ood_score": float(ood_score),
                "ood_threshold": float(ood_thresh),
                "pred_cls": pred_cls,
                "confidence": conf,
            }
            print(f"  {fname:30s} -> DIAGNOSED: {pred_cls} (conf={conf:.4f})")

        results.append(res_item)

    n_total = len(results)
    n_ood = sum(1 for r in results if r["status"] == "OOD_REJECTED")
    n_abs = sum(1 for r in results if r["status"] == "ABSTAINED")
    n_diag = sum(1 for r in results if r["status"] == "DIAGNOSED")

    ood_rate = n_ood / n_total if n_total > 0 else 0.0
    abs_rate_total = n_abs / n_total if n_total > 0 else 0.0
    abs_rate_indomain = n_abs / (n_total - n_ood) if (n_total - n_ood) > 0 else 0.0
    diag_rate = n_diag / n_total if n_total > 0 else 0.0

    # Prediction distribution for diagnosed
    pred_counts = {}
    for r in results:
        if r["status"] == "DIAGNOSED":
            cls = r["pred_cls"]
            pred_counts[cls] = pred_counts.get(cls, 0) + 1

    summary = {
        "crop": crop_name,
        "total_evaluated": n_total,
        "ood_rejected_count": n_ood,
        "ood_rejection_rate": ood_rate,
        "abstain_count": n_abs,
        "abstain_rate_total": abs_rate_total,
        "abstain_rate_indomain": abs_rate_indomain,
        "diagnosed_count": n_diag,
        "diagnosed_rate": diag_rate,
        "predicted_labels_diagnosed": pred_counts,
        "detailed_results": results
    }

    print("\n--- Summary for " + crop_name + " ---")
    print(f"  Total Images:      {n_total}")
    print(f"  OOD Rejections:    {n_ood} ({ood_rate:.4%})")
    print(f"  Abstentions:       {n_abs} ({abs_rate_total:.4%} of total, {abs_rate_indomain:.4%} of in-domain)")
    print(f"  Diagnosed:         {n_diag} ({diag_rate:.4%})")
    print(f"  Predicted Labels:  {pred_counts}")
    return summary

def main():
    base_dir = os.path.join(config.PROJECT_ROOT, "field_photos")
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        print("To run phone-photo robustness evaluation, create the folder structure:")
        print("  field_photos/paddy/")
        print("  field_photos/arecanut/")
        print("  field_photos/apple/")
        print("and add phone photos, then run: python field_photo_eval.py")
        return

    crops = ["Paddy", "Arecanut", "Apple"]
    overall_reports = {}

    for c in crops:
        # Check lowercase and capitalized
        candidate_dirs = [
            os.path.join(base_dir, c.lower()),
            os.path.join(base_dir, c),
        ]
        found_dir = None
        for cd in candidate_dirs:
            if os.path.exists(cd) and os.path.isdir(cd):
                found_dir = cd
                break

        if found_dir:
            res = evaluate_crop_field_photos(c, found_dir)
            if res:
                overall_reports[c] = res
        else:
            print(f"No subdirectory found for {c} in {base_dir}")

    if overall_reports:
        out_json = os.path.join(config.OUTPUTS_DIR, "field_photo_report.json")
        with open(out_json, "w") as f:
            json.dump(overall_reports, f, indent=2)
        print(f"\nFull report written to {out_json}")

if __name__ == "__main__":
    main()
