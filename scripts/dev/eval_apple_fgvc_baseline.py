"""
scripts/dev/eval_apple_fgvc_baseline.py - Evaluate Deployed Apple Classifier on FGVC Test Split

Stage 3 Step 3c:
- Evaluates outputs/models/apple_disease_classifier.pth on fgvc_test split (from outputs/apple_fgvc/fgvc_splits.csv).
- App mode: OOD guard check (ood_score > ood_threshold -> "rejected"), then confidence < 0.70 -> "abstained".
- Raw argmax mode: no guards.
- Computes:
  - N per class
  - Share rejected, share abstained
  - Accuracy on diagnosed images
  - Per-class recall over ALL images (rejections and abstains counted as misses)
  - Per-class recall over diagnosed only
  - Macro-F1 and accuracy with 95% bootstrap CIs (1,000 resamples, seed 42)
  - Confusion matrix with "rejected" and "abstained" columns
  - Mean confidence for correct vs wrong predictions
  - 12 example predictions with filenames
- Evaluates STOP RULE:
  If raw-argmax accuracy >= 90% AND scab & rust recall >= 85%, training is skipped.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torchvision import models, transforms

import sys
sys.path.insert(0, os.path.abspath("."))

import config
from ood_guard import check_ood, get_image_transform

SPLIT_CSV = "outputs/apple_fgvc/fgvc_splits.csv"
MODEL_PATH = "outputs/models/apple_disease_classifier.pth"
OUT_DIR = "outputs/apple_fgvc"


def load_classifier():
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = ck["classes"]
    model = models.resnet18(weights=None)
    if "fc.1.weight" in ck["model_state_dict"]:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.fc.in_features, len(classes)))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, classes


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_func(y_true[idx], y_pred[idx]))
    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return lower, upper


def evaluate_baseline():
    print(f"Loading FGVC splits from {SPLIT_CSV}...")
    df = pd.read_csv(SPLIT_CSV)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    n_test = len(test_df)
    print(f"Loaded {n_test} frozen fgvc_test images.")
    print("Class distribution:\n", test_df["disease_class"].value_counts())

    model, classes = load_classifier()
    tf = get_image_transform()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    results = []
    print("\nRunning inference and OOD guard across all test images...")
    t0 = time.time()

    for idx, row in test_df.iterrows():
        img_path = row["image_path"]
        true_cls = row["disease_class"]
        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            # 1. OOD check
            is_ood, ood_score, ood_thresh = check_ood(rgb_img, "Apple")

            # 2. Classifier
            tensor = tf(rgb_img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            raw_pred_idx = int(np.argmax(probs))
            raw_pred_cls = classes[raw_pred_idx]
            raw_conf = float(probs[raw_pred_idx])

            # Guarded pipeline
            if is_ood:
                guarded_status = "rejected"
                guarded_pred = "rejected"
            elif raw_conf < 0.70:
                guarded_status = "abstained"
                guarded_pred = "abstained"
            else:
                guarded_status = "diagnosed"
                guarded_pred = raw_pred_cls

            results.append({
                "image_id": row["image_id"],
                "filename": os.path.basename(img_path),
                "true_class": true_cls,
                "raw_pred": raw_pred_cls,
                "raw_conf": raw_conf,
                "probs": {c: float(probs[i]) for i, c in enumerate(classes)},
                "ood_score": ood_score,
                "ood_threshold": ood_thresh,
                "is_ood": is_ood,
                "guarded_status": guarded_status,
                "guarded_pred": guarded_pred,
            })

    print(f"Evaluated {n_test} test images in {time.time() - t0:.2f}s.")
    res_df = pd.DataFrame(results)

    # ── 1. GUARDED METRICS
    n_rejected = (res_df["guarded_status"] == "rejected").sum()
    n_abstained = (res_df["guarded_status"] == "abstained").sum()
    n_diagnosed = (res_df["guarded_status"] == "diagnosed").sum()

    share_rejected = n_rejected / n_test
    share_abstained = n_abstained / n_test

    diagnosed_df = res_df[res_df["guarded_status"] == "diagnosed"]
    acc_diagnosed = accuracy_score(diagnosed_df["true_class"], diagnosed_df["guarded_pred"]) if len(diagnosed_df) > 0 else 0.0

    # Recall over ALL images (rejections & abstains count as misses)
    recalls_all = {}
    recalls_diag = {}
    test_classes = ["apple_scab", "cedar_apple_rust", "healthy"]
    for c in test_classes:
        c_sub = res_df[res_df["true_class"] == c]
        recalls_all[c] = (c_sub["guarded_pred"] == c).sum() / len(c_sub)

        c_diag = c_sub[c_sub["guarded_status"] == "diagnosed"]
        recalls_diag[c] = (c_diag["guarded_pred"] == c).sum() / len(c_diag) if len(c_diag) > 0 else 0.0

    # Macro-F1 and accuracy on ALL images (guarded: missed if rejected/abstained)
    # Map predictions to integer indices, assigning a distinct index for rejected/abstained
    y_true = np.array([class_to_idx[c] for c in res_df["true_class"]])
    y_pred_guarded = np.array([class_to_idx.get(c, -1) for c in res_df["guarded_pred"]])

    # Overall guarded accuracy across ALL test images
    acc_guarded_all = (y_true == y_pred_guarded).mean()
    acc_g_ci = bootstrap_ci(y_true, y_pred_guarded, lambda yt, yp: (yt == yp).mean())

    # Macro-F1 across the 3 test classes
    def calc_macro_f1_3(yt, yp):
        f1s = []
        for c_idx in [class_to_idx[c] for c in test_classes]:
            tp = np.sum((yt == c_idx) & (yp == c_idx))
            fp = np.sum((yt != c_idx) & (yp == c_idx))
            fn = np.sum((yt == c_idx) & (yp != c_idx))
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s))

    macro_f1_guarded = calc_macro_f1_3(y_true, y_pred_guarded)
    macro_f1_g_ci = bootstrap_ci(y_true, y_pred_guarded, calc_macro_f1_3)

    # Confusion matrix (Guarded: Rows = True, Cols = apple_scab, black_rot, cedar_apple_rust, healthy, abstained, rejected)
    all_cols = classes + ["abstained", "rejected"]
    cm_guarded = pd.DataFrame(0, index=test_classes, columns=all_cols)
    for _, r in res_df.iterrows():
        cm_guarded.loc[r["true_class"], r["guarded_pred"]] += 1

    # Confidence analysis
    if len(diagnosed_df) > 0:
        correct_mask = (diagnosed_df["true_class"] == diagnosed_df["guarded_pred"])
        mean_conf_correct = float(diagnosed_df.loc[correct_mask, "raw_conf"].mean()) if correct_mask.sum() > 0 else 0.0
        mean_conf_wrong = float(diagnosed_df.loc[~correct_mask, "raw_conf"].mean()) if (~correct_mask).sum() > 0 else 0.0
    else:
        mean_conf_correct, mean_conf_wrong = 0.0, 0.0

    # ── 2. RAW ARGMAX METRICS
    y_pred_raw = np.array([class_to_idx[c] for c in res_df["raw_pred"]])
    acc_raw = accuracy_score(y_true, y_pred_raw)
    acc_raw_ci = bootstrap_ci(y_true, y_pred_raw, accuracy_score)
    macro_f1_raw = calc_macro_f1_3(y_true, y_pred_raw)
    macro_f1_raw_ci = bootstrap_ci(y_true, y_pred_raw, calc_macro_f1_3)

    recalls_raw = {}
    for c in test_classes:
        c_sub = res_df[res_df["true_class"] == c]
        recalls_raw[c] = (c_sub["raw_pred"] == c).sum() / len(c_sub)

    cm_raw = pd.DataFrame(0, index=test_classes, columns=classes)
    for _, r in res_df.iterrows():
        cm_raw.loc[r["true_class"], r["raw_pred"]] += 1

    # ── 3. 12 EXAMPLE PREDICTIONS (4 per class)
    examples = []
    for c in test_classes:
        sub = res_df[res_df["true_class"] == c].head(4)
        for _, r in sub.iterrows():
            examples.append({
                "filename": r["filename"],
                "true_class": r["true_class"],
                "raw_pred": r["raw_pred"],
                "raw_conf": f"{r['raw_conf']:.4f}",
                "ood_score": f"{r['ood_score']:.2f}",
                "ood_thresh": f"{r['ood_threshold']:.2f}",
                "guarded_pred": r["guarded_pred"],
                "status": r["guarded_status"]
            })

    # ── 4. STOP RULE CHECK
    scab_recall_raw = recalls_raw["apple_scab"]
    rust_recall_raw = recalls_raw["cedar_apple_rust"]
    stop_rule_met = (acc_raw >= 0.90) and (scab_recall_raw >= 0.85) and (rust_recall_raw >= 0.85)

    report_data = {
        "n_total": n_test,
        "class_counts": {c: int((test_df["disease_class"] == c).sum()) for c in test_classes},
        "guarded": {
            "n_rejected": int(n_rejected),
            "share_rejected": float(share_rejected),
            "n_abstained": int(n_abstained),
            "share_abstained": float(share_abstained),
            "n_diagnosed": int(n_diagnosed),
            "accuracy_diagnosed": float(acc_diagnosed),
            "accuracy_all": float(acc_guarded_all),
            "accuracy_all_ci_95": acc_g_ci,
            "macro_f1_all": float(macro_f1_guarded),
            "macro_f1_all_ci_95": macro_f1_g_ci,
            "recalls_all": {k: float(v) for k, v in recalls_all.items()},
            "recalls_diagnosed": {k: float(v) for k, v in recalls_diag.items()},
            "confusion_matrix": cm_guarded.to_dict(),
            "mean_conf_correct": mean_conf_correct,
            "mean_conf_wrong": mean_conf_wrong,
        },
        "raw_argmax": {
            "accuracy": float(acc_raw),
            "accuracy_ci_95": acc_raw_ci,
            "macro_f1": float(macro_f1_raw),
            "macro_f1_ci_95": macro_f1_raw_ci,
            "recalls": {k: float(v) for k, v in recalls_raw.items()},
            "confusion_matrix": cm_raw.to_dict(),
        },
        "stop_rule": {
            "raw_accuracy": float(acc_raw),
            "scab_recall": float(scab_recall_raw),
            "rust_recall": float(rust_recall_raw),
            "stop_rule_met": bool(stop_rule_met),
            "verdict": "training skipped: baseline adequate" if stop_rule_met else "training required"
        },
        "examples_12": examples
    }

    out_json = os.path.join(OUT_DIR, "baseline_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved baseline evaluation results to {out_json}")

    # Print summary to terminal
    print("\n" + "="*80)
    print("STAGE 3c BASELINE EVALUATION REPORT: DEPLOYED APPLE MODEL ON FGVC TEST")
    print("="*80)
    print(f"Total test images: {n_test}")
    for c in test_classes:
        print(f"  - {c}: {report_data['class_counts'][c]}")
    print(f"\n[Guarded Pipeline: OOD + Abstain 0.70]")
    print(f"  Share rejected (OOD): {share_rejected*100:.2f}% ({n_rejected}/{n_test})")
    print(f"  Share abstained (<0.70): {share_abstained*100:.2f}% ({n_abstained}/{n_test})")
    print(f"  Accuracy on diagnosed images: {acc_diagnosed*100:.2f}% ({int(acc_diagnosed * n_diagnosed)}/{n_diagnosed})")
    print(f"  Accuracy across ALL images: {acc_guarded_all*100:.2f}% (95% CI: [{acc_g_ci[0]*100:.2f}%, {acc_g_ci[1]*100:.2f}%])")
    print(f"  Macro-F1 across ALL images: {macro_f1_guarded:.4f} (95% CI: [{macro_f1_g_ci[0]:.4f}, {macro_f1_g_ci[1]:.4f}])")
    print(f"  Per-class recall (ALL images):")
    for c, r in recalls_all.items():
        print(f"    • {c}: {r*100:.2f}%")
    print(f"  Per-class recall (Diagnosed only):")
    for c, r in recalls_diag.items():
        print(f"    • {c}: {r*100:.2f}%")
    print(f"  Mean confidence (correct predictions): {mean_conf_correct:.4f}")
    print(f"  Mean confidence (wrong predictions):   {mean_conf_wrong:.4f}")
    print(f"\nGuarded Confusion Matrix:")
    print(cm_guarded)

    print(f"\n[Raw Argmax Pipeline (No Guards)]")
    print(f"  Raw Accuracy: {acc_raw*100:.2f}% (95% CI: [{acc_raw_ci[0]*100:.2f}%, {acc_raw_ci[1]*100:.2f}%])")
    print(f"  Raw Macro-F1: {macro_f1_raw:.4f} (95% CI: [{macro_f1_raw_ci[0]:.4f}, {macro_f1_raw_ci[1]:.4f}])")
    print(f"  Per-class recall:")
    for c, r in recalls_raw.items():
        print(f"    • {c}: {r*100:.2f}%")
    print(f"\nRaw Confusion Matrix:")
    print(cm_raw)

    print("\n" + "="*80)
    print("STOP RULE EVALUATION")
    print(f"  Criteria: Raw Accuracy >= 90.0% AND Scab Recall >= 85.0% AND Rust Recall >= 85.0%")
    print(f"  Achieved: Raw Accuracy = {acc_raw*100:.2f}%, Scab = {scab_recall_raw*100:.2f}%, Rust = {rust_recall_raw*100:.2f}%")
    print(f"  Verdict:  {report_data['stop_rule']['verdict']}")
    print("="*80 + "\n")

    return report_data


if __name__ == "__main__":
    evaluate_baseline()
