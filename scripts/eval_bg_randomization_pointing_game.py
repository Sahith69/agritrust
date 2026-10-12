import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

IMPROVEMENTS_DIR = "/home/sahith/Projects/Crop/outputs/improvements"
GALLERY_DIR = os.path.join(IMPROVEMENTS_DIR, "bg_randomization_gallery")
os.makedirs(GALLERY_DIR, exist_ok=True)


class GradCAMPlusPlus:
    """Grad-CAM++ formulation for higher-order gradient visualization with CORN ordinal support."""
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        target_layer = None
        if hasattr(self.model, 'backbone'):
            for name, module in self.model.backbone.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break
        else:
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break

        if target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found in model.")

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            if output.shape[1] == config.NUM_ORDINAL_TASKS:
                probs = torch.sigmoid(output)
                pred_grade, _ = corn_probs_to_grade(probs)
                target_class = pred_grade.item()
            else:
                target_class = output.argmax(dim=1).item()

        if output.shape[1] == config.NUM_ORDINAL_TASKS:
            if target_class == 0:
                target = -output[0, 0]
            elif target_class == config.NUM_ORDINAL_TASKS:
                target = output[0, target_class - 1]
            else:
                target = output[0, target_class - 1]
        else:
            target = output[0, target_class]

        target.backward()

        grads = self.gradients.detach()
        activations = self.activations.detach()

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-7

        ai_j = grads_power_2 / (2.0 * grads_power_2 + sum_activations * grads_power_3 + eps)
        ai_j = torch.where(grads != 0, ai_j, torch.zeros_like(ai_j))

        weights = torch.maximum(grads, torch.zeros_like(grads)) * ai_j
        weights = torch.sum(weights, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


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


def run_single_checkpoint_eval(model, gc, gcpp, precomputed_items):
    """Evaluates a single model checkpoint on precomputed items."""
    results = []
    hits_gc = 0
    hits_gcpp = 0
    hits_diseased_gc = 0
    hits_diseased_gcpp = 0
    total_samples = 0
    total_diseased = 0

    per_class = {}

    for item in precomputed_items:
        input_tensor = item['tensor']
        true_g = item['true_grade']
        cls_name = item['disease_class']
        mask_224 = item['mask_224']

        with torch.no_grad():
            out = model(input_tensor)
            probs = torch.sigmoid(out)
            pred_t, _ = corn_probs_to_grade(probs)
            pred_g = int(pred_t.item())

        cam_gc = gc.generate(input_tensor, target_class=pred_g)
        cam_gcpp = gcpp.generate(input_tensor, target_class=pred_g)

        # Pointing evaluation
        def check_hit(cam):
            max_idx = np.argmax(cam)
            r, c = np.unravel_index(max_idx, cam.shape)
            return bool(mask_224[r, c]), (int(r), int(c))

        hit_gc, peak_gc = check_hit(cam_gc)
        hit_gcpp, peak_gcpp = check_hit(cam_gcpp)

        is_diseased = item['is_diseased']

        total_samples += 1
        if hit_gc: hits_gc += 1
        if hit_gcpp: hits_gcpp += 1

        if is_diseased:
            total_diseased += 1
            if hit_gc: hits_diseased_gc += 1
            if hit_gcpp: hits_diseased_gcpp += 1

        if cls_name not in per_class:
            per_class[cls_name] = {
                'total': 0, 'diseased_total': 0,
                'gc_hits': 0, 'gcpp_hits': 0,
                'diseased_gc_hits': 0, 'diseased_gcpp_hits': 0
            }
        per_class[cls_name]['total'] += 1
        if hit_gc: per_class[cls_name]['gc_hits'] += 1
        if hit_gcpp: per_class[cls_name]['gcpp_hits'] += 1
        if is_diseased:
            per_class[cls_name]['diseased_total'] += 1
            if hit_gc: per_class[cls_name]['diseased_gc_hits'] += 1
            if hit_gcpp: per_class[cls_name]['diseased_gcpp_hits'] += 1

        results.append({
            'idx': item['idx'],
            'image_path': item['image_path'],
            'disease_class': cls_name,
            'true_grade': true_g,
            'pred_grade': pred_g,
            'is_diseased': is_diseased,
            'hit_gc': hit_gc,
            'peak_gc': peak_gc,
            'hit_gcpp': hit_gcpp,
            'peak_gcpp': peak_gcpp,
            'cam_gc': cam_gc,
            'cam_gcpp': cam_gcpp,
            'mask_224': mask_224,
            'image': item['image']
        })

    def calc_pct(hits, tot):
        return round(hits / tot * 100.0, 2) if tot > 0 else 0.0

    stats = {
        'total_samples': total_samples,
        'total_diseased': total_diseased,
        'overall_gc_hits': hits_gc,
        'overall_gcpp_hits': hits_gcpp,
        'overall_gc_pct': calc_pct(hits_gc, total_samples),
        'overall_gcpp_pct': calc_pct(hits_gcpp, total_samples),
        'diseased_gc_hits': hits_diseased_gc,
        'diseased_gcpp_hits': hits_diseased_gcpp,
        'diseased_gc_pct': calc_pct(hits_diseased_gc, total_diseased),
        'diseased_gcpp_pct': calc_pct(hits_diseased_gcpp, total_diseased),
        'per_class': {}
    }

    for c, d in per_class.items():
        stats['per_class'][c] = {
            'total_samples': d['total'],
            'diseased_samples': d['diseased_total'],
            'gc_hits': d['gc_hits'],
            'gcpp_hits': d['gcpp_hits'],
            'gc_pct': calc_pct(d['gc_hits'], d['total']),
            'gcpp_pct': calc_pct(d['gcpp_hits'], d['total']),
            'diseased_gc_pct': calc_pct(d['diseased_gc_hits'], d['diseased_total']) if d['diseased_total'] > 0 else None,
            'diseased_gcpp_pct': calc_pct(d['diseased_gcpp_hits'], d['diseased_total']) if d['diseased_total'] > 0 else None
        }

    return stats, results


def main():
    print("=" * 70)
    print("  STEP 3: POINTING-GAME PROTOCOL RE-RUN (BASELINE vs. BG-RANDOMIZATION)")
    print("=" * 70)

    transform = get_transforms('test')
    df = pd.read_csv(config.ARECANUT_SPLIT_CSV)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    print(f"Pre-computing masks for {len(test_df)} test samples...")
    t0 = time.time()

    precomputed = []
    for idx, row in test_df.iterrows():
        p = row['image_path']
        if not os.path.exists(p):
            continue
        true_g = int(row['severity_grade'])
        cls_name = row['disease_class']

        image = Image.open(p).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        w, h = image.size
        scale = 800 / max(w, h) if max(w, h) > 800 else 1.0
        img_for_mask = image.resize((int(w * scale), int(h * scale))) if scale < 1.0 else image
        symptom_mask = get_symptom_mask(np.array(img_for_mask), cls_name)
        if symptom_mask is None:
            continue

        mask_224 = np.array(Image.fromarray((symptom_mask * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)) > 0
        is_diseased = (true_g > 0 and cls_name not in ['normal', 'healthy_foot', 'healthy_leaf', 'healthy_nut', 'healthy_trunk'])

        precomputed.append({
            'idx': idx,
            'image_path': p,
            'disease_class': cls_name,
            'true_grade': true_g,
            'is_diseased': is_diseased,
            'image': image,
            'tensor': input_tensor,
            'mask_224': mask_224
        })

    print(f"Pre-computed {len(precomputed)} items in {time.time()-t0:.2f}s.")

    # 1. Evaluate Original Baseline
    orig_path = os.path.join(config.MODELS_DIR, "arecanut_ordinal.pth")
    print(f"\nEvaluating Original Baseline: {orig_path}...")
    model_orig = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False).to(device)
    model_orig.load_state_dict(torch.load(orig_path, map_location=device, weights_only=True))
    gc_orig = GradCAM(model_orig, target_layer_name='layer4')
    gcpp_orig = GradCAMPlusPlus(model_orig, target_layer_name='layer4')
    orig_stats, orig_results = run_single_checkpoint_eval(model_orig, gc_orig, gcpp_orig, precomputed)

    print(f"Baseline -> Diseased Grad-CAM: {orig_stats['diseased_gc_pct']}% ({orig_stats['diseased_gc_hits']}/{orig_stats['total_diseased']}) | "
          f"Grad-CAM++: {orig_stats['diseased_gcpp_pct']}% ({orig_stats['diseased_gcpp_hits']}/{orig_stats['total_diseased']})")
    print(f"Baseline -> Overall Grad-CAM: {orig_stats['overall_gc_pct']}% ({orig_stats['overall_gc_hits']}/{orig_stats['total_samples']}) | "
          f"Grad-CAM++: {orig_stats['overall_gcpp_pct']}% ({orig_stats['overall_gcpp_hits']}/{orig_stats['total_samples']})")

    # 2. Evaluate All 3 Background-Randomized Checkpoints
    seeds = [42, 100, 2024]
    new_stats_by_seed = {}
    new_results_seed42 = None

    for seed in seeds:
        ckpt_path = os.path.join(IMPROVEMENTS_DIR, f"arecanut_resnet18_corn_bgrand_seed{seed}.pth")
        print(f"\nEvaluating Bg-Randomized Model (Seed {seed}): {ckpt_path}...")
        model_new = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False).to(device)
        model_new.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        gc_new = GradCAM(model_new, target_layer_name='layer4')
        gcpp_new = GradCAMPlusPlus(model_new, target_layer_name='layer4')
        new_stats, new_results = run_single_checkpoint_eval(model_new, gc_new, gcpp_new, precomputed)

        print(f"Seed {seed} -> Diseased Grad-CAM: {new_stats['diseased_gc_pct']}% ({new_stats['diseased_gc_hits']}/{new_stats['total_diseased']}) | "
              f"Grad-CAM++: {new_stats['diseased_gcpp_pct']}% ({new_stats['diseased_gcpp_hits']}/{new_stats['total_diseased']})")
        print(f"Seed {seed} -> Overall Grad-CAM: {new_stats['overall_gc_pct']}% ({new_stats['overall_gc_hits']}/{new_stats['total_samples']}) | "
              f"Grad-CAM++: {new_stats['overall_gcpp_pct']}% ({new_stats['overall_gcpp_hits']}/{new_stats['total_samples']})")

        new_stats_by_seed[str(seed)] = new_stats
        if seed == 42:
            new_results_seed42 = new_results

    # 3. Side-by-side comparison gallery (Orig vs. Bg-Randomized Seed 42)
    # Find samples prioritizing attention differences and diseased tissue
    candidates = []
    for i in range(len(orig_results)):
        res_o = orig_results[i]
        res_n = new_results_seed42[i]
        diff_gc = int(res_o['hit_gc'] != res_n['hit_gc'])
        diff_gcpp = int(res_o['hit_gcpp'] != res_n['hit_gcpp'])
        diff_score = diff_gc + diff_gcpp

        candidates.append({
            'res_o': res_o,
            'res_n': res_n,
            'diff_score': diff_score,
            'is_diseased': res_o['is_diseased']
        })

    # Sort by diff_score descending, then diseased status
    candidates.sort(key=lambda x: (x['diff_score'], int(x['is_diseased'])), reverse=True)

    gallery_saved = []
    for i, c in enumerate(candidates[:6]):
        res_o = c['res_o']
        res_n = c['res_n']
        img = res_o['image']

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        # 1. Original Image
        axes[0].imshow(img.resize((224, 224)))
        axes[0].set_title(f"Test Image\n{res_o['disease_class']} (Grade {res_o['true_grade']})", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # 2. Ground Truth Mask Overlay
        img_arr = np.array(img.resize((224, 224)))
        mask_vis = img_arr.copy()
        mask_vis[res_o['mask_224']] = [220, 38, 38]  # red overlay
        axes[1].imshow(mask_vis)
        axes[1].set_title(f"Ground-Truth Mask\n(True Grade: {res_o['true_grade']})", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # 3. Original Baseline Model Grad-CAM
        ov_o = overlay_heatmap(img.resize((224, 224)), res_o['cam_gc'], alpha=0.5)
        axes[2].imshow(ov_o)
        status_o = "HIT (+)" if res_o['hit_gc'] else "MISS (-)"
        col_o = "darkgreen" if res_o['hit_gc'] else "firebrick"
        axes[2].plot(res_o['peak_gc'][1], res_o['peak_gc'][0], 'w*', markersize=14, markeredgecolor='black')
        axes[2].set_title(f"Baseline CAM: {status_o}\n(Pred: Grade {res_o['pred_grade']})", fontsize=11, fontweight='bold', color=col_o)
        axes[2].axis('off')

        # 4. Background-Randomized Model Grad-CAM
        ov_n = overlay_heatmap(img.resize((224, 224)), res_n['cam_gc'], alpha=0.5)
        axes[3].imshow(ov_n)
        status_n = "HIT (+)" if res_n['hit_gc'] else "MISS (-)"
        col_n = "darkgreen" if res_n['hit_gc'] else "firebrick"
        axes[3].plot(res_n['peak_gc'][1], res_n['peak_gc'][0], 'w*', markersize=14, markeredgecolor='black')
        axes[3].set_title(f"Bg-Rand CAM: {status_n}\n(Pred: Grade {res_n['pred_grade']})", fontsize=11, fontweight='bold', color=col_n)
        axes[3].axis('off')

        suptitle_text = f"Sample {i+1}: {res_o['disease_class']} (True Grade {res_o['true_grade']}) | Baseline Pred: G{res_o['pred_grade']} vs Bg-Rand Pred: G{res_n['pred_grade']}"
        plt.suptitle(suptitle_text, fontsize=12, y=0.98)
        plt.tight_layout()

        out_fname = f"bg_rand_vs_baseline_{res_o['disease_class']}_trueG{res_o['true_grade']}_ex{i+1}.png"
        out_fpath = os.path.join(GALLERY_DIR, out_fname)
        fig.savefig(out_fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gallery_saved.append(out_fpath)
        print(f"Saved gallery figure {i+1}: {out_fpath}")

    # 4. Save results to outputs/improvements/bg_randomization_pointing_game.json
    final_output = {
        "definition": "Pointing Game Accuracy measures the percentage of test images where the single maximum activation pixel falls directly inside the ground-truth symptom/lesion mask.",
        "baseline_reference": orig_stats,
        "bg_randomization_by_seed": new_stats_by_seed,
        "seed42_side_by_side": {
            "baseline": {
                "overall_gc_pct": orig_stats["overall_gc_pct"],
                "overall_gcpp_pct": orig_stats["overall_gcpp_pct"],
                "diseased_gc_pct": orig_stats["diseased_gc_pct"],
                "diseased_gcpp_pct": orig_stats["diseased_gcpp_pct"],
                "diseased_gc_hits": orig_stats["diseased_gc_hits"],
                "diseased_gcpp_hits": orig_stats["diseased_gcpp_hits"],
                "total_diseased": orig_stats["total_diseased"]
            },
            "bg_randomized": {
                "overall_gc_pct": new_stats_by_seed["42"]["overall_gc_pct"],
                "overall_gcpp_pct": new_stats_by_seed["42"]["overall_gcpp_pct"],
                "diseased_gc_pct": new_stats_by_seed["42"]["diseased_gc_pct"],
                "diseased_gcpp_pct": new_stats_by_seed["42"]["diseased_gcpp_pct"],
                "diseased_gc_hits": new_stats_by_seed["42"]["diseased_gc_hits"],
                "diseased_gcpp_hits": new_stats_by_seed["42"]["diseased_gcpp_hits"],
                "total_diseased": new_stats_by_seed["42"]["total_diseased"]
            },
            "per_class_comparison": {}
        },
        "gallery_images": gallery_saved
    }

    # Per-class side-by-side
    for c in orig_stats["per_class"]:
        orig_c = orig_stats["per_class"][c]
        new_c = new_stats_by_seed["42"]["per_class"].get(c, {})
        final_output["seed42_side_by_side"]["per_class_comparison"][c] = {
            "total_samples": orig_c["total_samples"],
            "diseased_samples": orig_c["diseased_samples"],
            "baseline_gc_pct": orig_c["gc_pct"],
            "baseline_gcpp_pct": orig_c["gcpp_pct"],
            "baseline_diseased_gc_pct": orig_c["diseased_gc_pct"],
            "baseline_diseased_gcpp_pct": orig_c["diseased_gcpp_pct"],
            "bg_rand_gc_pct": new_c.get("gc_pct"),
            "bg_rand_gcpp_pct": new_c.get("gcpp_pct"),
            "bg_rand_diseased_gc_pct": new_c.get("diseased_gc_pct"),
            "bg_rand_diseased_gcpp_pct": new_c.get("diseased_gcpp_pct")
        }

    # Compute 3-seed mean and std for pointing game
    diseased_gc_all_seeds = [new_stats_by_seed[str(s)]["diseased_gc_pct"] for s in seeds]
    diseased_gcpp_all_seeds = [new_stats_by_seed[str(s)]["diseased_gcpp_pct"] for s in seeds]
    overall_gc_all_seeds = [new_stats_by_seed[str(s)]["overall_gc_pct"] for s in seeds]
    overall_gcpp_all_seeds = [new_stats_by_seed[str(s)]["overall_gcpp_pct"] for s in seeds]

    final_output["bg_randomization_3seed_summary"] = {
        "diseased_gc_mean": round(float(np.mean(diseased_gc_all_seeds)), 2),
        "diseased_gc_std": round(float(np.std(diseased_gc_all_seeds, ddof=1)), 2),
        "diseased_gcpp_mean": round(float(np.mean(diseased_gcpp_all_seeds)), 2),
        "diseased_gcpp_std": round(float(np.std(diseased_gcpp_all_seeds, ddof=1)), 2),
        "overall_gc_mean": round(float(np.mean(overall_gc_all_seeds)), 2),
        "overall_gc_std": round(float(np.std(overall_gc_all_seeds, ddof=1)), 2),
        "overall_gcpp_mean": round(float(np.mean(overall_gcpp_all_seeds)), 2),
        "overall_gcpp_std": round(float(np.std(overall_gcpp_all_seeds, ddof=1)), 2)
    }

    out_file = os.path.join(IMPROVEMENTS_DIR, "bg_randomization_pointing_game.json")
    with open(out_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved pointing-game comparison JSON to {out_file}")


if __name__ == '__main__':
    main()
