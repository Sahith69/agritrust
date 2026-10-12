import os
import re
import json
import hashlib
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
import imagehash
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ─── CONFIGURATION & PATHS ───────────────────────────────────────────────────
PROJECT_ROOT = "/home/sahith/Projects/Hackathon"
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_PATH = os.path.join(PROJECT_ROOT, "audit_report.json")

SPLIT_FILES = {
    "apple": os.path.join(OUTPUTS_DIR, "apple_splits.csv"),
    "arecanut": os.path.join(OUTPUTS_DIR, "arecanut_splits.csv"),
    "paddy": os.path.join(OUTPUTS_DIR, "paddy_splits.csv")
}

# ImageNet transform for DINOv2 (224x224)
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─── DATASET FOR BATCH EMBEDDING EXTRACTION ──────────────────────────────────
class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if self.transform:
                    tensor = self.transform(img)
                else:
                    tensor = transforms.ToTensor()(img)
            return tensor, path, True
        except Exception:
            return torch.zeros(3, 224, 224), path, False


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def extract_md5(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest(), True
    except Exception:
        return None, False


def extract_phash(path):
    try:
        with Image.open(path) as img:
            h = imagehash.phash(img)
            # Convert 64-bit hex hash to uint64 for fast vectorized Hamming distance
            val = np.uint64(int(str(h), 16))
            return val, True
    except Exception:
        return None, False


def extract_exif_timestamp(path):
    try:
        with Image.open(path) as img:
            exif = img._getexif() if hasattr(img, '_getexif') and img._getexif() else img.getexif()
            if exif:
                # 36867: DateTimeOriginal, 306: DateTime, 36868: DateTimeDigitized
                for tag_id in [36867, 306, 36868]:
                    if tag_id in exif and exif[tag_id]:
                        dt_str = str(exif[tag_id]).strip()
                        try:
                            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S"), True
                        except Exception:
                            pass
    except Exception:
        pass
    return None, False


def extract_filename_timestamp_arecanut(fname):
    # Regex for IMG_YYYYMMDD_HHMMSS
    match = re.search(r'IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', fname)
    if match:
        try:
            dt = datetime(
                year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)),
                hour=int(match.group(4)), minute=int(match.group(5)), second=int(match.group(6))
            )
            return dt, True
        except Exception:
            pass
    return None, False


def compute_dinov2_embeddings(paths, model, batch_size=64):
    dataset = ImagePathDataset(paths, transform=DINO_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    embeddings = []
    failed_images = []

    model.eval()
    with torch.no_grad():
        for batch_tensors, batch_paths, valid_flags in loader:
            valid_mask = valid_flags.numpy()
            if not np.all(valid_mask):
                for p, v in zip(batch_paths, valid_mask):
                    if not v:
                        failed_images.append(p)
            
            feats = model(batch_tensors)
            # L2 normalize
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            embeddings.append(feats.cpu().numpy())

    if embeddings:
        all_embeddings = np.concatenate(embeddings, axis=0)
    else:
        all_embeddings = np.zeros((0, 384), dtype=np.float32)
    return all_embeddings, failed_images


# ─── CROP AUDIT PIPELINE ─────────────────────────────────────────────────────
def audit_crop(crop_name, split_csv_path, dinov2_model):
    print(f"\n{'='*75}")
    print(f"AUDITING CROP: {crop_name.upper()} ({split_csv_path})")
    print(f"{'='*75}")

    df = pd.read_csv(split_csv_path)
    total_records = len(df)
    
    # Identify key columns
    img_col = 'image_path'
    split_col = 'split'
    parent_col = 'parent_id' if 'parent_id' in df.columns else None

    # Load images, hashes, pHashes, EXIF
    print(f"[1/4] Extracting MD5, pHash, and EXIF across {total_records} images...")
    
    md5_list = []
    phash_list = []
    exif_dt_list = []
    filename_dt_list = []
    failed_loading = []

    for idx, row in df.iterrows():
        p = row[img_col]
        fname = row['filename'] if 'filename' in df.columns else os.path.basename(p)

        # MD5
        md5_val, ok_md5 = extract_md5(p)
        if not ok_md5:
            failed_loading.append(p)
        md5_list.append(md5_val)

        # pHash
        ph_val, _ = extract_phash(p)
        phash_list.append(ph_val)

        # EXIF
        ex_dt, _ = extract_exif_timestamp(p)
        exif_dt_list.append(ex_dt)

        # Filename timestamp (for Arecanut)
        fn_dt, _ = extract_filename_timestamp_arecanut(fname)
        filename_dt_list.append(fn_dt)

    df['computed_md5'] = md5_list
    df['computed_phash'] = phash_list
    df['exif_dt'] = exif_dt_list
    df['filename_dt'] = filename_dt_list

    # Separate splits
    df_train = df[df[split_col] == 'train'].reset_index(drop=True)
    df_val   = df[df[split_col] == 'val'].reset_index(drop=True)
    df_test  = df[df[split_col] == 'test'].reset_index(drop=True)

    n_train = len(df_train)
    n_val   = len(df_val)
    n_test  = len(df_test)

    print(f"      Split distribution: Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"      EXIF DateTime found: {sum(x is not None for x in exif_dt_list)}/{total_records} images")
    if crop_name == "arecanut":
        print(f"      Filename camera timestamps found: {sum(x is not None for x in filename_dt_list)}/{total_records} images")

    # [2/4] DINOv2 Embeddings
    print(f"[2/4] Extracting DINOv2 (ViT-S/14) embeddings...")
    train_embs, train_fails = compute_dinov2_embeddings(df_train[img_col].tolist(), dinov2_model)
    val_embs, val_fails     = compute_dinov2_embeddings(df_val[img_col].tolist(), dinov2_model)
    test_embs, test_fails   = compute_dinov2_embeddings(df_test[img_col].tolist(), dinov2_model)
    
    all_fails = list(set(failed_loading + train_fails + val_fails + test_fails))

    # [3/4] Comparative Leakage Analysis
    def analyze_pair(eval_df, eval_embs, eval_name):
        n_eval = len(eval_df)
        if n_eval == 0:
            return {}

        # 1. MD5 Exact Duplicates
        train_md5_set = set(df_train['computed_md5'].dropna())
        md5_match_count = 0
        md5_matching_files = []
        for _, row in eval_df.iterrows():
            if row['computed_md5'] in train_md5_set:
                md5_match_count += 1
                md5_matching_files.append((row['filename'], row['computed_md5']))

        # 2. pHash Near-Duplicates (Hamming <=0, <=8, <=16)
        train_phashes = np.array(df_train['computed_phash'].tolist(), dtype=np.uint64)
        eval_phashes  = np.array(eval_df['computed_phash'].tolist(), dtype=np.uint64)

        # Pairwise Hamming distance matrix: [N_eval, N_train]
        xor_matrix = np.bitwise_xor(eval_phashes[:, None], train_phashes[None, :])
        dist_matrix = np.bitwise_count(xor_matrix)

        min_dists = dist_matrix.min(axis=1)
        phash_le_0_count = int(np.sum(min_dists <= 0))
        phash_le_8_count = int(np.sum(min_dists <= 8))
        phash_le_16_count = int(np.sum(min_dists <= 16))

        # 3. EXIF Camera-burst overlap within 120s
        exif_overlap_count = 0
        train_exif_times = [t for t in df_train['exif_dt'] if t is not None]
        has_exif_data = len(train_exif_times) > 0

        if has_exif_data:
            for t_eval in eval_df['exif_dt']:
                if t_eval is not None:
                    if any(abs((t_eval - t_tr).total_seconds()) <= 120 for t_tr in train_exif_times):
                        exif_overlap_count += 1

        # Filename timestamp overlap within 120s (Specifically for Arecanut)
        filename_dt_overlap_count = 0
        train_fn_times = [t for t in df_train['filename_dt'] if t is not None]
        has_fn_times = len(train_fn_times) > 0
        if has_fn_times:
            for t_eval in eval_df['filename_dt']:
                if t_eval is not None:
                    if any(abs((t_eval - t_tr).total_seconds()) <= 120 for t_tr in train_fn_times):
                        filename_dt_overlap_count += 1

        # 4. DINOv2 Cosine Similarity >= 0.95
        # eval_embs @ train_embs.T -> [N_eval, N_train]
        sim_matrix = np.dot(eval_embs, train_embs.T)
        max_sims = sim_matrix.max(axis=1)
        dino_ge_95_count = int(np.sum(max_sims >= 0.95))

        return {
            "eval_name": eval_name,
            "total_eval_samples": n_eval,
            "md5_exact_matches": md5_match_count,
            "md5_exact_matches_pct": (md5_match_count / n_eval) * 100.0,
            "md5_matching_samples": md5_matching_files,
            "phash_hamming_le_0": phash_le_0_count,
            "phash_hamming_le_0_pct": (phash_le_0_count / n_eval) * 100.0,
            "phash_hamming_le_8": phash_le_8_count,
            "phash_hamming_le_8_pct": (phash_le_8_count / n_eval) * 100.0,
            "phash_hamming_le_16": phash_le_16_count,
            "phash_hamming_le_16_pct": (phash_le_16_count / n_eval) * 100.0,
            "exif_available": has_exif_data,
            "exif_burst_overlap_120s": exif_overlap_count,
            "exif_burst_overlap_120s_pct": (exif_overlap_count / n_eval) * 100.0 if has_exif_data else 0.0,
            "filename_dt_available": has_fn_times,
            "filename_dt_burst_overlap_120s": filename_dt_overlap_count,
            "filename_dt_burst_overlap_120s_pct": (filename_dt_overlap_count / n_eval) * 100.0 if has_fn_times else 0.0,
            "dinov2_cosine_ge_0_95": dino_ge_95_count,
            "dinov2_cosine_ge_0_95_pct": (dino_ge_95_count / n_eval) * 100.0
        }

    val_res = analyze_pair(df_val, val_embs, "val_vs_train")
    test_res = analyze_pair(df_test, test_embs, "test_vs_train")

    crop_summary = {
        "crop": crop_name,
        "split_file": split_csv_path,
        "columns": {
            "image_path_col": img_col,
            "split_col": split_col,
            "parent_id_col": parent_col
        },
        "total_images": total_records,
        "split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "failed_to_load_images": all_fails,
        "val_vs_train": val_res,
        "test_vs_train": test_res
    }

    # Apple specific checks
    if crop_name == "apple":
        print("[4/4] Running Apple-specific lineage & augmentation checks...")
        # Check parent_id multi-split overlap
        parent_splits = defaultdict(set)
        for _, row in df.iterrows():
            parent_splits[row['parent_id']].add(row['split'])
        
        leaked_parents = {pid: list(splits) for pid, splits in parent_splits.items() if len(splits) > 1}

        # Check for flip / rot / aug suffixes
        aug_patterns = ['flip', 'rot', 'rotate', 'aug', 'copy', 'zoom', 'shift']
        aug_files = [f for f in df['filename'] if any(p in f.lower() for p in aug_patterns)]

        crop_summary["apple_specific_checks"] = {
            "parent_id_derivation_explanation": (
                "PlantVillage filenames follow '<uuid>___<original_id>'. "
                "parent_id is extracted by taking parts[1] after splitting on '___'."
            ),
            "unique_parent_ids": df['parent_id'].nunique(),
            "parent_ids_in_multiple_splits_count": len(leaked_parents),
            "parent_ids_in_multiple_splits": leaked_parents,
            "augmented_variants_suffix_matches_count": len(aug_files),
            "augmented_variants_suffix_samples": aug_files[:10]
        }

    return crop_summary


# ─── MAIN EXECUTION ──────────────────────────────────────────────────────────
def main():
    print("="*75)
    print("WIDE DATA LEAKAGE AUDIT ENGINE")
    print("="*75)
    torch.set_num_threads(8)

    print("Loading DINOv2 ViT-S/14 model...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    full_report = {}
    for crop in ["apple", "arecanut", "paddy"]:
        split_csv = SPLIT_FILES[crop]
        if os.path.exists(split_csv):
            full_report[crop] = audit_crop(crop, split_csv, dinov2_model)
        else:
            print(f"[!] Warning: {split_csv} does not exist.")

    # Save to audit_report.json
    with open(REPORT_PATH, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[✓] Audit report saved successfully to: {REPORT_PATH}")

    # Print Formatted Tables per crop
    for crop, data in full_report.items():
        print("\n" + "="*80)
        print(f"LEAKAGE AUDIT REPORT: {crop.upper()}")
        print("="*80)
        print(f"Columns: Image Path = '{data['columns']['image_path_col']}', Split = '{data['columns']['split_col']}', Parent ID = '{data['columns']['parent_id_col']}'")
        print(f"Split Sizes: Train = {data['split_counts']['train']}, Val = {data['split_counts']['val']}, Test = {data['split_counts']['test']} (Total = {data['total_images']})")
        print(f"Failed to Load Images: {len(data['failed_to_load_images'])}")

        print("\n" + "-"*80)
        print(f"{'Metric':<35} | {'Val vs Train':<20} | {'Test vs Train':<20}")
        print("-"*80)

        v = data['val_vs_train']
        t = data['test_vs_train']

        print(f"{'MD5 Exact Duplicates':<35} | {v['md5_exact_matches']:>5} ({v['md5_exact_matches_pct']:>6.2f}%)    | {t['md5_exact_matches']:>5} ({t['md5_exact_matches_pct']:>6.2f}%)")
        print(f"{'pHash Hamming <= 0':<35} | {v['phash_hamming_le_0']:>5} ({v['phash_hamming_le_0_pct']:>6.2f}%)    | {t['phash_hamming_le_0']:>5} ({t['phash_hamming_le_0_pct']:>6.2f}%)")
        print(f"{'pHash Hamming <= 8':<35} | {v['phash_hamming_le_8']:>5} ({v['phash_hamming_le_8_pct']:>6.2f}%)    | {t['phash_hamming_le_8']:>5} ({t['phash_hamming_le_8_pct']:>6.2f}%)")
        print(f"{'pHash Hamming <= 16':<35} | {v['phash_hamming_le_16']:>5} ({v['phash_hamming_le_16_pct']:>6.2f}%)    | {t['phash_hamming_le_16']:>5} ({t['phash_hamming_le_16_pct']:>6.2f}%)")
        
        exif_v_str = f"{v['exif_burst_overlap_120s']} ({v['exif_burst_overlap_120s_pct']:.2f}%)" if v['exif_available'] else "EXIF Not Present"
        exif_t_str = f"{t['exif_burst_overlap_120s']} ({t['exif_burst_overlap_120s_pct']:.2f}%)" if t['exif_available'] else "EXIF Not Present"
        print(f"{'EXIF Camera Burst <= 120s':<35} | {exif_v_str:<20} | {exif_t_str:<20}")

        if crop == "arecanut" and v['filename_dt_available']:
            fn_v_str = f"{v['filename_dt_burst_overlap_120s']} ({v['filename_dt_burst_overlap_120s_pct']:.2f}%)"
            fn_t_str = f"{t['filename_dt_burst_overlap_120s']} ({t['filename_dt_burst_overlap_120s_pct']:.2f}%)"
            print(f"{'Filename Burst <= 120s':<35} | {fn_v_str:<20} | {fn_t_str:<20}")

        print(f"{'DINOv2 Cosine >= 0.95 Pairs':<35} | {v['dinov2_cosine_ge_0_95']:>5} ({v['dinov2_cosine_ge_0_95_pct']:>6.2f}%)    | {t['dinov2_cosine_ge_0_95']:>5} ({t['dinov2_cosine_ge_0_95_pct']:>6.2f}%)")
        print("-"*80)

        if crop == "apple" and "apple_specific_checks" in data:
            ap = data["apple_specific_checks"]
            print("\nApple Lineage Checks:")
            print(f"- Parent ID Derivation: {ap['parent_id_derivation_explanation']}")
            print(f"- Total Unique Parent IDs: {ap['unique_parent_ids']}")
            print(f"- Parent IDs in Multiple Splits: {ap['parent_ids_in_multiple_splits_count']}")
            print(f"- Augmented Suffix Matches (flip, rot, etc.): {ap['augmented_variants_suffix_matches_count']}")
        print("="*80)

if __name__ == "__main__":
    main()
