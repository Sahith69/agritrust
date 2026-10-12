import os
import sys
import json
import re
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
import cv2

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2c - PART C: CLEAN MANIFEST V2 & RESIDUAL CONTAMINATION AUDIT")
print("=" * 80)

# 1. Load splits
splits_path = 'outputs/arecanut_splits.csv'
df_all = pd.read_csv(splits_path)
n_before = len(df_all)
print(f"Total Arecanut rows before cleaning: {n_before}")

# Identify source
df_all['source'] = np.where(df_all['image_path'].str.contains('kaggle_merged'), 'kaggle_1896', 'original_871')

def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

df_all['base_id'] = df_all['filename'].apply(extract_base_id)

# 2. Load the 66 flagged images and the 24 previously removed images
with open('outputs/final_v2/stage2_section_e.json') as f:
    sec_e = json.load(f)
flagged_66 = sec_e['e4_scans']['flagged_images']
flagged_66_fns = set(x['filename'] for x in flagged_66)
flagged_66_dict = {x['filename']: x for x in flagged_66}

with open('outputs/improvements/removed_stock_images.json') as f:
    removed_24_paths = json.load(f)
removed_24_fns = set(os.path.basename(p) for p in removed_24_paths)

# Bad base IDs
bad_base_ids = set()
for fn in flagged_66_fns:
    bad_base_ids.add(extract_base_id(fn))
for fn in removed_24_fns:
    bad_base_ids.add(extract_base_id(fn))

print(f"Unique base IDs across 66 flagged + 24 previously removed: {len(bad_base_ids)}")

# 3. Identify dropped rows
dropped_records = []
keep_mask = []

for idx, row in df_all.iterrows():
    fn = row['filename']
    b_id = row['base_id']
    is_directly_flagged = fn in flagged_66_fns
    shares_bad_base = b_id in bad_base_ids
    
    if is_directly_flagged or shares_bad_base:
        reasons = []
        if is_directly_flagged:
            reasons.extend(flagged_66_dict[fn]['flags'])
        if shares_bad_base:
            reasons.append(f"shares base_id '{b_id}' with flagged/removed images")
        
        dropped_records.append({
            'filename': fn,
            'image_path': row['image_path'],
            'disease_class': row['disease_class'],
            'source': row['source'],
            'severity_grade': int(row['severity_grade']),
            'split': row['split'],
            'base_id': b_id,
            'reasons': reasons
        })
        keep_mask.append(False)
    else:
        keep_mask.append(True)

keep_mask = np.array(keep_mask)
df_clean = df_all[keep_mask].copy().reset_index(drop=True)
n_after = len(df_clean)
n_dropped = len(dropped_records)

print(f"Total dropped rows: {n_dropped}")
print(f"Total clean rows:   {n_after}")

# Mark yellow_leaf_disease rows exclude_from_severity=True
df_clean['exclude_from_severity'] = (df_clean['disease_class'] == 'yellow_leaf_disease')

# Save clean manifest v2
out_manifest = 'outputs/final_v2/arecanut_clean_v2.csv'
df_clean.to_csv(out_manifest, index=False)
print(f"Saved clean manifest to {out_manifest}")

# 4. Report counts per class x source x grade before/after
def get_crosstab(df):
    ct = pd.crosstab(
        [df['disease_class'], df['source']],
        df['severity_grade'],
        margins=True,
        margins_name='Total'
    )
    return ct

ct_before = get_crosstab(df_all)
ct_after = get_crosstab(df_clean)

# Also breakdown dropped rows by class and source
df_dropped = pd.DataFrame(dropped_records)
ct_dropped = pd.crosstab([df_dropped['disease_class'], df_dropped['source']], df_dropped['severity_grade'], margins=True, margins_name='Total')

print("\n--- COUNTS BEFORE CLEANING ---")
print(ct_before)
print("\n--- COUNTS AFTER CLEANING ---")
print(ct_after)
print("\n--- DROPPED ROWS BREAKDOWN ---")
print(ct_dropped)

# 5. Screen remaining files for residual scan/studio signatures
print("\nScreening remaining 2,674 images for aspect-ratio outliers, white-border >50%, and EXIF fields...")
suspicious_residuals = []

for idx, row in df_clean.iterrows():
    p = row['image_path']
    try:
        with Image.open(p) as img:
            w, h = img.size
            aspect = w / h
            
            # EXIF tags check
            exif = img.getexif()
            software = None
            copyright_field = None
            if exif:
                for k, v in exif.items():
                    tag = ExifTags.TAGS.get(k, k)
                    if tag == 'Software':
                        software = str(v)
                    elif tag == 'Copyright':
                        copyright_field = str(v)
            
            # Fast border white check
            # Read small thumbnail or open with OpenCV
            im_cv = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im_cv is not None:
                border_pixels = np.concatenate([
                    im_cv[:5, :].ravel(),
                    im_cv[-5:, :].ravel(),
                    im_cv[:, :5].ravel(),
                    im_cv[:, -5:].ravel()
                ])
                border_white_share = float(np.mean(border_pixels > 245))
            else:
                border_white_share = 0.0
                
            suspect_reasons = []
            if aspect < 0.60 or aspect > 1.67:
                suspect_reasons.append(f"aspect_ratio_outlier (aspect={aspect:.2f}, {w}x{h})")
            if border_white_share > 0.50:
                suspect_reasons.append(f"white_border_gt_50pct (share={border_white_share*100:.1f}%)")
            if software and any(bad in software.lower() for bad in ['adobe', 'photoshop', 'illustrator', 'gimp', 'canva']):
                suspect_reasons.append(f"editing_software_exif ({software})")
            if copyright_field:
                suspect_reasons.append(f"copyright_exif ({copyright_field})")
                
            if suspect_reasons:
                suspicious_residuals.append({
                    'filename': row['filename'],
                    'image_path': p,
                    'disease_class': row['disease_class'],
                    'source': row['source'],
                    'severity_grade': int(row['severity_grade']),
                    'aspect_ratio': float(aspect),
                    'border_white_share_pct': float(border_white_share * 100),
                    'software': software,
                    'copyright': copyright_field,
                    'reasons': suspect_reasons
                })
    except Exception as e:
        pass

print(f"\nResidual suspicious files found: {len(suspicious_residuals)}")
for item in suspicious_residuals[:15]:
    print(f"  [{item['disease_class']}] {item['filename']} -> {', '.join(item['reasons'])}")

part_c_summary = {
    'counts_before': ct_before.reset_index().to_dict(orient='records'),
    'counts_after': ct_after.reset_index().to_dict(orient='records'),
    'counts_dropped': ct_dropped.reset_index().to_dict(orient='records'),
    'total_before': n_before,
    'total_after': n_after,
    'total_dropped': n_dropped,
    'dropped_files': dropped_records,
    'residual_suspicious_files': suspicious_residuals
}

with open('outputs/final_v2/stage2c_part_c.json', 'w') as f:
    json.dump(part_c_summary, f, indent=2)

print("\nSaved Stage 2c Part C summary to outputs/final_v2/stage2c_part_c.json")
