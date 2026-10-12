import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.segmentation import segment_leaf_adaptive, compute_ratio, otsu_threshold, get_disk

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("SECTION F: BLIND HUMAN-GRADING SHEET (150 IMAGES)")
print("=" * 80)

np.random.seed(42)

# 1. Load Paddy data with ratios
paddy_labels_df = pd.read_csv('outputs/paddy_severity_labels.csv')
paddy_lesion_classes = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "downy_mildew",
    "tungro"
]
paddy_cand = paddy_labels_df[paddy_labels_df['disease_class'].isin(paddy_lesion_classes)].copy()
paddy_cand['crop'] = 'Paddy'

# Tag boundary and Grade 1
# Boundaries: [0.03, 0.07] and [0.13, 0.17]
paddy_cand['is_boundary'] = (
    ((paddy_cand['lesion_ratio'] >= 0.03) & (paddy_cand['lesion_ratio'] <= 0.07)) |
    ((paddy_cand['lesion_ratio'] >= 0.13) & (paddy_cand['lesion_ratio'] <= 0.17))
)
paddy_cand['is_g1'] = (paddy_cand['severity_grade'] == 1)

# Sample 60 Paddy lesion images:
paddy_b = paddy_cand[paddy_cand['is_boundary']]
paddy_g1 = paddy_cand[paddy_cand['is_g1'] & (~paddy_cand['is_boundary'])]
paddy_other = paddy_cand[(~paddy_cand['is_boundary']) & (~paddy_cand['is_g1'])]

paddy_sample_b = paddy_b.sample(n=min(25, len(paddy_b)), random_state=42)
paddy_sample_g1 = paddy_g1.sample(n=min(20, len(paddy_g1)), random_state=42)
needed_paddy = 60 - len(paddy_sample_b) - len(paddy_sample_g1)
paddy_sample_other = paddy_other.sample(n=needed_paddy, random_state=42)
paddy_selected = pd.concat([paddy_sample_b, paddy_sample_g1, paddy_sample_other]).copy()
print(f"Selected {len(paddy_selected)} Paddy lesion images (Boundary={paddy_selected['is_boundary'].sum()}, G1={paddy_selected['is_g1'].sum()})")

# 2. Arecanut lesion candidates
DISK_RADIUS = config.LEAF_MASK_EROSION_PX
DISK_STRUCT = get_disk(DISK_RADIUS)
MIN_LESION_BLOB = config.MIN_LESION_BLOB_PX

def fast_detect(img_rgb, leaf_mask):
    eroded = scipy.ndimage.binary_erosion(leaf_mask, structure=DISK_STRUCT)
    gray = (0.2989 * img_rgb[:,:,0] + 0.5870 * img_rgb[:,:,1] + 0.1140 * img_rgb[:,:,2]).astype(np.uint8)
    th = otsu_threshold(gray, mask=eroded)
    l_mask = (gray < th) & eroded
    labeled, num = scipy.ndimage.label(l_mask)
    if num > 0:
        counts = np.bincount(labeled.ravel())
        valid = np.where(counts >= MIN_LESION_BLOB)[0]
        valid = valid[valid > 0]
        l_mask = np.isin(labeled, valid)
    else:
        l_mask = np.zeros_like(l_mask, dtype=bool)
    return l_mask

arec_df = pd.read_csv('outputs/arecanut_splits.csv')
arec_lesion_classes = ['mahali_koleroga', 'stem_cracking', 'stem_bleeding', 'bud_borer']
arec_lesion_cand = arec_df[arec_df['disease_class'].isin(arec_lesion_classes)].copy()

# Sample a candidate pool of 200 images to find boundary & G1 cases quickly
cand_pool = arec_lesion_cand.sample(n=min(200, len(arec_lesion_cand)), random_state=42).copy()
cand_pool['crop'] = 'Arecanut'

ratios = []
for p in cand_pool['image_path']:
    im = cv2.imread(p)
    if im is None:
        ratios.append(0.0)
        continue
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    lf = segment_leaf_adaptive(im_rgb)
    ls = fast_detect(im_rgb, lf)
    r = compute_ratio(ls, lf)
    ratios.append(r)

cand_pool['lesion_ratio'] = ratios

def r2g(r):
    if r <= 0.0: return 0
    elif r <= 0.05: return 1
    elif r <= 0.15: return 2
    else: return 3

cand_pool['pipeline_grade'] = cand_pool['lesion_ratio'].apply(r2g)

cand_pool['is_boundary'] = (
    ((cand_pool['lesion_ratio'] >= 0.03) & (cand_pool['lesion_ratio'] <= 0.07)) |
    ((cand_pool['lesion_ratio'] >= 0.13) & (cand_pool['lesion_ratio'] <= 0.17))
)
cand_pool['is_g1'] = (cand_pool['pipeline_grade'] == 1)

arec_b = cand_pool[cand_pool['is_boundary']]
arec_g1 = cand_pool[cand_pool['is_g1'] & (~cand_pool['is_boundary'])]
arec_other = cand_pool[(~cand_pool['is_boundary']) & (~cand_pool['is_g1'])]

arec_sample_b = arec_b.sample(n=min(20, len(arec_b)), random_state=42)
arec_sample_g1 = arec_g1.sample(n=min(15, len(arec_g1)), random_state=42)
needed_arec = 60 - len(arec_sample_b) - len(arec_sample_g1)
arec_sample_other = arec_other.sample(n=needed_arec, random_state=42)
arec_lesion_selected = pd.concat([arec_sample_b, arec_sample_g1, arec_sample_other]).copy()
print(f"Selected {len(arec_lesion_selected)} Arecanut lesion images (Boundary={arec_lesion_selected['is_boundary'].sum()}, G1={arec_lesion_selected['is_g1'].sum()})")

# 3. Sample 15 healthy_trunk images
ht_cand = arec_df[arec_df['disease_class'] == 'healthy_trunk'].copy()
ht_cand['crop'] = 'Arecanut'
ht_cand['lesion_ratio'] = 0.0
ht_cand['pipeline_grade'] = 0
ht_cand['is_boundary'] = False
ht_cand['is_g1'] = False
ht_selected = ht_cand.sample(n=15, random_state=42)

# 4. Sample 15 healthy nut/leaf images (8 nut, 7 leaf)
nut_cand = arec_df[arec_df['disease_class'] == 'healthy_nut'].copy()
leaf_cand = arec_df[arec_df['disease_class'] == 'healthy_leaf'].copy()
for df in [nut_cand, leaf_cand]:
    df['crop'] = 'Arecanut'
    df['lesion_ratio'] = 0.0
    df['pipeline_grade'] = 0
    df['is_boundary'] = False
    df['is_g1'] = False

nut_selected = nut_cand.sample(n=8, random_state=42)
leaf_selected = leaf_cand.sample(n=7, random_state=42)
healthy_nut_leaf_selected = pd.concat([nut_selected, leaf_selected])

# Standardize columns
cols = ['crop', 'disease_class', 'filename', 'image_path', 'lesion_ratio', 'pipeline_grade', 'is_boundary', 'is_g1']
paddy_sub = paddy_selected.rename(columns={'severity_grade': 'pipeline_grade'})[cols]
arec_sub = arec_lesion_selected[cols]
ht_sub = ht_selected[cols]
hnl_sub = healthy_nut_leaf_selected[cols]

# Combine all 150
combined_150 = pd.concat([paddy_sub, arec_sub, ht_sub, hnl_sub]).reset_index(drop=True)
# Randomly shuffle so grader cannot anticipate block structure
combined_150 = combined_150.sample(frac=1.0, random_state=42).reset_index(drop=True)
combined_150['image_id'] = [f"{i+1:03d}" for i in range(len(combined_150))]

print(f"\nTotal combined images: {len(combined_150)}")

# Save Answer Key (separate file)
answer_key = []
for idx, row in combined_150.iterrows():
    answer_key.append({
        'image_id': row['image_id'],
        'crop': row['crop'],
        'disease_class': row['disease_class'],
        'filename': row['filename'],
        'image_path': row['image_path'],
        'pipeline_ratio': round(float(row['lesion_ratio']), 4),
        'pipeline_grade': int(row['pipeline_grade']),
        'is_boundary_case': bool(row['is_boundary']),
        'is_grade1_candidate': bool(row['is_g1'])
    })

answer_key_path = 'outputs/final_v2/blind_grading_answer_key.json'
with open(answer_key_path, 'w') as f:
    json.dump(answer_key, f, indent=2)
print(f"Saved secret answer key to {answer_key_path}")

# Save Blank Grading Sheet CSV (no grades)
blank_csv_df = pd.DataFrame({
    'image_id': combined_150['image_id'],
    'assigned_grade': [''] * len(combined_150),
    'notes': [''] * len(combined_150)
})
blank_csv_path = 'outputs/final_v2/blind_grading_sheet.csv'
blank_csv_df.to_csv(blank_csv_path, index=False)
print(f"Saved blank grading sheet to {blank_csv_path}")

# Create PDF booklet (6 images per page: 2 columns x 3 rows, 25 pages)
pdf_path = 'outputs/final_v2/blind_grading_sheet.pdf'
print(f"Rendering 25-page PDF booklet to {pdf_path}...")

with PdfPages(pdf_path) as pdf:
    n_pages = int(np.ceil(len(combined_150) / 6))
    for page_i in range(n_pages):
        fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
        page_items = combined_150.iloc[page_i * 6 : (page_i + 1) * 6]
        
        for slot_i, ax in enumerate(axes.flat):
            if slot_i < len(page_items):
                item = page_items.iloc[slot_i]
                img_p = item['image_path']
                try:
                    im = Image.open(img_p).convert('RGB')
                    ax.imshow(im)
                except Exception:
                    ax.text(0.5, 0.5, 'Image load error', ha='center', va='center')
                ax.set_title(f"Image ID: {item['image_id']}", fontsize=13, fontweight='bold', pad=8)
            ax.axis('off')
            
        fig.suptitle(f"Blind Agricultural Disease Severity Grading Sheet — Page {page_i+1} of {n_pages}", fontsize=11, y=0.98, color='gray')
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

print(f"PDF creation complete: {pdf_path}")

# Print and save allocation table
alloc_summary = {
    'total_images': len(combined_150),
    'categories': {
        'paddy_lesion_classes': {
            'target': 60,
            'actual': len(paddy_sub),
            'by_class': paddy_selected['disease_class'].value_counts().to_dict(),
            'boundary_cases': int(paddy_selected['is_boundary'].sum()),
            'grade_1_candidates': int(paddy_selected['is_g1'].sum())
        },
        'arecanut_lesion_classes': {
            'target': 60,
            'actual': len(arec_sub),
            'by_class': arec_lesion_selected['disease_class'].value_counts().to_dict(),
            'boundary_cases': int(arec_lesion_selected['is_boundary'].sum()),
            'grade_1_candidates': int(arec_lesion_selected['is_g1'].sum())
        },
        'healthy_trunk': {
            'target': 15,
            'actual': len(ht_sub)
        },
        'healthy_nut_and_leaf': {
            'target': 15,
            'actual': len(hnl_sub),
            'healthy_nut': 8,
            'healthy_leaf': 7
        }
    }
}

with open('outputs/final_v2/stage2_section_f_allocation.json', 'w') as f:
    json.dump(alloc_summary, f, indent=2)

print("\nSection F complete! All files generated successfully.")
