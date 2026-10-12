import os
import sys
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2b - PART C: CONTAMINATION & CROSS-SPLIT DUPLICATES")
print("=" * 80)

# 1. Load the 66 flagged images from stage2_section_e.json
with open('outputs/final_v2/stage2_section_e.json') as f:
    sec_e = json.load(f)

flagged = sec_e['e4_scans']['flagged_images']
print(f"Total flagged images: {len(flagged)}")

df_all = pd.read_csv('outputs/arecanut_splits.csv')
fn_to_path = dict(zip(df_all['filename'], df_all['image_path']))

def extract_base_id(fn):
    m = re.match(r'^(.*?)(?:_|\.)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?: - Copy)?\.(?:jpg|png|jpeg)$', fn, re.I)
    if m:
        return m.group(1).lower()
    return fn.lower()

# 2. Group by reason
group_studio = []
group_watermark = []
group_black = []

for item in flagged:
    fn = item['filename']
    flags = item['flags']
    path = fn_to_path.get(fn, '')
    base_id = extract_base_id(fn)
    entry = {
        'filename': fn,
        'image_path': path,
        'disease_class': item['disease_class'],
        'split': item['split'],
        'flags': flags,
        'base_id': base_id
    }
    
    # Categorize
    is_studio = any('studio_white' in f or 'solid_white_border' in f for f in flags)
    is_wm = any('watermark' in f for f in flags)
    is_black = any('solid_black_border' in f for f in flags)
    
    if is_studio:
        group_studio.append(entry)
    elif is_wm:
        group_watermark.append(entry)
    elif is_black:
        group_black.append(entry)
    else:
        group_studio.append(entry)

print(f"Group Studio White / White Border: {len(group_studio)}")
print(f"Group Corner Watermark:            {len(group_watermark)}")
print(f"Group Solid Black Border:          {len(group_black)}")

# 3. Create Montages for each group
def create_montage(items, title, out_path, cols=4):
    n = len(items)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 4.0, rows * 4.2))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.2)
    
    for i, item in enumerate(items):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        p = item['image_path']
        try:
            im = Image.open(p).convert('RGB')
            ax.imshow(im)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        
        # Caption
        flags_str = "\n".join(item['flags'])
        caption = f"[{item['split']}] {item['disease_class']}\n{item['filename'][:32]}...\n{flags_str}"
        ax.set_title(caption, fontsize=9, pad=4)
        ax.axis('off')
        
    # Hide any unused subplots
    for j in range(n, rows * cols):
        ax = fig.add_subplot(gs[j // cols, j % cols])
        ax.axis('off')
        
    fig.suptitle(f"{title} (N={n})", fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved montage: {out_path}")

create_montage(group_studio, "Flagged Contamination: Studio White Background & Solid White Border", "outputs/final_v2/contamination_studio_white.png", cols=5)
create_montage(group_watermark, "Flagged Contamination: High-Contrast Corner Watermarks", "outputs/final_v2/contamination_watermarks.png", cols=4)
create_montage(group_black, "Flagged Contamination: Solid Black Borders / Scanner Edges", "outputs/final_v2/contamination_black_border.png", cols=4)

# 4. Check against the 24 images already removed
with open('outputs/improvements/removed_stock_images.json') as f:
    removed_24_paths = json.load(f)

removed_24_filenames = [os.path.basename(p) for p in removed_24_paths]
removed_24_base_ids = set([extract_base_id(fn) for fn in removed_24_filenames])
print(f"\nUnique base IDs among 24 already removed: {len(removed_24_base_ids)}")
print(f"Removed base IDs: {sorted(list(removed_24_base_ids))}")

related_to_24 = []
not_related_to_24 = []
for item in flagged:
    b_id = extract_base_id(item['filename'])
    item_with_base = dict(item)
    item_with_base['base_id'] = b_id
    if b_id in removed_24_base_ids:
        related_to_24.append(item_with_base)
    else:
        not_related_to_24.append(item_with_base)

print(f"\nFlagged images sharing base IDs with the 24 removed images: {len(related_to_24)} / {len(flagged)}")
print(f"Flagged images with NEW base IDs: {len(not_related_to_24)} / {len(flagged)}")

# 5. Detail the 2 remaining cross-split pairs
cross_pairs_details = [
    {
        'pair_id': 1,
        'image_1': {
            'filename': 'Mahali_Koleroga_original_778.jpg_801ed8f9-f513-48f7-b70f-d04261752c25.jpg',
            'split': 'train',
            'disease_class': 'mahali_koleroga',
            'severity_grade': 3
        },
        'image_2': {
            'filename': 'Mahali_Koleroga_original_776.jpg_1de389b0-5a91-49e6-bbd3-9226cd62093a.jpg',
            'split': 'test',
            'disease_class': 'mahali_koleroga',
            'severity_grade': 3
        },
        'hamming_distance_cv2_inter_area': 4,
        'hamming_distance_pil_lanczos_no_dc': 7,
        'why_earlier_check_missed': (
            "In merge_and_retrain_expanded_arecanut.py line 77, pHash was computed using PIL Image.Resampling.LANCZOS "
            "and excluded the DC coefficient (low[1:, 1:]). Under that implementation, the distance was 7 (strictly > 5), "
            "so the threshold condition min_dist <= 5 did not trigger."
        )
    },
    {
        'pair_id': 2,
        'image_1': {
            'filename': 'stem cracking_original_8160.jpg_062e7c85-7df7-48e1-bac6-bce639870d6d.jpg',
            'split': 'val',
            'disease_class': 'stem_cracking',
            'severity_grade': 2
        },
        'image_2': {
            'filename': 'stem cracking_original_8163.jpg_29b038e7-abf5-45a0-80f1-912ad1fec30c.jpg',
            'split': 'train',
            'disease_class': 'stem_cracking',
            'severity_grade': 2
        },
        'hamming_distance_cv2_inter_area': 4,
        'hamming_distance_pil_lanczos_no_dc': 6,
        'why_earlier_check_missed': (
            "Under the LANCZOS + no-DC implementation used in merge_and_retrain_expanded_arecanut.py, "
            "the distance was 6 (strictly > 5), so it was not clustered into the same group and passed the earlier check."
        )
    }
]

part_c_summary = {
    'total_flagged_images': len(flagged),
    'counts_by_reason': {
        'studio_white_background_or_solid_white_border': len(group_studio),
        'high_contrast_corner_watermark': len(group_watermark),
        'solid_black_border_or_scanner_edge': len(group_black)
    },
    'montages_saved': {
        'studio_white': 'outputs/final_v2/contamination_studio_white.png',
        'watermarks': 'outputs/final_v2/contamination_watermarks.png',
        'black_border': 'outputs/final_v2/contamination_black_border.png'
    },
    'overlap_with_24_already_removed': {
        'removed_24_base_ids': sorted(list(removed_24_base_ids)),
        'flagged_sharing_base_id_with_removed_count': len(related_to_24),
        'flagged_sharing_base_id_with_removed_list': [
            {'filename': x['filename'], 'base_id': x['base_id'], 'split': x['split'], 'flags': x['flags']}
            for x in related_to_24
        ],
        'flagged_with_new_base_ids_count': len(not_related_to_24),
        'flagged_with_new_base_ids_list': [
            {'filename': x['filename'], 'base_id': x['base_id'], 'split': x['split'], 'flags': x['flags']}
            for x in not_related_to_24
        ]
    },
    'remaining_cross_split_near_duplicate_pairs': cross_pairs_details
}

with open('outputs/final_v2/stage2b_part_c.json', 'w') as f:
    json.dump(part_c_summary, f, indent=2)

print("\nPart C complete! Saved to outputs/final_v2/stage2b_part_c.json")
