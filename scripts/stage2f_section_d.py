import os
import sys
import json
import re
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, f_oneway

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2f - SECTION D: HUMAN GRADING SHEET V4 & GOLD LOCK V3")
print("=" * 80)

# Load Arecanut robust clusters & Paddy dataset
df_areca = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
c_sizes = df_areca.groupby('robust_cluster_id').size()
df_areca['cluster_size'] = df_areca['robust_cluster_id'].map(c_sizes)

df_paddy = pd.read_csv('outputs/paddy_splits.csv')

# Paddy twin-free at Hamming <= 8 mask
cache_phash_path = 'outputs/final_v2/paddy_phashes_cache.npz'
npz = np.load(cache_phash_path)
h_orig = npz['orig']
h_hflip = npz['hflip']
h_vflip = npz['vflip']

def fast_popcount_matrix(m):
    m = m - ((m >> np.uint64(1)) & np.uint64(0x5555555555555555))
    m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
    m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return m.astype(np.int32)

splits = df_paddy['split'].values
test_idx = np.where(splits == 'test')[0]
train_val_idx = np.where(splits != 'test')[0]

def find_twins_le8():
    h_t = h_orig[test_idx]
    h_tv_o = h_orig[train_val_idx]
    h_tv_h = h_hflip[train_val_idx]
    h_tv_v = h_vflip[train_val_idx]
    twins = set()
    chunk_a = 500
    chunk_b = 2500
    for a_start in range(0, len(test_idx), chunk_a):
        a_end = min(a_start + chunk_a, len(test_idx))
        sub_a = h_t[a_start:a_end]
        for b_start in range(0, len(train_val_idx), chunk_b):
            b_end = min(b_start + chunk_b, len(train_val_idx))
            d1 = fast_popcount_matrix(sub_a[:, None] ^ h_tv_o[b_start:b_end][None, :])
            d2 = fast_popcount_matrix(sub_a[:, None] ^ h_tv_h[b_start:b_end][None, :])
            d3 = fast_popcount_matrix(sub_a[:, None] ^ h_tv_v[b_start:b_end][None, :])
            min_d = np.minimum(d1, np.minimum(d2, d3))
            hits = np.where(min_d <= 8)
            for r_i, c_j in zip(hits[0], hits[1]):
                twins.add(test_idx[a_start + r_i])
    return twins

paddy_twins_le8_idx = find_twins_le8()
paddy_test_df = df_paddy.iloc[test_idx].copy().reset_index(drop=True)
paddy_test_df['is_twin_le8'] = [test_idx[i] in paddy_twins_le8_idx for i in range(len(test_idx))]
paddy_twinfree = paddy_test_df[~paddy_test_df['is_twin_le8']].copy()

print(f"Paddy twin-free test pool (Hamming <= 8): {len(paddy_twinfree)} / {len(paddy_test_df)}")

# Load precomputed paddy lesion ratios if cached
cache_paddy_ratios_path = 'outputs/final_v2/paddy_lesion_ratios_cache.csv'
if os.path.exists(cache_paddy_ratios_path):
    paddy_ratios_df = pd.read_csv(cache_paddy_ratios_path)
    fn_to_ratio = dict(zip(paddy_ratios_df['filename'], paddy_ratios_df['lesion_ratio_pct']))
    paddy_twinfree['lesion_ratio_pct'] = paddy_twinfree['filename'].map(fn_to_ratio).fillna(0.0)
else:
    paddy_twinfree['lesion_ratio_pct'] = 0.0

# Sampling 150 Unique Slot Images (Seed 42)
np.random.seed(42)

selected_slots = []
used_areca_clusters = set()

# D.1 Arecanut (60 images): mahali_koleroga 35, stem_cracking 25 (clusters <=10, <=2 per cluster)
areca_le10 = df_areca[df_areca['cluster_size'] <= 10].copy()

def sample_areca_class(cls_name, target_n):
    pool = areca_le10[areca_le10['disease_class'] == cls_name].copy()
    c_ids = sorted(pool['robust_cluster_id'].unique())
    selected_indices = []
    selected_c = []
    
    np.random.shuffle(c_ids)
    for c_id in c_ids:
        c_sub = pool[pool['robust_cluster_id'] == c_id]
        n_take = min(2, len(c_sub))
        if len(selected_indices) + n_take > target_n:
            n_take = target_n - len(selected_indices)
        if n_take > 0:
            sub_s = c_sub.sample(n=n_take, random_state=42)
            selected_indices.extend(sub_s.index.tolist())
            selected_c.append(c_id)
        if len(selected_indices) >= target_n:
            break
    return df_areca.loc[selected_indices].copy(), set(selected_c)

mk_df, mk_clusters = sample_areca_class('mahali_koleroga', 35)
sc_df, sc_clusters = sample_areca_class('stem_cracking', 25)

used_areca_clusters.update(mk_clusters)
used_areca_clusters.update(sc_clusters)

for idx, r in mk_df.iterrows():
    selected_slots.append({
        'crop': 'Arecanut',
        'disease_class': r['disease_class'],
        'source': r['source'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r['robust_cluster_id'],
        'original_grade': r['severity_grade'],
        'lesion_ratio_pct': r.get('lesion_ratio', 0.0) * 100.0 if 'lesion_ratio' in r else 0.0,
        'stratum': 'arecanut_lesion_le10'
    })

for idx, r in sc_df.iterrows():
    selected_slots.append({
        'crop': 'Arecanut',
        'disease_class': r['disease_class'],
        'source': r['source'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r['robust_cluster_id'],
        'original_grade': r['severity_grade'],
        'lesion_ratio_pct': r.get('lesion_ratio', 0.0) * 100.0 if 'lesion_ratio' in r else 0.0,
        'stratum': 'arecanut_lesion_le10'
    })

print(f"Selected Arecanut lesion slots: mahali_koleroga={len(mk_df)}, stem_cracking={len(sc_df)} across {len(used_areca_clusters)} clusters.")

# D.2 Paddy Lesion (65 images): 7 classes, half within +/- 3% of thresholds (5%, 15%)
paddy_lesion_classes = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
                        'blast', 'brown_spot', 'leaf_scald', 'narrow_brown_spot']
paddy_lesion_pool = paddy_twinfree[paddy_twinfree['disease_class'].isin(paddy_lesion_classes)].copy()

near_thresh_mask = ((paddy_lesion_pool['lesion_ratio_pct'] >= 2.0) & (paddy_lesion_pool['lesion_ratio_pct'] <= 8.0)) | \
                  ((paddy_lesion_pool['lesion_ratio_pct'] >= 12.0) & (paddy_lesion_pool['lesion_ratio_pct'] <= 18.0))

near_thresh_pool = paddy_lesion_pool[near_thresh_mask]
other_lesion_pool = paddy_lesion_pool[~near_thresh_mask]

s_near = near_thresh_pool.sample(n=min(32, len(near_thresh_pool)), random_state=42)
s_other = other_lesion_pool.sample(n=65 - len(s_near), random_state=42)
paddy_lesion_sample = pd.concat([s_near, s_other]).reset_index(drop=True)

for idx, r in paddy_lesion_sample.iterrows():
    selected_slots.append({
        'crop': 'Paddy',
        'disease_class': r['disease_class'],
        'source': 'Paddy_Dataset',
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r.get('comp_le8', -1),
        'original_grade': r['severity_grade'],
        'lesion_ratio_pct': r['lesion_ratio_pct'],
        'stratum': 'paddy_lesion_near_thresh' if idx < len(s_near) else 'paddy_lesion_general'
    })

print(f"Selected Paddy lesion slots: {len(paddy_lesion_sample)} images across 7 classes (near threshold: {len(s_near)}).")

# D.3 Healthy (25 images): Paddy normal 8, healthy_trunk 6 (distinct clusters), leaf 5, nut 6
paddy_norm_sample = paddy_twinfree[paddy_twinfree['disease_class'] == 'normal'].sample(n=8, random_state=42)
for idx, r in paddy_norm_sample.iterrows():
    selected_slots.append({
        'crop': 'Paddy',
        'disease_class': 'normal',
        'source': 'Paddy_Dataset',
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r.get('comp_le8', -1),
        'original_grade': 0,
        'lesion_ratio_pct': 0.0,
        'stratum': 'paddy_normal'
    })

# Arecanut Healthy: healthy_trunk 6 (distinct clusters), healthy_leaf 5, healthy_nut 6
def sample_healthy_areca(cls_name, target_n):
    pool = areca_le10[areca_le10['disease_class'] == cls_name].copy()
    c_ids = sorted(pool['robust_cluster_id'].unique())
    np.random.shuffle(c_ids)
    sel_rows = []
    sel_c = []
    for c_id in c_ids:
        c_sub = pool[pool['robust_cluster_id'] == c_id]
        r_one = c_sub.sample(n=1, random_state=42)
        sel_rows.append(r_one)
        sel_c.append(c_id)
        if len(sel_rows) >= target_n: break
    df_res = pd.concat(sel_rows).reset_index(drop=True)
    return df_res, set(sel_c)

ht_df, ht_c = sample_healthy_areca('healthy_trunk', 6)
hl_df, hl_c = sample_healthy_areca('healthy_leaf', 5)
hn_df, hn_c = sample_healthy_areca('healthy_nut', 6)

used_areca_clusters.update(ht_c)
used_areca_clusters.update(hl_c)
used_areca_clusters.update(hn_c)

for h_df, h_cls in [(ht_df, 'healthy_trunk'), (hl_df, 'healthy_leaf'), (hn_df, 'healthy_nut')]:
    for idx, r in h_df.iterrows():
        selected_slots.append({
            'crop': 'Arecanut',
            'disease_class': r['disease_class'],
            'source': r['source'],
            'filename': r['filename'],
            'image_path': r['image_path'],
            'cluster_id': r['robust_cluster_id'],
            'original_grade': 0,
            'lesion_ratio_pct': 0.0,
            'stratum': f'arecanut_{h_cls}'
        })

print(f"Selected Healthy slots: Paddy normal=8, healthy_trunk=6, healthy_leaf=5, healthy_nut=6 (Total healthy = 25).")
print(f"Total Unique Evaluation Slots: {len(selected_slots)} (Target = 150).")

# D.4 Shuffling & 15 Blind Repeat Pairs Insertion (Seed 2026)
df_unique = pd.DataFrame(selected_slots)

SHUFFLE_SEED = 2026
np.random.seed(SHUFFLE_SEED)
unique_indices = np.random.permutation(len(df_unique))
df_unique_shuffled = df_unique.iloc[unique_indices].reset_index(drop=True)
df_unique_shuffled['unique_image_id'] = range(1, len(df_unique_shuffled) + 1)

# Sample 15 repeat images across strata
repeat_candidates = df_unique_shuffled.index.tolist()
repeat_sampled_orig_indices = np.random.choice(repeat_candidates, size=15, replace=False)

final_slots = []
for idx, r in df_unique_shuffled.iterrows():
    row_dict = r.to_dict()
    row_dict['is_repeat'] = False
    row_dict['repeat_of_slot_id'] = None
    final_slots.append(row_dict)

# Insert 15 repeats into slots ensuring >= 15 separation and different pages
for rep_i, orig_i in enumerate(repeat_sampled_orig_indices):
    orig_slot_1based = orig_i + 1
    orig_page = (orig_slot_1based - 1) // 6 + 1
    
    rep_row = df_unique_shuffled.iloc[orig_i].to_dict()
    rep_row['is_repeat'] = True
    rep_row['repeat_of_slot_id'] = orig_slot_1based
    
    # Target insertion slot at least 15 slots away and on different page
    target_pos = (orig_i + 35 + rep_i * 7) % (len(final_slots) + 1)
    if target_pos <= orig_i + 15:
        target_pos = min(len(final_slots), orig_i + 20 + rep_i * 3)
        
    final_slots.insert(target_pos, rep_row)

df_v4_key = pd.DataFrame(final_slots)
df_v4_key['slot_id'] = range(1, len(df_v4_key) + 1)
df_v4_key['page_number'] = (df_v4_key['slot_id'] - 1) // 6 + 2 # Page 1 is Instruction Card

print(f"\nFinal V4 Booklet Slots: {len(df_v4_key)} (150 unique + 15 repeats). Total Pages = {df_v4_key['page_number'].max()}.")

# Audit runs of same-class or same-source consecutive slots
classes_seq = df_v4_key['disease_class'].values
sources_seq = df_v4_key['source'].values

def max_run_length(arr):
    max_r = 1
    curr_r = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            curr_r += 1
            if curr_r > max_r: max_r = curr_r
        else:
            curr_r = 1
    return max_r

max_class_run = max_run_length(classes_seq)
max_source_run = max_run_length(sources_seq)

print(f"Shuffling Run Audit:")
print(f"  • Maximum run length of consecutive SAME-CLASS slots:  {max_class_run}")
print(f"  • Maximum run length of consecutive SAME-SOURCE slots: {max_source_run}")

# D.5 Save V4 CSVs and PDF Booklet
blank_v4_df = pd.DataFrame({
    'slot_id': df_v4_key['slot_id'],
    'human_grade': [''] * len(df_v4_key),
    'confidence_1_to_5': [''] * len(df_v4_key),
    'notes': [''] * len(df_v4_key)
})
blank_v4_path = 'outputs/final_v2/human_grading_sheet_v4_blank.csv'
blank_v4_df.to_csv(blank_v4_path, index=False)
print(f"Saved blank V4 CSV: {blank_v4_path}")

key_v4_csv_path = 'outputs/final_v2/human_grading_answer_key_v4.csv'
df_v4_key.to_csv(key_v4_csv_path, index=False)
print(f"Saved V4 answer key CSV: {key_v4_csv_path}")

key_v4_json_path = 'outputs/final_v2/human_grading_answer_key_v4.json'
with open(key_v4_json_path, 'w') as f:
    json.dump(df_v4_key.to_dict(orient='records'), f, indent=2)

# PDF Generation (29 pages: Page 1 Instruction Card + 28 Grader Pages with RAW images only)
pdf_v4_path = 'outputs/final_v2/human_grading_sheet_v4.pdf'
print(f"Generating PDF Booklet {pdf_v4_path}...")

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(pdf_v4_path) as pdf:
    # Page 1: Instruction Card
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    p1_title = "AGRICULTURAL DISEASE SEVERITY GRADING PROTOCOL (v4 BLIND)"
    ax.text(0.5, 0.95, p1_title, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', color='#1A252C')
    ax.text(0.5, 0.92, "Standardized Protocol & Visual Calibration Booklet (Raw Images Only)", transform=ax.transAxes, fontsize=10, fontstyle='italic', ha='center', color='#555555')
    
    instruct_text = (
        "OBJECTIVE:\n"
        "Accurately estimate disease severity by evaluating the proportion of leaf/organ surface covered by visible lesions.\n"
        "Assess severity strictly on an ordinal 4-grade scale [Grade 0 to Grade 3].\n\n"
        "SEVERITY SCALE DEFINITIONS (Standard Equation 2 Thresholds):\n"
        "  • Grade 0: No visible symptoms. Organ surface displays healthy, uniform green/bark appearance with 0% necrosis.\n"
        "  • Grade 1: > 0% to ≤ 5% lesion area. Localized specks, tiny spots, or minor chlorotic discoloration.\n"
        "  • Grade 2: > 5% to ≤ 15% lesion area. Moderate coalesced lesions occupying up to 15% of total organ area.\n"
        "  • Grade 3: > 15% lesion area. Severe necrosis, extensive blight/rots occupying over 15% of surface.\n\n"
        "GRADING INSTRUCTIONS:\n"
        "1. Evaluate each raw image slot independently.\n"
        "2. Record your grade (0, 1, 2, or 3) and confidence score (1 = Lowest, 5 = Highest) in the blank CSV sheet.\n"
        "3. All slot images are RAW RGB captures with zero overlay masks or annotations to prevent rater bias."
    )
    ax.text(0.05, 0.60, instruct_text, transform=ax.transAxes, fontsize=9.5, va='top', ha='left', color='#2C3E50', linespacing=1.4)
    
    # Area calibration diagrams (2%, 5%, 10%, 15%, 25%)
    ax.text(0.5, 0.42, "VISUAL AREA PROPORTION CALIBRATION SWATCHES", transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', color='#1A252C')
    swatch_percents = [2, 5, 10, 15, 25]
    for idx_sw, pct in enumerate(swatch_percents):
        cx = 0.12 + idx_sw * 0.19
        cy = 0.25
        circle_outer = plt.Circle((cx, cy), 0.07, color='#ECEFF1', ec='#37474F', lw=1.5, transform=ax.transAxes)
        ax.add_patch(circle_outer)
        r_inner = 0.07 * np.sqrt(pct / 100.0)
        circle_inner = plt.Circle((cx, cy), r_inner, color='#D32F2F', transform=ax.transAxes)
        ax.add_patch(circle_inner)
        ax.text(cx, cy - 0.10, f"{pct}% Area", transform=ax.transAxes, fontsize=9, fontweight='bold', ha='center', color='#2C3E50')
        
    ax.text(0.5, 0.05, "Page 1 of 29 — Instruction Card", transform=ax.transAxes, fontsize=9, ha='center', color='#7F8C8D')
    pdf.savefig(fig, dpi=120)
    plt.close(fig)
    
    # Pages 2-29: 28 Grader Pages with 6 RAW images per page
    total_slots = len(df_v4_key)
    slots_per_page = 6
    total_pages = 28
    
    for page_i in range(total_pages):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        gs = GridSpec(3, 2, figure=fig, hspace=0.28, wspace=0.20, left=0.06, right=0.94, top=0.94, bottom=0.05)
        
        page_slots = df_v4_key[df_v4_key['page_number'] == page_i + 2]
        
        for slot_idx_in_page, (_, r) in enumerate(page_slots.iterrows()):
            r_i = slot_idx_in_page // 2
            c_i = slot_idx_in_page % 2
            ax_slot = fig.add_subplot(gs[r_i, c_i])
            
            # Load RAW RGB image ONLY (no overlays!)
            img_bgr = cv2.imread(r['image_path'])
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax_slot.imshow(img_rgb)
            else:
                ax_slot.text(0.5, 0.5, "Image Load Error", ha='center', va='center')
                
            ax_slot.axis('off')
            
            # Blind slot title
            title_str = f"Slot #{r['slot_id']:03d}"
            ax_slot.set_title(title_str, fontsize=10, fontweight='bold', pad=4, color='#1A252C')
            
            for spine in ax_slot.spines.values():
                spine.set_color('#BDC3C7'); spine.set_linewidth(1)
                
        fig.suptitle(f"Blind Agricultural Disease Severity Grading Sheet — Page {page_i+2} of {total_pages+1}", fontsize=9.5, y=0.985, color='#555555')
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

print(f"Generated 29-page PDF booklet: {pdf_v4_path}")

# D.6 FULL PDF LEAK CHECK AUDIT
print("\n--- FULL PDF LEAK CHECK AUDIT ---")
info_proc = subprocess.run(['pdfinfo', pdf_v4_path], capture_output=True, text=True, check=True)
pdf_info_text = info_proc.stdout

text_proc = subprocess.run(['pdftotext', pdf_v4_path, '-'], capture_output=True, text=True, check=True)
full_pdf_text = text_proc.stdout
pages_text = full_pdf_text.split('\x0c')

images_proc = subprocess.run(['pdfimages', '-list', pdf_v4_path], capture_output=True, text=True, check=True)
pdf_images_text = images_proc.stdout

forbidden_terms = [
    'mahali', 'koleroga', 'stem_cracking', 'stem_bleeding', 'bud_borer', 'yellow_leaf_disease',
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'panicle_blight', 'blast', 'brown_spot',
    'hispa', 'dead_heart', 'tungro', 'healthy_foot', 'healthy_nut', 'healthy_trunk', 'healthy_leaf',
    'arecanut', 'paddy', 'kaggle', 'ratio:', '.jpg', '.png', '.jpeg'
]

p2_text = "".join(pages_text[1:])
leak_results = {}
for term in forbidden_terms:
    cnt = len(re.findall(re.escape(term), p2_text, re.IGNORECASE))
    leak_results[term] = cnt

total_leaks = sum(leak_results.values())

print("\nPDF METADATA (pdfinfo):")
print(pdf_info_text)

print("\nEMBEDDED IMAGE OBJECT STREAM NAMES (pdfimages -list | head -n 20):")
for line in pdf_images_text.splitlines()[:20]:
    print("  " + line)

print(f"\nTEXT LAYER LEAK SCAN ON GRADER PAGES (Pages 2 to 29):")
print(f"Total Leaks Found = {total_leaks}")

# D.7 Write New Gold Lock V3 (gold_lock_v3.csv)
paddy_locked = paddy_test_df[paddy_test_df['is_twin_le8']].copy()

v4_areca_locked_clusters = set(df_v4_key[df_v4_key['crop'] == 'Arecanut']['cluster_id'].unique())
areca_locked_v3 = df_areca[df_areca['robust_cluster_id'].isin(v4_areca_locked_clusters)].copy()

print(f"\nGold Lock V3:")
print(f"  • Arecanut Gold Images: {len(df_v4_key[df_v4_key['crop'] == 'Arecanut'])} across {len(v4_areca_locked_clusters)} clusters.")
print(f"  • Arecanut Total Locked (Gold + Mates): {len(areca_locked_v3)} images.")
print(f"  • Paddy Total Locked (Gold + Twins <=8): {len(paddy_locked)} images.")

v3_lock_rows = []
for idx, r in areca_locked_v3.iterrows():
    v3_lock_rows.append({
        'crop_type': 'Arecanut',
        'disease_class': r['disease_class'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'is_direct_gold': r['filename'] in set(df_v4_key['filename']),
        'lock_reason': 'direct_gold_sample' if r['filename'] in set(df_v4_key['filename']) else 'cluster_mate_burst',
        'cluster_id': r['robust_cluster_id']
    })

for idx, r in paddy_locked.iterrows():
    v3_lock_rows.append({
        'crop_type': 'Paddy',
        'disease_class': r['disease_class'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'is_direct_gold': r['filename'] in set(df_v4_key['filename']),
        'lock_reason': 'direct_gold_sample' if r['filename'] in set(df_v4_key['filename']) else 'phash_twin_le8',
        'cluster_id': r.get('comp_le8', r.get('component_le8', -1))
    })

df_v3_lock = pd.DataFrame(v3_lock_rows)
v3_lock_path = 'outputs/final_v2/gold_lock_v3.csv'
df_v3_lock.to_csv(v3_lock_path, index=False)
print(f"Saved new gold lock V3: {v3_lock_path} (Total locked = {len(df_v3_lock)}).")

stage2f_d_output = {
    'pdf_path': pdf_v4_path,
    'blank_csv_path': blank_v4_path,
    'answer_key_csv_path': key_v4_csv_path,
    'answer_key_json_path': key_v4_json_path,
    'gold_lock_v3_path': v3_lock_path,
    'total_slots': len(df_v4_key),
    'unique_images': 150,
    'repeat_pairs': 15,
    'shuffling_audit': {
        'seed': SHUFFLE_SEED,
        'max_run_same_class': int(max_class_run),
        'max_run_same_source': int(max_source_run)
    },
    'total_leaks_found': total_leaks,
    'v3_lock_total_images': len(df_v3_lock),
    'v3_lock_arecanut_images': len(areca_locked_v3),
    'v3_lock_paddy_images': len(paddy_locked)
}

with open('outputs/final_v2/stage2f_section_d.json', 'w') as f:
    json.dump(stage2f_d_output, f, indent=2)

print("\nSaved Stage 2f Section D results to outputs/final_v2/stage2f_section_d.json")
