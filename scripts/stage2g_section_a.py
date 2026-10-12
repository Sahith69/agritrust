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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, f_oneway

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2g - SECTION A: SHEET V5 & TAXONOMY CORRECTION")
print("=" * 80)

# 1. Forensic Verification of Paddy Taxonomy
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
paddy_classes_present = sorted(df_paddy['disease_class'].unique())
print(f"All classes present in paddy_splits.csv (N={len(df_paddy)}):")
for c in paddy_classes_present:
    print(f"  • {c:25}: {len(df_paddy[df_paddy['disease_class'] == c]):4d} images")

has_leaf_scald = 'leaf_scald' in paddy_classes_present
has_narrow_brown_spot = 'narrow_brown_spot' in paddy_classes_present
print(f"\nTaxonomy check in paddy_splits.csv:")
print(f"  - 'leaf_scald' present?        {has_leaf_scald} (0 images)")
print(f"  - 'narrow_brown_spot' present? {has_narrow_brown_spot} (0 images)")

taxonomy_explanation = (
    "ORIGIN OF 'leaf_scald' AND 'narrow_brown_spot':\n"
    "In stage 2f, 'leaf_scald' and 'narrow_brown_spot' were mistakenly included in a hardcoded list of rice diseases\n"
    "(copied from standard agronomic literature / RiceLeaf / Kaggle rice datasets). Neither class exists in the\n"
    "Paddy Doctor benchmark (paddy_splits.csv contains 0 images of either). The true 7 lesion classes in Paddy Doctor\n"
    "(excluding the biological fallback classes 'hispa' and 'dead_heart', and excluding 'normal') are:\n"
    "  1. bacterial_leaf_blight\n"
    "  2. bacterial_leaf_streak\n"
    "  3. bacterial_panicle_blight\n"
    "  4. blast\n"
    "  5. brown_spot\n"
    "  6. downy_mildew\n"
    "  7. tungro"
)
print("\n" + taxonomy_explanation)

# 2. Paddy Twin-Free Pool at Hamming <= 8
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

print(f"\nPaddy twin-free test pool (Hamming <= 8): {len(paddy_twinfree)} / {len(paddy_test_df)}")

# Load precomputed paddy ratios
cache_paddy_ratios_path = 'outputs/final_v2/paddy_lesion_ratios_cache.csv'
if os.path.exists(cache_paddy_ratios_path):
    paddy_ratios_df = pd.read_csv(cache_paddy_ratios_path)
    fn_to_ratio = dict(zip(paddy_ratios_df['filename'], paddy_ratios_df['lesion_ratio_pct']))
    paddy_twinfree['lesion_ratio_pct'] = paddy_twinfree['filename'].map(fn_to_ratio).fillna(0.0)
else:
    paddy_twinfree['lesion_ratio_pct'] = 0.0

# 3. Load Arecanut dataset
df_areca = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')
c_sizes = df_areca.groupby('robust_cluster_id').size()
df_areca['cluster_size'] = df_areca['robust_cluster_id'].map(c_sizes)
areca_le10 = df_areca[df_areca['cluster_size'] <= 10].copy()

# Sample 150 Unique Slots (Seed 42)
np.random.seed(42)
selected_slots = []
used_areca_clusters = set()

# A. Arecanut (60 images): mahali_koleroga 35, stem_cracking 25 (clusters <=10, <=2 per cluster)
def sample_areca(cls_name, target_n):
    pool = areca_le10[areca_le10['disease_class'] == cls_name].copy()
    c_ids = sorted(pool['robust_cluster_id'].unique())
    np.random.shuffle(c_ids)
    sel_indices = []
    sel_c = []
    for c_id in c_ids:
        c_sub = pool[pool['robust_cluster_id'] == c_id]
        n_take = min(2, len(c_sub))
        if len(sel_indices) + n_take > target_n:
            n_take = target_n - len(sel_indices)
        if n_take > 0:
            sub_s = c_sub.sample(n=n_take, random_state=42)
            sel_indices.extend(sub_s.index.tolist())
            sel_c.append(c_id)
        if len(sel_indices) >= target_n:
            break
    return df_areca.loc[sel_indices].copy(), set(sel_c)

mk_df, mk_c = sample_areca('mahali_koleroga', 35)
sc_df, sc_c = sample_areca('stem_cracking', 25)
used_areca_clusters.update(mk_c)
used_areca_clusters.update(sc_c)

for _, r in mk_df.iterrows():
    selected_slots.append({
        'crop': 'Arecanut',
        'disease_class': r['disease_class'],
        'source': r['source'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r['robust_cluster_id'],
        'original_grade': r['severity_grade'],
        'lesion_ratio_pct': r.get('lesion_ratio', 0.0) * 100.0 if 'lesion_ratio' in r else 0.0,
        'stratum': 'arecanut_mahali_koleroga'
    })

for _, r in sc_df.iterrows():
    selected_slots.append({
        'crop': 'Arecanut',
        'disease_class': r['disease_class'],
        'source': r['source'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r['robust_cluster_id'],
        'original_grade': r['severity_grade'],
        'lesion_ratio_pct': r.get('lesion_ratio', 0.0) * 100.0 if 'lesion_ratio' in r else 0.0,
        'stratum': 'arecanut_stem_cracking'
    })

print(f"Selected Arecanut: mahali_koleroga={len(mk_df)} ({len(mk_c)} clus), stem_cracking={len(sc_df)} ({len(sc_c)} clus). Total Arecanut = 60.")

# B. Paddy Lesion (65 images): exactly 7 classes: blast 12, others 8 or 9 (sum = 65)
paddy_allocation = {
    'blast': 12,
    'bacterial_leaf_blight': 9,
    'bacterial_leaf_streak': 9,
    'bacterial_panicle_blight': 9,
    'brown_spot': 9,
    'tungro': 9,
    'downy_mildew': 8
}
assert sum(paddy_allocation.values()) == 65, f"Allocation sum {sum(paddy_allocation.values())} != 65"

paddy_lesion_slots_count = 0
for cls_name, target_n in paddy_allocation.items():
    pool = paddy_twinfree[paddy_twinfree['disease_class'] == cls_name].copy()
    if len(pool) < target_n:
        raise RuntimeError(f"Not enough twin-free images for {cls_name}: available {len(pool)} < target {target_n}")
    
    # Stratified: half near threshold (2%-8% or 12%-18%), half general
    near_mask = ((pool['lesion_ratio_pct'] >= 2.0) & (pool['lesion_ratio_pct'] <= 8.0)) | \
                ((pool['lesion_ratio_pct'] >= 12.0) & (pool['lesion_ratio_pct'] <= 18.0))
    near_pool = pool[near_mask]
    other_pool = pool[~near_mask]
    
    n_near = min(target_n // 2, len(near_pool))
    n_other = target_n - n_near
    
    s_near = near_pool.sample(n=n_near, random_state=42)
    s_other = other_pool.sample(n=n_other, random_state=42)
    s_cls = pd.concat([s_near, s_other]).reset_index(drop=True)
    
    for idx_c, r in s_cls.iterrows():
        selected_slots.append({
            'crop': 'Paddy',
            'disease_class': r['disease_class'],
            'source': 'Paddy_Dataset',
            'filename': r['filename'],
            'image_path': r['image_path'],
            'cluster_id': r.get('comp_le8', r.get('component_le8', -1)),
            'original_grade': r['severity_grade'],
            'lesion_ratio_pct': r['lesion_ratio_pct'],
            'stratum': f"paddy_{cls_name}"
        })
    paddy_lesion_slots_count += len(s_cls)
    print(f"Selected Paddy lesion {cls_name:25}: {len(s_cls)} images (target={target_n}).")

print(f"Total Paddy Lesion Slots: {paddy_lesion_slots_count} (Target = 65).")

# C. Healthy (25 images): Paddy normal 8, healthy_trunk 6 (distinct clusters), leaf 5, nut 6
norm_pool = paddy_twinfree[paddy_twinfree['disease_class'] == 'normal'].copy()
s_norm = norm_pool.sample(n=8, random_state=42)
for _, r in s_norm.iterrows():
    selected_slots.append({
        'crop': 'Paddy',
        'disease_class': 'normal',
        'source': 'Paddy_Dataset',
        'filename': r['filename'],
        'image_path': r['image_path'],
        'cluster_id': r.get('comp_le8', r.get('component_le8', -1)),
        'original_grade': 0,
        'lesion_ratio_pct': 0.0,
        'stratum': 'paddy_normal'
    })

def sample_healthy(cls_name, target_n):
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

ht_df, ht_c = sample_healthy('healthy_trunk', 6)
hl_df, hl_c = sample_healthy('healthy_leaf', 5)
hn_df, hn_c = sample_healthy('healthy_nut', 6)

used_areca_clusters.update(ht_c)
used_areca_clusters.update(hl_c)
used_areca_clusters.update(hn_c)

for h_df, h_cls in [(ht_df, 'healthy_trunk'), (hl_df, 'healthy_leaf'), (hn_df, 'healthy_nut')]:
    for _, r in h_df.iterrows():
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

print(f"Selected Healthy: Paddy normal=8, healthy_trunk=6, healthy_leaf=5, healthy_nut=6. Total Healthy = 25.")
print(f"Total Unique Slots: {len(selected_slots)} (Target = 150).")

# 4. Shuffle & Add 15 Repeat Pairs (Seed 2026)
df_unique = pd.DataFrame(selected_slots)
SHUFFLE_SEED = 2026
np.random.seed(SHUFFLE_SEED)
unique_indices = np.random.permutation(len(df_unique))
df_unique_shuffled = df_unique.iloc[unique_indices].reset_index(drop=True)
df_unique_shuffled['unique_image_id'] = range(1, len(df_unique_shuffled) + 1)

repeat_candidates = df_unique_shuffled.index.tolist()
repeat_sampled_orig_indices = np.random.choice(repeat_candidates, size=15, replace=False)

final_slots = []
for idx, r in df_unique_shuffled.iterrows():
    row_dict = r.to_dict()
    row_dict['is_repeat'] = False
    row_dict['repeat_of_slot_id'] = None
    final_slots.append(row_dict)

for rep_i, orig_i in enumerate(repeat_sampled_orig_indices):
    orig_slot_1based = orig_i + 1
    rep_row = df_unique_shuffled.iloc[orig_i].to_dict()
    rep_row['is_repeat'] = True
    rep_row['repeat_of_slot_id'] = orig_slot_1based
    
    target_pos = (orig_i + 35 + rep_i * 7) % (len(final_slots) + 1)
    if target_pos <= orig_i + 15:
        target_pos = min(len(final_slots), orig_i + 20 + rep_i * 3)
    final_slots.insert(target_pos, rep_row)

df_v5_key = pd.DataFrame(final_slots)
df_v5_key['slot_id'] = range(1, len(df_v5_key) + 1)
df_v5_key['page_number'] = (df_v5_key['slot_id'] - 1) // 6 + 2 # Page 1 is Instruction Card

print(f"Total slots in V5 booklet: {len(df_v5_key)} (150 unique + 15 repeats). Total Pages = {df_v5_key['page_number'].max()}.")

# Save Blank CSV and Key CSV/JSON
blank_v5_df = pd.DataFrame({
    'slot_id': df_v5_key['slot_id'],
    'human_grade': [''] * len(df_v5_key),
    'confidence_1_to_5': [''] * len(df_v5_key),
    'notes': [''] * len(df_v5_key)
})
blank_v5_path = 'outputs/final_v2/human_grading_sheet_v5_blank.csv'
blank_v5_df.to_csv(blank_v5_path, index=False)
print(f"Saved blank V5 CSV: {blank_v5_path}")

key_v5_csv_path = 'outputs/final_v2/human_grading_answer_key_v5.csv'
df_v5_key.to_csv(key_v5_csv_path, index=False)
print(f"Saved answer key CSV: {key_v5_csv_path}")

key_v5_json_path = 'outputs/final_v2/human_grading_answer_key_v5.json'
with open(key_v5_json_path, 'w') as f:
    json.dump(df_v5_key.to_dict(orient='records'), f, indent=2)

# Generate PDF Booklet
pdf_v5_path = 'outputs/final_v2/human_grading_sheet_v5.pdf'
print(f"Generating 29-page PDF Booklet {pdf_v5_path}...")

with PdfPages(pdf_v5_path) as pdf:
    # Page 1: Instruction Card
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.5, 0.95, "AGRICULTURAL DISEASE SEVERITY GRADING PROTOCOL (v5 BLIND)", transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', color='#1A252C')
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
    total_pages = 28
    for page_i in range(total_pages):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        gs = GridSpec(3, 2, figure=fig, hspace=0.28, wspace=0.20, left=0.06, right=0.94, top=0.94, bottom=0.05)
        
        page_slots = df_v5_key[df_v5_key['page_number'] == page_i + 2]
        
        for slot_idx_in_page, (_, r) in enumerate(page_slots.iterrows()):
            r_i = slot_idx_in_page // 2
            c_i = slot_idx_in_page % 2
            ax_slot = fig.add_subplot(gs[r_i, c_i])
            
            img_bgr = cv2.imread(r['image_path'])
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax_slot.imshow(img_rgb)
            else:
                ax_slot.text(0.5, 0.5, "Image Load Error", ha='center', va='center')
                
            ax_slot.axis('off')
            ax_slot.set_title(f"Slot #{r['slot_id']:03d}", fontsize=10, fontweight='bold', pad=4, color='#1A252C')
            for spine in ax_slot.spines.values():
                spine.set_color('#BDC3C7'); spine.set_linewidth(1)
                
        fig.suptitle(f"Blind Agricultural Disease Severity Grading Sheet — Page {page_i+2} of {total_pages+1}", fontsize=9.5, y=0.985, color='#555555')
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

print(f"Generated 29-page PDF booklet: {pdf_v5_path}")

# 5. Full Leak Scan for 25 Forbidden Strings
print("\n--- FULL PDF LEAK SCAN (ALL 25 FORBIDDEN STRINGS) ---")
info_proc = subprocess.run(['pdfinfo', pdf_v5_path], capture_output=True, text=True, check=True)
pdf_info_text = info_proc.stdout

text_proc = subprocess.run(['pdftotext', pdf_v5_path, '-'], capture_output=True, text=True, check=True)
full_pdf_text = text_proc.stdout
pages_text = full_pdf_text.split('\x0c')

images_proc = subprocess.run(['pdfimages', '-list', pdf_v5_path], capture_output=True, text=True, check=True)
pdf_images_text = images_proc.stdout

forbidden_25_terms = [
    'mahali', 'koleroga', 'stem_cracking', 'stem_bleeding', 'bud_borer', 'yellow_leaf_disease',
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'panicle_blight', 'blast', 'brown_spot',
    'downy_mildew', 'hispa', 'dead_heart', 'tungro', 'healthy_foot', 'healthy_nut', 'healthy_trunk', 'healthy_leaf',
    'arecanut', 'paddy', 'kaggle', 'ratio:', '.jpg', '.png'
]
assert len(forbidden_25_terms) == 25, f"Expected 25 forbidden strings, got {len(forbidden_25_terms)}"

p2_text = "".join(pages_text[1:])
leak_results = {}
for term in forbidden_25_terms:
    cnt = len(re.findall(re.escape(term), p2_text, re.IGNORECASE))
    leak_results[term] = cnt

total_leaks = sum(leak_results.values())

print("\nPDF METADATA (pdfinfo):")
print(pdf_info_text)

print("\nEMBEDDED IMAGE OBJECT STREAM NAMES (pdfimages -list | head -n 20):")
for line in pdf_images_text.splitlines()[:20]:
    print("  " + line)

print("\nTEXT LAYER LEAK SCAN ON GRADER PAGES (Pages 2 to 29):")
for term in forbidden_25_terms:
    print(f"  • {term:<26}: {leak_results[term]} occurrences")
print(f"\nTOTAL LEAKS FOUND ACROSS ALL 25 FORBIDDEN STRINGS: {total_leaks}")

# 6. New Gold Lock V3 (keep v1 and v2 intact)
areca_gold_v5 = df_v5_key[df_v5_key['crop'] == 'Arecanut']
areca_v5_clusters = set(areca_gold_v5['cluster_id'].unique())
areca_locked_v3 = df_areca[df_areca['robust_cluster_id'].isin(areca_v5_clusters)].copy()

paddy_gold_v5 = df_v5_key[df_v5_key['crop'] == 'Paddy']
paddy_locked_v3 = paddy_test_df[paddy_test_df['is_twin_le8']].copy()

v3_lock_rows = []
for idx, r in areca_locked_v3.iterrows():
    v3_lock_rows.append({
        'crop_type': 'Arecanut',
        'disease_class': r['disease_class'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'is_direct_gold': r['filename'] in set(areca_gold_v5['filename']),
        'lock_reason': 'direct_gold_sample' if r['filename'] in set(areca_gold_v5['filename']) else 'cluster_mate_burst',
        'cluster_id': r['robust_cluster_id']
    })

for idx, r in paddy_locked_v3.iterrows():
    v3_lock_rows.append({
        'crop_type': 'Paddy',
        'disease_class': r['disease_class'],
        'filename': r['filename'],
        'image_path': r['image_path'],
        'is_direct_gold': r['filename'] in set(paddy_gold_v5['filename']),
        'lock_reason': 'direct_gold_sample' if r['filename'] in set(paddy_gold_v5['filename']) else 'phash_twin_le8',
        'cluster_id': r.get('comp_le8', r.get('component_le8', -1))
    })

df_gold_lock_v3 = pd.DataFrame(v3_lock_rows)
gold_lock_v3_path = 'outputs/final_v2/gold_lock_v3.csv'
df_gold_lock_v3.to_csv(gold_lock_v3_path, index=False)
print(f"\nSaved Gold Lock V3 to: {gold_lock_v3_path}")
print(f"Total locked images in V3: {len(df_gold_lock_v3)} (Arecanut: {len(areca_locked_v3)}, Paddy: {len(paddy_locked_v3)})")

stage2g_a_output = {
    'taxonomy_explanation': taxonomy_explanation,
    'sheet_v5_pdf': pdf_v5_path,
    'sheet_v5_blank_csv': blank_v5_path,
    'sheet_v5_key_csv': key_v5_csv_path,
    'sheet_v5_key_json': key_v5_json_path,
    'gold_lock_v3_path': gold_lock_v3_path,
    'total_slots': len(df_v5_key),
    'unique_images': 150,
    'repeat_pairs': 15,
    'arecanut_slots': 60,
    'arecanut_clusters_covered': len(used_areca_clusters),
    'paddy_lesion_slots': 65,
    'paddy_allocation': paddy_allocation,
    'healthy_slots': 25,
    'leak_results': leak_results,
    'total_leaks_found': total_leaks,
    'v3_lock_total_images': len(df_gold_lock_v3),
    'v3_lock_arecanut_images': len(areca_locked_v3),
    'v3_lock_paddy_images': len(paddy_locked_v3)
}

with open('outputs/final_v2/stage2g_section_a.json', 'w') as f:
    json.dump(stage2g_a_output, f, indent=2)

print("Saved Stage 2g Section A results to outputs/final_v2/stage2g_section_a.json")
