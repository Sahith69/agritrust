import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('/home/sahith/Projects/Crop/figures', exist_ok=True)

# -------------------------------------------------------------
# Figure 1: The Process Pipeline Diagram (fig_process_pipeline.png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.8)

box_1 = dict(boxstyle='round,pad=0.45', facecolor='#E3F2FD', edgecolor='#1976D2', lw=1.8)
box_2 = dict(boxstyle='round,pad=0.45', facecolor='#E8F5E9', edgecolor='#388E3C', lw=1.8)
box_3 = dict(boxstyle='round,pad=0.45', facecolor='#FFF3E0', edgecolor='#F57C00', lw=1.8)
box_4 = dict(boxstyle='round,pad=0.45', facecolor='#F3E5F5', edgecolor='#7B1FA2', lw=1.8)
box_5 = dict(boxstyle='round,pad=0.45', facecolor='#ECEFF1', edgecolor='#455A64', lw=1.8)
box_6 = dict(boxstyle='round,pad=0.45', facecolor='#E0F2F1', edgecolor='#00897B', lw=1.8)

# Stage 1: Data Ingestion & Auto-Labeling
ax.text(1.2, 3.6, "1. Weakly Supervised\nAuto-Labeling\n\n• Adaptive HSV/Lab Seg.\n• Otsu Morphological Filter\n• Lesion Area Ratio (rho)\n• Horsfall-Barratt Grades:\n  G0 (0%), G1 (<5%),\n  G2 (5-15%), G3 (>15%)\n• Boundary Conf. Filter", 
        ha='center', va='center', fontsize=8, bbox=box_1)

# Stage 2: Leak-Free Group Splitting
ax.text(3.7, 3.6, "2. Leak-Free\nStratified Split\n\n• Parent Timestamp Regex:\n  IMG_YYYYMMDD_HHMMSS\n• GroupShuffleSplit on Trees\n• 0% Cross-Split Leakage\n• Paddy: MD5 Deduplication\n  (70 / 15 / 15 Splits)", 
        ha='center', va='center', fontsize=8, bbox=box_2)

# Stage 3: Ordinal Learning (CORN)
ax.text(6.3, 3.6, "3. Deep Ordinal\nClassification\n\n• ResNet-18, ResNet-34,\n  MobileNetV2 Backbones\n• CORN Ordinal Loss Head\n• Cumulative Rank Logits\n• Strong Penalty on\n  Severe Errors (|Delta| >= 2)", 
        ha='center', va='center', fontsize=8, bbox=box_3)

# Stage 4: Explainability Audit
ax.text(8.8, 3.6, "4. Explainability &\nAttention Audit\n\n• Grad-CAM on Layer4\n• Mask IoU vs Lesion Area\n• Anatomical Shortcuts\n  (Panicle vs Lesion)\n• Mask-Size Confound\n  Identification", 
        ha='center', va='center', fontsize=8, bbox=box_4)

# Arrows between top stages
arrow_props = dict(arrowstyle='->', lw=2, color='#37474F')
ax.annotate('', xy=(2.45, 3.6), xytext=(1.95, 3.6), arrowprops=arrow_props)
ax.annotate('', xy=(5.0, 3.6), xytext=(4.5, 3.6), arrowprops=arrow_props)
ax.annotate('', xy=(7.55, 3.6), xytext=(7.05, 3.6), arrowprops=arrow_props)

# Row 2: Evaluation & Practical Deliverables
ax.text(3.7, 1.1, "5. Empirical Validation & Benchmarking\n• Quadratic Weighted Kappa (Paddy Full QWK: 0.939, Lesion-Only: 0.706; Arecanut Audited N=111 QWK: 0.700)\n• Paired Correctness McNemar's Test (p = 0.656 > 0.05)\n• Zero-Shot Cross-Crop Evaluation (Uncovering Bipolar Prediction Collapse)\n• Multi-Seed Robustness Benchmarking (3 Random Seeds: 42, 100, 2024)", 
        ha='center', va='center', fontsize=8, bbox=box_5)

ax.text(8.2, 1.1, "6. Field Deployment\n\n• Interactive Streamlit GUI (app.py)\n• MobileNetV2 Edge Checkpoint\n• Real-Time Grad-CAM Inspection\n• Offline Agronomic Utility", 
        ha='center', va='center', fontsize=8, bbox=box_6)

# Connector arrows to row 2
ax.annotate('', xy=(3.7, 2.05), xytext=(6.3, 2.7), arrowprops=arrow_props)
ax.annotate('', xy=(8.2, 2.05), xytext=(8.8, 2.7), arrowprops=arrow_props)
ax.annotate('', xy=(7.05, 1.1), xytext=(6.5, 1.1), arrowprops=arrow_props)

plt.title('Figure 1: Complete End-to-End Weakly Supervised & Explainable Ordinal Crop Disease Severity Framework', fontsize=10.5, fontweight='bold', pad=10)
plt.savefig('/home/sahith/Projects/Crop/figures/fig_process_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created fig_process_pipeline.png')

# -------------------------------------------------------------
# Figure 2: Architecture Comparison (fig_backbone_comparison.png)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=300)

models = ['ResNet-18', 'ResNet-34', 'MobileNetV2']
qwk_means = [0.8552, 0.9566, 0.9174]
qwk_stds = [0.0599, 0.0164, 0.0302]
acc_means = [84.95, 89.78, 89.78]
acc_stds = [4.29, 3.04, 1.01]
model_sizes = [44.8, 85.3, 9.2] # MB

x = np.arange(len(models))
width = 0.35

ax1.bar(x, qwk_means, width, yerr=qwk_stds, capsize=5, color=['#1976D2', '#388E3C', '#F57C00'], alpha=0.85, edgecolor='black')
ax1.set_ylabel('Quadratic Weighted Kappa (QWK)', fontsize=9, fontweight='bold')
ax1.set_title('(a) Multi-Seed QWK on Arecanut (N=124)', fontsize=9.5, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8.5, fontweight='bold')
ax1.set_ylim(0.7, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(qwk_means):
    ax1.text(i, v + qwk_stds[i] + 0.012, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

# Accuracy & Model Size on ax2
ax2_twin = ax2.twinx()
b1 = ax2.bar(x - width/2, acc_means, width, yerr=acc_stds, capsize=4, color='#3F51B5', label='Exact Acc (%)', alpha=0.85, edgecolor='black')
b2 = ax2_twin.bar(x + width/2, model_sizes, width, color='#E53935', label='Disk Size (MB)', alpha=0.85, edgecolor='black')

ax2.set_ylabel('Exact Accuracy (%)', fontsize=9, fontweight='bold', color='#1A237E')
ax2_twin.set_ylabel('Checkpoint Footprint (MB)', fontsize=9, fontweight='bold', color='#B71C1C')
ax2.set_title('(b) Accuracy vs. Parameter Footprint', fontsize=9.5, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8.5, fontweight='bold')
ax2.set_ylim(75, 100)
ax2_twin.set_ylim(0, 100)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7.5)

plt.tight_layout()
plt.savefig('/home/sahith/Projects/Crop/figures/fig_backbone_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created fig_backbone_comparison.png')

# -------------------------------------------------------------
# Figure 3: Zero-Shot Transfer & Metric Distortion (fig_zeroshot_transfer.png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=300)

eval_types = ['Paddy (In-Domain)', 'Arecanut (In-Domain)', 'Paddy -> Areca\n(Zero-Shot)', 'Areca -> Paddy\n(Zero-Shot)']
qwk_scores = [0.9390, 0.9177, 0.1396, 0.2051]
exact_accs = [90.98, 90.32, 29.84, 30.52]
within1_accs = [98.81, 97.58, 47.58, 64.83]

x = np.arange(len(eval_types))
w = 0.25

rects1 = ax.bar(x - w, qwk_scores, w, label='QWK (Rank Correlation)', color='#1E88E5', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, [a/100 for a in exact_accs], w, label='Exact Accuracy', color='#43A047', edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + w, [a/100 for a in within1_accs], w, label='Within-1 Accuracy (Distorted)', color='#E53935', edgecolor='black', alpha=0.9)

ax.set_ylabel('Score / Ratio', fontsize=9, fontweight='bold')
ax.set_title('Figure 3: In-Domain vs. Zero-Shot Cross-Crop Evaluation & Metric Distortion', fontsize=9.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(eval_types, fontsize=8, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.annotate('Metric Distortion:\nWithin-1 (64.8%) masks\nbipolar ordinal collapse',
            xy=(3 + w, 0.65), xytext=(2.15, 0.88),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=5),
            fontsize=7.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#FBC02D'))

plt.tight_layout()
plt.savefig('/home/sahith/Projects/Crop/figures/fig_zeroshot_transfer.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created fig_zeroshot_transfer.png')

# -------------------------------------------------------------
# Figure 4: Grad-CAM IoU vs Lesion Area Confound (fig_iou_confound.png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)

classes = [
    ('Dead Heart', 57.9, 0.1369, '#D32F2F'),
    ('Bacterial Leaf Streak', 8.9, 0.0276, '#1976D2'),
    ('Bacterial Panicle Blight', 11.2, 0.0224, '#7B1FA2'),
    ('Bacterial Leaf Blight', 11.9, 0.0294, '#388E3C'),
    ('Brown Spot', 14.5, 0.0381, '#F57C00'),
    ('Blast', 16.2, 0.0410, '#00796B'),
    ('Tungro', 24.8, 0.0652, '#C2185B'),
    ('Normal (Healthy)', 100.0, 0.1273, '#616161')
]

areas = [c[1] for c in classes]
ious = [c[2] for c in classes]

ax.scatter(areas, ious, s=100, c=[c[3] for c in classes], edgecolors='black', zorder=4)

z = np.polyfit(areas, ious, 1)
p = np.poly1d(z)
x_line = np.linspace(5, 100, 100)
ax.plot(x_line, p(x_line), 'r--', lw=1.5, alpha=0.7, label='Linear Confound Trend (R^2 > 0.82)')

for name, a, iou, col in classes:
    offset_y = 0.007 if name != 'Dead Heart' else -0.012
    offset_x = 1.5 if a < 80 else -20.0
    ax.annotate(name, (a, iou), xytext=(a + offset_x, iou + offset_y), fontsize=7, fontweight='bold', color='#212121')

ax.set_xlabel('Mean Ground Truth Lesion/Leaf Area Coverage (%)', fontsize=8.5, fontweight='bold')
ax.set_ylabel('Mean Grad-CAM IoU (Threshold = 0.5)', fontsize=8.5, fontweight='bold')
ax.set_title('Figure 4: Grad-CAM IoU Confound with Lesion Coverage Area', fontsize=9.5, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0, 0.16)

plt.tight_layout()
plt.savefig('/home/sahith/Projects/Crop/figures/fig_iou_confound.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created fig_iou_confound.png')
