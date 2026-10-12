import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
from src.model import CropSeverityModel
from src.dataset import get_transforms
from src.gradcam import GradCAM

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("GENERATING PUBLICATION GRAD-CAM ATTENTION MAPS (CORN VS SOFTMAX)")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Paddy ordinal & baseline models
m_corn = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=True, pretrained=False)
ckpt_c = torch.load('outputs/models/paddy_ordinal.pth', map_location='cpu')
m_corn.load_state_dict(ckpt_c if 'model_state_dict' not in ckpt_c else ckpt_c['model_state_dict'])
m_corn.to(device).eval()

m_soft = CropSeverityModel(backbone='resnet18', num_classes=4, ordinal=False, pretrained=False)
ckpt_s = torch.load('outputs/models/paddy_baseline.pth', map_location='cpu')
m_soft.load_state_dict(ckpt_s if 'model_state_dict' not in ckpt_s else ckpt_s['model_state_dict'])
m_soft.to(device).eval()

gradcam_corn = GradCAM(m_corn, target_layer_name='layer4')
gradcam_soft = GradCAM(m_soft, target_layer_name='layer4')

# Load predictions & image paths
df_paddy = pd.read_csv('outputs/paddy_splits.csv')
fn_to_path = dict(zip(df_paddy['filename'], df_paddy['image_path']))

df_preds = pd.read_csv('outputs/final_v2/paddy_corn_predictions_all_3_decodes.csv')
df_preds['image_path'] = df_preds['filename'].map(fn_to_path)

classes_7 = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'downy_mildew', 'tungro'
]

tf_7 = df_preds[(~df_preds['has_twin_le8']) & (df_preds['disease_class'].isin(classes_7))].copy()
sample_rows = []
for c in classes_7:
    sub = tf_7[tf_7['disease_class'] == c]
    if len(sub) > 0:
        sample_rows.append(sub.iloc[0])

tf_transform = get_transforms('test')

fig = plt.figure(figsize=(15, 22))
gs = GridSpec(7, 3, figure=fig, hspace=0.35, wspace=0.20)

for i, row in enumerate(sample_rows):
    img_bgr = cv2.imread(row['image_path'])
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(img_rgb)
    tensor_img = tf_transform(pil_img).unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    cam_c = gradcam_corn.generate(tensor_img)
    cam_s = gradcam_soft.generate(tensor_img)
    
    h, w, _ = img_rgb.shape
    cam_c_resized = cv2.resize(cam_c, (w, h))
    cam_s_resized = cv2.resize(cam_s, (w, h))
    
    heatmap_c = cv2.applyColorMap(np.uint8(255 * cam_c_resized), cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
    vis_c = cv2.addWeighted(img_rgb, 0.6, heatmap_c, 0.4, 0)
    
    heatmap_s = cv2.applyColorMap(np.uint8(255 * cam_s_resized), cv2.COLORMAP_JET)
    heatmap_s = cv2.cvtColor(heatmap_s, cv2.COLOR_BGR2RGB)
    vis_s = cv2.addWeighted(img_rgb, 0.6, heatmap_s, 0.4, 0)
    
    gt_g = row['severity_grade']
    p_c = row['pred_rule_ii_chained']
    p_s = row['pred_softmax']
    cls = row['disease_class']
    fname = row['filename']
    
    # Col 1: Raw RGB
    ax1 = fig.add_subplot(gs[i, 0])
    ax1.imshow(img_rgb)
    ax1.set_title(f"Raw RGB Image\n{cls} ({fname})\nGT Grade: {gt_g}", fontsize=9, fontweight='bold', pad=3)
    ax1.axis('off')
    
    # Col 2: Softmax Grad-CAM
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.imshow(vis_s)
    ax2.set_title(f"Softmax Attention\nPred Grade: {p_s} ({'Correct' if p_s==gt_g else 'Error'})", fontsize=9, pad=3)
    ax2.axis('off')
    
    # Col 3: CORN Grad-CAM
    ax3 = fig.add_subplot(gs[i, 2])
    ax3.imshow(vis_c)
    ax3.set_title(f"CORN Ordinal Attention\nPred Grade: {p_c} ({'Correct' if p_c==gt_g else 'Error'})", fontsize=9, pad=3, color='#1B5E20')
    ax3.axis('off')

fig.suptitle("Publication Grad-CAM Visual Attention Comparison (Twin-Free Paddy Test Set)\n[CORN Ordinal Attention vs Standard Softmax Baseline]", fontsize=13, fontweight='bold', y=0.995)

gradcam_out_path = 'outputs/final_v2/paddy_gradcam_comparison_montage.png'
plt.savefig(gradcam_out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved Grad-CAM publication figure to: {gradcam_out_path}")
