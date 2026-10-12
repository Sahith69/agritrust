import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, roc_auc_score

sys.path.insert(0, '/home/sahith/Projects/Crop')
import config

os.makedirs('outputs/final_v2', exist_ok=True)

print("=" * 80)
print("STAGE 2f - SECTION C: ARECANUT SOURCE CONFOUND & METADATA CLASSIFIER")
print("=" * 80)

# Load Arecanut dataset with robust clusters & sources
df_areca = pd.read_csv('outputs/final_v2/arecanut_clean_v2_robust_clusters.csv')

# C.1 Grade Distribution by Source
print("\n--- C.1 GRADE DISTRIBUTION BY SOURCE (ORIGINAL 871 vs KAGGLE 1,896) ---")
sources = sorted(df_areca['source'].unique())
classes = sorted(df_areca['disease_class'].unique())

source_dist_summary = {}

header_c1 = f"{'Disease Class':<22} | {'Source':<15} | {'G0':<6} | {'G1':<6} | {'G2':<6} | {'G3':<6} | {'Total':<6}"
sep_c1 = "-" * len(header_c1)
print(sep_c1)
print(header_c1)
print(sep_c1)

for cls in classes:
    c_df = df_areca[df_areca['disease_class'] == cls]
    source_dist_summary[cls] = {}
    for src in sources:
        s_df = c_df[c_df['source'] == src]
        g_counts = s_df['severity_grade'].value_counts().to_dict()
        g0 = g_counts.get(0, 0)
        g1 = g_counts.get(1, 0)
        g2 = g_counts.get(2, 0)
        g3 = g_counts.get(3, 0)
        tot = len(s_df)
        source_dist_summary[cls][src] = {'G0': g0, 'G1': g1, 'G2': g2, 'G3': g3, 'total': tot}
        if tot > 0:
            print(f"{cls:<22} | {src:<15} | {g0:<6d} | {g1:<6d} | {g2:<6d} | {g3:<6d} | {tot:<6d}")

print(sep_c1)

# Map image_path to source
path_to_source = dict(zip(df_areca['image_path'], df_areca['source']))

# Extract Metadata Features for all images in df_areca
print("\nExtracting metadata features for all 2,661 Arecanut images...")
meta_features = []

for idx, row in df_areca.iterrows():
    p = row['image_path']
    file_size = os.path.getsize(p) if os.path.exists(p) else 0
    try:
        with Image.open(p) as im:
            w, h = im.size
            aspect = w / float(h)
            
            q_lum_sum = 0
            q_chrom_sum = 0
            if hasattr(im, 'quantization') and im.quantization:
                q_tables = list(im.quantization.values())
                if len(q_tables) > 0:
                    q_lum_sum = sum(q_tables[0])
                if len(q_tables) > 1:
                    q_chrom_sum = sum(q_tables[1])
    except Exception:
        w, h, aspect, q_lum_sum, q_chrom_sum = 0, 0, 1.0, 0, 0

    meta_features.append({
        'width': w,
        'height': h,
        'aspect_ratio': aspect,
        'file_size_bytes': file_size,
        'q_lum_sum': q_lum_sum,
        'q_chrom_sum': q_chrom_sum
    })

meta_df = pd.DataFrame(meta_features)
X_all = meta_df.values

# C.2 Evaluate on 502 Test and Clean-146 Test
split_502_path = 'outputs/arecanut_splits.csv'
df_502 = pd.read_csv(split_502_path)
df_502['source'] = df_502['image_path'].map(path_to_source).fillna('original_871')

split_146_path = 'outputs/improvements/arecanut_splits_clean_871.csv'
df_146 = pd.read_csv(split_146_path)
df_146['source'] = df_146['image_path'].map(path_to_source).fillna('original_871')

# 1. Evaluate Mode Grade per (Class x Source)
print("\n--- C.2 MODE GRADE PER (CLASS x SOURCE) BASELINE ---")

df_502_train = df_502[df_502['split'] == 'train']
mode_class_source_502 = df_502_train.groupby(['disease_class', 'source'])['severity_grade'].agg(lambda x: s.mode()[0] if len(s:=x.dropna())>0 else 3).to_dict()
mode_class_502 = df_502_train.groupby('disease_class')['severity_grade'].agg(lambda x: s.mode()[0] if len(s:=x.dropna())>0 else 3).to_dict()

df_502_test = df_502[df_502['split'] == 'test'].copy()
pred_mode_cs_502 = []
for idx, r in df_502_test.iterrows():
    key = (r['disease_class'], r['source'])
    pred_mode_cs_502.append(mode_class_source_502.get(key, mode_class_502.get(r['disease_class'], 3)))

qwk_mode_cs_502 = cohen_kappa_score(df_502_test['severity_grade'], pred_mode_cs_502, weights='quadratic')
acc_mode_cs_502 = accuracy_score(df_502_test['severity_grade'], pred_mode_cs_502) * 100.0

print(f"502 Test (N={len(df_502_test)}): Mode (Class x Source) QWK = {qwk_mode_cs_502:.4f}, Accuracy = {acc_mode_cs_502:.2f}%")

df_146_train = df_146[df_146['split'] == 'train']
mode_class_source_146 = df_146_train.groupby(['disease_class', 'source'])['severity_grade'].agg(lambda x: s.mode()[0] if len(s:=x.dropna())>0 else 3).to_dict()
mode_class_146 = df_146_train.groupby('disease_class')['severity_grade'].agg(lambda x: s.mode()[0] if len(s:=x.dropna())>0 else 3).to_dict()

df_146_test = df_146[df_146['split'] == 'test'].copy()
pred_mode_cs_146 = []
for idx, r in df_146_test.iterrows():
    key = (r['disease_class'], r['source'])
    pred_mode_cs_146.append(mode_class_source_146.get(key, mode_class_146.get(r['disease_class'], 3)))

qwk_mode_cs_146 = cohen_kappa_score(df_146_test['severity_grade'], pred_mode_cs_146, weights='quadratic')
acc_mode_cs_146 = accuracy_score(df_146_test['severity_grade'], pred_mode_cs_146) * 100.0

print(f"146 Test (N={len(df_146_test)}): Mode (Class x Source) QWK = {qwk_mode_cs_146:.4f}, Accuracy = {acc_mode_cs_146:.2f}%")

# 2. Metadata-only Classifier
print("\n--- METADATA-ONLY CLASSIFIER EVALUATION ---")
fn_to_meta_idx = {row['image_path']: idx for idx, row in df_areca.iterrows()}

def get_meta_features(df_subset):
    feats = []
    for p in df_subset['image_path']:
        if p in fn_to_meta_idx:
            feats.append(X_all[fn_to_meta_idx[p]])
        else:
            file_size = os.path.getsize(p) if os.path.exists(p) else 0
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    aspect = w / float(h)
                    q_lum_sum = sum(list(im.quantization.values())[0]) if hasattr(im, 'quantization') and im.quantization else 0
                    q_chrom_sum = sum(list(im.quantization.values())[1]) if hasattr(im, 'quantization') and len(im.quantization) > 1 else 0
            except Exception:
                w, h, aspect, q_lum_sum, q_chrom_sum = 0, 0, 1.0, 0, 0
            feats.append([w, h, aspect, file_size, q_lum_sum, q_chrom_sum])
    return np.array(feats)

X_502_train = get_meta_features(df_502_train)
y_502_train = df_502_train['severity_grade'].values

X_502_test = get_meta_features(df_502_test)
y_502_test = df_502_test['severity_grade'].values

rf_meta_502 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_meta_502.fit(X_502_train, y_502_train)
pred_meta_502 = rf_meta_502.predict(X_502_test)

qwk_meta_502 = cohen_kappa_score(y_502_test, pred_meta_502, weights='quadratic')
acc_meta_502 = accuracy_score(y_502_test, pred_meta_502) * 100.0

X_146_train = get_meta_features(df_146_train)
y_146_train = df_146_train['severity_grade'].values

X_146_test = get_meta_features(df_146_test)
y_146_test = df_146_test['severity_grade'].values

rf_meta_146 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_meta_146.fit(X_146_train, y_146_train)
pred_meta_146 = rf_meta_146.predict(X_146_test)

qwk_meta_146 = cohen_kappa_score(y_146_test, pred_meta_146, weights='quadratic')
acc_meta_146 = accuracy_score(y_146_test, pred_meta_146) * 100.0

print(f"502 Test: Metadata Classifier QWK = {qwk_meta_502:.4f}, Accuracy = {acc_meta_502:.2f}%")
print(f"146 Test: Metadata Classifier QWK = {qwk_meta_146:.4f}, Accuracy = {acc_meta_146:.2f}%")

# 3. Predict Data Source using Metadata Features
print("\n--- PREDICTING DATA SOURCE (ORIGINAL vs KAGGLE) USING METADATA ---")
y_source_502_train = (df_502_train['source'] == 'original_871').astype(int).values
y_source_502_test = (df_502_test['source'] == 'original_871').astype(int).values

rf_source = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_source.fit(X_502_train, y_source_502_train)
pred_source_prob = rf_source.predict_proba(X_502_test)[:, 1]
pred_source_bin = (pred_source_prob >= 0.5).astype(int)

acc_source = accuracy_score(y_source_502_test, pred_source_bin) * 100.0
auc_source = roc_auc_score(y_source_502_test, pred_source_prob)

print(f"Source Prediction Accuracy: {acc_source:.2f}% | ROC-AUC: {auc_source:.4f}")

stage2f_c_output = {
    'source_grade_distribution': source_dist_summary,
    '502_test_eval': {
        'mode_class_x_source': {'qwk': float(qwk_mode_cs_502), 'accuracy': float(acc_mode_cs_502)},
        'metadata_classifier': {'qwk': float(qwk_meta_502), 'accuracy': float(acc_meta_502)},
        'source_prediction_accuracy': float(acc_source),
        'source_prediction_auc': float(auc_source)
    },
    '146_test_eval': {
        'mode_class_x_source': {'qwk': float(qwk_mode_cs_146), 'accuracy': float(acc_mode_cs_146)},
        'metadata_classifier': {'qwk': float(qwk_meta_146), 'accuracy': float(acc_meta_146)}
    }
}

with open('outputs/final_v2/stage2f_section_c.json', 'w') as f:
    json.dump(stage2f_c_output, f, indent=2)

print("\nSaved Stage 2f Section C results to outputs/final_v2/stage2f_section_c.json")
