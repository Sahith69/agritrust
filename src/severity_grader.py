import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
from PIL import Image
import numpy as np

from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    detect_dead_heart,
    detect_hispa,
    compute_ratio
)
from src.confidence import composite_confidence

def ratio_to_grade(ratio):
    if ratio <= config.SEVERITY_THRESHOLDS['grade_0_max']:
        return 0
    elif ratio <= config.SEVERITY_THRESHOLDS['grade_1_max']:
        return 1
    elif ratio <= config.SEVERITY_THRESHOLDS['grade_2_max']:
        return 2
    else:
        return 3

def grade_image(img_path, disease_class, crop_type):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    if disease_class in config.ARECANUT_HEALTHY_CLASSES or disease_class == config.PADDY_NORMAL:
        return {
            "grade": 0,
            "ratio": 0.0,
            "confidence": 100.0,
            "method": "auto_healthy"
        }
    
    leaf_mask = segment_leaf_adaptive(img_array)
    method = "segmentation"
        
    if crop_type == "paddy":
        if disease_class == config.PADDY_GROUP_B_DEAD_HEART:
            lesion_mask = detect_dead_heart(img_array, leaf_mask)
            method = "dead_heart"
        elif disease_class == config.PADDY_GROUP_B_HISPA:
            lesion_mask = detect_hispa(img_array, leaf_mask)
            method = "hispa"
        else:
            lesion_mask = detect_lesions_group_a(img_array, leaf_mask)
            method = "group_a"
    else:
        lesion_mask = detect_lesions_group_a(img_array, leaf_mask)
        method = "group_a"
        
    ratio = compute_ratio(lesion_mask, leaf_mask)
    grade = ratio_to_grade(ratio)
    confidence = composite_confidence(leaf_mask, lesion_mask, ratio, img_array.shape)
    
    return {
        "grade": grade,
        "ratio": ratio,
        "confidence": confidence,
        "method": method
    }
