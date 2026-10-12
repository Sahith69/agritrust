import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
import numpy as np

def leaf_plausibility_score(leaf_mask, img_shape):
    total_px = img_shape[0] * img_shape[1]
    leaf_px = np.sum(leaf_mask)
    leaf_pct = leaf_px / total_px
    
    score = 100.0
    if leaf_pct < config.LEAF_AREA_MIN_PCT:
        score -= (config.LEAF_AREA_MIN_PCT - leaf_pct) / config.LEAF_AREA_MIN_PCT * 100
    elif leaf_pct > config.LEAF_AREA_MAX_PCT:
        score -= (leaf_pct - config.LEAF_AREA_MAX_PCT) / (1.0 - config.LEAF_AREA_MAX_PCT) * 100
    
    return max(0.0, min(100.0, score))

def boundary_distance_score(ratio):
    """
    Evaluates confidence based on proximity to severity boundaries (0.05 and 0.15).
    Only applies a linear penalty when the ratio falls within a narrow window of ±0.5%
    around a boundary:
      - 0.05 boundary: [0.045, 0.055]
      - 0.15 boundary: [0.145, 0.155]
    Outside these ±0.5% windows, returns 100.0 (no boundary penalty).
    """
    window = 0.005  # ±0.5%
    boundaries = [
        config.SEVERITY_THRESHOLDS['grade_1_max'],  # 0.05
        config.SEVERITY_THRESHOLDS['grade_2_max']   # 0.15
    ]
    
    min_score = 100.0
    for b in boundaries:
        dist = abs(ratio - b)
        if dist < window:
            b_score = (dist / window) * 100.0
            if b_score < min_score:
                min_score = b_score
                
    return max(0.0, min(100.0, min_score))

def fragmentation_score(lesion_mask):
    ys, xs = np.nonzero(lesion_mask)
    lesion_px = len(ys)
    if lesion_px == 0:
        return 100.0
    
    padded = np.pad(lesion_mask, 1)
    boundary_px = np.sum(lesion_mask & ~padded[1:-1, 2:]) + \
                  np.sum(lesion_mask & ~padded[1:-1, :-2]) + \
                  np.sum(lesion_mask & ~padded[2:, 1:-1]) + \
                  np.sum(lesion_mask & ~padded[:-2, 1:-1])
                  
    frag_idx = boundary_px / lesion_px
    score = 100.0 - (frag_idx * 50)
    return max(0.0, min(100.0, score))

def composite_confidence(leaf_mask, lesion_mask, ratio, img_shape):
    s1 = leaf_plausibility_score(leaf_mask, img_shape)
    s2 = boundary_distance_score(ratio)
    s3 = fragmentation_score(lesion_mask)
    
    score = 0.4 * s1 + 0.4 * s2 + 0.2 * s3
    return max(0.0, min(100.0, score))
