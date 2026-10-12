import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import numpy as np

def otsu_threshold(gray_img, mask=None):
    if mask is None:
        mask = np.ones_like(gray_img, dtype=bool)
    pixels = gray_img[mask]
    if pixels.size == 0:
        return 0
    
    hist, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    mean1 = np.cumsum(hist * bin_mids) / (weight1 + 1e-8)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / (weight2[::-1] + 1e-8))[::-1]
    
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    if len(variance12) == 0:
        return 0
    
    idx = np.argmax(variance12)
    threshold = bin_mids[idx]
    return threshold

def get_disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask

def binary_erosion(mask, radius):
    if radius == 0:
        return mask
    disk = get_disk(radius)
    offsets = np.argwhere(disk) - radius
    padded = np.pad(mask, radius, mode='constant', constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy, dx in offsets:
        eroded &= padded[radius+dy : radius+dy+mask.shape[0], radius+dx : radius+dx+mask.shape[1]]
    return eroded

def filter_small_components(mask, min_size):
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    
    ys, xs = np.nonzero(mask)
    
    for start_y, start_x in zip(ys, xs):
        if not visited[start_y, start_x]:
            queue = [(start_y, start_x)]
            visited[start_y, start_x] = True
            component = []
            
            head = 0
            while head < len(queue):
                y, x = queue[head]
                head += 1
                component.append((y, x))
                
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            
            if len(component) >= min_size:
                for cy, cx in component:
                    output[cy, cx] = True
    return output

def segment_leaf_adaptive(img_array):
    img_array = img_array.astype(float)
    R = img_array[:,:,0]
    G = img_array[:,:,1]
    B = img_array[:,:,2]
    ExG = 2 * G - R - B
    
    ExG_min, ExG_max = ExG.min(), ExG.max()
    if ExG_max > ExG_min:
        ExG_norm = ((ExG - ExG_min) / (ExG_max - ExG_min) * 255).astype(np.uint8)
    else:
        ExG_norm = np.zeros_like(ExG, dtype=np.uint8)
    
    thresh = otsu_threshold(ExG_norm)
    leaf_mask = ExG_norm > thresh
    return leaf_mask

def detect_lesions_group_a(img_array, leaf_mask):
    eroded_mask = binary_erosion(leaf_mask, config.LEAF_MASK_EROSION_PX)
    
    img_float = img_array.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    
    thresh = otsu_threshold(gray, mask=eroded_mask)
    lesion_mask = (gray < thresh) & eroded_mask
    
    lesion_mask = filter_small_components(lesion_mask, config.MIN_LESION_BLOB_PX)
    return lesion_mask

def detect_dead_heart(img_array, leaf_mask):
    img_float = img_array.astype(float) + 1e-5
    r_g_ratio = img_float[:,:,0] / img_float[:,:,1]
    
    r_g_ratio = np.clip(r_g_ratio, 0, 3)
    r_g_norm = (r_g_ratio / 3.0 * 255).astype(np.uint8)
    
    thresh = otsu_threshold(r_g_norm, mask=leaf_mask)
    discoloration_mask = (r_g_norm > thresh) & leaf_mask
    
    discoloration_mask = filter_small_components(discoloration_mask, config.MIN_LESION_BLOB_PX)
    return discoloration_mask

def detect_hispa(img_array, leaf_mask):
    img_float = img_array.astype(float)
    gray = (0.2989 * img_float[:,:,0] + 0.5870 * img_float[:,:,1] + 0.1140 * img_float[:,:,2]).astype(np.uint8)
    
    padded = np.pad(gray, 1, mode='edge')
    var_map = np.zeros_like(gray, dtype=float)
    for dy in range(3):
        for dx in range(3):
            var_map += (padded[dy:dy+gray.shape[0], dx:dx+gray.shape[1]] - gray)**2
    var_map = var_map / 9.0
    
    var_norm = np.clip(var_map, 0, 255).astype(np.uint8)
    thresh = otsu_threshold(var_norm, mask=leaf_mask)
    
    hispa_mask = (var_norm > thresh) & leaf_mask
    hispa_mask = filter_small_components(hispa_mask, config.MIN_LESION_BLOB_PX)
    return hispa_mask

def compute_ratio(lesion_mask, leaf_mask):
    leaf_px = np.sum(leaf_mask)
    if leaf_px == 0:
        return 0.0
    lesion_px = np.sum(lesion_mask & leaf_mask)
    return float(lesion_px) / float(leaf_px)
