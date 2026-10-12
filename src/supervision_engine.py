import sys
sys.path.insert(0, '/home/sahith/Projects/Hackathon')

import os
import numpy as np
from PIL import Image
import torch
import config

from src.segmentation import (
    segment_leaf_adaptive,
    detect_lesions_group_a,
    detect_dead_heart,
    detect_hispa,
    compute_ratio
)
from src.confidence import composite_confidence
from src.severity_grader import ratio_to_grade

class PathologicalSupervisionEngine:
    """
    Autonomous Pathological Supervision Engine.
    
    Acts as a weakly supervised teacher that derives continuous lesion-to-organ
    ratios (rho) using dual-color-space chromatic segmentation, maps them onto
    Horsfall-Barratt clinical ordinal grades (0, 1, 2, 3), and generates binary
    target vectors b in {0, 1}^3 for Consistent Rank Logits (CORN) ordinal loss.
    """
    def __init__(self, confidence_threshold=0.80):
        self.conf_threshold = confidence_threshold

    def grade_to_corn_targets(self, grade: int) -> torch.Tensor:
        """
        Converts discrete grade y in {0, 1, 2, 3} into K-1=3 binary vectors:
        b_j = 1 if y > j else 0, for j in {0, 1, 2}.
        """
        return torch.tensor([
            1.0 if grade > 0 else 0.0,
            1.0 if grade > 1 else 0.0,
            1.0 if grade > 2 else 0.0
        ], dtype=torch.float32)

    def process_image(self, img_input, disease_class: str = "", crop_type: str = "paddy") -> dict:
        """
        Processes a single RGB image and returns complete pathological supervision targets.
        
        Args:
            img_input: File path (str) or PIL Image or np.ndarray (H, W, 3).
            disease_class: Diagnostic class name.
            crop_type: "paddy" or "arecanut".
            
        Returns:
            dict with:
                - grade: int in {0, 1, 2, 3}
                - ratio: float (% infection)
                - corn_targets: torch.Tensor of shape (3,)
                - confidence: float (0.0 to 100.0)
                - method: str
                - pass_filter: bool (True if confidence >= threshold)
        """
        if isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
            img_array = np.array(img)
        elif isinstance(img_input, Image.Image):
            img_array = np.array(img_input.convert('RGB'))
        else:
            img_array = np.array(img_input)

        # 1. Healthy baseline bypass
        if (crop_type == "arecanut" and disease_class in config.ARECANUT_HEALTHY_CLASSES) or \
           (crop_type == "paddy" and disease_class == config.PADDY_NORMAL):
            grade = 0
            ratio = 0.0
            confidence = 100.0
            method = "auto_healthy"
            corn_targets = self.grade_to_corn_targets(grade)
            return {
                "grade": grade,
                "ratio": ratio,
                "corn_targets": corn_targets,
                "confidence": confidence,
                "method": method,
                "pass_filter": True
            }

        # 2. Chromatic organ canopy extraction
        leaf_mask = segment_leaf_adaptive(img_array)
        method = "chromatic_morphology"

        # 3. Pathological symptom extraction by crop & disease
        if crop_type == "paddy":
            if disease_class == config.PADDY_GROUP_B_DEAD_HEART:
                lesion_mask = detect_dead_heart(img_array, leaf_mask)
                method = "dead_heart_fallback"
            elif disease_class == config.PADDY_GROUP_B_HISPA:
                lesion_mask = detect_hispa(img_array, leaf_mask)
                method = "hispa_fallback"
            else:
                lesion_mask = detect_lesions_group_a(img_array, leaf_mask)
        else:
            lesion_mask = detect_lesions_group_a(img_array, leaf_mask)

        # 4. Continuous area morphometry
        ratio = float(compute_ratio(lesion_mask, leaf_mask))
        grade = int(ratio_to_grade(ratio))

        # 5. Calibration confidence filtering
        confidence = float(composite_confidence(leaf_mask, lesion_mask, ratio, img_array.shape))
        pass_filter = (confidence / 100.0) >= self.conf_threshold

        # 6. Binary ordinal supervision vector
        corn_targets = self.grade_to_corn_targets(grade)

        return {
            "grade": grade,
            "ratio": ratio,
            "corn_targets": corn_targets,
            "confidence": confidence,
            "method": method,
            "pass_filter": pass_filter
        }

if __name__ == "__main__":
    engine = PathologicalSupervisionEngine()
    print("PathologicalSupervisionEngine initialized successfully.")
