import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import config

import logging
import streamlit as st
import os, glob, json
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade
from src.segmentation import segment_leaf_adaptive, detect_lesions_group_a, compute_ratio
from src.severity_grader import ratio_to_grade
from src.multimodal_fusion import evaluate_environmental_risk, MultimodalCropFusionNet
from ood_guard import check_ood
from weather_live import DISTRICT_COORDINATES, fetch_forecast, compute_48h_disease_risk

from ui_theme import (
    APP_NAME, THEME_CONFIG, TRUST_AND_AUDIT, CSS_STYLES,
    UI_STRINGS, get_ui_string, format_local_date, build_chat_response, render_sidebar
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — Multimodal Crop Disease & Severity Engine",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─── CSS STYLING (SINGLE BLOCK INJECTION FROM UI_THEME) ─────────────────────
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# ─── DISPLAY NAMES & FACTS ───────────────────────────────────────────────────
PADDY_DISPLAY = {
    "bacterial_leaf_blight":   "Bacterial Leaf Blight",
    "bacterial_leaf_streak":   "Bacterial Leaf Streak",
    "bacterial_panicle_blight":"Bacterial Panicle Blight",
    "blast":                   "Rice Blast",
    "brown_spot":              "Brown Spot",
    "dead_heart":              "Dead Heart",
    "downy_mildew":            "Downy Mildew",
    "hispa":                   "Hispa",
    "normal":                  "Healthy (No Disease)",
    "tungro":                  "Tungro Virus",
}

ARECA_DISPLAY = {
    "bud_borer":           "Bud Borer",
    "healthy_foot":        "Healthy (Foot)",
    "healthy_leaf":        "Healthy (Leaf)",
    "healthy_nut":         "Healthy (Nut)",
    "healthy_trunk":       "Healthy (Trunk)",
    "mahali_koleroga":     "Mahali / Koleroga (Rot Disease)",
    "stem_cracking":       "Stem Cracking",
    "stem_bleeding":       "Stem Bleeding",
    "yellow_leaf_disease": "Yellow Leaf Disease",
}

APPLE_DISPLAY = {
    "apple_scab":        "Apple Scab",
    "black_rot":         "Black Rot (Frog-Eye Leaf Spot)",
    "cedar_apple_rust":  "Cedar Apple Rust",
    "healthy":           "Healthy (No Disease)",
}

DISEASE_FACTS = {
    "Rice Blast":               "Caused by *Magnaporthe oryzae*. Creates diamond-shaped gray lesions with brown borders.",
    "Bacterial Leaf Blight":    "Caused by *Xanthomonas oryzae*. Water-soaked margins turn yellow-white; high humidity accelerates spread.",
    "Brown Spot":               "Caused by *Helminthosporium oryzae*. Circular brown lesions with yellow halos; linked to nutrient deficiency.",
    "Mahali / Koleroga (Rot Disease)": "Caused by *Phytophthora* sp. Nut rot; fruits fall prematurely with dark ooze.",
    "Yellow Leaf Disease":      "Phytoplasma-induced yellowing of arecanut fronds; spread by leafhopper vectors.",
    "Apple Scab":               "Caused by *Venturia inaequalis*. Produces olive-green to velvet-black spots on apple foliage.",
    "Black Rot (Frog-Eye Leaf Spot)": "Caused by *Botryosphaeria obtusa*. Creates purple-black spots expanding into circular frog-eye lesions.",
    "Cedar Apple Rust":         "Caused by *Gymnosporangium juniperi-virginianae*. Produces bright orange-yellow spots on upper leaf surfaces.",
    "Healthy (No Disease)":     "No visible signs of disease. Leaf tissue appears uniform green and undamaged."
}

# ─── PESTICIDE & REMEDIATION DATABASE (EXPLICIT ENTRIES FOR ALL CLASSES) ──────
TREATMENT_DATABASE = {
    "Paddy": {
        "Rice Blast": {
            0: ("Healthy Crop", "No treatment needed. Maintain optimal water depth and balanced N-P-K fertilizer."),
            1: ("Mild Infection", "Apply bio-fungicide Trichoderma viride or neem-based formulation; follow product label and local KVK advice."),
            2: ("Moderate Infection", "Apply systemic fungicide Tricyclazole 75 WP; follow product label and local KVK advice."), # VERIFY WITH CIB&RC: Tricyclazole
            3: ("Severe Infection!", "Apply Isoprothiolane 40 EC and drain excess field water; follow product label and local KVK advice.") # VERIFY WITH CIB&RC: Isoprothiolane
        },
        "Bacterial Leaf Blight": {
            0: ("Healthy Crop", "No treatment needed. Maintain field sanitation and balanced nitrogen application."),
            1: ("Mild Blight", "Spray neem seed kernel extract. Avoid excess nitrogen top-dressing; follow product label and local KVK advice."),
            2: ("Moderate Blight", "Apply Copper Oxychloride + Streptocycline; follow product label and local KVK advice."), # VERIFY WITH CIB&RC: Copper Oxychloride + Streptocycline
            3: ("Severe Blight!", "Drain field water, withhold nitrogen, apply copper-based bactericide; follow product label and local KVK advice.")
        },
        "Bacterial Leaf Streak": {
            0: ("Healthy Crop", "No treatment needed. Ensure field drainage."),
            1: ("Mild Infection", "Promote balanced fertilizer application and avoid excess urea; follow local KVK advice."),
            2: ("Moderate Infection", "Apply copper hydroxide spray; follow product label and local KVK advice."), # VERIFY WITH CIB&RC: Copper Hydroxide
            3: ("Severe Infection!", "Drain standing water, avoid field operations when leaves are wet; follow product label and local KVK advice.")
        },
        "Bacterial Panicle Blight": {
            0: ("Healthy Crop", "No treatment needed. Use clean certified seed."),
            1: ("Mild Blight", "Avoid high nitrogen application at boot stage; follow local KVK advice."),
            2: ("Moderate Blight", "Apply preventive copper-based bactericide at early heading; follow product label and local KVK advice."),
            3: ("Severe Blight!", "Quarantine field patch and rogue severely infected panicles; follow product label and local KVK advice.")
        },
        "Brown Spot": {
            0: ("Healthy Crop", "No treatment needed. Correct soil nutrient balance."),
            1: ("Mild Infection", "Apply compost and correct potassium deficiency; follow local KVK advice."),
            2: ("Moderate Infection", "Apply Mancozeb protective spray; follow product label and local KVK advice."), # VERIFY WITH CIB&RC: Mancozeb
            3: ("Severe Infection!", "Apply compost and correct soil deficiency; consult your KVK for a registered product; follow product label and local KVK advice.")
        },
        "Dead Heart": {
            0: ("Healthy Crop", "No treatment needed. Conserve natural egg parasitoids."),
            1: ("Mild Borer Damage", "Clip seedling tips before transplanting; install yellow stem borer pheromone traps; follow local KVK advice."),
            2: ("Moderate Borer Damage", "Release Trichogramma japonicum egg parasitoids; install pheromone traps; follow local KVK advice."),
            3: ("Severe Borer Infestation!", "Install pheromone traps; consult your KVK for a registered product; follow product label and local KVK advice.")
        },
        "Downy Mildew": {
            0: ("Healthy Crop", "No treatment needed. Ensure good water drainage."),
            1: ("Mild Mildew", "Rogue out infected volunteer plants and weed hosts; follow local KVK advice."),
            2: ("Moderate Mildew", "Apply Metalaxyl-M foliar spray; follow product label and local KVK advice."), # VERIFY WITH CIB&RC: Metalaxyl-M
            3: ("Severe Mildew!", "Apply registered systemic oomycete fungicide; follow product label and local KVK advice.")
        },
        "Hispa": {
            0: ("Healthy Crop", "No treatment needed. Monitor leaf margins."),
            1: ("Mild Leaf Scrape", "Manually clip infected leaf tips containing grubs; use sweep nets; follow local KVK advice."),
            2: ("Moderate Infestation", "Manually clip affected leaf tips containing grubs; consult your KVK for a registered product; follow product label and local KVK advice."),
            3: ("Severe Outbreak!", "Clip affected leaves; consult your KVK for a registered product; follow product label and local KVK advice.")
        },
        "Healthy (No Disease)": {
            0: ("Healthy Crop", "No treatment needed."),
            1: ("Healthy Crop", "No treatment needed."),
            2: ("Healthy Crop", "No treatment needed."),
            3: ("Healthy Crop", "No treatment needed.")
        },
        "Tungro Virus": {
            0: ("Viral Infection Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."),
            1: ("Viral Infection Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."),
            2: ("Viral Infection Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."),
            3: ("Viral Infection Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK.")
        }
    },
    "Apple": {
        "Apple Scab": {
            "disease_advice": (
                "Apple Scab Management",
                "Rake and compost fallen leaves to reduce overwintering ascospore inoculum. Maintain open canopy pruning for air circulation and rapid leaf drying. Apply protective fungicides such as Captan or Mancozeb, or systemic Difenoconazole; follow product label and local KVK advice." # VERIFY WITH CIB&RC: Captan, Mancozeb, Difenoconazole
            ),
            0: ("Healthy Foliage", "No treatment needed."),
            1: ("Apple Scab Management", "Rake and compost fallen leaves to reduce overwintering ascospore inoculum. Maintain open canopy pruning for air circulation and rapid leaf drying. Apply protective fungicides such as Captan or Mancozeb, or systemic Difenoconazole; follow product label and local KVK advice."),
            2: ("Apple Scab Management", "Rake and compost fallen leaves to reduce overwintering ascospore inoculum. Maintain open canopy pruning for air circulation and rapid leaf drying. Apply protective fungicides such as Captan or Mancozeb, or systemic Difenoconazole; follow product label and local KVK advice."),
            3: ("Apple Scab Management", "Rake and compost fallen leaves to reduce overwintering ascospore inoculum. Maintain open canopy pruning for air circulation and rapid leaf drying. Apply protective fungicides such as Captan or Mancozeb, or systemic Difenoconazole; follow product label and local KVK advice.")
        },
        "Black Rot (Frog-Eye Leaf Spot)": {
            "disease_advice": (
                "Black Rot Management",
                "Prune out dead wood and cankers during winter dormancy. Remove mummified fruit from trees and orchard floor to eliminate pycnidial sources. Apply protective sprays of Captan or Thiophanate-Methyl; follow product label and local KVK advice." # VERIFY WITH CIB&RC: Thiophanate-Methyl, Captan
            ),
            0: ("Healthy Foliage", "No treatment needed."),
            1: ("Black Rot Management", "Prune out dead wood and cankers during winter dormancy. Remove mummified fruit from trees and orchard floor to eliminate pycnidial sources. Apply protective sprays of Captan or Thiophanate-Methyl; follow product label and local KVK advice."),
            2: ("Black Rot Management", "Prune out dead wood and cankers during winter dormancy. Remove mummified fruit from trees and orchard floor to eliminate pycnidial sources. Apply protective sprays of Captan or Thiophanate-Methyl; follow product label and local KVK advice."),
            3: ("Black Rot Management", "Prune out dead wood and cankers during winter dormancy. Remove mummified fruit from trees and orchard floor to eliminate pycnidial sources. Apply protective sprays of Captan or Thiophanate-Methyl; follow product label and local KVK advice.")
        },
        "Cedar Apple Rust": {
            "disease_advice": (
                "Cedar Apple Rust Management",
                "Remove eastern red cedar or juniper galls if present nearby. Plant rust-resistant cultivars. Apply protective fungicide sprays such as Myclobutanil or Mancozeb during pink through petal-fall stages; follow product label and local KVK advice." # VERIFY WITH CIB&RC: Myclobutanil, Mancozeb
            ),
            0: ("Healthy Foliage", "No treatment needed."),
            1: ("Cedar Apple Rust Management", "Remove eastern red cedar or juniper galls if present nearby. Plant rust-resistant cultivars. Apply protective fungicide sprays such as Myclobutanil or Mancozeb during pink through petal-fall stages; follow product label and local KVK advice."),
            2: ("Cedar Apple Rust Management", "Remove eastern red cedar or juniper galls if present nearby. Plant rust-resistant cultivars. Apply protective fungicide sprays such as Myclobutanil or Mancozeb during pink through petal-fall stages; follow product label and local KVK advice."),
            3: ("Cedar Apple Rust Management", "Remove eastern red cedar or juniper galls if present nearby. Plant rust-resistant cultivars. Apply protective fungicide sprays such as Myclobutanil or Mancozeb during pink through petal-fall stages; follow product label and local KVK advice.")
        },
        "Healthy (No Disease)": {
            "disease_advice": ("Healthy Foliage", "No treatment needed."),
            0: ("Healthy Foliage", "No treatment needed."),
            1: ("Healthy Foliage", "No treatment needed."),
            2: ("Healthy Foliage", "No treatment needed."),
            3: ("Healthy Foliage", "No treatment needed.")
        }
    },
    "Arecanut": {
        "Bud Borer": {
            0: ("Healthy Palm", "No treatment needed. Inspect palm crowns periodically."),
            1: ("Mild Borer Damage", "Clear leaf axils of organic debris; hook out larvae; apply light traps; follow local KVK advice."),
            2: ("Moderate Damage", "Apply neem oil formulation into the crown; consult your KVK for a registered product; follow product label and local KVK advice."),
            3: ("Severe Infestation!", "Clear crown debris; consult your KVK for a registered systemic product; follow product label and local KVK advice.")
        },
        "Healthy (Foot)": {
            0: ("Healthy Palm", "No treatment needed."), 1: ("Healthy Palm", "No treatment needed."),
            2: ("Healthy Palm", "No treatment needed."), 3: ("Healthy Palm", "No treatment needed.")
        },
        "Healthy (Leaf)": {
            0: ("Healthy Palm", "No treatment needed."), 1: ("Healthy Palm", "No treatment needed."),
            2: ("Healthy Palm", "No treatment needed."), 3: ("Healthy Palm", "No treatment needed.")
        },
        "Healthy (Nut)": {
            0: ("Healthy Palm", "No treatment needed."), 1: ("Healthy Palm", "No treatment needed."),
            2: ("Healthy Palm", "No treatment needed."), 3: ("Healthy Palm", "No treatment needed.")
        },
        "Healthy (Trunk)": {
            0: ("Healthy Palm", "No treatment needed."), 1: ("Healthy Palm", "No treatment needed."),
            2: ("Healthy Palm", "No treatment needed."), 3: ("Healthy Palm", "No treatment needed.")
        },
        "Mahali / Koleroga (Rot Disease)": {
            0: ("Healthy Palms", "No treatment needed. Clear drainage channels before monsoon onset."),
            1: ("Mild Rot", "Tie polythene covers over developing bunches; apply preventive 1% Bordeaux Mixture before monsoon; follow product label and local KVK advice."), # VERIFY WITH CPCRI: Bordeaux Mixture
            2: ("Moderate Rot", "Spray 1% Bordeaux Mixture with adhesive sticker on all nut bunches; follow product label and local KVK advice."),
            3: ("Severe Rot!", "Apply Metalaxyl + Mancozeb formulation; collect and destroy fallen rotting nuts; follow product label and local KVK advice.") # VERIFY WITH CPCRI: Metalaxyl + Mancozeb
        },
        "Stem Bleeding": {
            0: ("Healthy Palm", "No treatment needed. Maintain field drainage."),
            1: ("Early Bleeding", "Chisel out bleeding bark patch down to healthy tissue; apply Bordeaux paste; consult your KVK for a registered product; follow local KVK advice."), # VERIFY WITH CPCRI: Bordeaux paste
            2: ("Active Bleeding", "Chisel out infected bark; apply Bordeaux paste; consult your KVK for a registered product (e.g. Tridemorph); follow product label and local KVK advice."), # VERIFY WITH CPCRI: Tridemorph
            3: ("Severe Bleeding", "Drench soil basin around palm; chisel infected wood; consult your KVK for a registered product; follow product label and local KVK advice.")
        },
        "Stem Cracking": {
            0: ("Healthy Palm", "No treatment needed. Avoid extreme soil drying."),
            1: ("Mild Cracks", "Maintain even irrigation schedule and organic mulching around palm basin; follow local KVK advice."),
            2: ("Moderate Cracking", "Conduct a soil test for boron deficiency; maintain even irrigation and organic mulching; follow local KVK advice."),
            3: ("Severe Cracking", "Conduct a soil test for boron; ensure uniform irrigation and organic mulching to prevent moisture fluctuations; consult your KVK for a registered product; follow local KVK advice.")
        },
        "Yellow Leaf Disease": {
            0: ("Phytoplasma Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."),
            1: ("Phytoplasma Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."),
            2: ("Phytoplasma Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."),
            3: ("Phytoplasma Alert", "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK.")
        }
    }
}

# ─── LOCALIZED DISEASE & CONDITION NAMES (KANNADA & HINDI) ────────────────────
DISEASE_LOCALIZATION = {
    "Paddy": {
        "bacterial_leaf_blight": {
            "en": "Bacterial Leaf Blight",
            "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಕರಕು ರೋಗ",  # REVIEW (Kannada): ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಕರಕು ರೋಗ (Bacterial Leaf Blight)
            "hi": "जीवाणु पत्ती झुलसा",  # REVIEW (Hindi): जीवाणु पत्ती झुलसा (Bacterial Leaf Blight)
        },
        "bacterial_leaf_streak": {
            "en": "Bacterial Leaf Streak",
            "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಪಟ್ಟೆ ರೋಗ",  # REVIEW (Kannada): ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಪಟ್ಟೆ ರೋಗ (Bacterial Leaf Streak)
            "hi": "जीवाणु पत्ती धारी",  # REVIEW (Hindi): जीवाणु पत्ती धारी (Bacterial Leaf Streak)
        },
        "bacterial_panicle_blight": {
            "en": "Bacterial Panicle Blight",
            "kn": "ತೆನೆ ಕರಕು ರೋಗ",  # REVIEW (Kannada): ತೆನೆ ಕರಕು ರೋಗ (Bacterial Panicle Blight)
            "hi": "जीवाणु बाली झुलसा",  # REVIEW (Hindi): जीवाणु बाली झुलसा (Bacterial Panicle Blight)
        },
        "blast": {
            "en": "Rice Blast",
            "kn": "ಬೆಂಕಿ ರೋಗ (ಬ್ಲಾಸ್ಟ್)",  # REVIEW (Kannada): ಬೆಂಕಿ ರೋಗ / ಬ್ಲಾಸ್ಟ್ (Rice Blast)
            "hi": "झुलसा रोग (ब्लास्ट)",  # REVIEW (Hindi): झुलसा रोग / ब्लास्ट (Rice Blast)
        },
        "brown_spot": {
            "en": "Brown Spot",
            "kn": "ಕಂದು ಚುಕ್ಕೆ ರೋಗ",  # REVIEW (Kannada): ಕಂದು ಚುಕ್ಕೆ ರೋಗ (Brown Spot)
            "hi": "भूरा धब्बा रोग",  # REVIEW (Hindi): भूरा धब्बा रोग (Brown Spot)
        },
        "dead_heart": {
            "en": "Dead Heart",
            "kn": "ಸುಳಿ ಒಣಗು ರೋಗ (ಡೆಡ್ ಹಾರ್ಟ್)",  # REVIEW (Kannada): ಸುಳಿ ಒಣಗು ರೋಗ (Dead Heart)
            "hi": "डेड हार्ट (तना छेदक प्रकोप)",  # REVIEW (Hindi): डेड हार्ट (Dead Heart)
        },
        "downy_mildew": {
            "en": "Downy Mildew",
            "kn": "ಬೂದಿ ರೋಗ",  # REVIEW (Kannada): ಬೂದಿ ರೋಗ (Downy Mildew)
            "hi": "डाउनी फफूंदी",  # REVIEW (Hindi): डाउनी फफूंदी (Downy Mildew)
        },
        "hispa": {
            "en": "Hispa",
            "kn": "ಹಿಸ್ಪಾ ಕೀಟ ಬಾಧೆ",  # REVIEW (Kannada): ಹಿಸ್ಪಾ ಕೀಟ ಬಾಧೆ (Rice Hispa)
            "hi": "हिस्पा कीट प्रकोप",  # REVIEW (Hindi): हिस्पा कीट प्रकोप (Rice Hispa)
        },
        "normal": {
            "en": "Healthy (No Disease)",
            "kn": "ಆರೋಗ್ಯಕರ ಬೆಳೆ",  # REVIEW (Kannada): ಆರೋಗ್ಯಕರ ಬೆಳೆ (Healthy Crop)
            "hi": "स्वस्थ फसल",  # REVIEW (Hindi): स्वस्थ फसल (Healthy Crop)
        },
        "tungro": {
            "en": "Tungro Virus",
            "kn": "ತುಂಗ್ರೋ ವೈರಸ್ ರೋಗ",  # REVIEW (Kannada): ತುಂಗ್ರೋ ವೈರಸ್ ರೋಗ (Tungro Virus)
            "hi": "तुंग्रो वायरस",  # REVIEW (Hindi): तुंग्रो वायरस (Tungro Virus)
        },
    },
    "Arecanut": {
        "bud_borer": {
            "en": "Bud Borer",
            "kn": "ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು",  # REVIEW (Kannada): ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು (Bud Borer)
            "hi": "कली छेदक कीट",  # REVIEW (Hindi): कली छेदक कीट (Bud Borer)
        },
        "healthy_foot": {
            "en": "Healthy (Foot)",
            "kn": "ಆರೋಗ್ಯಕರ ಬುಡ",  # REVIEW (Kannada): ಆರೋಗ್ಯಕರ ಬುಡ (Healthy Foot)
            "hi": "स्वस्थ तना आधार",  # REVIEW (Hindi): स्वस्थ तना आधार (Healthy Foot)
        },
        "healthy_leaf": {
            "en": "Healthy (Leaf)",
            "kn": "ಆರೋಗ್ಯಕರ ಗರಿ",  # REVIEW (Kannada): ಆರೋಗ್ಯಕರ ಗರಿ (Healthy Leaf)
            "hi": "स्वस्थ पत्ती",  # REVIEW (Hindi): स्वस्थ पत्ती (Healthy Leaf)
        },
        "healthy_nut": {
            "en": "Healthy (Nut)",
            "kn": "ಆರೋಗ್ಯಕರ ಅಡಿಕೆ ಕಾಯಿ",  # REVIEW (Kannada): ಆರೋಗ್ಯಕರ ಅಡಿಕೆ ಕಾಯಿ (Healthy Nut)
            "hi": "स्वस्थ सुपारी फल",  # REVIEW (Hindi): स्वस्थ सुपारी फल (Healthy Nut)
        },
        "healthy_trunk": {
            "en": "Healthy (Trunk)",
            "kn": "ಆರೋಗ್ಯಕರ ಮರ",  # REVIEW (Kannada): ಆರೋಗ್ಯಕರ ಮರ (Healthy Trunk)
            "hi": "स्वस्थ मुख्य तना",  # REVIEW (Hindi): स्वस्थ मुख्य तना (Healthy Trunk)
        },
        "mahali_koleroga": {
            "en": "Mahali / Koleroga (Rot Disease)",
            "kn": "ಮಹಾಳಿ / ಕೊಳೆರೋಗ",  # REVIEW (Kannada): ಮಹಾಳಿ / ಕೊಳೆರೋಗ (Mahali / Koleroga)
            "hi": "महाली / कोलेरोगा सड़ांध",  # REVIEW (Hindi): महाली / कोलेरोगा सड़ांध (Mahali / Koleroga)
        },
        "stem_bleeding": {
            "en": "Stem Bleeding",
            "kn": "ಕಾಂಡ ಸೋರುವಿಕೆ ರೋಗ",  # REVIEW (Kannada): ಕಾಂಡ ಸೋರುವಿಕೆ ರೋಗ (Stem Bleeding)
            "hi": "तना स्राव रोग",  # REVIEW (Hindi): तना स्राव रोग (Stem Bleeding)
        },
        "stem_cracking": {
            "en": "Stem Cracking",
            "kn": "ಕಾಂಡ ಬಿರುಕು ರೋಗ",  # REVIEW (Kannada): ಕಾಂಡ ಬಿರುಕು ರೋಗ (Stem Cracking)
            "hi": "तना फटना रोग",  # REVIEW (Hindi): तना फटना रोग (Stem Cracking)
        },
        "yellow_leaf_disease": {
            "en": "Yellow Leaf Disease",
            "kn": "ಹಳದಿ ಎಲೆ ರೋಗ",  # REVIEW (Kannada): ಹಳದಿ ಎಲೆ ರೋಗ (Yellow Leaf Disease)
            "hi": "पीली पत्ती रोग",  # REVIEW (Hindi): पीली पत्ती रोग (Yellow Leaf Disease)
        },
    },
    "Apple": {
        "apple_scab": {
            "en": "Apple Scab",
            "kn": "ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ರೋಗ",  # REVIEW (Kannada): ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ರೋಗ (Apple Scab)
            "hi": "सेब का स्केब रोग",  # REVIEW (Hindi): सेब का स्केब रोग (Apple Scab)
        },
        "black_rot": {
            "en": "Black Rot (Frog-Eye Leaf Spot)",
            "kn": "ಕಪ್ಪು ಕೊಳೆರೋಗ",  # REVIEW (Kannada): ಕಪ್ಪು ಕೊಳೆರೋಗ (Black Rot)
            "hi": "ब्लैक रॉट (काला सड़न रोग)",  # REVIEW (Hindi): ब्लैक रॉट (Black Rot)
        },
        "cedar_apple_rust": {
            "en": "Cedar Apple Rust",
            "kn": "ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ)",  # REVIEW (Kannada): ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್ (Cedar Apple Rust)
            "hi": "सीडर एप्पल रस्ट (गेरुआ रोग)",  # REVIEW (Hindi): सीडर एप्पल रस्ट (गेरुआ रोग)
        },
        "healthy": {
            "en": "Healthy (No Disease)",
            "kn": "ಆರೋಗ್ಯಕರ ಎಲೆ",  # REVIEW (Kannada): ಆರೋಗ್ಯಕರ ಎಲೆ (Healthy Foliage)
            "hi": "स्वस्थ पत्तियां",  # REVIEW (Hindi): स्वस्थ पत्तियां (Healthy Foliage)
        },
    }
}

ABSTAIN_THRESHOLDS = {
    "Paddy": 0.70,
    "Arecanut": 0.70,
    "Apple": 0.70
}

# ─── TUNGRO SAFETY WARNING NOTE (EN / KN / HI) ──────────────────────────────
TUNGRO_SAFETY_NOTE = {
    "en": "Stunted plants with yellow-orange leaves can also be Tungro virus (no cure). If so, consult your KVK.",
    "kn": "ಗಿಡ್ಡವಾದ ಸಸಿಗಳು ಹಾಗೂ ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣದ ಎಲೆಗಳು ಕಂಡುಬಂದರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಆಗಿರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದಲ್ಲಿ, ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",  # REVIEW (Kannada): Tungro safety note
    "hi": "बौने पौधे और पीली-नारंगी पत्तियां तुंग्रो वायरस (जिसका कोई इलाज नहीं है) भी हो सकती हैं। यदि ऐसा है, तो अपने केवीके से सलाह लें।",  # REVIEW (Hindi): Tungro safety note
}

# ─── SOIL HEALTH ADVISORIES (PER CROP, LOCALIZED) ───────────────────────────
SOIL_HEALTH_ADVISORY = {
    "Paddy": {
        "en": (
            "Soil Health & Nutrient Management",
            "Incorporate organic compost or green manure (Dhaincha). Inoculate with Azolla or biofertilisers to reduce synthetic urea reliance. Conduct a soil test before adding nutrients."
        ),
        "kn": (
            "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಪೋಷಕಾಂಶ ನಿರ್ವಹಣೆ",
            "ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಅಥವಾ ಹಸಿರೆಲೆ ಗೊಬ್ಬರ (ಡೈಂಚಾ) ಬಳಸಿ. ಯೂರಿಯಾ ಅವಲಂಬನೆ ಕಡಿಮೆ ಮಾಡಲು ಅಜೋಲ್ಲಾ ಅಥವಾ ಜೈವಿಕ ಗೊಬ್ಬರ ಸೇರಿಸಿ. ಪೋಷಕಾಂಶಗಳನ್ನು ನೀಡುವ ಮುನ್ನ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."
        ),
        "hi": (
            "मृदा स्वास्थ्य एवं पोषक तत्व प्रबंधन",
            "सड़ा गोबर का खाद या हरी खाद (ढैंचा) मिलाएं। यूरिया पर निर्भरता घटाने के लिए अजोला या जैव उर्वरक का उपयोग करें। पोषक तत्व देने से पहले मिट्टी परीक्षण अवश्य करवाएं।"
        ),
    },
    "Arecanut": {
        "en": (
            "Soil Health & Nutrient Management",
            "Mulch the palm basin with fallen arecanut leaves. Apply well-decomposed farmyard manure and Trichoderma-enriched compost. Conduct a soil test for boron and potash deficiency."
        ),
        "kn": (
            "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಪೋಷಕಾಂಶ ನಿರ್ವಹಣೆ",
            "ಅಡಿಕೆ ಬುಡಕ್ಕೆ ಒಣಗಿದ ಅಡಿಕೆ ಗರಿಗಳಿಂದ ಮಲ್ಚಿಂಗ್ ಮಾಡಿ. ಚೆನ್ನಾಗಿ ಕೊಳೆತ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಮತ್ತು ಟ್ರೈಕೋಡರ್ಮ ಮಿಶ್ರಿತ ಕಾಂಪೋಸ್ಟ್ ಹಾಕಿ. ಬೋರಾನ್ ಮತ್ತು ಪೊಟ್ಯಾಷ್ ಕೊರತೆಗಾಗಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."
        ),
        "hi": (
            "मृदा स्वास्थ्य एवं पोषक तत्व प्रबंधन",
            "सुपारी के तने के चारों ओर पत्तियों से मल्चिंग करें। अच्छी सड़ी गोबर की खाद और ट्राइकोडर्मा युक्त कंपोस्ट डालें। बोरॉन और पोटाश के लिए मिट्टी परीक्षण करवाएं।"
        ),
    },
    "Apple": {
        "en": (
            "Soil Health & Nutrient Management",
            "Apply well-rotted farmyard manure and maintain organic tree-basin mulching during dormancy. Conduct a soil test before applying calcium or boron amendments."
        ),
        "kn": (
            "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಪೋಷಕಾಂಶ ನಿರ್ವಹಣೆ",
            "ವಿಶ್ರಾಂತ ಕಾಲದಲ್ಲಿ ಮರದ ಬುಡಕ್ಕೆ ಚೆನ್ನಾಗಿ ಕೊಳೆತ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಹಾಕಿ ಮಲ್ಚಿಂಗ್ ಮಾಡಿ. ಕ್ಯಾಲ್ಸಿಯಂ ಅಥವಾ ಬೋರಾನ್ ನೀಡುವ ಮುನ್ನ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."
        ),
        "hi": (
            "मृदा स्वास्थ्य एवं पोषक तत्व प्रबंधन",
            "सुप्त अवस्था के दौरान थालों में सड़ी गोबर की खाद डालें और मल्चिंग करें। कैल्शियम या बोरॉन डालने से पहले मिट्टी परीक्षण करवाएं।"
        ),
    },
}

# ─── VOICE ADVISORY BUILDER ──────────────────────────────────────────────────
def build_voice_advisory(crop_name: str, disease_name: str, pred_class: str, grade_idx: int, lang: str, weather_sentence: str = "") -> str:
    def _get_base_advisory():
        crop_map = {
            "en": {"Paddy": "Paddy", "Arecanut": "Arecanut", "Apple": "Apple"},
            "kn": {"Paddy": "ಭತ್ತ", "Arecanut": "ಅಡಿಕೆ", "Apple": "ಸೇಬು"},
            "hi": {"Paddy": "धान", "Arecanut": "सुपारी", "Apple": "सेब"}
        }
        crop_lbl = crop_map.get(lang, crop_map["en"]).get(crop_name, crop_name)

        is_healthy_leaf = (pred_class in ["normal", "healthy", "healthy_foot", "healthy_leaf", "healthy_nut", "healthy_trunk"]
                           or disease_name.startswith("Healthy"))
        is_incurable_dis = (pred_class in ["tungro", "yellow_leaf_disease"])

        loc_dict = DISEASE_LOCALIZATION.get(crop_name, {}).get(pred_class, {})
        dis_lbl = loc_dict.get(lang, disease_name)

        # 1. Healthy foliage
        if is_healthy_leaf:
            if lang == "kn":
                return f"ಬೆಳೆ: {crop_lbl}. ನಿಮ್ಮ ಬೆಳೆ ತುಂಬಾ ಹೆಲ್ದಿ ಆಗಿದೆ ಮತ್ತು ಎಲೆಯಲ್ಲಿ ಯಾವುದೇ ರೋಗ ಇಲ್ಲ. ಯಾವುದೇ ಸ್ಪ್ರೇ ಬೇಕಾಗಿಲ್ಲ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। आपकी फसल बिल्कुल हेल्दी है और पत्तों पर कोई बीमारी नहीं है। किसी दवा या स्प्रे की ज़रूरत नहीं है।"
            else:
                return f"Crop: {crop_lbl}. Foliage is healthy with no disease symptoms. No treatment needed."

        # 2. Apple special handling (Never mention severity; disease-specific organic steps first; chemicals only as consult KVK)
        if crop_name == "Apple":
            if pred_class == "apple_scab":
                if lang == "kn":
                    return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಉದುರಿದ ಎಲೆಗಳನ್ನು ಹೆಕ್ಕಿ ಗೊಬ್ಬರ ಮಾಡಿ ಮತ್ತು ಎಲೆಗಳು ಬೇಗನೆ ಒಣಗಲು ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ ಗಾಳಿಯಾಡುವಂತೆ ಮಾಡಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
                elif lang == "hi":
                    return f"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय: गिरी हुई पत्तियों को इकट्ठा करके नष्ट करें और हवा के आवागमन के लिए पेड़ों की छंटाई करें ताकि पत्तियां जल्दी सूखें। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"
                else:
                    return f"Crop: Apple. Disease: Apple Scab. Organic and cultural steps: rake and compost fallen leaves to eliminate overwintering spores, and maintain open canopy pruning for air circulation and fast leaf drying. For chemical options, consult your KVK for a registered product."
            elif pred_class == "black_rot":
                if lang == "kn":
                    return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಒಣಗಿದ ಕೊಂಬೆಗಳು ಮತ್ತು ಒಣಗಿದ ಹಣ್ಣುಗಳನ್ನು ಮರ ಮತ್ತು ನೆಲದಿಂದ ತೆಗೆದುಹಾಕಿ ಶಿಲೀಂಧ್ರ ಮೂಲಗಳನ್ನು ನಾಶಪಡಿಸಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
                elif lang == "hi":
                    return f"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय: सूखी टहनियों, घावों और सड़े-सूखे फलों को पेड़ों और बगीचे से हटाकर नष्ट करें ताकि फफूंद के स्रोत खत्म हों। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"
                else:
                    return f"Crop: Apple. Disease: Black Rot. Organic and cultural steps: prune dead wood, cankers, and mummified fruits from trees and orchard floor to reduce fungal sources. For chemical options, consult your KVK for a registered product."
            elif pred_class == "cedar_apple_rust":
                if lang == "kn":
                    return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಹತ್ತಿರದಲ್ಲಿ ಜೂನಿಪರ್ ಅಥವಾ ಸೀಡರ್ ಮರಗಳಿದ್ದರೆ ಅವುಗಳಲ್ಲಿನ ಗಂಟುಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬೆಳೆಯಿರಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
                elif lang == "hi":
                    return f"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय: यदि आसपास जुनिपर या सीडर के पेड़ मौजूद हों तो रस्ट की गांठों को हटाएं और प्रतिरोधी किस्में लगाएं। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"
                else:
                    return f"Crop: Apple. Disease: Cedar Apple Rust. Organic and cultural steps: remove nearby cedar or juniper rust galls if present nearby, and plant rust-resistant cultivars. For chemical options, consult your KVK for a registered product."
            else:
                if lang == "kn":
                    return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
                elif lang == "hi":
                    return f"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय अपनाएं। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"
                else:
                    return f"Crop: Apple. Disease: {disease_name}. Start with organic and cultural orchard management. For chemical options, consult your KVK for a registered product."

        # 3. Incurable disease (Paddy Tungro / Arecanut YLD)
        if is_incurable_dis:
            if lang == "kn":
                return f"ಬೆಳೆ: {crop_lbl}. ಅಲರ್ಟ್: {dis_lbl} ಬಂದಿದೆ. ಇದಕ್ಕೆ ಯಾವುದೇ ಕೆಮಿಕಲ್ ಮೆಡಿಸಿನ್ ಇಲ್ಲ. ರೋಗ ಹರಡಿಸೋ ಕೀಟಗಳನ್ನ ಕಂಟ್ರೋಲ್ ಮಾಡಿ, ರೋಗ ಬಂದಿರೋ ಗಿಡನ ಕಿತ್ತು ಬಿಸಾಕಿ, ಮತ್ತು ಕೆವಿಕೆನ ಕಾಂಟ್ಯಾಕ್ಟ್ ಮಾಡಿ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। अलर्ट: {dis_lbl} पाया गया है। इसका कोई केमिकल इलाज नहीं है। रस चूसने वाले कीड़ों को रोको, बीमार पौधों को उखाड़ कर खेत से बाहर फेंक दो, और केवीके से संपर्क करो।"
            else:
                return f"Crop: {crop_lbl}. Warning: {disease_name} detected. No chemical cure exists. Manage insect vectors, remove infected plants, and consult your local KVK."

        # 4. Specific treatable diseases: card-aligned organic (grade 1) / chemical (grade >= 2)
        SPECIFIC_VOICE_RULES = {
            "blast": {
                1: {
                    "en": "Apply bio-fungicide Trichoderma viride or neem-based formulation; follow product label and local KVK advice.",
                    "kn": "ಟ್ರೈಕೋಡರ್ಮ ವಿರಿಡೆ ಜೈವಿಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ ಬೇವಿನ ಸೂತ್ರೀಕರಣವನ್ನು ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆಯನ್ನು ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "जैव कवकनाशी ट्राइकोडर्मा विरिडे या नीम आधारित कीटनाशक का छिड़काव करें; लेबल और स्थानीय केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Apply systemic fungicide Tricyclazole 75 WP; follow product label and local KVK advice.",
                    "kn": "ಶಿಫಾರಸು ಮಾಡಿದ ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ 75 ಡಬ್ಲ್ಯೂಪಿ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "प्रणालीगत कवकनाशी ट्राइसाइक्लाजोल 75 डब्ल्यूपी का छिड़काव करें; लेबल निर्देश और केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                }
            },
            "mahali_koleroga": {
                1: {
                    "en": "Tie polythene covers over developing bunches; apply preventive 1% Bordeaux Mixture before monsoon; follow product label and local KVK advice.",
                    "kn": "ಬೆಳೆಯುತ್ತಿರುವ ಅಡಿಕೆ ಗೊನೆಗಳಿಗೆ ಪಾಲಿಥಿನ್ ಹೊದಿಕೆ ಕಟ್ಟಿ; ಮುಂಗಾರಿಗೆ ಮುನ್ನ ಮುನ್ನೆಚ್ಚರಿಕೆಯಾಗಿ ಶೇಕಡಾ 1 ರ ಬೋರ್ಡೋ ದ್ರಾವಣ ಸಿಂಪಡಿಸಿ ಮತ್ತು ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "सुपारी के गुच्छों पर पॉलीथिन बांधें; मानसून से पहले बचाव के लिए 1% बोर्डो मिश्रण का छिड़काव करें और केवीके की सलाह मानें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Spray 1% Bordeaux Mixture with adhesive sticker on all nut bunches; follow product label and local KVK advice.",
                    "kn": "ಎಲ್ಲಾ ಅಡಿಕೆ ಗೊನೆಗಳ ಮೇಲೆ ಅಂಟು ದ್ರವದೊಂದಿಗೆ ಶೇಕಡಾ 1 ರ ಬೋರ್ಡೋ ಮಿಶ್ರಣವನ್ನು ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "सभी गुच्छों पर चिपकाने वाले पदार्थ के साथ 1% बोर्डो मिश्रण का छिड़काव करें; लेबल और केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                }
            },
            "bacterial_leaf_blight": {
                1: {
                    "en": "Spray neem seed kernel extract. Avoid excess nitrogen top-dressing; follow product label and local KVK advice.",
                    "kn": "ಬೇವಿನ ಬೀಜದ ಕಷಾಯ ಸಿಂಪಡಿಸಿ. ಹೆಚ್ಚಿನ ಸಾರಜನಕ ಗೊಬ್ಬರ ನೀಡುವುದನ್ನು ತಪ್ಪಿಸಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "नीम के बीज का काढ़ा छिड़कें। यूरिया का अधिक उपयोग न करें; स्थानीय केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Apply Copper Oxychloride + Streptocycline; follow product label and local KVK advice.",
                    "kn": "ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಮತ್ತು ಸ್ಟ್ರೆಪ್ಟೊಸೈಕ್ಲಿನ್ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "कॉपर ऑक्सीक्लोराइड और स्ट्रेप्टोसाइक्लिन का छिड़काव करें; लेबल और केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                }
            },
            "brown_spot": {
                1: {
                    "en": "Apply compost and correct potassium deficiency; follow local KVK advice.",
                    "kn": "ಕಾಂಪೋಸ್ಟ್ ಗೊಬ್ಬರ ಹಾಕಿ ಮತ್ತು ಪೊಟ್ಯಾಷ್ ಕೊರತೆಯನ್ನು ಸರಿಪಡಿಸಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "कम्पोस्ट खाद डालें और पोटाश की कमी को दूर करें; स्थानीय केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Apply Mancozeb protective spray; follow product label and local KVK advice.",
                    "kn": "ಮ್ಯಾಂಕೋಜೆಬ್ ರಕ್ಷಣಾತ್ಮಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "मैंकोजेब कवकनाशी का सुरक्षात्मक छिड़काव करें; लेबल और केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                }
            },
            "dead_heart": {
                1: {
                    "en": "Clip seedling tips before transplanting; install yellow stem borer pheromone traps; follow local KVK advice.",
                    "kn": "ನಾಟಿ ಮಾಡುವ ಮುನ್ನ ಸಸಿಗಳ ತುದಿಯನ್ನು ಕತ್ತರಿಸಿ; ಹಳದಿ ಕಾಂಡ ಕೊರೆಯುವ ಕೀಟದ ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "रोपाई से पहले पौध की नोक काटें; तना छेदक के लिए फेरोमोन ट्रैप लगाएं; केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Release Trichogramma japonicum egg parasitoids; install pheromone traps; follow local KVK advice.",
                    "kn": "ಟ್ರೈಕೋಗ್ರಾಮಾ ಜಪಾನಿಕಂ ಮೊಟ್ಟೆ ಪರಾವಲಂಬಿಗಳನ್ನು ಬಿಡಿ; ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "ट्राइकोग्रामा जैपोनिकम परजीवी कीट छोड़ें; फेरोमोन ट्रैप लगाएं; स्थानीय केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                }
            },
            "hispa": {
                1: {
                    "en": "Manually clip infected leaf tips containing grubs; use sweep nets; follow local KVK advice.",
                    "kn": "ಮರಿ ಕೀಟಗಳಿರುವ ಎಲೆಗಳ ತುದಿಯನ್ನು ಕೈಯಿಂದ ಕತ್ತರಿಸಿ ನಾಶಪಡಿಸಿ; ಜಾಲರಿ ಬಳಸಿ ಕೀಟ ಹಿಡಿಯಿರಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "कीट युक्त पत्तियों की नोक को काटकर नष्ट करें; कीट पकड़ने वाले जाल का उपयोग करें; स्थानीय केवीके की सलाह मानें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Manually clip affected leaf tips containing grubs; consult your KVK for a registered product; follow product label and local KVK advice.",
                    "kn": "ಕೀಟವಿರುವ ಎಲೆಗಳ ತುದಿಯನ್ನು ಕತ್ತರಿಸಿ; ನೋಂದಾಯಿತ ಔಷಧಿಗಾಗಿ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಲೇಬಲ್ ಸೂಚನೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "संक्रमित पत्तियों की नोक काटें; अनुमोदित कीटनाशक के लिए केवीके से संपर्क करें और लेबल का पालन करें।"  # REVIEW (Hindi)
                }
            },
            "stem_bleeding": {
                1: {
                    "en": "Chisel out bleeding bark patch down to healthy tissue; apply Bordeaux paste; consult your KVK for a registered product; follow local KVK advice.",
                    "kn": "ರಸ ಸೋರುವ ತೊಗಟೆಯನ್ನು ಆರೋಗ್ಯಕರ ಭಾಗದವರೆಗೆ ಕೆತ್ತಿ ತೆಗೆಯಿರಿ; ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಹಚ್ಚಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "तने से रस बहने वाले हिस्से को स्वस्थ लकड़ी तक छीलें; बोर्डो पेस्ट लगाएं; स्थानीय केवीके की सलाह का पालन करें।"  # REVIEW (Hindi)
                },
                2: {
                    "en": "Chisel out infected bark; apply Bordeaux paste; consult your KVK for a registered product (e.g. Tridemorph); follow product label and local KVK advice.",
                    "kn": "ಸೋಂಕಿತ ತೊಗಟೆಯನ್ನು ಕೆತ್ತಿ ತೆಗೆದು ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಹಚ್ಚಿ; ಶಿಫಾರಸು ಮಾಡಿದ ಔಷಧಿಗಾಗಿ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",  # REVIEW (Kannada)
                    "hi": "संक्रमित छाल को छीलें और बोर्डो पेस्ट लगाएं; अनुशंसित दवा के लिए अपने केवीके से संपर्क करें।"  # REVIEW (Hindi)
                }
            }
        }

        if pred_class in SPECIFIC_VOICE_RULES:
            tier = 1 if grade_idx < 2 else 2
            act_dict = SPECIFIC_VOICE_RULES[pred_class][tier]
            act_text = act_dict.get(lang, act_dict["en"])
            if lang == "kn":
                sev_prefix = "ಲಘು ತೀವ್ರತೆ. " if tier == 1 else "ಹೆಚ್ಚಿನ ತೀವ್ರತೆ. "
                return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. {sev_prefix}{act_text}"
            elif lang == "hi":
                sev_prefix = "शुरुआती अवस्था। " if tier == 1 else "गंभीर अवस्था। "
                return f"फसल: {crop_lbl}। रोग: {dis_lbl}। {sev_prefix}{act_text}"
            else:
                sev_prefix = "Mild severity. " if tier == 1 else "Moderate to severe infection. "
                return f"Crop: {crop_lbl}. Disease: {disease_name}. {sev_prefix}{act_text}"

        # 5. Other Treatable disease (Paddy & Arecanut): organic steps first; chemical only if grade >= 2
        if grade_idx < 2:
            if lang == "kn":
                return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಇನ್‌ಫೆಕ್ಷನ್ ಈಗ ಸ್ಟಾರ್ಟ್ ಆಗಿದೆ. ಮೊದಲು ಆರ್ಗ್ಯಾನಿಕ್ ಸ್ಟೆಪ್ಸ್ ಮಾಡಿ: ರೋಗ ಬಂದಿರೋ ಎಲೆ ಕಿತ್ತು ಬಿಸಾಕಿ ಮತ್ತು ತೋಟ ಕ್ಲೀನ್ ಆಗಿ ಇಡಿ. ಕೆಮಿಕಲ್ ಸ್ಪ್ರೇ ಮಾಡೋಕಿಂತ ಮುಂಚೆ ಕೆವಿಕೆಯಲ್ಲಿ ಕೇಳಿ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। रोग: {dis_lbl}। इन्फेक्शन अभी शुरुआत में है। पहले ऑर्गेनिक स्टेप्स लें: खराब पत्तों को तोड़कर हटा दें और खेत में सफाई रखें। केमिकल स्प्रे से पहले अपने केवीके से ज़रूर पूछें।"
            else:
                return f"Crop: {crop_lbl}. Disease: {disease_name}. Mild severity. Start with organic and cultural steps: remove infected leaves and maintain field sanitation. Consult your local KVK before using chemicals."
        else:
            if lang == "kn":
                return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ರೋಗ ತುಂಬಾ ಜಾಸ್ತಿ ಆಗಿದೆ ಮತ್ತು ಬರೀ ಆರ್ಗ್ಯಾನಿಕ್ ಸ್ಟೆಪ್ಸ್ ಸಾಲಲ್ಲ. ಕೂಡಲೇ ಕೆವಿಕೆ ಕಾಂಟ್ಯಾಕ್ಟ್ ಮಾಡಿ, ಅವರು ರೆಕಮೆಂಡ್ ಮಾಡಿರೋ ಕೆಮಿಕಲ್ ಸ್ಪ್ರೇನ ಲೇಬಲ್ ಇನ್‌ಸ್ಟ್ರಕ್ಷನ್ ತರ ಯೂಸ್ ಮಾಡಿ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। रोग: {dis_lbl}। बीमारी काफी बढ़ चुकी है और सिर्फ ऑर्गेनिक स्टेप्स काफी नहीं होंगे। तुरंत अपने केवीके से कांटेक्ट करें और उनके बताए केमिकल को पैकेट पर लिखे इंस्ट्रक्शन के हिसाब से स्प्रे करें।"
            else:
                return f"Crop: {crop_lbl}. Disease: {disease_name}. Moderate to severe infection. Cultural measures alone are insufficient. Consult your local KVK for registered chemical treatments and follow product label advice."

    base_text = _get_base_advisory()
    if weather_sentence:
        return f"{base_text} {weather_sentence}".strip()
    return base_text

REQUIRED_MODEL_FILES = [
    "outputs/models/paddy_disease_classifier.pth",
    "outputs/models/paddy_ordinal.pth",
    "outputs/models/paddy_baseline.pth",
    "outputs/models/arecanut_disease_classifier.pth",
    "outputs/models/arecanut_ordinal.pth",
    "outputs/models/arecanut_baseline.pth",
    "outputs/apple_fgvc/apple_disease_fgvc_v1.pth",
    "outputs/models/apple_ordinal.pth",
    "outputs/ood_params_v2.pt"
]

def check_missing_models() -> List[str]:
    missing = []
    for rel in REQUIRED_MODEL_FILES:
        pth = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(pth):
            missing.append(rel)
    return missing

# ─── LOAD DISEASE CLASSIFIER ─────────────────────────────────────────────────
@st.cache_resource
def get_disease_classifier(crop_name: str):
    key = crop_name.lower()
    if key == "apple":
        pth = os.path.join(config.OUTPUTS_DIR, "apple_fgvc", "apple_disease_fgvc_v1.pth")
    else:
        pth = os.path.join(config.OUTPUTS_DIR, "models", f"{key}_disease_classifier.pth")
    if not os.path.exists(pth):
        return None, None, None

    ck = torch.load(pth, map_location='cpu', weights_only=True)
    classes = ck['classes']
    model = models.resnet18(weights=None)

    if 'fc.1.weight' in ck['model_state_dict']:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.fc.in_features, len(classes)))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, classes, ck.get('idx_to_class', {str(i): c for i, c in enumerate(classes)})

# ─── LOAD SEVERITY MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def get_severity_model(crop_name, model_type="Ordinal CORN"):
    key = crop_name.lower()
    t   = "ordinal" if "Ordinal" in model_type else "baseline"
    path = os.path.join(config.OUTPUTS_DIR, "models", f"{key}_{t}.pth")
    if not os.path.exists(path):
        return None
    try:
        m = CropSeverityModel(backbone=config.BACKBONE, num_classes=config.NUM_SEVERITY_GRADES,
                              ordinal=(t == "ordinal"), pretrained=False)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        m.load_state_dict(sd)
        m.eval()
        return m
    except Exception as e:
        logging.warning(f"Severity model load error: {e}")
        return None

# ─── GRAD-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model    = model
        self._feats   = None
        self._grads   = None
        target_layer = model.layer4[-1] if hasattr(model, 'layer4') else model.layer4
        self._hook_f  = target_layer.register_forward_hook(self._save_feats)
        self._hook_b  = target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, _, __, out): self._feats = out
    def _save_grads(self, _, __, grad_out): self._grads = grad_out[0]

    def __call__(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._feats).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        if cam.max() > 0: cam = cam / cam.max()
        return cam

    def remove(self):
        self._hook_f.remove()
        self._hook_b.remove()

def make_gradcam_overlay(original_pil: Image.Image, cam: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    img_np = np.array(original_pil.convert("RGB"))
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * img_np)

# ─── INFERENCE HELPERS ───────────────────────────────────────────────────────
def run_disease_classification(image: Image.Image, crop_name: str):
    model, classes, idx_to_class = get_disease_classifier(crop_name)
    if model is None:
        return None, 0.0, {}, None, "Classifier not trained yet", False

    tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy().tolist()

    pred_idx  = int(np.argmax(probs))
    pred_cls  = classes[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict  = {c: float(probs[i]) for i, c in enumerate(classes)}

    gradcam_overlay = None
    try:
        model_for_cam = get_disease_classifier(crop_name)[0]
        gcam = GradCAM(model_for_cam)
        tensor_grad = tf(image).unsqueeze(0).requires_grad_(False)
        cam = gcam(tensor_grad, pred_idx)
        gradcam_overlay = make_gradcam_overlay(image, cam)
        gcam.remove()
    except Exception as e:
        logging.warning(f"Grad-CAM failed: {e}")

    if crop_name == "Paddy": disp_map = PADDY_DISPLAY
    elif crop_name == "Apple": disp_map = APPLE_DISPLAY
    else: disp_map = ARECA_DISPLAY

    display_name = disp_map.get(pred_cls, pred_cls.replace("_", " ").title())
    return pred_cls, confidence, prob_dict, gradcam_overlay, display_name, True

def run_severity(image: Image.Image, crop_name: str, model_type: str):
    model = get_severity_model(crop_name, model_type)
    img_np = np.array(image)

    try:
        leaf_mask   = segment_leaf_adaptive(img_np)
        lesion_mask = detect_lesions_group_a(img_np, leaf_mask)
        cv_ratio    = compute_ratio(lesion_mask, leaf_mask)
    except Exception:
        leaf_mask   = np.ones(img_np.shape[:2], dtype=bool)
        lesion_mask = np.zeros(img_np.shape[:2], dtype=bool)
        cv_ratio    = 0.0

    overlay = img_np.copy()
    overlay[lesion_mask] = [239, 68, 68]
    overlay[~leaf_mask]  = (overlay[~leaf_mask] * 0.4).astype(np.uint8)

    if model is None:
        grade = ratio_to_grade(cv_ratio)
        return grade, 0.90, [0.1, 0.7, 0.15, 0.05], cv_ratio, overlay

    tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    ordinal = "Ordinal" in model_type
    t = tf(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)
        if ordinal:
            p_raw   = torch.sigmoid(logits)
            grade_t, gprobs = corn_probs_to_grade(p_raw[0])
            grade = grade_t.item()
            probs = gprobs.cpu().numpy().tolist()
        else:
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            grade = int(np.argmax(probs))

    return grade, float(probs[grade]), probs, cv_ratio, overlay

# ─── SESSION STATE INITIALIZATION ────────────────────────────────────────────
if "lang_code" not in st.session_state:
    st.session_state["lang_code"] = "en"
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = "Mangaluru"
if "active_crop" not in st.session_state:
    st.session_state["active_crop"] = "Paddy"
if "scan_history" not in st.session_state:
    st.session_state["scan_history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "view_result" not in st.session_state:
    st.session_state["view_result"] = False
if "severity_arch" not in st.session_state:
    st.session_state["severity_arch"] = "Ordinal CORN"
if "use_live_weather" not in st.session_state:
    st.session_state["use_live_weather"] = True

# ─── LANGUAGE SYNCHRONIZATION ────────────────────────────────────────────────
lang_opts = ["English", "ಕನ್ನಡ", "हिन्दी"]
lang_map = {"English": "en", "ಕನ್ನಡ": "kn", "हिन्दी": "hi"}
inv_lang_map = {v: k for k, v in lang_map.items()}

if "lang_radio_choice" in st.session_state and st.session_state["lang_radio_choice"] in lang_map:
    st.session_state["lang_code"] = lang_map[st.session_state["lang_radio_choice"]]

cur_lang = st.session_state.get("lang_code", "en")

# ─── HEADER (APP NAME, LOCATION PIN, LOCALIZED DATE, LANGUAGE SWITCHER) ──────
st.markdown(f"""
<div style="padding: 2px 0 4px 0;">
    <h1 style="margin:0;font-size:26px;color:#275637;display:flex;align-items:center;gap:8px;">
        🌾 {APP_NAME}
    </h1>
    <div style="font-size:13px;color:#556B5C;margin-top:2px;">
        📅 {format_local_date(datetime.date.today(), cur_lang)}
    </div>
</div>
""", unsafe_allow_html=True)

chosen_lang_name = st.segmented_control(
    "Language",
    options=lang_opts,
    default=inv_lang_map.get(st.session_state.get("lang_code", "en"), "English"),
    selection_mode="single",
    label_visibility="collapsed",
    key="lang_segmented_choice"
)
if chosen_lang_name and chosen_lang_name in lang_map:
    st.session_state["lang_code"] = lang_map[chosen_lang_name]
lang_code = st.session_state.get("lang_code", "en")

# Render unified sidebar branding and telemetry
render_sidebar(current_page="home", lang_code=lang_code)

# ─── LIVE WEATHER DATA FETCH ─────────────────────────────────────────────────
districts_list = list(DISTRICT_COORDINATES.keys())
selected_district = st.session_state.get("selected_district", "Mangaluru")
use_live_weather = st.session_state.get("use_live_weather", True)
forecast_data = None
fetch_success = False

coords = DISTRICT_COORDINATES[selected_district]
if use_live_weather:
    try:
        forecast_data = fetch_forecast(coords["lat"], coords["lon"], timeout=8.0)
        fetch_success = True
        hourly = forecast_data["hourly"]
        temp_c = float(hourly["temperature_2m"][0])
        humidity = float(hourly["relative_humidity_2m"][0])
        precip = float(hourly["precipitation"][0])
        precip_48h = float(sum(hourly["precipitation"][:48]))

        sm_arr = hourly.get("soil_moisture_0_to_1cm", [])
        if sm_arr and sm_arr[0] is not None:
            sm_val = float(sm_arr[0])
            soil_moist_display = f"{sm_val:.3f} m³/m³ ({sm_val * 100:.1f}%)"
            soil_moist_pct = float(sm_val * 100.0)
            soil_moist_available = True
        else:
            sm_val = None
            soil_moist_display = "not available"
            soil_moist_pct = 72.0
            soil_moist_available = False
    except Exception as e:
        fetch_success = False
        forecast_data = None
        temp_c, humidity, precip_48h, soil_moist_pct = 26.0, 75.0, 0.0, 72.0
        soil_moist_display = "not available"
        soil_moist_available = False
else:
    temp_c, humidity, precip_48h, soil_moist_pct = 26.0, 75.0, 0.0, 72.0
    soil_moist_display = f"{soil_moist_pct:.0f}%"
    soil_moist_available = True

# ─── TOP TABS NAVIGATION ─────────────────────────────────────────────────────
tab_home, tab_chat, tab_weather = st.tabs([
    f"🏠 {get_ui_string('tab_home', lang_code)}",
    f"💬 {get_ui_string('tab_chat', lang_code)}",
    f"🌦️ {get_ui_string('tab_weather', lang_code)}"
])

# ═════════════════════════════════════════════════════════════════════════════
# ─── VIEW 1: HOME TAB ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
with tab_home:
    if not st.session_state.get("view_result", False):
        # 1. Location Bar & Greeting
        dist_col, _ = st.columns([1.2, 1.0])
        with dist_col:
            selected_district = st.selectbox(
                "📍 " + get_ui_string("district_label", lang_code),
                districts_list,
                index=districts_list.index(selected_district) if selected_district in districts_list else 0,
                key="home_district_selector"
            )
            st.session_state["selected_district"] = selected_district

        # 2. Greeting & Weather summary
        col_greet, col_wx = st.columns([1.2, 1.0])
        with col_greet:
            st.markdown(f"""
            <div style="padding: 10px 0;">
                <h2 style="margin:0;color:#275637;font-size:22px;">{get_ui_string('greeting', lang_code)}</h2>
                <p style="margin:4px 0 0 0;color:#556B5C;font-size:14px;">Instant leaf pathology, ordinal severity & 48-hour preventive agronomy.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_wx:
            badge_text = get_ui_string('live_forecast_badge' if fetch_success else 'offline_badge', lang_code)
            status_word = get_ui_string('weather_status_rain' if precip_48h > 2.0 else ('weather_status_dry' if humidity < 40 else 'weather_status_normal'), lang_code)
            st.markdown(f"""
            <div class="ui-card" style="padding:12px 16px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:700;font-size:14px;color:#275637;">📍 {selected_district}</span>
                    <span style="font-size:12px;background:#91BA94;color:#275637;padding:2px 8px;border-radius:12px;font-weight:600;">{badge_text}</span>
                </div>
                <div style="margin-top:6px;font-size:14px;color:#275637;">
                    🌡️ <b>{temp_c:.1f}°C</b> &nbsp;|&nbsp; 🌧️ <b>{humidity:.0f}%</b> &nbsp;|&nbsp; 💧 <b>{precip_48h:.1f} mm</b>
                </div>
                <div style="margin-top:4px;font-size:12px;color:#556B5C;font-weight:600;">
                    {status_word}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Large Scan / Upload Photo Card
        st.markdown(f"""
        <div class="ui-card">
            <h3 style="margin:0;color:#275637;font-size:18px;">📷 {get_ui_string('scan_card_title', lang_code)}</h3>
            <p style="margin:4px 0 12px 0;font-size:13px;color:#556B5C;">ℹ️ {get_ui_string('tip_line', lang_code)}</p>
        </div>
        """, unsafe_allow_html=True)

        # Crop chooser buttons inside the card
        crop_c1, crop_c2, crop_c3 = st.columns(3)
        with crop_c1:
            if st.button("🌾 " + get_ui_string("crop_paddy", lang_code),
                         type="primary" if st.session_state.get("active_crop") == "Paddy" else "secondary",
                         key="btn_crop_paddy"):
                st.session_state["active_crop"] = "Paddy"
                st.rerun()
        with crop_c2:
            if st.button("🌴 " + get_ui_string("crop_arecanut", lang_code),
                         type="primary" if st.session_state.get("active_crop") == "Arecanut" else "secondary",
                         key="btn_crop_areca"):
                st.session_state["active_crop"] = "Arecanut"
                st.rerun()
        with crop_c3:
            if st.button("🍎 " + get_ui_string("crop_apple", lang_code),
                         type="primary" if st.session_state.get("active_crop") == "Apple" else "secondary",
                         key="btn_crop_apple"):
                st.session_state["active_crop"] = "Apple"
                st.rerun()

        missing_models = check_missing_models()
        if missing_models:
            st.markdown(f"""
            <div class="ui-card" style="border-left: 5px solid {THEME_CONFIG['warn_yellow']}; background:#FFFFFF; padding: 14px 18px; margin: 12px 0;">
                <div style="font-weight:600; font-size:15px; color:#275637;">
                    ℹ️ {get_ui_string('models_missing_msg', lang_code)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Camera input first, upload second
        cam_file = st.camera_input(get_ui_string("take_photo", lang_code), key="home_camera_input")
        up_file = st.file_uploader(get_ui_string("upload_file", lang_code), type=["jpg", "jpeg", "png"], key="home_file_input")

        input_file = cam_file if cam_file is not None else up_file

        # Process image when submitted
        if input_file is not None and not missing_models:
            active_crop = st.session_state.get("active_crop", "Paddy")
            input_image = Image.open(input_file).convert("RGB")
            image_title = getattr(input_file, "name", "Captured Photo")

            # ── 0. Multi-Crop OOD & Cross-Crop Guard Check
            crops_list = ["Paddy", "Arecanut", "Apple"]
            ood_ratios = {}
            for c in crops_list:
                _, sc, th = check_ood(input_image, c)
                ood_ratios[c] = sc / th

            sel_ratio = ood_ratios[active_crop]
            other_crops = [c for c in crops_list if c != active_crop]
            min_other_crop = min(other_crops, key=lambda c: ood_ratios[c])
            min_other_ratio = ood_ratios[min_other_crop]

            all_above_1 = all(ood_ratios[c] > 1.0 for c in crops_list)
            is_cross_crop = ((sel_ratio > 1.0 and min_other_ratio < 1.0) or (sel_ratio - min_other_ratio >= 0.25)) and not all_above_1

            if is_cross_crop:
                other_crop_loc = get_ui_string(f"crop_{min_other_crop.lower()}", lang_code)
                refusal_msg = get_ui_string("cross_crop_refusal_msg", lang_code).replace("{other_crop}", other_crop_loc)
                st.session_state["last_result"] = {
                    "is_cross_crop": True,
                    "suggested_crop": min_other_crop,
                    "is_ood": False,
                    "is_abstain": False,
                    "status": "unsure",
                    "crop": active_crop,
                    "display_name": f"Check Selection ({min_other_crop})",
                    "confidence": 0.0,
                    "organic_advice": refusal_msg,
                    "chemical_advice": "",
                    "weather_alert_text": None,
                    "date": format_local_date(datetime.date.today(), lang_code),
                    "image": input_image,
                    "image_title": image_title
                }
                st.session_state["scan_history"].append(st.session_state["last_result"])
                if len(st.session_state["scan_history"]) > 10:
                    st.session_state["scan_history"].pop(0)
                st.session_state["view_result"] = True
                st.session_state["current_scan_eval"] = {
                    "is_cross_crop": True, "is_ood": False, "is_abstain": False, "image": input_image, "title": image_title
                }
                st.rerun()

            is_ood = (sel_ratio > 1.0)

            if is_ood:
                st.session_state["last_result"] = {
                    "is_cross_crop": False,
                    "is_ood": True, "is_abstain": False, "status": "unsure",
                    "crop": active_crop, "display_name": "Unknown / Out-of-Distribution",
                    "confidence": 0.0, "organic_advice": get_ui_string("refusal_ood_msg", lang_code),
                    "chemical_advice": "", "weather_alert_text": None, "date": format_local_date(datetime.date.today(), lang_code),
                    "image": input_image,
                    "image_title": image_title
                }
                st.session_state["scan_history"].append(st.session_state["last_result"])
                if len(st.session_state["scan_history"]) > 10:
                    st.session_state["scan_history"].pop(0)
                st.session_state["view_result"] = True
                st.session_state["current_scan_eval"] = {
                    "is_cross_crop": False, "is_ood": True, "is_abstain": False, "image": input_image, "title": image_title
                }
                st.rerun()

            # ── 1. Disease Classification
            pred_cls, dis_conf, dis_probs, gradcam_img, display_name, clf_available = \
                run_disease_classification(input_image, active_crop)

            # ── 2. Abstain Check
            abstain_tau = ABSTAIN_THRESHOLDS.get(active_crop, 0.70)
            if dis_conf < abstain_tau:
                st.session_state["last_result"] = {
                    "is_ood": False, "is_abstain": True, "status": "unsure",
                    "crop": active_crop, "display_name": "Withheld (Low Confidence)",
                    "confidence": dis_conf, "organic_advice": get_ui_string("refusal_abstain_msg", lang_code),
                    "chemical_advice": "", "weather_alert_text": None, "date": format_local_date(datetime.date.today(), lang_code)
                }
                st.session_state["scan_history"].append(st.session_state["last_result"])
                if len(st.session_state["scan_history"]) > 10:
                    st.session_state["scan_history"].pop(0)
                st.session_state["view_result"] = True
                st.session_state["current_scan_eval"] = {
                    "is_ood": False, "is_abstain": True, "image": input_image, "title": image_title, "conf": dis_conf
                }
                st.rerun()

            # ── 3. Un-Abstained Successful Diagnosis
            is_healthy = (pred_cls in ["normal", "healthy", "healthy_foot", "healthy_leaf", "healthy_nut", "healthy_trunk"]
                          or display_name.startswith("Healthy"))

            severity_arch = st.session_state.get("severity_arch", "Ordinal CORN")
            if active_crop != "Apple":
                if is_healthy:
                    grade = 0
                    sev_conf = 1.0
                    sev_probs = [1.0, 0.0, 0.0, 0.0]
                    cv_ratio = 0.0
                    lesion_overlay = None
                elif active_crop == "Paddy" and pred_cls == "hispa":
                    grade = 1
                    sev_conf = 1.0
                    sev_probs = [0.0, 1.0, 0.0, 0.0]
                    cv_ratio = 0.0
                    lesion_overlay = None
                elif active_crop == "Paddy" and pred_cls == "dead_heart":
                    grade = 3
                    sev_conf = 1.0
                    sev_probs = [0.0, 0.0, 0.0, 1.0]
                    cv_ratio = 0.0
                    lesion_overlay = None
                else:
                    grade, sev_conf, sev_probs, cv_ratio, lesion_overlay = \
                        run_severity(input_image, active_crop, severity_arch)
                    if grade == 0:
                        grade = 1
            else:
                grade = 0
                sev_conf = 0.0
                sev_probs = [0.0] * 4
                cv_ratio = 0.0
                lesion_overlay = None

            # ── 4. Bioclimatic Risk Evaluation
            risk_soil_pct = 50.0 if (use_live_weather and fetch_success) else soil_moist_pct
            risk_score, risk_level, env_alert, action_modifier = evaluate_environmental_risk(
                crop=active_crop, disease_name=display_name, temp_c=temp_c, soil_moisture_pct=risk_soil_pct, humidity_pct=humidity
            )

            # ── 5. 48-Hour Alert Gating
            weather_alert_res = None
            weather_spoken_alert = ""
            if (not is_healthy) and use_live_weather and fetch_success and (forecast_data is not None):
                weather_alert_res = compute_48h_disease_risk(forecast_data, active_crop, display_name, pred_cls=pred_cls)
                if weather_alert_res is not None:
                    weather_spoken_alert = weather_alert_res["voice_alerts"].get(lang_code, weather_alert_res["voice_alerts"]["en"])

            # ── 6. Treatment Recommendations
            crop_tx = TREATMENT_DATABASE.get(active_crop, {})
            dis_tx  = crop_tx.get(display_name, {})
            if active_crop == "Apple":
                if is_healthy:
                    title_tx, desc_tx = dis_tx.get(0, ("Healthy Foliage", "No treatment needed."))
                    org_steps = "No treatment needed."
                    chem_steps = "No chemical treatment needed."
                else:
                    title_tx, desc_tx = dis_tx.get("disease_advice", ("Apple Scab Management", "Follow local KVK advice."))
                    # Split into cultural/organic and chemical
                    if "Apply protective" in desc_tx:
                        parts = desc_tx.split("Apply protective")
                        org_steps = parts[0].strip()
                        chem_steps = "Apply protective " + parts[1].strip()
                    elif "Apply" in desc_tx:
                        parts = desc_tx.split("Apply")
                        org_steps = parts[0].strip()
                        chem_steps = "Apply " + parts[1].strip()
                    else:
                        org_steps = desc_tx
                        chem_steps = "Consult your KVK for registered chemical treatments."
            else:
                title_tx, desc_tx = dis_tx.get(grade, ("Action Plan", "Follow local KVK advice."))
                if is_healthy:
                    org_steps = "No treatment needed. Maintain optimal water and balanced fertilizer."
                    chem_steps = "No chemical treatment needed."
                elif pred_cls in ["tungro", "yellow_leaf_disease"]:
                    org_steps = "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK."
                    chem_steps = "No chemical cure exists for viral/phytoplasma infections."
                else:
                    g1_tx = dis_tx.get(1, ("", "Start with cultural steps and bio-fungicides."))[1]
                    g_hi_tx = dis_tx.get(max(2, grade), ("", "Apply registered systemic treatment under KVK advice."))[1]
                    org_steps = g1_tx
                    chem_steps = g_hi_tx

            # ── 7. Soil Health Card
            soil_info = SOIL_HEALTH_ADVISORY.get(active_crop, SOIL_HEALTH_ADVISORY["Paddy"])
            soil_title, soil_desc = soil_info.get(lang_code, soil_info["en"])

            # ── 8. Multilingual Spoken Advisory
            advisory_text = build_voice_advisory(active_crop, display_name, pred_cls, grade, lang_code, weather_spoken_alert)

            # Determine traffic light status
            if is_healthy:
                status_type = "healthy"
            elif pred_cls in ["tungro", "yellow_leaf_disease"]:
                status_type = "no_cure"
            else:
                status_type = "treat"

            scan_record = {
                "is_ood": False,
                "is_abstain": False,
                "status": status_type,
                "crop": active_crop,
                "pred_cls": pred_cls,
                "display_name": display_name,
                "confidence": dis_conf,
                "dis_probs": dis_probs,
                "grade": grade,
                "sev_conf": sev_conf,
                "cv_ratio": cv_ratio,
                "gradcam_img": gradcam_img,
                "lesion_overlay": lesion_overlay,
                "organic_advice": org_steps,
                "chemical_advice": chem_steps,
                "action_modifier": action_modifier,
                "weather_alert_res": weather_alert_res,
                "weather_alert_text": weather_alert_res["alert_text"] if weather_alert_res else None,
                "soil_title": soil_title,
                "soil_desc": soil_desc,
                "advisory_text": advisory_text,
                "date": format_local_date(datetime.date.today(), lang_code),
                "tungro_note": TUNGRO_SAFETY_NOTE.get(lang_code, TUNGRO_SAFETY_NOTE["en"]) if (active_crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]) else None,
                "cedar_note": get_ui_string("cedar_rust_caption", lang_code) if (active_crop == "Apple" and pred_cls == "cedar_apple_rust") else None,
                "image": input_image,
                "image_title": image_title
            }
            st.session_state["last_result"] = scan_record
            st.session_state["scan_history"].append(scan_record)
            if len(st.session_state["scan_history"]) > 10:
                st.session_state["scan_history"].pop(0)
            st.session_state["view_result"] = True
            st.rerun()

        # 4. "Your Last Result" Card
        scan_hist = st.session_state.get("scan_history", [])
        if not scan_hist:
            st.markdown(f"""
            <div class="ui-card">
                <h4 style="margin:0;color:#275637;font-size:16px;">📋 {get_ui_string('last_result_title', lang_code)}</h4>
                <p style="margin:8px 0 0 0;font-size:14px;color:#556B5C;">{get_ui_string('no_scans_yet', lang_code)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            latest = scan_hist[-1]
            st.markdown(f"""
            <div class="ui-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h4 style="margin:0;color:#275637;font-size:16px;">📋 {get_ui_string('last_result_title', lang_code)}</h4>
                    <span style="font-size:12px;color:#556B5C;">{latest.get('date', '')}</span>
                </div>
                <div style="margin-top:8px;font-size:15px;color:#275637;">
                    <b>{get_ui_string('crop_label', lang_code)}:</b> {latest.get('crop', '')} &nbsp;|&nbsp;
                    <b>{latest.get('display_name', '')}</b> &nbsp;|&nbsp;
                    <b>{get_ui_string('confidence_label', lang_code)}:</b> {latest.get('confidence', 0.0):.1%}
                </div>
                <div style="margin-top:6px;font-size:14px;color:#556B5C;">
                    <b>{get_ui_string('recommendation_label', lang_code)}:</b> {latest.get('organic_advice', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ═════════════════════════════════════════════════════════════════════
        # ─── RESULT SCREEN (AFTER PHOTO IS SUBMITTED) ─────────────────────────
        # ═════════════════════════════════════════════════════════════════════
        if st.button(get_ui_string("back_btn", lang_code), key="btn_back_home"):
            st.session_state["view_result"] = False
            st.rerun()

        res = st.session_state.get("last_result")
        if res is not None:
            # ── A. REFUSAL STATES (CROSS-CROP / OOD / ABSTAIN)
            if res.get("is_cross_crop"):
                other_crop_loc = get_ui_string(f"crop_{res.get('suggested_crop', 'Paddy').lower()}", lang_code)
                refusal_text = get_ui_string("cross_crop_refusal_msg", lang_code).replace("{other_crop}", other_crop_loc)
                st.markdown(f"""
                <div class="ui-card" style="border-left: 5px solid #888888; background:#F9F9F6; padding: 18px 20px; margin: 16px 0;">
                    <div style="font-size:28px;margin-bottom:8px;">⚠️</div>
                    <h3 style="margin:0;color:#275637;font-size:18px;">{get_ui_string('cross_crop_refusal_title', lang_code)}</h3>
                    <p style="margin:10px 0 0 0;font-size:15px;color:#556B5C;line-height:1.5;">
                        {refusal_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if res.get("image") is not None:
                    st.image(res["image"], caption=res.get("image_title", "Uploaded Image"), width=360)

            elif res.get("is_ood"):
                st.markdown(f"""
                <div class="refusal-card">
                    <div style="font-size:32px;margin-bottom:8px;">⚠️</div>
                    <h3 style="margin:0;color:#275637;font-size:18px;">{get_ui_string('refusal_ood_title', lang_code)}</h3>
                    <p style="margin:10px 0 0 0;font-size:14px;color:#556B5C;line-height:1.5;">
                        {get_ui_string('refusal_ood_msg', lang_code)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if res.get("image") is not None:
                    st.image(res["image"], caption=res.get("image_title", "Uploaded Image"), width=360)

            elif res.get("is_abstain"):
                st.markdown(f"""
                <div class="refusal-card">
                    <div style="font-size:32px;margin-bottom:8px;">⚠️</div>
                    <h3 style="margin:0;color:#275637;font-size:18px;">{get_ui_string('refusal_abstain_title', lang_code)}</h3>
                    <p style="margin:10px 0 0 0;font-size:14px;color:#556B5C;line-height:1.5;">
                        {get_ui_string('refusal_abstain_msg', lang_code)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if res.get("image") is not None:
                    st.image(res["image"], caption=res.get("image_title", "Uploaded Image"), width=360)

            else:
                # ── B. SUCCESSFUL SCAN RESULT SCREEN
                status = res.get("status", "healthy")
                if status == "healthy":
                    badge_cls, badge_icon, badge_word = "badge-healthy", "🟢", get_ui_string("status_healthy", lang_code)
                elif status == "no_cure":
                    badge_cls, badge_icon, badge_word = "badge-no-cure", "🔴", get_ui_string("status_no_cure", lang_code)
                else:
                    badge_cls, badge_icon, badge_word = "badge-treat", "🟡", get_ui_string("status_treat", lang_code)

                # 1. Status Banner
                st.markdown(f"""
                <div class="badge-status {badge_cls}">
                    {badge_icon} {badge_word}
                </div>
                """, unsafe_allow_html=True)

                # 2. Disease Name (Localized large, English small below)
                active_crop = res.get("crop", "Paddy")
                pred_cls = res.get("pred_cls", "normal")
                display_name = res.get("display_name", "Disease")
                loc_name = DISEASE_LOCALIZATION.get(active_crop, {}).get(pred_cls, {}).get(lang_code, display_name)

                if lang_code != "en" and loc_name != display_name:
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <h2 style="margin:0;color:#275637;font-size:24px;">{loc_name}</h2>
                        <div style="font-size:14px;color:#556B5C;">{display_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <h2 style="margin:0;color:#275637;font-size:24px;">{display_name}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Confidence Bar
                dis_conf = res.get("confidence", 0.0)
                st.progress(dis_conf, text=f"{get_ui_string('confidence_label', lang_code)}: {dis_conf:.1%}")

                # 3b. Severity Card
                res_grade = res.get("grade", 0)
                sev_conf_val = res.get("sev_conf", 1.0)
                cv_ratio_val = res.get("cv_ratio", 0.0)
                if active_crop != "Apple" and status != "healthy":
                    grade_labels = {
                        1: "Grade 1: Mild Severity",
                        2: "Grade 2: Moderate Severity",
                        3: "Grade 3: Severe Severity"
                    }
                    grade_str = grade_labels.get(res_grade, f"Grade {res_grade}")
                    st.markdown(f"""
                    <div class="ui-card" style="padding:12px 16px;margin:10px 0;">
                        <div style="font-weight:700;font-size:15px;color:#275637;">📊 {grade_str}</div>
                        <div style="font-size:13px;color:#556B5C;margin-top:2px;">
                            Confidence: {sev_conf_val:.1%} &nbsp;|&nbsp; Estimated Lesion Area: {cv_ratio_val:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if active_crop == "Arecanut":
                        st.caption(get_ui_string("areca_provisional", lang_code))
                    elif active_crop == "Paddy":
                        if pred_cls in ["hispa", "dead_heart"]:
                            st.caption(get_ui_string("paddy_severity_fixed", lang_code))
                        elif pred_cls not in ["normal"]:
                            st.caption(get_ui_string("paddy_severity_qwk", lang_code))

                # 4. Mandatory Captions & Warnings
                if active_crop == "Apple":
                    st.markdown(f"""
                    <div style="font-size:13px;color:#B5483A;background:#FFFFFF;border:1px solid #91BA94;border-radius:10px;padding:8px 12px;margin:10px 0;">
                        {get_ui_string('apple_public_banner', lang_code)}
                    </div>
                    """, unsafe_allow_html=True)

                if active_crop == "Apple" and pred_cls == "cedar_apple_rust":
                    st.caption(get_ui_string("cedar_rust_caption", lang_code))

                if active_crop == "Arecanut" and pred_cls in ["bud_borer", "yellow_leaf_disease"]:
                    st.markdown(f"""
                    <div style="font-size:13px;color:#275637;background:#F6D05F;border-radius:10px;padding:8px 12px;margin:10px 0;font-weight:600;">
                        {get_ui_string('areca_bud_borer_warning', lang_code)}
                    </div>
                    """, unsafe_allow_html=True)

                if active_crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    tungro_txt = res.get("tungro_note") or TUNGRO_SAFETY_NOTE.get(lang_code, TUNGRO_SAFETY_NOTE["en"])
                    st.markdown(f"""
                    <div style="font-size:13px;color:#275637;background:#F6D05F;border-radius:10px;padding:8px 12px;margin:10px 0;">
                        ℹ️ <b>{get_ui_string('tungro_warning_title', lang_code)}:</b> {tungro_txt}
                    </div>
                    """, unsafe_allow_html=True)

                # 5. "What to do" with organic steps first and chemical in expander
                st.markdown(f"""
                <div class="ui-card">
                    <h3 style="margin:0;color:#275637;font-size:17px;">📋 {get_ui_string('what_to_do', lang_code)}</h3>
                    <div style="margin-top:10px;font-size:15px;color:#275637;line-height:1.5;">
                        <b>{get_ui_string('organic_title', lang_code)}:</b> {res.get('organic_advice', '')}
                    </div>
                    {f'<div style="margin-top:8px;font-size:14px;color:#275637;font-weight:600;">{res.get("action_modifier", "")}</div>' if res.get("action_modifier") else ''}
                </div>
                """, unsafe_allow_html=True)

                # Chemical option inside expander
                with st.expander(get_ui_string("severe_expander", lang_code), expanded=False):
                    st.markdown(f"""
                    <div style="font-size:15px;color:#275637;line-height:1.5;">
                        {res.get('chemical_advice', '')}
                    </div>
                    <div style="margin-top:10px;font-size:12px;color:#556B5C;font-style:italic;">
                        {get_ui_string('footer_disclaimer_tx', lang_code)}
                    </div>
                    """, unsafe_allow_html=True)

                # 6. 48-Hour Alert Strip
                weather_alert_res = res.get("weather_alert_res")
                if weather_alert_res is not None:
                    rcat = weather_alert_res["risk_cat"]
                    border_color = THEME_CONFIG['danger'] if rcat == 'High' else THEME_CONFIG['warn_yellow']
                    st.markdown(f"""
                    <div class="ui-card" style="border-left: 5px solid {border_color};">
                        <h4 style="margin:0;color:#275637;font-size:16px;">⏱️ {get_ui_string('weather_alert_title', lang_code)} ({rcat} Risk)</h4>
                        <p style="margin:6px 0 0 0;font-size:14px;color:#275637;line-height:1.5;">
                            {weather_alert_res['alert_text']}
                        </p>
                        <p style="margin:6px 0 0 0;font-size:11px;color:#556B5C;font-style:italic;">
                            ⚠️ {get_ui_string('rule_based_tag', lang_code)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if use_live_weather and fetch_success:
                        st.markdown(f"""
                        <div class="ui-card" style="padding:10px 14px;">
                            <span style="font-size:13px;color:#556B5C;font-style:italic;">ℹ️ No weather rule for this disease.</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ui-card" style="padding:10px 14px;">
                            <span style="font-size:13px;color:#556B5C;font-style:italic;">ℹ️ 48-hour alert needs the live forecast.</span>
                        </div>
                        """, unsafe_allow_html=True)

                # 7. Soil Card
                st.markdown(f"""
                <div class="ui-card">
                    <h4 style="margin:0;color:#275637;font-size:16px;">🌱 {res.get('soil_title', '')}</h4>
                    <p style="margin:8px 0 0 0;font-size:14px;color:#275637;line-height:1.5;">
                        {res.get('soil_desc', '')}
                    </p>
                    <div style="margin-top:8px;font-size:12px;color:#556B5C;font-style:italic;">
                        {get_ui_string('footer_disclaimer_soil', lang_code)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 8. Collapsed "For experts and judges" Expander
                st.markdown("---")
                with st.expander("🔬 " + get_ui_string("experts_expander", lang_code), expanded=False):
                    # Trust & Audit Panel
                    st.markdown(f"#### 🛡️ {get_ui_string('trust_audit_title', lang_code)}")
                    for k, v in TRUST_AND_AUDIT.items():
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

                    st.markdown("---")
                    # Explainability Panels
                    st.markdown("#### 🖼️ Explainability & Attention Panels")
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        if res.get("gradcam_img") is not None:
                            st.markdown("**Attention Heatmap (Grad-CAM)**")
                            st.image(res["gradcam_img"], use_container_width=True)
                            st.caption(f"Grad-CAM Heatmap — red areas show features driving model decision for {display_name}. (Note: Pointing accuracy is only 5–14% on leaf pathology; heatmaps indicate salient regions rather than verified disease localization).")
                    with exp_col2:
                        if res.get("lesion_overlay") is not None:
                            st.markdown("**Color-Threshold Lesion Mask**")
                            st.image(res["lesion_overlay"], use_container_width=True)
                            cv_ratio = res.get("cv_ratio", 0.0)
                            st.caption(f"Lesion Mask — {cv_ratio:.1%} estimated lesion area.")

                    # Paddy & Arecanut severity captions
                    if active_crop == "Arecanut":
                        st.caption(get_ui_string("areca_provisional", lang_code))
                    elif active_crop == "Paddy":
                        if pred_cls in ["hispa", "dead_heart"]:
                            st.caption(get_ui_string("paddy_severity_fixed", lang_code))
                        elif pred_cls not in ["normal"]:
                            st.caption(get_ui_string("paddy_severity_qwk", lang_code))

                    st.markdown("---")
                    # Probability Distribution
                    st.markdown("#### 📊 Disease Probability Distribution")
                    dis_probs = res.get("dis_probs", {})
                    if dis_probs:
                        disp_map = PADDY_DISPLAY if active_crop == "Paddy" else (APPLE_DISPLAY if active_crop == "Apple" else ARECA_DISPLAY)
                        df_dis = pd.DataFrame({
                            "Disease": [disp_map.get(c, c) for c in dis_probs.keys()],
                            "Probability": list(dis_probs.values())
                        }).sort_values("Probability", ascending=False)
                        st.bar_chart(df_dis.set_index("Disease"), height=180)

                    st.markdown("---")
                    # Severity Architecture Selector & Manual Weather Sliders
                    st.markdown("#### ⚙️ Expert Controls")
                    st.session_state["severity_arch"] = st.selectbox(
                        "Severity AI Architecture", ["Ordinal CORN", "Baseline Softmax"],
                        index=0 if st.session_state.get("severity_arch") == "Ordinal CORN" else 1
                    )
                    st.session_state["use_live_weather"] = st.toggle("Use live forecast", value=st.session_state.get("use_live_weather", True))
                    if not st.session_state["use_live_weather"]:
                        _ = st.slider("Manual Temperature (°C)", 10.0, 45.0, 26.0, 0.5)
                        _ = st.slider("Manual Soil Moisture (%)", 10.0, 100.0, 72.0, 1.0)
                        _ = st.slider("Manual Humidity (%)", 20.0, 100.0, 75.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# ─── VIEW 2: AI CHAT TAB (GROUNDED, NO LLM) ───────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(f"""
    <div style="padding: 6px 0 12px 0;">
        <h2 style="margin:0;color:#275637;font-size:20px;">💬 {get_ui_string('chat_title', lang_code)}</h2>
    </div>
    """, unsafe_allow_html=True)

    last_scan = st.session_state.get("last_result")

    # If no scan yet, show initial grounding notice
    if not last_scan:
        st.markdown(f"""
        <div class="chat-bubble-ai">
            🤖 {get_ui_string('chat_no_context', lang_code)}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show render of chat message history
        for msg in st.session_state.get("chat_history", []):
            if msg["role"] == "user":
                st.markdown(f"""<div class="chat-bubble-user">🧑‍🌾 {msg['text']}</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-bubble-ai">🤖 {msg['text']}</div>""", unsafe_allow_html=True)

    # Quick reply chips
    st.markdown("<p style='font-size:13px;color:#556B5C;margin:12px 0 6px 0;font-weight:600;'>Quick Questions:</p>", unsafe_allow_html=True)
    chip_keys = [
        ("chip_first", get_ui_string("chip_first", lang_code)),
        ("chip_cure", get_ui_string("chip_cure", lang_code)),
        ("chip_spray", get_ui_string("chip_spray", lang_code)),
        ("chip_soil", get_ui_string("chip_soil", lang_code)),
        ("chip_weather", get_ui_string("chip_weather", lang_code)),
        ("chip_unsure", get_ui_string("chip_unsure", lang_code)),
    ]

    # Render chips as 2 rows of 3 buttons
    r1_cols = st.columns(3)
    for i in range(3):
        k, lbl = chip_keys[i]
        with r1_cols[i]:
            if st.button(lbl, key=f"btn_chip_{k}"):
                st.session_state["chat_history"].append({"role": "user", "text": lbl})
                ans, _ = build_chat_response(lbl, last_scan, lang=lang_code)
                st.session_state["chat_history"].append({"role": "ai", "text": ans})
                st.rerun()

    r2_cols = st.columns(3)
    for i in range(3, 6):
        k, lbl = chip_keys[i]
        with r2_cols[i - 3]:
            if st.button(lbl, key=f"btn_chip_{k}"):
                st.session_state["chat_history"].append({"role": "user", "text": lbl})
                ans, _ = build_chat_response(lbl, last_scan, lang=lang_code)
                st.session_state["chat_history"].append({"role": "ai", "text": ans})
                st.rerun()

    # Free-form chat input
    user_query = st.chat_input(get_ui_string("chat_placeholder", lang_code), key="chat_freeform_input")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "text": user_query})
        ans, _ = build_chat_response(user_query, last_scan, lang=lang_code)
        st.session_state["chat_history"].append({"role": "ai", "text": ans})
        st.rerun()

    # Fixed disclaimer and Clear Chat button
    st.markdown(f"""
    <div style="font-size:12px;color:#556B5C;margin-top:16px;font-style:italic;">
        ⚠️ {get_ui_string('chat_disclaimer', lang_code)}
    </div>
    """, unsafe_allow_html=True)

    if st.button(get_ui_string("chat_clear_btn", lang_code), key="btn_clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ─── VIEW 3: WEATHER VIEW ────────────────═════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
with tab_weather:
    st.markdown(f"""
    <div style="padding: 6px 0 12px 0;">
        <h2 style="margin:0;color:#275637;font-size:20px;">🌦️ {get_ui_string('weather_view_title', lang_code)}</h2>
    </div>
    """, unsafe_allow_html=True)

    # District Selector & Crop Chooser for Weather Risk
    wx_c1, wx_c2 = st.columns([1.2, 1.0])
    with wx_c1:
        wx_district = st.selectbox(
            "📍 " + get_ui_string("district_label", lang_code),
            districts_list,
            index=districts_list.index(selected_district) if selected_district in districts_list else 0,
            key="wx_view_district"
        )
    with wx_c2:
        wx_crop = st.selectbox(
            "🌱 " + get_ui_string("choose_crop", lang_code),
            ["Paddy", "Arecanut", "Apple"],
            index=["Paddy", "Arecanut", "Apple"].index(st.session_state.get("active_crop", "Paddy")),
            key="wx_view_crop"
        )

    # Fetch forecast for Weather view
    wx_coords = DISTRICT_COORDINATES[wx_district]
    try:
        wx_forecast = fetch_forecast(wx_coords["lat"], wx_coords["lon"], timeout=8.0)
        wx_hourly = wx_forecast["hourly"]
        wx_success = True
    except Exception:
        wx_forecast = None
        wx_success = False

    if not wx_success or wx_forecast is None:
        st.warning(f"⚠️ Unable to fetch live weather forecast for {wx_district}.")
    else:
        # 48-Hour Forecast Strip in 6-Hour Blocks
        st.markdown(f"#### ⏱️ {get_ui_string('weather_6h_blocks', lang_code)}")

        # 8 blocks of 6 hours
        b_cols1 = st.columns(4)
        b_cols2 = st.columns(4)

        for b_idx in range(8):
            start_h = b_idx * 6
            end_h = start_h + 6
            b_temp = np.mean(wx_hourly["temperature_2m"][start_h:end_h])
            b_rh = np.mean(wx_hourly["relative_humidity_2m"][start_h:end_h])
            b_rain = np.sum(wx_hourly["precipitation"][start_h:end_h])

            target_col = b_cols1[b_idx] if b_idx < 4 else b_cols2[b_idx - 4]
            with target_col:
                st.markdown(f"""
                <div class="weather-block-card">
                    <div style="font-weight:700;font-size:12px;color:#275637;">+{start_h}h..+{end_h}h</div>
                    <div style="font-size:13px;color:#275637;margin-top:4px;">🌡️ {b_temp:.1f}°C</div>
                    <div style="font-size:12px;color:#556B5C;">🌧️ {b_rh:.0f}%</div>
                    <div style="font-size:12px;color:#275637;font-weight:600;">💧 {b_rain:.1f}mm</div>
                </div>
                """, unsafe_allow_html=True)

        # Crop Weather Risk Analysis
        st.markdown("---")
        st.markdown(f"#### 🌾 {get_ui_string('crop_weather_risk', lang_code)}: {wx_crop}")

        # Compute rule-based hours of favourable weather
        if wx_crop == "Paddy":
            dis_key, dis_name = "blast", "Rice Blast"
        elif wx_crop == "Arecanut":
            dis_key, dis_name = "mahali_koleroga", "Mahali / Koleroga (Rot Disease)"
        else:
            dis_key, dis_name = "apple_scab", "Apple Scab"

        crop_risk_res = compute_48h_disease_risk(wx_forecast, wx_crop, dis_name, pred_cls=dis_key)

        if crop_risk_res is not None:
            rcat = crop_risk_res["risk_cat"]
            m_cnt = crop_risk_res["matched_count"]
            r_border = THEME_CONFIG['danger'] if rcat == 'High' else THEME_CONFIG['warn_yellow']
            st.markdown(f"""
            <div class="ui-card" style="border-left: 5px solid {r_border};">
                <h4 style="margin:0;color:#275637;font-size:16px;">
                    {crop_risk_res['rule_name']} ({rcat} Risk — {m_cnt} Favourable Hours in Next 48h)
                </h4>
                <p style="margin:8px 0 0 0;font-size:14px;color:#275637;line-height:1.5;">
                    {crop_risk_res['alert_text']}
                </p>
                <p style="margin:6px 0 0 0;font-size:11px;color:#556B5C;font-style:italic;">
                    ⚠️ {get_ui_string('rule_based_tag', lang_code)}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ui-card">
                <p style="margin:0;font-size:14px;color:#556B5C;">
                    {get_ui_string('no_rule_for_crop', lang_code)}
                </p>
            </div>
            """, unsafe_allow_html=True)
