import sys
sys.path.insert(0, '/home/sahith/Projects/Hackathon')
import config

import logging
import streamlit as st
import os, glob, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from gtts import gTTS

from src.model import CropSeverityModel
from src.corn_loss import corn_probs_to_grade
from src.segmentation import segment_leaf_adaptive, detect_lesions_group_a, compute_ratio
from src.severity_grader import ratio_to_grade
from src.multimodal_fusion import evaluate_environmental_risk, MultimodalCropFusionNet
from ood_guard import check_ood
from weather_live import DISTRICT_COORDINATES, fetch_forecast, compute_48h_disease_risk

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriTrust AI — Multimodal Crop Disease & Severity Engine",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS STYLING ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); color: #f8fafc; }
    .upload-card { background: rgba(255,255,255,0.04); border: 2px dashed rgba(56,189,248,0.4);
        border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 24px; }
    .disease-card { background: rgba(15,23,42,0.7); border: 1px solid rgba(56,189,248,0.35);
        border-radius: 16px; padding: 18px 22px; margin-bottom: 18px; }
    .multimodal-card { background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.22));
        border: 1px solid rgba(139,92,246,0.5); border-radius: 16px; padding: 20px; margin-bottom: 18px; }
    .pesticide-card { background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
        border: 1px solid rgba(16,185,129,0.5); border-radius: 16px; padding: 20px; margin-top: 18px; }
    .result-card-g0 { background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.25));
        border: 1px solid rgba(34,197,94,0.4); border-radius: 16px; padding: 20px; margin-bottom: 16px; }
    .result-card-g1 { background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(202,138,4,0.25));
        border: 1px solid rgba(234,179,8,0.4); border-radius: 16px; padding: 20px; margin-bottom: 16px; }
    .result-card-g2 { background: linear-gradient(135deg, rgba(249,115,22,0.15), rgba(234,88,12,0.25));
        border: 1px solid rgba(249,115,22,0.4); border-radius: 16px; padding: 20px; margin-bottom: 16px; }
    .result-card-g3 { background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.25));
        border: 1px solid rgba(239,68,68,0.4); border-radius: 16px; padding: 20px; margin-bottom: 16px; }
    .metric-badge { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px; padding: 14px; text-align: center; margin-bottom: 8px; }
    .metric-val { font-size: 24px; font-weight: 700; color: #38bdf8; }
    .metric-lbl { font-size: 13px; color: #94a3b8; margin-top: 4px; }
    .explainer-note { font-size: 12px; color: #64748b; font-style: italic; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

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
# Each entry is explicitly tagged for native-speaker agricultural review.
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
            "hi": "सीडर एप्पल रस्ट (गेरुआ रोग)",  # REVIEW (Hindi): सीडर एप्पल रस्ट (Cedar Apple Rust)
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


# TODO: Review voice advisory templates with native Kannada and Hindi speakers before field deployment.
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
                return f"ಬೆಳೆ: {crop_lbl}. ಎಲೆಗಳು ಆರೋಗ್ಯಕರವಾಗಿವೆ ಮತ್ತು ಯಾವುದೇ ರೋಗದ ಲಕ್ಷಣಗಳಿಲ್ಲ. ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। पत्तियां स्वस्थ हैं और किसी रोग के लक्षण नहीं हैं। किसी उपचार की आवश्यकता नहीं है।"
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
                return f"ಬೆಳೆ: {crop_lbl}. ಎಚ್ಚರಿಕೆ: {dis_lbl} ಕಂಡುಬಂದಿದೆ. ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ, ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। चेतावनी: {dis_lbl} का संक्रमण पाया गया है। इसका कोई रासायनिक उपचार उपलब्ध नहीं है। रसचूसक कीटों को नियंत्रित करें, रोगी पौधों को हटाएं और तुरंत कृषि विज्ञान केंद्र से संपर्क करें।"
            else:
                return f"Crop: {crop_lbl}. Warning: {disease_name} detected. No chemical cure exists. Manage insect vectors, remove infected plants, and consult your local KVK."

        # 4. Treatable disease (Paddy & Arecanut): organic steps first; chemical only if grade >= 2
        if grade_idx < 2:
            if lang == "kn":
                return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಲಘು ತೀವ್ರತೆ. ಮೊದಲು ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ: ರೋಗಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ತೋಟವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ. ರಾಸಾಯನಿಕ ಬಳಸುವ ಮುನ್ನ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। रोग: {dis_lbl}। शुरुआती अवस्था। पहले जैविक और पारंपरिक उपाय अपनाएं: संक्रमित पत्तियों को हटाएं और खेत साफ रखें। रसायन छिड़कने से पहले केवीके से सलाह लें।"
            else:
                return f"Crop: {crop_lbl}. Disease: {disease_name}. Mild severity. Start with organic and cultural steps: remove infected leaves and maintain field sanitation. Consult your local KVK before using chemicals."
        else:
            if lang == "kn":
                return f"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ತೀವ್ರತೆಯು ಹೆಚ್ಚಾಗಿದೆ. ಸಾವಯವ ಕ್ರಮಗಳು ಮಾತ್ರ ಸಾಕಾಗುವುದಿಲ್ಲ. ಶಿಫಾರಸು ಮಾಡಿದ ಔಷಧಿಗಳಿಗಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಲೇಬಲ್ ಸೂಚನೆ ಪಾಲಿಸಿ."
            elif lang == "hi":
                return f"फसल: {crop_lbl}। रोग: {dis_lbl}। संक्रमण गंभीर है। केवल जैविक उपाय पर्याप्त नहीं हैं। अनुशंसित दवाओं के लिए अपने नजदीकी कृषि विज्ञान केंद्र से संपर्क करें और लेबल के अनुसार छिड़काव करें।"
            else:
                return f"Crop: {crop_lbl}. Disease: {disease_name}. Moderate to severe infection. Cultural measures alone are insufficient. Consult your local KVK for registered chemical treatments and follow product label advice."


    base_text = _get_base_advisory()
    if weather_sentence:
        return f"{base_text} {weather_sentence}".strip()
    return base_text

# ─── LOAD DISEASE CLASSIFIER ─────────────────────────────────────────────────
@st.cache_resource
def get_disease_classifier(crop_name: str):
    key = crop_name.lower()
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

# ─── SIDEBAR & IOT SENSOR INPUTS ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=64)
    st.title("AgriTrust AI Controls")

    crop_mode = st.selectbox("🌱 Select Crop", ["Paddy", "Arecanut", "Apple"])
    model_type_choice = st.selectbox("🧠 Severity AI Architecture", [
        "Ordinal CORN", "Baseline Softmax"
    ])
    voice_lang = st.selectbox("🎙️ Voice Advisory Language", [
        "English (en)", "Hindi (hi)", "Kannada (kn)"
    ])
    
    st.markdown("---")
    st.subheader("📝 Field conditions (entered by user)")
    
    st.markdown("---")
    st.subheader("🌦️ Weather & Field Conditions")

    selected_district = st.selectbox("📍 District", [
        "Mangaluru", "Udupi", "Shivamogga", "Mysuru",
        "Srinagar", "Shimla", "Kullu", "Dehradun"
    ])
    use_live_weather = st.toggle("🛰️ Use live forecast", value=True)

    forecast_data = None
    fetch_success = False

    if use_live_weather:
        coords = DISTRICT_COORDINATES[selected_district]
        try:
            forecast_data = fetch_forecast(coords["lat"], coords["lon"], timeout=8.0)
            fetch_success = True
            fetch_time_str = forecast_data.get("_fetched_at", "Recent")
            st.success(f"Live forecast: {selected_district}, fetched {fetch_time_str}")
            hourly = forecast_data["hourly"]
            temp_c = float(hourly["temperature_2m"][0])
            humidity = float(hourly["relative_humidity_2m"][0])
            precip = float(hourly["precipitation"][0])

            # Fetch soil_moisture_0_to_1cm from forecast
            sm_arr = hourly.get("soil_moisture_0_to_1cm", [])
            if sm_arr and sm_arr[0] is not None:
                sm_val = float(sm_arr[0])
                soil_moist_display = f"{sm_val:.3f} m³/m³ ({sm_val * 100:.1f}%)"
                soil_moist_pct = float(sm_val * 100.0)
                soil_moist_available = True
            else:
                sm_val = None
                soil_moist_display = "not available"
                soil_moist_pct = 72.0  # existing default
                soil_moist_available = False

            if soil_moist_available:
                st.caption(f"Current: 🌡️ {temp_c:.1f}°C | 💧 Topsoil moisture (model estimate): {soil_moist_display} | 🌧️ RH {humidity:.0f}% (Precip: {precip:.1f} mm/h)")
            else:
                st.caption(f"Current: 🌡️ {temp_c:.1f}°C | 💧 Topsoil moisture (model estimate): not available | 🌧️ RH {humidity:.0f}% (Precip: {precip:.1f} mm/h)")
        except Exception as e:
            fetch_success = False
            forecast_data = None
            st.warning(f"⚠️ Live forecast fetch failed for {selected_district}. Falling back to manual values.")

    if not use_live_weather or not fetch_success:
        st.info("Offline: using entered values.")
        env_preset = st.selectbox("⚡ Weather / Field Presets", [
            "Custom Sliders",
            "🌧️ Monsoon Rain / High Humidity (Rot Risk)",
            "☀️ Sunny & Dry Day (Low Climatic Pressure)",
            "🍎 Cool Wet Orchard (Mills Period / Scab Risk)"
        ])

        if "Monsoon" in env_preset:
            def_temp, def_moist, def_hum = 24.0, 88.0, 92.0
        elif "Sunny" in env_preset:
            def_temp, def_moist, def_hum = 33.0, 28.0, 42.0
        elif "Cool Wet" in env_preset:
            def_temp, def_moist, def_hum = 18.5, 78.0, 86.0
        else:
            def_temp, def_moist, def_hum = 26.0, 72.0, 75.0

        temp_c = st.slider("🌡️ Temperature (°C)", min_value=10.0, max_value=45.0, value=def_temp, step=0.5)
        soil_moist_pct = st.slider("💧 Soil Moisture (%)", min_value=10.0, max_value=100.0, value=def_moist, step=1.0)
        humidity = st.slider("🌧️ Relative Humidity (%)", min_value=20.0, max_value=100.0, value=def_hum, step=1.0)
        soil_moist_display = f"{soil_moist_pct:.0f}%"
        soil_moist_available = True

    st.markdown("---")
    show_gradcam = st.checkbox("🔥 Show Grad-CAM Heatmap", value=True)
    show_lesion  = st.checkbox("🔴 Show Color-Threshold Lesion Mask", value=True)

# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
# ─── LANGUAGE SELECTION (DEFINED ONCE FOR ENTIRE APPLICATION) ───────────────
lang_code = voice_lang.split("(")[1].replace(")", "").strip()

st.title("🌾 AgriTrust AI — Multimodal Crop Disease & Severity Engine")
st.markdown(
    "Upload a leaf image to receive: **① AI Vision Diagnosis** · **② Field Condition Assessment** "
    "· **③ Ordinal Severity Grade (0–3)** · **④ Grad-CAM Attention** · **⑤ Management Plan & Multilingual Voice**"
)

uploaded = st.file_uploader("📷 Upload Crop Leaf Image", type=["jpg", "jpeg", "png"])

input_image = None
if uploaded:
    input_image = Image.open(uploaded).convert("RGB")
    image_title = uploaded.name

# ─── INFERENCE WORKFLOW ───────────────────────────────────────────────────────
if input_image is not None:
    st.markdown("---")
    active_crop = crop_mode

    if active_crop == "Apple":
        st.warning("⚠️ **Trained on public lab data, not field-validated in orchards.**")

    # ── 0. OOD GUARD CHECK ────────────────────────────────────────────────────
    with st.spinner(f"🛡️ Verifying image domain for {active_crop}..."):
        is_ood, ood_score, ood_thresh = check_ood(input_image, active_crop)

    if is_ood:
        if os.path.exists("advisory_voice.mp3"):
            try:
                os.remove("advisory_voice.mp3")
            except Exception:
                pass
        st.error(f"⚠️ **This does not look like a close-up {active_crop.lower()} photo. Retake a well-lit close-up of the affected leaf, trunk or nut.**")
        st.info("The image features deviate significantly from close-up crop pathology training distributions. Diagnosis, severity grading, advisory, Grad-CAM, and voice are suppressed for safety.")
        st.image(input_image, caption=f"Uploaded photo: {image_title}", width=350)
    else:
        # 1. DISEASE CLASSIFICATION
        with st.spinner(f"🔬 Analyzing {active_crop} leaf..."):
            pred_cls, dis_conf, dis_probs, gradcam_img, display_name, clf_available = \
                run_disease_classification(input_image, active_crop)

        # ── ABSTAIN SAFETY RULE CHECK ─────────────────────────────────────────────
        abstain_tau = ABSTAIN_THRESHOLDS.get(active_crop, 0.70)
        if dis_conf < abstain_tau:
            if os.path.exists("advisory_voice.mp3"):
                try:
                    os.remove("advisory_voice.mp3")
                except Exception:
                    pass
            st.error("⚠️ **Not sure about this photo. Retake it in daylight or consult your KVK / agriculture officer.**")
            st.info("The classifier confidence fell below the safety validation threshold. Disease diagnosis, pesticide prescription, Grad-CAM heatmap, and voice advisory are withheld.")
            st.image(input_image, caption=f"Uploaded photo: {image_title}", width=350)
        else:
            # ── UN-ABSTAINED DIAGNOSIS & REMEDIATION PIPELINE ──────────────────────
            if active_crop == "Arecanut" and pred_cls in ["bud_borer", "yellow_leaf_disease"]:
                st.warning("⚠️ **Low Reliability Notice:** The model frequently confuses Bud Borer and Yellow Leaf Disease (YLD). An agricultural expert from your local KVK or CPCRI should confirm this diagnosis before treatment.")

            is_healthy = (pred_cls in ["normal", "healthy", "healthy_foot", "healthy_leaf", "healthy_nut", "healthy_trunk"]
                          or display_name.startswith("Healthy"))

            # 2. SEVERITY GRADING (Suppressed for Apple)
            if active_crop != "Apple":
                if is_healthy:
                    grade = 0
                    sev_conf = 1.0
                    sev_probs = [1.0, 0.0, 0.0, 0.0]
                    cv_ratio = 0.0
                    lesion_overlay = None
                else:
                    with st.spinner(f"📊 Grading disease severity ({model_type_choice})..."):
                        grade, sev_conf, sev_probs, cv_ratio, lesion_overlay = \
                            run_severity(input_image, active_crop, model_type_choice)
                    if grade == 0:
                        grade = 1  # if non-healthy predicted grade 0, promote to grade 1 for advice and display

                grade_info = [
                    {"name": "Grade 0: Healthy",          "card": "result-card-g0", "color": "#22c55e", "badge": "🟢"},
                    {"name": "Grade 1: Mild Severity",     "card": "result-card-g1", "color": "#eab308", "badge": "🟡"},
                    {"name": "Grade 2: Moderate Severity", "card": "result-card-g2", "color": "#f97316", "badge": "🟠"},
                    {"name": "Grade 3: Severe Severity",   "card": "result-card-g3", "color": "#ef4444", "badge": "🔴"},
                ][grade]
            else:
                grade = 0
                sev_conf = 0.0
                sev_probs = [0.0] * 4
                cv_ratio = 0.0
                lesion_overlay = None

            # 3. BIOCLIMATIC SENSOR RISK EVALUATION (MULTIMODAL FUSION)
            if use_live_weather and fetch_success:
                # Live Open-Meteo soil moisture is volumetric fraction (0-1 cm, typically 10-45%),
                # whereas the legacy risk engine thresholds expect sensor saturation (25-85%).
                # In live mode, ignore soil moisture by passing neutral 50.0% so it neither
                # triggers nor suppresses rules.
                risk_soil_pct = 50.0
            else:
                risk_soil_pct = soil_moist_pct

            risk_score, risk_level, env_alert, action_modifier = evaluate_environmental_risk(
                crop=active_crop, disease_name=display_name, temp_c=temp_c, soil_moisture_pct=risk_soil_pct, humidity_pct=humidity
            )

            # ── DISEASE BANNER ───────────────────────────────────────────────────
            if clf_available:
                fact = DISEASE_FACTS.get(display_name, "")
                loc_dict = DISEASE_LOCALIZATION.get(active_crop, {}).get(pred_cls, {})
                localized_name = loc_dict.get(lang_code, display_name)
                banner_name = f"{display_name} ({localized_name})" if (lang_code != "en" and localized_name != display_name) else display_name

                rust_caption = '<p style="margin-top:6px;font-size:12px;color:#fef08a;font-style:italic;">⚠️ Class from US public data; confirm locally with your KVK.</p>' if (active_crop == "Apple" and pred_cls == "cedar_apple_rust") else ''

                st.markdown(f"""
                <div class="disease-card">
                    <h2 style="margin:0;color:#38bdf8;">🦠 Detected Disease: {banner_name}</h2>
                    <p style="margin-top:8px;font-size:15px;color:#cbd5e1;">
                        Vision Confidence: <b style="color:#38bdf8;">{dis_conf:.1%}</b>
                        &nbsp;|&nbsp; Target Crop: <b>{active_crop}</b>
                    </p>
                    {rust_caption}
                    {'<p style="margin-top:8px;font-size:13px;color:#94a3b8;font-style:italic;">' + fact + '</p>' if fact else ''}
                </div>""", unsafe_allow_html=True)

            # ── MULTIMODAL FUSION PANEL (VISION + FIELD CONDITIONS) ──────────────
            risk_color = "#ef4444" if risk_score >= 70 else ("#f97316" if risk_score >= 45 else "#22c55e")
            if use_live_weather and fetch_success:
                if soil_moist_available:
                    moist_str = f"<b>💧 Topsoil moisture (model estimate): {soil_moist_display}</b>"
                else:
                    moist_str = "<b>💧 Topsoil moisture (model estimate): not available</b>"
                env_source_label = f"Live forecast ({selected_district})"
                soil_engine_note = '<p style="margin-top:4px;font-size:12px;color:#94a3b8;font-style:italic;">ℹ️ Soil moisture neutralised (50%) in risk engine due to volumetric scale difference.</p>'
            else:
                moist_str = f"<b>💧 Soil Moisture: {soil_moist_pct:.0f}%</b>"
                env_source_label = "Field conditions (entered by user)"
                soil_engine_note = ""

            # 48-Hour Preventive Weather Alert gating:
            # ONLY for non-healthy predictions matching the 3 rules:
            # apple_scab (Apple), blast (Paddy), mahali_koleroga (Arecanut).
            weather_alert_res = None
            weather_spoken_alert = ""

            if (not is_healthy) and use_live_weather and fetch_success and (forecast_data is not None):
                weather_alert_res = compute_48h_disease_risk(forecast_data, active_crop, display_name, pred_cls=pred_cls)
                if weather_alert_res is not None:
                    weather_spoken_alert = weather_alert_res["voice_alerts"].get(lang_code, weather_alert_res["voice_alerts"]["en"])

            if not (use_live_weather and fetch_success):
                alert_html = """
                <div style="background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.3);
                            border-radius: 12px; padding: 12px 16px; margin-top: 14px;">
                    <p style="margin:0;font-size:14px;color:#cbd5e1;">
                        ℹ️ 48-hour alert needs the live forecast.
                    </p>
                </div>
                """
            elif weather_alert_res is not None:
                rcat = weather_alert_res["risk_cat"]
                rcol = "#ef4444" if rcat == "High" else ("#f97316" if rcat == "Moderate" else "#22c55e")
                alert_html = f"""
                <div style="background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.35);
                            border-radius: 12px; padding: 14px 18px; margin-top: 14px;">
                    <h4 style="margin:0;color:{rcol};">⏱️ 48-Hour Preventive Disease Alert ({rcat} Risk)</h4>
                    <p style="margin:8px 0 0 0;font-size:14.5px;color:#f1f5f9;line-height:1.5;">
                        {weather_alert_res['alert_text']}
                    </p>
                    <p style="margin:6px 0 0 0;font-size:11.5px;color:#94a3b8;font-style:italic;">
                        ⚠️ rule-based risk estimate, not a validated forecast
                    </p>
                </div>
                """
            else:
                alert_html = ""

            st.markdown(f"""
            <div class="multimodal-card">
                <h3 style="margin:0;color:#a78bfa;">🌦️ Environmental Evaluation (Vision + Field Conditions)</h3>
                <p style="margin-top:6px;font-size:14px;color:#cbd5e1;">
                    {env_source_label}: <b>🌡️ {temp_c:.1f}°C</b> &nbsp;|&nbsp; {moist_str} &nbsp;|&nbsp; <b>🌧️ Humidity: {humidity:.0f}%</b>
                </p>
                {soil_engine_note}
                <div style="margin-top:12px;display:flex;align-items:center;gap:12px;">
                    <span style="font-weight:600;color:{risk_color};font-size:16px;">Bioclimatic Outbreak Risk: {risk_score:.0f}% ({risk_level})</span>
                </div>
                <p style="margin-top:10px;font-size:14px;color:#e2e8f0;line-height:1.5;">
                    <b>Agronomic Assessment:</b> {env_alert}
                </p>
                {alert_html}
            </div>
            """, unsafe_allow_html=True)

            # ── THREE-PANEL VISUAL ROW ───────────────────────────────────────────
            st.subheader("🖼️ Explainability & Attention Panels")
            panels = [("Original Photo", input_image, f"Raw uploaded image: {image_title}", "image")]
            if show_gradcam and gradcam_img is not None and clf_available:
                panels.append(("🔥 Attention Heatmap (Grad-CAM)", gradcam_img,
                               f"Grad-CAM Heatmap — red areas show features driving model decision for {display_name}. (Note: Pointing accuracy is only 5–14% on leaf pathology; heatmaps indicate salient regions rather than verified disease localization).", "image"))
            if show_lesion and lesion_overlay is not None:
                panels.append(("🔴 Color-Threshold Lesion Mask", lesion_overlay,
                               f"Lesion Mask — {cv_ratio:.1%} estimated lesion area.", "image"))

            cols = st.columns(len(panels))
            for p_idx, (p_title, p_content, p_note, p_type) in enumerate(panels):
                with cols[p_idx]:
                    st.markdown(f"**{p_title}**")
                    st.image(p_content, caption="", use_container_width=True)
                    st.markdown(f'<p class="explainer-note">{p_note}</p>', unsafe_allow_html=True)

            # ── SEVERITY & DISEASE DISTRIBUTION ROW ──────────────────────────────
            st.markdown("---")
            left, right = st.columns([1, 1])

            with left:
                if active_crop == "Apple":
                    st.info("🍎 **Severity grading is withheld for Apple:** Model trained on public lab data, not field-validated in orchards.")
                elif is_healthy:
                    st.success("🌿 **Foliage Healthy:** No disease lesions detected. Severity grading is withheld.")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""<div class="metric-badge">
                            <div class="metric-val">{dis_conf:.1%}</div>
                            <div class="metric-lbl">Vision Confidence</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="metric-badge">
                            <div class="metric-val">{cv_ratio:.1%}</div>
                            <div class="metric-lbl">Lesion Coverage</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="{grade_info['card']}">
                        <h2 style="margin:0;color:{grade_info['color']};">{grade_info['badge']} {grade_info['name']}</h2>
                    </div>""", unsafe_allow_html=True)

                    if active_crop == "Arecanut":
                        st.caption("Provisional: not independently validated.")
                    elif active_crop == "Paddy":
                        st.caption("Independent-test QWK about 0.67 (twin-free, lesion classes); headline 0.94 is inflated.")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="metric-badge">
                            <div class="metric-val">{dis_conf:.1%}</div>
                            <div class="metric-lbl">Vision Confidence</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="metric-badge">
                            <div class="metric-val">{sev_conf:.1%}</div>
                            <div class="metric-lbl">Severity Confidence</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="metric-badge">
                            <div class="metric-val">{cv_ratio:.1%}</div>
                            <div class="metric-lbl">Lesion Coverage</div>
                        </div>""", unsafe_allow_html=True)

            with right:
                if clf_available:
                    st.markdown("**🦠 Disease Probability Distribution**")
                    if active_crop == "Paddy": disp_map = PADDY_DISPLAY
                    elif active_crop == "Apple": disp_map = APPLE_DISPLAY
                    else: disp_map = ARECA_DISPLAY
                    
                    df_dis = pd.DataFrame({
                        "Disease": [disp_map.get(c, c) for c in dis_probs.keys()],
                        "Probability": list(dis_probs.values())
                    }).sort_values("Probability", ascending=False)
                    st.bar_chart(df_dis.set_index("Disease"), height=160, color="#38bdf8")

            # ── PESTICIDE & TREATMENT RECOMMENDER ─────────────────────────────────
            st.markdown("---")
            st.subheader("💊 Actionable Management Plan")
            
            crop_tx = TREATMENT_DATABASE.get(active_crop, {})
            dis_tx  = crop_tx.get(display_name, {})
            
            if active_crop == "Apple":
                if display_name == "Healthy (No Disease)":
                    title_tx, desc_tx = dis_tx.get(0, ("Healthy Foliage", "No treatment needed."))
                else:
                    title_tx, desc_tx = dis_tx.get("disease_advice", ("Disease Advisory", "Follow local KVK advice."))
            else:
                title_tx, desc_tx = dis_tx.get(grade, ("Action Plan", "Follow local KVK advice."))
            
            rust_box_note = '<p style="margin-top:6px;font-size:12px;color:#fef08a;font-style:italic;">⚠️ Class from US public data; confirm locally with your KVK.</p>' if (active_crop == "Apple" and pred_cls == "cedar_apple_rust") else ''

            st.markdown(f"""
            <div class="pesticide-card">
                <h3 style="margin:0;color:#10b981;">📋 Action Plan: {title_tx}</h3>
                {rust_box_note}
                <p style="margin-top:10px;font-size:16px;color:#e2e8f0;line-height:1.6;">
                    {desc_tx}
                </p>
                <p style="margin-top:10px;font-size:14px;color:#fef08a;font-weight:600;">
                    {action_modifier}
                </p>
                <div style="margin-top:12px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.15);font-size:12px;color:#94a3b8;font-style:italic;">
                    ⚠️ Verify against ICAR / CPCRI / CIB&RC before use.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── SOIL HEALTH CARD ──────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🌱 Soil Health Card")
            soil_info = SOIL_HEALTH_ADVISORY.get(active_crop, SOIL_HEALTH_ADVISORY["Paddy"])
            soil_title, soil_desc = soil_info.get(lang_code, soil_info["en"])

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(14, 116, 144, 0.20));
                        border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 16px; padding: 18px 22px; margin-top: 14px;">
                <h3 style="margin:0;color:#34d399;">🌱 {soil_title}</h3>
                <p style="margin-top:10px;font-size:15px;color:#e2e8f0;line-height:1.6;">
                    {soil_desc}
                </p>
                <div style="margin-top:12px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.15);font-size:12px;color:#94a3b8;font-style:italic;">
                    ⚠️ Verify with your KVK / soil health card.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── MULTILINGUAL VOICE ASSISTANT ──────────────────────────────────────
            st.markdown("---")
            st.subheader("🎙️ Multilingual Voice Advisory")
            advisory_text = build_voice_advisory(active_crop, display_name, pred_cls, grade, lang_code, weather_spoken_alert)
            
            try:
                tts = gTTS(text=advisory_text, lang=lang_code, slow=False)
                tts.save("advisory_voice.mp3")
                st.audio("advisory_voice.mp3", format="audio/mp3")
                st.caption(f"🔊 Spoken Advisory in **{voice_lang}**")
            except Exception as e:
                st.info(f"🔊 Voice Advisory Text: {advisory_text}")

else:
    st.info("👆 Upload a crop leaf image above (and adjust soil moisture & temperature in the sidebar) to begin multimodal analysis.")
