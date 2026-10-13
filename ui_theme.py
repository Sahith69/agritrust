"""
ui_theme.py — AgriTrust AI Theme, Navigation, UI Strings & Grounded Chat Logic.

Minimal, mobile-first design system:
- Palette: bg #F9F9F6; primary #275637; sage #91BA94; light_green #B1D27D; warn_yellow #F6D05F; danger #B5483A.
- Text on sage, light green, and warn yellow is ALWAYS #275637, NEVER white.
- Cards: white, 1px sage border, 14px radius, no heavy shadows.
- Max content width: 760px, tap targets >= 48px.
"""

import datetime
from typing import Dict, Any, Optional, Tuple, List

# ─── APP IDENTITY & AUDIT BENCHMARKS ─────────────────────────────────────────
APP_NAME = "AgriTrust AI"

TRUST_AND_AUDIT = {
    "paddy_twin_rate": "51.4% (775/1507 twins in test set)",
    "paddy_severity_qwk": "0.94 headline vs 0.67 twin-free lesion classes",
    "class_only_baseline": "0.91 full vs 0.00 lesion-only baseline",
    "abstain_threshold": "0.70 confidence threshold",
    "apple_orchard_result": "Held-out orchard split of one public dataset, N=346: raw accuracy 96.0% [93.6%, 98.0%], guarded accuracy 89.9% (coverage 92.5%, diagnosed accuracy 97.2%); on images farthest from training: 95.7%. Not northern-India validated.",
    "arecanut_weak_guard": "Arecanut weak OOD detection (heterogeneous palm views yield ~40% non-crop block)",
    "arecanut_severity": "Arecanut severity provisional (not independently validated)",
}

# ─── THEME CONFIGURATION (ONE DICT) ──────────────────────────────────────────
THEME_CONFIG = {
    "bg": "#F9F9F6",
    "primary": "#275637",
    "sage": "#91BA94",
    "light_green": "#B1D27D",
    "warn_yellow": "#F6D05F",
    "danger": "#B5483A",
    "card_bg": "#FFFFFF",
    "refusal_bg": "#F1F3F0",
    "text_dark": "#275637",
    "text_muted": "#556B5C",
    "font_stack": 'system-ui, "Noto Sans", "Noto Sans Kannada", "Noto Sans Devanagari", sans-serif',
    "base_font_size": "16px",
    "border_radius": "14px",
    "max_width": "760px",
    "min_tap_target": "48px",
}

# ─── LOCALIZED MONTH NAMES ───────────────────────────────────────────────────
MONTHS_KN = {
    1: "ಜನವರಿ", 2: "ಫೆಬ್ರವರಿ", 3: "ಮಾರ್ಚ್", 4: "ಏಪ್ರಿಲ್", 5: "ಮೇ", 6: "ಜೂನ್",
    7: "ಜುಲೈ", 8: "ಆಗಸ್ಟ್", 9: "ಸೆಪ್ಟೆಂಬರ್", 10: "ಅಕ್ಟೋಬರ್", 11: "ನವೆಂಬರ್", 12: "ಡಿಸೆಂಬರ್"
}
MONTHS_HI = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल", 5: "मई", 6: "जून",
    7: "जुलाई", 8: "अगस्त", 9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर"
}

def format_local_date(dt: datetime.date, lang: str = "en") -> str:
    if lang == "kn":
        return f"{dt.day} {MONTHS_KN.get(dt.month, '')} {dt.year}"
    elif lang == "hi":
        return f"{dt.day} {MONTHS_HI.get(dt.month, '')} {dt.year}"
    return dt.strftime("%d %B %Y")

# ─── UI STRINGS (EN, KN, HI) ─────────────────────────────────────────────────
UI_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_name": "AgriTrust AI",
        "greeting": "Hey, welcome farmer",
        "district_label": "District",
        "live_forecast_badge": "Live Forecast",
        "offline_badge": "Offline",
        "temp_label": "Temp",
        "humidity_label": "Humidity",
        "rain_48h_label": "Rain in 48h",
        "weather_status_normal": "Normal",
        "weather_status_rain": "Rain Expected",
        "weather_status_dry": "Dry",
        # Scan Card
        "scan_card_title": "Scan / Upload Photo",
        "tip_line": "Close-up, daylight, one leaf or one nut",
        "choose_crop": "Choose Crop",
        "crop_paddy": "Paddy",
        "crop_arecanut": "Arecanut",
        "crop_apple": "Apple",
        "take_photo": "Take photo with camera",
        "upload_file": "Or upload photo from device",
        # Last result card
        "last_result_title": "Your Last Result",
        "no_scans_yet": "No scans yet.",
        "crop_label": "Crop",
        "date_label": "Date",
        "confidence_label": "Confidence",
        "recommendation_label": "Recommendation",
        # Result Screen
        "back_btn": "← Back",
        "status_healthy": "Healthy",
        "status_treat": "Treatable",
        "status_no_cure": "No cure / See expert",
        "status_unsure": "Not sure",
        "what_to_do": "What to do",
        "organic_title": "Organic & Natural Steps",
        "severe_expander": "If the problem is severe (Chemical Treatment)",
        "weather_alert_title": "48-Hour Preventive Alert",
        "soil_card_title": "Soil Health Card",
        "listen_audio": "Voice Advisory",
        "experts_expander": "For experts and judges",
        "trust_audit_title": "Trust and Audit Benchmarks",
        "apple_public_banner": "Fine-tuned on lab (PlantVillage) and public orchard photos (Cornell dataset, not northern India). Black rot was trained on lab photos only. Severity grading is withheld.",
        "areca_provisional": "Provisional: not independently validated.",
        "paddy_severity_qwk": "Independent-test QWK about 0.67 (twin-free, lesion classes); headline 0.94 is inflated.",
        "paddy_severity_fixed": "Severity is assigned by a fixed rule for this class.",
        "tungro_warning_title": "Tungro Warning",
        "cedar_rust_caption": "⚠️ Class from US public data; confirm locally with your KVK.",
        "areca_bud_borer_warning": "⚠️ Low Reliability Notice: The model frequently confuses Bud Borer and Yellow Leaf Disease (YLD). Confirm with KVK / CPCRI before treatment.",
        "footer_disclaimer_tx": "⚠️ Verify against ICAR / CPCRI / CIB&RC before use.",
        "footer_disclaimer_soil": "⚠️ Verify with your KVK / soil health card.",
        # Refusal states
        "refusal_ood_title": "Image Outside Field Scope",
        "refusal_ood_msg": "This does not look like a close-up photo of the selected crop. Retake a well-lit close-up of the affected leaf, trunk or nut.",
        "refusal_abstain_title": "Diagnosis Withheld for Safety",
        "refusal_abstain_msg": "Not sure about this photo. Retake it in daylight or consult your KVK / agriculture officer.",
        # Navigation
        "tab_home": "Home",
        "tab_chat": "AI Chat",
        "tab_weather": "Weather",
        # AI Chat
        "chat_title": "Grounded Agricultural Assistant",
        "chat_no_context": "Scan a leaf first and I can explain the result.",
        "chat_disclaimer": "Answers come from the advice shown for your last scan. Verify with your KVK.",
        "chat_clear_btn": "Clear Chat",
        "chat_placeholder": "Ask a question about your scan...",
        "chip_first": "What should I do first?",
        "chip_cure": "Is there a cure?",
        "chip_spray": "When should I spray?",
        "chip_soil": "Soil tips",
        "chip_weather": "Weather risk",
        "chip_unsure": "Why are you unsure?",
        "chat_unmatched": "I can only explain your last result. Please ask your KVK / agriculture officer.",
        # Weather View
        "weather_view_title": "48-Hour Live Weather & Disease Pressure",
        "weather_6h_blocks": "48-Hour Forecast Summary (6-Hour Blocks)",
        "crop_weather_risk": "Crop Weather Risk Analysis",
        "rule_based_tag": "rule-based estimate, not a forecast",
        "no_rule_for_crop": "ℹ️ No weather rule active for this condition in the next 48 hours.",
        "models_missing_msg": "Model files not found. Run: python download_models.py",
        "cross_crop_refusal_title": "Crop Selection Mismatch",
        "cross_crop_refusal_msg": "This looks more like a {other_crop} photo. Check your crop selection.",
    },
    "kn": {
        "app_name": "AgriTrust AI",
        "greeting": "ನಮಸ್ಕಾರ, ರೈತ ಮಿತ್ರರಿಗೆ ಸುಸ್ವಾಗತ",
        "district_label": "ಜಿಲ್ಲೆ",
        "live_forecast_badge": "ನೇರ ಮುನ್ಸೂಚನೆ",
        "offline_badge": "ಆಫ್‌ಲೈನ್",
        "temp_label": "ತಾಪಮಾನ",
        "humidity_label": "ಆರ್ದ್ರತೆ",
        "rain_48h_label": "48 ಗಂ ಮಳೆ",
        "weather_status_normal": "ಸಾಮಾನ್ಯ",
        "weather_status_rain": "ಮಳೆಯ ಸಾಧ್ಯತೆ",
        "weather_status_dry": "ಶುಷ್ಕ",
        # Scan Card
        "scan_card_title": "ಫೋಟೋ ಸ್ಕ್ಯಾನ್ / ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "tip_line": "ಬೆಳಕಿನಲ್ಲಿ, ಹತ್ತಿರದಿಂದ ಒಂದೇ ಎಲೆ ಅಥವಾ ಅಡಿಕೆ ಫೋಟೋ ತೆಗೀರಿ",
        "choose_crop": "ಬೆಳೆ ಆಯ್ಕೆಮಾಡಿ",
        "crop_paddy": "ಭತ್ತ",
        "crop_arecanut": "ಅಡಿಕೆ",
        "crop_apple": "ಸೇಬು",
        "take_photo": "ಕ್ಯಾಮರಾದಿಂದ ಫೋಟೋ ತೆಗೀರಿ",
        "upload_file": "ಅಥವಾ ಮೊಬೈಲ್‌ನಿಂದ ಫೋಟೋ ಹಾಕಿ",
        # Last result card
        "last_result_title": "ನಿಮ್ಮ ಹಿಂದಿನ ಫಲಿತಾಂಶ",
        "no_scans_yet": "ಇನ್ನೂ ಯಾವುದೇ ಸ್ಕ್ಯಾನ್ ಮಾಡಿಲ್ಲ.",
        "crop_label": "ಬೆಳೆ",
        "date_label": "ದಿನಾಂಕ",
        "confidence_label": "ವಿಶ್ವಾಸಾರ್ಹತೆ",
        "recommendation_label": "ಶಿಫಾರಸು",
        # Result Screen
        "back_btn": "← ವಾಪಸ್ ಹೋಗಿ (ಹೊಸ ಫೋಟೋ)",
        "status_healthy": "ಆರೋಗ್ಯಕರ",
        "status_treat": "ಚಿಕಿತ್ಸೆ ಸಾಧ್ಯ",
        "status_no_cure": "ಪರಿಹಾರವಿಲ್ಲ / ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ",
        "status_unsure": "ಖಚಿತವಿಲ್ಲ",
        "what_to_do": "ಮಾಡಬೇಕಾದ ಕ್ರಮಗಳು",
        "organic_title": "ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು",
        "severe_expander": "ಸಮಸ್ಯೆ ಗಂಭೀರವಾಗಿದ್ದರೆ (ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ)",
        "weather_alert_title": "48 ಗಂಟೆ ಹವಾಮಾನ ಅಲರ್ಟ್",
        "soil_card_title": "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್",
        "listen_audio": "ವಾಯ್ಸ್‌ನಲ್ಲಿ ಕೇಳಿ (ಆಡಿಯೋ)",
        "experts_expander": "ತಜ್ಞರು ಮತ್ತು ತೀರ್ಪುಗಾರರಿಗೆ",
        "trust_audit_title": "ವಿಶ್ವಾಸಾರ್ಹತೆ ಮತ್ತು ಆಡಿಟ್ ಅಂಕಿಅಂಶಗಳು",
        "apple_public_banner": "ಪ್ರಯೋಗಾಲಯ (PlantVillage) ಮತ್ತು ಸಾರ್ವಜನಿಕ ತೋಟದ ಫೋಟೋಗಳಲ್ಲಿ (Cornell ಡೇಟಾಸೆಟ್, ಉತ್ತರ ಭಾರತವಲ್ಲ) ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಲಾಗಿದೆ. Black rot ಕೇವಲ ಲ್ಯಾಬ್ ಫೋಟೋಗಳಲ್ಲಿ ತರಬೇತಿ ಪಡೆದಿದೆ. ತೀವ್ರತೆಯ ಶ್ರೇಣೀಕರಣವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ.",
        "areca_provisional": "ತಾತ್ಕಾಲಿಕ: ಸ್ವತಂತ್ರವಾಗಿ ಮೌಲ್ಯೀಕರಿಸಲಾಗಿಲ್ಲ.",
        "paddy_severity_qwk": "ನಿಜ ತಪಾಸಣಾ ರಿಪೋರ್ಟ್: ರಿಯಲ್ ಫೀಲ್ಡ್ ಟೆಸ್ಟ್‌ನಲ್ಲಿ ಸ್ಕೋರ್ ಸುಮಾರು 0.67 ಇದೆ; ಹೊರಗಡೆ ಹೇಳುವ 0.94 ಸ್ಕೋರ್ ಜಾಸ್ತಿ ಮಾಡಿ ಹೇಳಿದ್ದು.",
        "paddy_severity_fixed": "ಈ ರೋಗಕ್ಕೆ ಸ್ಥಿರ ನಿಯಮದ ಮೂಲಕ ತೀವ್ರತೆಯನ್ನು ನಿಗದಿಪಡಿಸಲಾಗಿದೆ.",
        "tungro_warning_title": "ತುಂಗ್ರೋ ಎಚ್ಚರಿಕೆ",
        "cedar_rust_caption": "⚠️ ಯುಎಸ್ ಸಾರ್ವಜನಿಕ ಡೇಟಾದ ವರ್ಗ; ಸ್ಥಳೀಯವಾಗಿ ನಿಮ್ಮ ಕೆವಿಕೆ ಜೊತೆ ದೃಢೀಕರಿಸಿ.",
        "areca_bud_borer_warning": "⚠️ ಕಡಿಮೆ ವಿಶ್ವಾಸಾರ್ಹತೆಯ ಸೂಚನೆ: ಮಾದರಿಯು ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು ಮತ್ತು ಹಳದಿ ಎಲೆ ರೋಗವನ್ನು ಗೊಂದಲಗೊಳಿಸುತ್ತದೆ. ಚಿಕಿತ್ಸೆಗೆ ಮುನ್ನ ಕೆವಿಕೆ/ಸಿಪಿಸಿಆರ್‌ಐ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        "footer_disclaimer_tx": "⚠️ ಬಳಸುವ ಮುನ್ನ ICAR / CPCRI / CIB&RC ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        "footer_disclaimer_soil": "⚠️ ನಿಮ್ಮ ಕೆವಿಕೆ / ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        # Refusal states
        "refusal_ood_title": "ಫೋಟೋ ಐಡೆಂಟಿಫೈ ಆಗಿಲ್ಲ",
        "refusal_ood_msg": "ಇದು ಭತ್ತ, ಅಡಿಕೆ ಅಥವಾ ಆಪಲ್‌ನ ಕ್ಲಿಯರ್ ಆದ ಫೋಟೋ ತರ ಕಾಣ್ತಾ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಒಳ್ಳೆ ಲೈಟ್‌ನಲ್ಲಿ, ರೋಗ ಬಂದಿರೋ ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಅಡಿಕೆ ಹತ್ತಿರ ಹೋಗಿ ಇನ್ನೊಂದು ಸಲ ಕ್ಲಿಯರ್ ಆಗಿ ಫೋಟೋ ತೆಗೀರಿ.",
        "refusal_abstain_title": "ರಿಸಲ್ಟ್ ನಿಲ್ಲಿಸಿದೆ (ಕಮ್ಮಿ ಕಾನ್ಫಿಡೆನ್ಸ್)",
        "refusal_abstain_msg": "ರೋಗ ಪಕ್ಕಾ ಆಗಿ ಕನ್ಫರ್ಮ್ ಆಗಿಲ್ಲ (ಕಾನ್ಫಿಡೆನ್ಸ್ 70% ಗಿಂತ ಕಮ್ಮಿ ಇದೆ). ಫೋಟೋ ಬ್ಲರ್ ಆಗಿದೆ ಅಥವಾ ಲೈಟ್ ಸಾಲಲ್ಲ. ತಪ್ಪು ಗೆಸ್ ಮಾಡೋದು ಬೇಡ ಅಂತ ಆ್ಯಪ್ ನಿಲ್ಲಿಸಿದೆ. ದಯವಿಟ್ಟು ಕ್ಲಿಯರ್ ಫೋಟೋ ಇನ್ನೊಂದು ಸಲ ತೆಗೀರಿ ಅಥವಾ ಕೆವಿಕೆ ಆಫೀಸರ್‌ನ ಕಾಂಟ್ಯಾಕ್ಟ್ ಮಾಡಿ.",
        # Navigation
        "tab_home": "ಹೋಮ್",
        "tab_chat": "ಕೃಷಿ ಚಾಟ್",
        "tab_weather": "ಹವಾಮಾನ",
        # AI Chat
        "chat_title": "ಕೃಷಿ ಸಹಾಯಕ ಸಂಭಾಷಣೆ",
        "chat_no_context": "ಮೊದಲು ಎಲೆಯ ಫೋಟೋ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ, ನಂತರ ನಾನು ಫಲಿತಾಂಶವನ್ನು ವಿವರಿಸಬಲ್ಲೆ.",
        "chat_disclaimer": "ಉತ್ತರಗಳು ನಿಮ್ಮ ಕೊನೆಯ ಸ್ಕ್ಯಾನ್‌ನಲ್ಲಿ ತೋರಿಸಲಾದ ಸಲಹೆಯಿಂದ ಬಂದಿವೆ. ಕೆವಿಕೆ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        "chat_clear_btn": "ಸಂಭಾಷಣೆ ಅಳಿಸಿ",
        "chat_placeholder": "ನಿಮ್ಮ ಸ್ಕ್ಯಾನ್ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ...",
        "chip_first": "ನಾನು ಮೊದಲು ಏನು ಮಾಡ್ಬೇಕು?",
        "chip_cure": "ಈ ರೋಗ ವಾಸಿ ಆಗುತ್ತಾ?",
        "chip_spray": "ಮದ್ದು ಯಾವಾಗ ಸ್ಪ್ರೇ ಮಾಡ್ಬೇಕು?",
        "chip_soil": "ಮಣ್ಣಿನ ಸಲಹೆಗಳು",
        "chip_weather": "ಹವಾಮಾನ ಅಪಾಯ",
        "chip_unsure": "ಆ್ಯಪ್‌ಗೆ ಯಾಕೆ ಪಕ್ಕಾ ನಂಬಿಕೆ ಇಲ್ಲ?",
        "chat_unmatched": "ನಾನು ನಿಮ್ಮ ಕೊನೆಯ ಫಲಿತಾಂಶವನ್ನು ಮಾತ್ರ ವಿವರಿಸಬಲ್ಲೆ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        # Weather View
        "weather_view_title": "48 ಗಂಟೆಗಳ ನೇರ ಹವಾಮಾನ ಮತ್ತು ರೋಗದ ಒತ್ತಡ",
        "weather_6h_blocks": "48 ಗಂಟೆಗಳ ಹವಾಮಾನ ಸಾರಾಂಶ (6 ಗಂಟೆಗಳ ವಿವರ)",
        "crop_weather_risk": "ಬೆಳೆ ಹವಾಮಾನ ಅಪಾಯದ ವಿಶ್ಲೇಷಣೆ",
        "rule_based_tag": "ನಿಯಮ-ಆಧಾರಿತ ಅಂದಾಜು, ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಯಲ್ಲ",
        "no_rule_for_crop": "ℹ️ ಮುಂದಿನ 48 ಗಂಟೆಗಳಲ್ಲಿ ಈ ಸ್ಥಿತಿಗೆ ಯಾವುದೇ ಹವಾಮಾನ ನಿಯಮ ಅನ್ವಯಿಸುವುದಿಲ್ಲ.",
        "models_missing_msg": "ಮಾದರಿ ಕಡತಗಳು ಕಂಡುಬಂದಿಲ್ಲ. ಚಲಾಯಿಸಿ: python download_models.py",
        "cross_crop_refusal_title": "ಬೆಳೆ ಆಯ್ಕೆ ಹೊಂದಾಣಿಕೆಯಾಗುತ್ತಿಲ್ಲ",
        "cross_crop_refusal_msg": "ಇದು {other_crop} ಫೋಟೋದಂತೆ ಕಾಣುತ್ತಿದೆ. ನಿಮ್ಮ ಬೆಳೆ ಆಯ್ಕೆಯನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    "hi": {
        "app_name": "AgriTrust AI",
        "greeting": "नमस्ते, किसान भाई आपका स्वागत है",
        "district_label": "ज़िला",
        "live_forecast_badge": "लाइव पूर्वानुमान",
        "offline_badge": "ऑफलाइन",
        "temp_label": "तापमान",
        "humidity_label": "नमी",
        "rain_48h_label": "48 घंटे में बारिश",
        "weather_status_normal": "सामान्य",
        "weather_status_rain": "बारिश की संभावना",
        "weather_status_dry": "शुष्क",
        # Scan Card
        "scan_card_title": "फोटो स्कैन / अपलोड करें",
        "tip_line": "दिन की रोशनी में, पास से एक पत्ती या फल की फोटो लें",
        "choose_crop": "फसल चुनें",
        "crop_paddy": "धान",
        "crop_arecanut": "सुपारी",
        "crop_apple": "सेब",
        "take_photo": "कैमरा से फोटो खींचो",
        "upload_file": "या फोन से फोटो अपलोड करो",
        # Last result card
        "last_result_title": "आपका पिछला परिणाम",
        "no_scans_yet": "अभी तक कोई स्कैन नहीं हुआ।",
        "crop_label": "फसल",
        "date_label": "दिनांक",
        "confidence_label": "सटीकता",
        "recommendation_label": "अनुशंसा",
        # Result Screen
        "back_btn": "← वापस जाएं (नई फोटो लें)",
        "status_healthy": "स्वस्थ",
        "status_treat": "उपचार संभव",
        "status_no_cure": "इलाज नहीं / विशेषज्ञ से मिलें",
        "status_unsure": "संदेहास्पद",
        "what_to_do": "क्या करें",
        "organic_title": "जैविक और प्राकृतिक उपाय",
        "severe_expander": "यदि समस्या गंभीर हो (रासायनिक उपचार)",
        "weather_alert_title": "48 घंटे का मौसम अलर्ट",
        "soil_card_title": "मृदा स्वास्थ्य कार्ड",
        "listen_audio": "बोलकर सुनें (आवाज में सलाह)",
        "experts_expander": "विशेषज्ञों और परीक्षकों के लिए",
        "trust_audit_title": "विश्वसनीयता एवं ऑडिट रिपोर्ट",
        "apple_public_banner": "प्रयोगशाला (PlantVillage) और सार्वजनिक बगीचे की तस्वीरों (Cornell डेटासेट, उत्तर भारत नहीं) पर फाइन-ट्यून किया गया। Black rot को केवल लैब तस्वीरों पर प्रशिक्षित किया गया था। गंभीरता ग्रेडिंग रोक दी गई है।",
        "areca_provisional": "अनंतिम: स्वतंत्र रूप से मान्य नहीं किया गया।",
        "paddy_severity_qwk": "ईमानदार टेस्ट स्कोर: असली खेत के टेस्ट में स्कोर लगभग 0.67 है; मार्केट में बोला जाने वाला 0.94 का स्कोर बढ़ा-चढ़ाकर दिखाया गया है।",
        "paddy_severity_fixed": "इस वर्ग के लिए गंभीरता एक निश्चित नियम द्वारा निर्धारित की गई है।",
        "tungro_warning_title": "तुंग्रो चेतावनी",
        "cedar_rust_caption": "⚠️ अमेरिकी सार्वजनिक डेटा से वर्ग; स्थानीय केवीके से पुष्टि करें।",
        "areca_bud_borer_warning": "⚠️ कम विश्वसनीयता सूचना: मॉडल कली छेदक और पीली पत्ती रोग में भ्रमित होता है। उपचार से पहले केवीके से पुष्टि करें।",
        "footer_disclaimer_tx": "⚠️ उपयोग से पहले ICAR / CPCRI / CIB&RC से पुष्टि करें।",
        "footer_disclaimer_soil": "⚠️ अपने केवीके / मृदा स्वास्थ्य कार्ड से जांच करें।",
        # Refusal states
        "refusal_ood_title": "फोटो पहचान में नहीं आई",
        "refusal_ood_msg": "यह फोटो धान, सुपारी या सेब की साफ और पास से ली गई फोटो नहीं लग रही है। कृपया अच्छी रोशनी में बीमारी वाली पत्ती, तने या फल के बिल्कुल पास जाकर दोबारा फोटो खींचो।",
        "refusal_abstain_title": "नतीजा रोक दिया गया (कम भरोसा)",
        "refusal_abstain_msg": "बीमारी का पक्का पता नहीं चल पाया (भरोसा 70% से कम है)। फोटो धुंधली है या रोशनी कम है। गलत अंदाज़ा लगाने से बचने के लिए ऐप रुक गया है। कृपया साफ फोटो दोबारा खींचो या अपने केवीके से सलाह लो।",
        # Navigation
        "tab_home": "होम",
        "tab_chat": "किसान चैट",
        "tab_weather": "मौसम",
        # AI Chat
        "chat_title": "कृषि सहायक चैट",
        "chat_no_context": "पहले एक पत्ती स्कैन करें, फिर मैं परिणाम समझा सकता हूँ।",
        "chat_disclaimer": "उत्तर आपके पिछले स्कैन में दिखाई गई सलाह पर आधारित हैं। अपने केवीके से पुष्टि करें।",
        "chat_clear_btn": "चैट साफ करें",
        "chat_placeholder": "अपने स्कैन के बारे में सवाल पूछें...",
        "chip_first": "मुझे पहले क्या करना चाहिए?",
        "chip_cure": "क्या ये बीमारी ठीक हो सकती है?",
        "chip_spray": "स्प्रे कब और कैसे करना चाहिए?",
        "chip_soil": "मिट्टी की सलाह",
        "chip_weather": "मौसम का जोखिम",
        "chip_unsure": "ऐप को पक्का भरोसा क्यों नहीं है?",
        "chat_unmatched": "मैं केवल आपके पिछले परिणाम की व्याख्या कर सकता हूँ। कृपया अपने नजदीकी केवीके या कृषि अधिकारी से संपर्क करें।",
        # Weather View
        "weather_view_title": "48 घंटे का मौसम और रोग का दबाव",
        "weather_6h_blocks": "48 घंटे का मौसम सारांश (6 घंटे के अंतराल पर)",
        "crop_weather_risk": "फसल मौसम जोखिम विश्लेषण",
        "rule_based_tag": "नियम-आधारित अनुमान, मौसम पूर्वानुमान नहीं",
        "no_rule_for_crop": "ℹ️ अगले 48 घंटों में इस रोग के लिए कोई मौसम नियम सक्रिय नहीं है।",
        "models_missing_msg": "मॉडल फ़ाइलें नहीं मिलीं। चलाएं: python download_models.py",
        "cross_crop_refusal_title": "गलत फसल का चयन",
        "cross_crop_refusal_msg": "यह {other_crop} की तस्वीर जैसी लग रही है। कृपया अपनी चुनी हुई फसल की जांच करें।",
    }
}

def get_ui_string(key: str, lang: str = "en") -> str:
    lang_dict = UI_STRINGS.get(lang, UI_STRINGS["en"])
    return lang_dict.get(key, UI_STRINGS["en"].get(key, key))

# ─── MINIMAL THEME CSS (ALL IN ONE BLOCK, NO EXTERNAL CSS/JS) ────────────────
CSS_STYLES = f"""
<style>
    /* App Canvas & Font Stack */
    .stApp,
    [data-testid="stAppViewContainer"],
    section.main,
    .main {{
        background-color: {THEME_CONFIG['bg']} !important;
        font-family: {THEME_CONFIG['font_stack']} !important;
        font-size: {THEME_CONFIG['base_font_size']} !important;
        color: {THEME_CONFIG['text_dark']} !important;
    }}

    /* Mobile First Centered Container (Max Width 760px) */
    header[data-testid="stHeader"], [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        z-index: 999999 !important;
        color: #275637 !important;
    }}

    .block-container {{
        max-width: {THEME_CONFIG['max_width']} !important;
        margin: 0 auto !important;
        padding-top: 2.8rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    /* Global Headings */
    h1, h2, h3, h4, h5, h6 {{
        font-family: {THEME_CONFIG['font_stack']} !important;
        color: {THEME_CONFIG['primary']} !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }}

    /* Clean Card System (14px radius, 1px sage border, no heavy shadow) */
    .ui-card {{
        background: {THEME_CONFIG['card_bg']} !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
        border-radius: {THEME_CONFIG['border_radius']} !important;
        padding: 18px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: none !important;
    }}

    /* Refusal / Neutral Card */
    .refusal-card {{
        background: {THEME_CONFIG['refusal_bg']} !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
        border-radius: {THEME_CONFIG['border_radius']} !important;
        padding: 24px 20px !important;
        text-align: center !important;
        margin-bottom: 16px !important;
    }}

    /* Traffic-Light Status Badges */
    /* Rule: Text on sage (#91BA94), light green (#B1D27D) and warn yellow (#F6D05F) is ALWAYS #275637, NEVER white */
    .badge-status {{
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 8px 16px !important;
        border-radius: 24px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;
    }}
    .badge-healthy {{
        background-color: {THEME_CONFIG['light_green']} !important;
        color: {THEME_CONFIG['primary']} !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
    }}
    .badge-treat {{
        background-color: {THEME_CONFIG['warn_yellow']} !important;
        color: {THEME_CONFIG['primary']} !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
    }}
    .badge-no-cure {{
        background-color: {THEME_CONFIG['danger']} !important;
        color: #FFFFFF !important;
    }}
    .badge-unsure {{
        background-color: {THEME_CONFIG['sage']} !important;
        color: {THEME_CONFIG['primary']} !important;
    }}

    /* Tap Targets >= 48px */
    button, .stButton > button, select, input:not([type="radio"]):not([type="checkbox"]) {{
        min-height: {THEME_CONFIG['min_tap_target']} !important;
        font-family: {THEME_CONFIG['font_stack']} !important;
        border-radius: 12px !important;
    }}

    /* Language Selector & Radio Group Styling */
    div[data-testid="stRadio"] {{
        z-index: 1000 !important;
        position: relative !important;
        width: 100% !important;
        margin: 6px 0 16px 0 !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioGroup"],
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: stretch !important;
        gap: 8px !important;
        width: 100% !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"],
    div[data-testid="stRadio"] label {{
        flex: 1 1 0 !important;
        justify-content: center !important;
        cursor: pointer !important;
        padding: 8px 6px !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid {THEME_CONFIG['sage']} !important;
        border-radius: 12px !important;
        display: inline-flex !important;
        align-items: center !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: {THEME_CONFIG['primary']} !important;
        min-height: 44px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"] *,
    div[data-testid="stRadio"] label *,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label div {{
        color: {THEME_CONFIG['primary']} !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"]:hover,
    div[data-testid="stRadio"] label:hover {{
        background-color: #F1F5EE !important;
        border-color: {THEME_CONFIG['primary']} !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
    div[data-testid="stRadio"] label[data-selected="true"],
    div[data-testid="stRadio"] label:has(input:checked) {{
        background-color: {THEME_CONFIG['light_green']} !important;
        border-color: {THEME_CONFIG['primary']} !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] *,
    div[data-testid="stRadio"] label[data-selected="true"] *,
    div[data-testid="stRadio"] label:has(input:checked) *,
    div[data-testid="stRadio"] label:has(input:checked) p {{
        font-weight: 700 !important;
        color: {THEME_CONFIG['primary']} !important;
    }}

    /* Button Styling: Secondary (Default) & Primary (Active/Highlighted) */
    .stButton > button,
    button[data-testid="stBaseButton-secondary"],
    .stButton > button:not([data-testid="stBaseButton-primary"]) {{
        background-color: #FFFFFF !important;
        color: {THEME_CONFIG['primary']} !important;
        border: 1.5px solid {THEME_CONFIG['sage']} !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100% !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }}
    .stButton > button:not([data-testid="stBaseButton-primary"]) *,
    button[data-testid="stBaseButton-secondary"] *,
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] span,
    button[data-testid="stBaseButton-secondary"] div {{
        color: {THEME_CONFIG['primary']} !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    .stButton > button:not([data-testid="stBaseButton-primary"]):hover,
    button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: #F1F5EE !important;
        border-color: {THEME_CONFIG['primary']} !important;
    }}
    .stButton > button:not([data-testid="stBaseButton-primary"]):hover *,
    button[data-testid="stBaseButton-secondary"]:hover * {{
        color: {THEME_CONFIG['primary']} !important;
    }}

    button[data-testid="stBaseButton-primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background-color: {THEME_CONFIG['primary']} !important;
        color: #FFFFFF !important;
        border: 1.5px solid {THEME_CONFIG['primary']} !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        width: 100% !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }}
    button[data-testid="stBaseButton-primary"] *,
    .stButton > button[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] span,
    button[data-testid="stBaseButton-primary"] div {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background-color: #1c3f28 !important;
        border-color: #1c3f28 !important;
    }}
    button[data-testid="stBaseButton-primary"]:hover * {{
        color: #FFFFFF !important;
    }}

    /* Top Tabs Styling */
    div[data-testid="stTabs"] {{
        width: 100% !important;
        margin-top: 8px !important;
    }}
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 4px !important;
        width: 100% !important;
        display: flex !important;
        justify-content: space-between !important;
        background-color: transparent !important;
    }}
    button[data-baseweb="tab"] {{
        flex: 1 1 0 !important;
        min-height: 46px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: {THEME_CONFIG['primary']} !important;
        padding: 8px 6px !important;
        text-align: center !important;
        white-space: nowrap !important;
        overflow: visible !important;
        background-color: transparent !important;
    }}
    button[data-baseweb="tab"] *,
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {{
        color: {THEME_CONFIG['primary']} !important;
        overflow: visible !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {THEME_CONFIG['primary']} !important;
        border-bottom-color: {THEME_CONFIG['primary']} !important;
        border-bottom-width: 3px !important;
        font-weight: 700 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] * {{
        color: {THEME_CONFIG['primary']} !important;
        font-weight: 700 !important;
    }}

    /* Chat Bubbles */
    .chat-bubble-ai {{
        background-color: #FFFFFF !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
        border-radius: 14px 14px 14px 4px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        color: {THEME_CONFIG['primary']} !important;
        max-width: 88% !important;
        line-height: 1.5 !important;
    }}
    .chat-bubble-user {{
        background-color: {THEME_CONFIG['light_green']} !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
        border-radius: 14px 14px 4px 14px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        color: {THEME_CONFIG['primary']} !important;
        max-width: 88% !important;
        margin-left: auto !important;
        line-height: 1.5 !important;
    }}

    /* Weather 6-Hour Block Card */
    .weather-block-card {{
        background: #FFFFFF !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
        border-radius: 10px !important;
        padding: 10px 8px !important;
        text-align: center !important;
        margin-bottom: 8px !important;
    }}

    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {THEME_CONFIG['primary']} !important;
    }}
</style>
"""

# ─── LOCALIZED DISEASE NAMES (ALL 23 CLASSES) ─────────────────────────────────
DISEASE_LOCALIZATION: Dict[str, Dict[str, Dict[str, str]]] = {
    "Paddy": {
        "bacterial_leaf_blight": {
            "en": "Bacterial Leaf Blight",
            "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಕರಕು ರೋಗ",
            "hi": "जीवाणु पत्ती झुलसा",
        },
        "bacterial_leaf_streak": {
            "en": "Bacterial Leaf Streak",
            "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಪಟ್ಟೆ ರೋಗ",
            "hi": "जीवाणु पत्ती धारी",
        },
        "bacterial_panicle_blight": {
            "en": "Bacterial Panicle Blight",
            "kn": "ತೆನೆ ಕರಕು ರೋಗ",
            "hi": "जीवाणु बाली झुलसा",
        },
        "blast": {
            "en": "Rice Blast",
            "kn": "ಬೆಂಕಿ ರೋಗ (ಬ್ಲಾಸ್ಟ್)",
            "hi": "झुलसा रोग (ब्लास्ट)",
        },
        "brown_spot": {
            "en": "Brown Spot",
            "kn": "ಕಂದು ಚುಕ್ಕೆ ರೋಗ",
            "hi": "भूरा धब्बा रोग",
        },
        "dead_heart": {
            "en": "Dead Heart",
            "kn": "ಸುಳಿ ಒಣಗು ರೋಗ (ಡೆಡ್ ಹಾರ್ಟ್)",
            "hi": "डेड हार्ट (तना छेदक प्रकोप)",
        },
        "downy_mildew": {
            "en": "Downy Mildew",
            "kn": "ಬೂದಿ ರೋಗ",
            "hi": "डाउनी फफूंदी",
        },
        "hispa": {
            "en": "Hispa",
            "kn": "ಹಿಸ್ಪಾ ಕೀಟ ಬಾಧೆ",
            "hi": "हिस्पा कीट प्रकोप",
        },
        "normal": {
            "en": "Healthy (No Disease)",
            "kn": "ಆರೋಗ್ಯಕರ ಬೆಳೆ",
            "hi": "स्वस्थ फसल",
        },
        "tungro": {
            "en": "Tungro Virus",
            "kn": "ತುಂಗ್ರೋ ವೈರಸ್ ರೋಗ",
            "hi": "तुंग्रो वायरस",
        },
    },
    "Arecanut": {
        "bud_borer": {
            "en": "Bud Borer",
            "kn": "ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು",
            "hi": "कली छेदक कीट",
        },
        "healthy_foot": {
            "en": "Healthy (Foot)",
            "kn": "ಆರೋಗ್ಯಕರ ಬುಡ",
            "hi": "स्वस्थ तना आधार",
        },
        "healthy_leaf": {
            "en": "Healthy (Leaf)",
            "kn": "ಆರೋಗ್ಯಕರ ಗರಿ",
            "hi": "स्वस्थ पत्ती",
        },
        "healthy_nut": {
            "en": "Healthy (Nut)",
            "kn": "ಆರೋಗ್ಯಕರ ಅಡಿಕೆ ಕಾಯಿ",
            "hi": "स्वस्थ सुपारी फल",
        },
        "healthy_trunk": {
            "en": "Healthy (Trunk)",
            "kn": "ಆರೋಗ್ಯಕರ ಮರ",
            "hi": "स्वस्थ मुख्य तना",
        },
        "mahali_koleroga": {
            "en": "Mahali / Koleroga (Rot Disease)",
            "kn": "ಮಹಾಳಿ / ಕೊಳೆರೋಗ",
            "hi": "महाली / कोलेरोगा सड़ांध",
        },
        "stem_bleeding": {
            "en": "Stem Bleeding",
            "kn": "ಕಾಂಡ ಸೋರುವಿಕೆ ರೋಗ",
            "hi": "तना स्राव रोग",
        },
        "stem_cracking": {
            "en": "Stem Cracking",
            "kn": "ಕಾಂಡ ಬಿರುಕು ರೋಗ",
            "hi": "तना फटना रोग",
        },
        "yellow_leaf_disease": {
            "en": "Yellow Leaf Disease",
            "kn": "ಹಳದಿ ಎಲೆ ರೋಗ",
            "hi": "पीली पत्ती रोग",
        },
    },
    "Apple": {
        "apple_scab": {
            "en": "Apple Scab",
            "kn": "ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ರೋಗ",
            "hi": "सेब का स्केब रोग",
        },
        "black_rot": {
            "en": "Black Rot (Frog-Eye Leaf Spot)",
            "kn": "ಕಪ್ಪು ಕೊಳೆರೋಗ",
            "hi": "ब्लैक रॉट (काला सड़न रोग)",
        },
        "cedar_apple_rust": {
            "en": "Cedar Apple Rust",
            "kn": "ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ)",
            "hi": "सीडर एप्पल रस्ट (गेरुआ रोग)",
        },
        "healthy": {
            "en": "Healthy (No Disease)",
            "kn": "ಆರೋಗ್ಯಕರ ಎಲೆ",
            "hi": "स्वस्थ पत्तियां",
        },
    },
}

# ─── LOCALIZED CARD TREATMENT DICTIONARY (ALL 23 CLASSES) ────────────────────
TREATMENT_LOCALIZATION: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Paddy": {
        "bacterial_leaf_blight": {
            "status": "treat",
            "organic_advice": {
                "en": "Spray neem seed kernel extract. Avoid excess nitrogen top-dressing; follow product label and local KVK advice.",
                "kn": "ಬೇವಿನ ಬೀಜದ ಕಷಾಯ ಸಿಂಪಡಿಸಿ. ಹೆಚ್ಚಿನ ಸಾರಜನಕ ಗೊಬ್ಬರ ನೀಡುವುದನ್ನು ತಪ್ಪಿಸಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "नीम के बीज का काढ़ा छिड़कें। यूरिया का अधिक उपयोग न करें; स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Apply Copper Oxychloride + Streptocycline; follow product label and local KVK advice.",
                "kn": "Copper Oxychloride ಮತ್ತು Streptocycline ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Copper Oxychloride और Streptocycline का छिड़काव करें; लेबल और KVK की सलाह का पालन करें।",
            },
        },
        "bacterial_leaf_streak": {
            "status": "treat",
            "organic_advice": {
                "en": "Promote balanced fertilizer application and avoid excess urea; follow local KVK advice.",
                "kn": "ಸಮತೋಲಿತ ಗೊಬ್ಬರ ಬಳಸಿ ಮತ್ತು ಹೆಚ್ಚಿನ ಯೂರಿಯಾ ತಪ್ಪಿಸಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "संतुलित उर्वरक का प्रयोग करें और अतिरिक्त यूरिया से बचें; स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Apply copper hydroxide spray; follow product label and local KVK advice.",
                "kn": "ತಾಮ್ರದ ಹೈಡ್ರಾಕ್ಸೈಡ್ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "कॉपर हाइड्रॉक्साइड का छिड़काव करें; उत्पाद लेबल और स्थानीय KVK की सलाह का पालन करें।",
            },
        },
        "bacterial_panicle_blight": {
            "status": "treat",
            "organic_advice": {
                "en": "Avoid high nitrogen application at boot stage; follow local KVK advice.",
                "kn": "ತೆನೆ ಬರುವ ಹಂತದಲ್ಲಿ ಹೆಚ್ಚಿನ ಸಾರಜನಕ ಹಾಕಬೇಡಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "बाली निकलने की अवस्था में अधिक नाइट्रोजन न डालें; स्थानीय KVK की सलाह लें।",
            },
            "chemical_advice": {
                "en": "Apply preventive copper-based bactericide at early heading; follow product label and local KVK advice.",
                "kn": "ತೆನೆ ಮೂಡುವ ಆರಂಭದಲ್ಲಿ ತಾಮ್ರ ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಯಾನಾಶಕ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "शुरुआती बाली अवस्था में तांबा आधारित जीवाणुनाशी का छिड़काव करें; लेबल और KVK की सलाह का पालन करें।",
            },
        },
        "blast": {
            "status": "treat",
            "organic_advice": {
                "en": "Apply bio-fungicide Trichoderma viride or neem-based formulation; follow product label and local KVK advice.",
                "kn": "Trichoderma viride ಜೈವಿಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ ಬೇವಿನ ಸೂತ್ರೀಕರಣವನ್ನು ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆಯನ್ನು ಪಾಲಿಸಿ.",
                "hi": "जैव कवकनाशी Trichoderma viride या नीम आधारित कीटनाशक का छिड़काव करें; लेबल और स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Apply systemic fungicide Tricyclazole 75 WP; follow product label and local KVK advice.",
                "kn": "ಶಿಫಾರಸು ಮಾಡಿದ Tricyclazole 75 WP ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "प्रणालीगत कवकनाशी Tricyclazole 75 WP का छिड़काव करें; लेबल निर्देश और KVK की सलाह का पालन करें।",
            },
        },
        "brown_spot": {
            "status": "treat",
            "organic_advice": {
                "en": "Apply compost and correct potassium deficiency; follow local KVK advice.",
                "kn": "ಕಾಂಪೋಸ್ಟ್ ಗೊಬ್ಬರ ಹಾಕಿ ಮತ್ತು ಪೊಟ್ಯಾಷ್ ಕೊರತೆಯನ್ನು ಸರಿಪಡಿಸಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "कम्पोस्ट खाद डालें और पोटाश की कमी को दूर करें; स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Apply Mancozeb protective spray; follow product label and local KVK advice.",
                "kn": "Mancozeb ರಕ್ಷಣಾತ್ಮಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Mancozeb कवकनाशी का सुरक्षात्मक छिड़काव करें; लेबल और KVK की सलाह का पालन करें।",
            },
        },
        "dead_heart": {
            "status": "treat",
            "organic_advice": {
                "en": "Clip seedling tips before transplanting; install yellow stem borer pheromone traps; follow local KVK advice.",
                "kn": "ನಾಟಿ ಮಾಡುವ ಮುನ್ನ ಸಸಿಗಳ ತುದಿಯನ್ನು ಕತ್ತರಿಸಿ; ಹಳದಿ ಕಾಂಡ ಕೊರೆಯುವ ಕೀಟದ ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "रोपाई से पहले पौध की नोक काटें; तना छेदक के लिए फेरोमोन ट्रैप लगाएं; KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Release Trichogramma japonicum egg parasitoids; install pheromone traps; follow local KVK advice.",
                "kn": "Trichogramma japonicum ಮೊಟ್ಟೆ ಪರಾವಲಂಬಿಗಳನ್ನು ಬಿಡಿ; ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Trichogramma japonicum परजीवी कीट छोड़ें; फेरोमोन ट्रैप लगाएं; स्थानीय KVK की सलाह का पालन करें।",
            },
        },
        "downy_mildew": {
            "status": "treat",
            "organic_advice": {
                "en": "Rogue out infected volunteer plants and weed hosts; follow local KVK advice.",
                "kn": "ಸೋಂಕಿತ ಸಸಿಗಳು ಮತ್ತು ಕಳೆ ಸಸ್ಯಗಳನ್ನು ಕಿತ್ತು ನಾಶಪಡಿಸಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "संक्रमित पौधों और खरपतवारों को उखाड़कर नष्ट करें; स्थानीय KVK की सलाह लें।",
            },
            "chemical_advice": {
                "en": "Apply Metalaxyl-M foliar spray; follow product label and local KVK advice.",
                "kn": "Metalaxyl-M ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Metalaxyl-M का छिड़काव करें; उत्पाद लेबल और स्थानीय KVK की सलाह का पालन करें।",
            },
        },
        "hispa": {
            "status": "treat",
            "organic_advice": {
                "en": "Manually clip infected leaf tips containing grubs; use sweep nets; follow local KVK advice.",
                "kn": "ಮರಿ ಕೀಟಗಳಿರುವ ಎಲೆಗಳ ತುದಿಯನ್ನು ಕೈಯಿಂದ ಕತ್ತರಿಸಿ ನಾಶಪಡಿಸಿ; ಜಾಲರಿ ಬಳಸಿ ಕೀಟ ಹಿಡಿಯಿರಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "कीट युक्त पत्तियों की नोक को काटकर नष्ट करें; कीट पकड़ने वाले जाल का उपयोग करें; स्थानीय KVK की सलाह मानें।",
            },
            "chemical_advice": {
                "en": "Manually clip affected leaf tips containing grubs; consult your KVK for a registered product; follow product label and local KVK advice.",
                "kn": "ಕೀಟವಿರುವ ಎಲೆಗಳ ತುದಿಯನ್ನು ಕತ್ತರಿಸಿ; ನೋಂದಾಯಿತ ಔಷಧಿಗಾಗಿ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಲೇಬಲ್ ಸೂಚನೆ ಪಾಲಿಸಿ.",
                "hi": "संक्रमित पत्तियों की नोक काटें; अनुमोदित कीटनाशक के लिए KVK से संपर्क करें और लेबल का पालन करें।",
            },
        },
        "normal": {
            "status": "healthy",
            "organic_advice": {
                "en": "No treatment needed. Maintain optimal water depth and balanced N-P-K fertilizer.",
                "kn": "ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ಸಮತೋಲಿತ ಗೊಬ್ಬರ ಮತ್ತು ನೀರಿನ ನಿರ್ವಹಣೆ ಮಾಡಿ.",
                "hi": "किसी उपचार की आवश्यकता नहीं है। संतुलित खाद और पानी का उचित प्रबंधन रखें।",
            },
            "chemical_advice": {
                "en": "No chemical treatment needed.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "किसी रासायनिक उपचार की आवश्यकता नहीं है।",
            },
        },
        "tungro": {
            "status": "no_cure",
            "organic_advice": {
                "en": "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ, ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
                "hi": "कोई रासायनिक उपचार उपलब्ध नहीं है। रसचूसक कीटों को नियंत्रित करें, रोगी पौधों को हटाएं और तुरंत KVK से संपर्क करें।",
            },
            "chemical_advice": {
                "en": "No chemical cure exists for viral/phytoplasma infections.",
                "kn": "ವೈರಸ್ ಸೋಂಕಿಗೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ.",
                "hi": "वायरस संक्रमण के लिए कोई रासायनिक इलाज नहीं है।",
            },
        },
    },
    "Arecanut": {
        "bud_borer": {
            "status": "treat",
            "organic_advice": {
                "en": "Clear leaf axils of organic debris; hook out larvae; apply light traps; follow local KVK advice.",
                "kn": "ಗರಿಗಳ ಸಂದಿಗಳಿಂದ ತ್ಯಾಜ್ಯವನ್ನು ತೆಗೆದು ಸ್ವಚ್ಛಗೊಳಿಸಿ; ಹುಳುಗಳನ್ನು ಹೆಕ್ಕಿ ತೆಗೆಯಿರಿ; ಬೆಳಕಿನ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "पत्तियों के जोड़ से कचरा साफ करें; कीड़ों को बाहर निकालें; प्रकाश प्रपंच लगाएं; स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Apply neem oil formulation into the crown; consult your KVK for a registered product; follow product label and local KVK advice.",
                "kn": "ಮರದ ಸುಳಿಗೆ ಬೇವಿನ ಎಣ್ಣೆ ಸೂತ್ರೀಕರಣ ಹಾಕಿ; ನೋಂದಾಯಿತ ಔಷಧಿಗಾಗಿ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಲೇಬಲ್ ಪಾಲಿಸಿ.",
                "hi": "सुपारी की गोफ में नीम का तेल डालें; अनुमोदित कीटनाशक के लिए KVK से संपर्क करें और लेबल का पालन करें।",
            },
        },
        "healthy_foot": {
            "status": "healthy",
            "organic_advice": {
                "en": "No treatment needed. Inspect palm crowns periodically.",
                "kn": "ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ನಿಯಮಿತವಾಗಿ ತೋಟವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "hi": "किसी उपचार की आवश्यकता नहीं है। समय-समय पर बाग का निरीक्षण करें।",
            },
            "chemical_advice": {
                "en": "No chemical treatment needed.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "किसी रासायनिक उपचार की आवश्यकता नहीं है।",
            },
        },
        "healthy_leaf": {
            "status": "healthy",
            "organic_advice": {
                "en": "No treatment needed. Inspect palm crowns periodically.",
                "kn": "ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ನಿಯಮಿತವಾಗಿ ತೋಟವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "hi": "کسی उपचार की आवश्यकता नहीं है। समय-समय पर बाग का निरीक्षण करें।",
            },
            "chemical_advice": {
                "en": "No chemical treatment needed.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "किसी रासायनिक उपचार की आवश्यकता नहीं है।",
            },
        },
        "healthy_nut": {
            "status": "healthy",
            "organic_advice": {
                "en": "No treatment needed. Inspect palm crowns periodically.",
                "kn": "ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ನಿಯಮಿತವಾಗಿ ತೋಟವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "hi": "किसी उपचार की आवश्यकता नहीं है। समय-समय पर बाग का निरीक्षण करें।",
            },
            "chemical_advice": {
                "en": "No chemical treatment needed.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "کسی रासायनिक उपचार की आवश्यकता नहीं है।",
            },
        },
        "healthy_trunk": {
            "status": "healthy",
            "organic_advice": {
                "en": "No treatment needed. Inspect palm crowns periodically.",
                "kn": "ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ನಿಯಮಿತವಾಗಿ ತೋಟವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "hi": "किसी उपचार की आवश्यकता नहीं है। समय-समय पर बाग का निरीक्षण करें।",
            },
            "chemical_advice": {
                "en": "No chemical treatment needed.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "کسی रासायनिक उपचार की आवश्यकता नहीं है।",
            },
        },
        "mahali_koleroga": {
            "status": "treat",
            "organic_advice": {
                "en": "Tie polythene covers over developing bunches; apply preventive 1% Bordeaux Mixture before monsoon; follow product label and local KVK advice.",
                "kn": "ಬೆಳೆಯುತ್ತಿರುವ ಅಡಿಕೆ ಗೊನೆಗಳಿಗೆ ಪಾಲಿಥಿನ್ ಹೊದಿಕೆ ಕಟ್ಟಿ; ಮುಂಗಾರಿಗೆ ಮುನ್ನ ಮುನ್ನೆಚ್ಚರಿಕೆಯಾಗಿ ಶೇಕಡಾ 1 ರ Bordeaux Mixture ಸಿಂಪಡಿಸಿ ಮತ್ತು KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "सुपारी के गुच्छों पर पॉलीथिन बांधें; मानसून से पहले बचाव के लिए 1% Bordeaux Mixture का छिड़काव करें और KVK की सलाह मानें।",
            },
            "chemical_advice": {
                "en": "Spray 1% Bordeaux Mixture with adhesive sticker on all nut bunches; follow product label and local KVK advice.",
                "kn": "ಎಲ್ಲಾ ಅಡಿಕೆ ಗೊನೆಗಳ ಮೇಲೆ ಅಂಟು ದ್ರವದೊಂದಿಗೆ ಶೇಕಡಾ 1 ರ Bordeaux Mixture ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "सभी गुच्छों पर चिपकाने वाले पदार्थ के साथ 1% Bordeaux Mixture का छिड़काव करें; लेबल और KVK की सलाह का पालन करें।",
            },
        },
        "stem_bleeding": {
            "status": "treat",
            "organic_advice": {
                "en": "Chisel out bleeding bark patch down to healthy tissue; apply Bordeaux paste; consult your KVK for a registered product; follow local KVK advice.",
                "kn": "ರಸ ಸೋರುವ ತೊಗಟೆಯನ್ನು ಆರೋಗ್ಯಕರ ಭಾಗದವರೆಗೆ ಕೆತ್ತಿ ತೆಗೆಯಿರಿ; Bordeaux paste ಹಚ್ಚಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "तने से रस बहने वाले हिस्से को स्वस्थ लकड़ी तक छीलें; Bordeaux paste लगाएं; स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Chisel out infected bark; apply Bordeaux paste; consult your KVK for a registered product (e.g. Tridemorph); follow product label and local KVK advice.",
                "kn": "ಸೋಂಕಿತ ತೊಗಟೆಯನ್ನು ಕೆತ್ತಿ ತೆಗೆದು Bordeaux paste ಹಚ್ಚಿ; ಶಿಫಾರಸು ಮಾಡಿದ ಔಷಧಿಗಾಗಿ (Tridemorph) ನಿಮ್ಮ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
                "hi": "संक्रमित छाल को छीलें और Bordeaux paste लगाएं; अनुशंसित दवा के लिए (Tridemorph) अपने KVK से संपर्क करें।",
            },
        },
        "stem_cracking": {
            "status": "treat",
            "organic_advice": {
                "en": "Maintain even irrigation schedule and organic mulching around palm basin; follow local KVK advice.",
                "kn": "ಸಮತೋಲಿತ ನೀರಾವರಿ ನಿರ್ವಹಣೆ ಮಾಡಿ ಮತ್ತು ಮರದ ಬುಡಕ್ಕೆ ಸಾವಯವ ಮಲ್ಚಿಂಗ್ ಹಾಕಿ; ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "नियमित सिंचाई करें और तने के चारों ओर जैविक मल्चिंग करें; स्थानीय KVK की सलाह का पालन करें।",
            },
            "chemical_advice": {
                "en": "Conduct a soil test for boron deficiency; maintain even irrigation and organic mulching; follow local KVK advice.",
                "kn": "ಬೋರಾನ್ ಕೊರತೆಗಾಗಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ; ಸಮತೋಲಿತ ನೀರಾವರಿ ಮತ್ತು ಸಾವಯವ ಮಲ್ಚಿಂಗ್ ಮುಂದುವರಿಸಿ; KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "बोरॉन की कमी के लिए मिट्टी परीक्षण करवाएं; नियमित सिंचाई और जैविक मल्चिंग बनाए रखें; KVK की सलाह लें।",
            },
        },
        "yellow_leaf_disease": {
            "status": "no_cure",
            "organic_advice": {
                "en": "No chemical cure exists. Manage insect vectors, remove infected plants, consult your KVK.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ, ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
                "hi": "कोई रासायनिक उपचार उपलब्ध नहीं है। रसचूसक कीटों को नियंत्रित करें, रोगी पौधों को हटाएं और तुरंत KVK से संपर्क करें।",
            },
            "chemical_advice": {
                "en": "No chemical cure exists for viral/phytoplasma infections.",
                "kn": "ಫೈಟೋಪ್ಲಾಸ್ಮಾ ಸೋಂಕಿಗೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ.",
                "hi": "फाइटोप्लाज्मा संक्रमण के लिए कोई रासायनिक इलाज नहीं है।",
            },
        },
    },
    "Apple": {
        "apple_scab": {
            "status": "treat",
            "organic_advice": {
                "en": "Rake and compost fallen leaves to reduce overwintering ascospore inoculum. Maintain open canopy pruning for air circulation and rapid leaf drying.",
                "kn": "ಉದುರಿದ ಎಲೆಗಳನ್ನು ಹೆಕ್ಕಿ ಗೊಬ್ಬರ ಮಾಡಿ ಮತ್ತು ಎಲೆಗಳು ಬೇಗನೆ ಒಣಗಲು ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ ಗಾಳಿಯಾಡುವಂತೆ ಮಾಡಿ.",
                "hi": "गिरी हुई पत्तियों को इकट्ठा करके नष्ट करें और हवा के आवागमन के लिए पेड़ों की छंटाई करें ताकि पत्तियां जल्दी सूखें।",
            },
            "chemical_advice": {
                "en": "Apply protective fungicides such as Captan or Mancozeb, or systemic Difenoconazole; follow product label and local KVK advice.",
                "kn": "Captan ಅಥವಾ Mancozeb ರಕ್ಷಣಾತ್ಮಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ Difenoconazole ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Captan या Mancozeb जैसे सुरक्षात्मक कवकनाशी या Difenoconazole का छिड़काव करें; लेबल और KVK की सलाह का पालन करें।",
            },
        },
        "black_rot": {
            "status": "treat",
            "organic_advice": {
                "en": "Prune out dead wood and cankers during winter dormancy. Remove mummified fruit from trees and orchard floor to eliminate pycnidial sources.",
                "kn": "ಒಣಗಿದ ಕೊಂಬೆಗಳು ಮತ್ತು ಒಣಗಿದ ಹಣ್ಣುಗಳನ್ನು ಮರ ಮತ್ತು ನೆಲದಿಂದ ತೆಗೆದುಹಾಕಿ ಶಿಲೀಂಧ್ರ ಮೂಲಗಳನ್ನು ನಾಶಪಡಿಸಿ.",
                "hi": "सूखी टहनियों, घावों और सड़े-सूखे फलों को पेड़ों और बगीचे से हटाकर नष्ट करें ताकि फफूंद के स्रोत खत्म हों।",
            },
            "chemical_advice": {
                "en": "Apply protective sprays of Captan or Thiophanate-Methyl; follow product label and local KVK advice.",
                "kn": "Captan ಅಥವಾ Thiophanate-Methyl ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Captan या Thiophanate-Methyl का छिड़काव करें; लेबल निर्देश और KVK की सलाह का पालन करें।",
            },
        },
        "cedar_apple_rust": {
            "status": "treat",
            "organic_advice": {
                "en": "Remove eastern red cedar or juniper galls if present nearby. Plant rust-resistant cultivars.",
                "kn": "ಹತ್ತಿರದಲ್ಲಿ ಜೂನಿಪರ್ ಅಥವಾ ಸೀಡರ್ ಮರಗಳಿದ್ದರೆ ಅವುಗಳಲ್ಲಿನ ಗಂಟುಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬೆಳೆಯಿರಿ.",
                "hi": "यदि आसपास जुनिपर या सीडर के पेड़ मौजूद हों तो रस्ट की गांठों को हटाएं और प्रतिरोधी किस्में लगाएं।",
            },
            "chemical_advice": {
                "en": "Apply protective fungicide sprays such as Myclobutanil or Mancozeb during pink through petal-fall stages; follow product label and local KVK advice.",
                "kn": "Myclobutanil ಅಥವಾ Mancozeb ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ KVK ಸಲಹೆ ಪಾಲಿಸಿ.",
                "hi": "Myclobutanil या Mancozeb का छिड़काव करें; लेबल निर्देश और KVK की सलाह का पालन करें।",
            },
        },
        "healthy": {
            "status": "healthy",
            "organic_advice": {
                "en": "No treatment needed.",
                "kn": "ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "किसी उपचार की आवश्यकता नहीं है।",
            },
            "chemical_advice": {
                "en": "No chemical treatment needed.",
                "kn": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
                "hi": "किसी रासायनिक उपचार की आवश्यकता नहीं है।",
            },
        },
    },
}

WEATHER_ALERT_LOCALIZATION = {
    ("Paddy", "blast"): {
        "en": "High risk of disease pressure in the next 48 hours. Apply protective measures before conditions worsen as advised by your KVK.",
        "kn": "ಮುಂದಿನ 48 ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು. ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ KVK ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ.",
        "hi": "अगले 48 घंटों में रोग के अनुकूल मौसम की संभावना है। स्थिति बिगड़ने से पहले KVK की सलाह अनुसार बचाव के उपाय करें।",
    },
    ("Arecanut", "mahali_koleroga"): {
        "en": "High risk of disease pressure in the next 48 hours. Apply protective measures before the wet spell as advised by your KVK.",
        "kn": "ಮುಂದಿನ 48 ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು. ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ KVK ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ.",
        "hi": "अगले 48 घंटों में रोग के अनुकूल मौसम की संभावना है। बारिश से पहले KVK की सलाह अनुसार बचाव के उपाय करें।",
    },
    ("Apple", "apple_scab"): {
        "en": "High risk of disease pressure in the next 48 hours. Apply protective measures before the wet spell as advised by your KVK.",
        "kn": "ಮುಂದಿನ 48 ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು. ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ KVK ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ.",
        "hi": "अगले 48 घंटों में रोग के अनुकूल मौसम की संभावना है। बारिश से पहले KVK की सलाह अनुसार बचाव के उपाय करें।",
    },
}

# ─── GROUNDED CHAT ENGINE (NO LLM, NO FREE-FORM AGRONOMY) ────────────────────
def build_chat_response(query: str, last_scan: Optional[Dict[str, Any]], lang: str = "en") -> Tuple[str, bool]:
    """
    Answers user queries strictly by composing existing, grounded strings from the last scan result.
    Never hallucinates or generates new agronomy, product names, or dosages.
    Returns: (response_text, is_matched)
    """
    if not last_scan:
        return (get_ui_string("chat_no_context", lang), True)

    q = query.lower().strip()

    # Intent 1: "What should I do first?" / Steps
    first_keywords = [
        "first", "start", "organic", "step", "what should i do", "action",
        "ಮೊದಲು", "ಪ್ರಾರಂಭ", "ಕ್ರಮ", "ಸಾವಯವ", "ಮಾಡಬೇಕು", "ಮಾಡಬೇಕಾ", "ಮಾಡ್ದೆ",
        "पहले", "शुरुआत", "उपाय", "जैविक", "कदम"
    ]
    # Intent 2: "Is there a cure?"
    cure_keywords = [
        "cure", "heal", "treatable", "curable", "fix",
        "ಪರಿಹಾರ", "ಗುಣ", "ವಾಸಿಯಾಗ", "ಚಿಕಿತ್ಸೆ", "ವಾಸಿ",
        "इलाज", "उपचार", "ठीक", "संभव"
    ]
    # Intent 3: "When should I spray?" / Spraying advice
    spray_keywords = [
        "spray", "chemical", "pesticide", "fungicide", "medicine", "when should i spray",
        "ಸಿಂಪರಣೆ", "ಸಿಂಪಡಿಸ", "ಔಷಧಿ", "ರಾಸಾಯನಿಕ", "ಸ್ಪ್ರೇ", "ಮದ್ದು",
        "छिड़काव", "दवा", "कीटनाशक", "फफूंदनाशी", "रासायनिक", "स्प्रे"
    ]
    # Intent 4: "Soil tips" / Soil health
    soil_keywords = [
        "soil", "fertilizer", "compost", "manure", "nutrient", "urea",
        "ಮಣ್ಣು", "ಗೊಬ್ಬರ", "ಕಾಂಪೋಸ್ಟ್", "ಪೋಷಕಾಂಶ",
        "मिट्टी", "खाद", "कम्पोस्ट", "उर्वरक", "पोषण"
    ]
    # Intent 5: "Weather risk" / Forecast
    weather_keywords = [
        "weather", "rain", "humidity", "risk", "forecast", "monsoon",
        "ಹವಾಮಾನ", "ಮಳೆ", "ಆರ್ದ್ರತೆ", "ಅಪಾಯ",
        "मौसम", "बारिश", "नमी", "जोखिम"
    ]
    # Intent 6: "Why are you unsure?" / Refusal / Confidence
    unsure_keywords = [
        "unsure", "sure", "why", "confidence", "abstain", "ood", "doubt", "retake",
        "ಖಚಿತ", "ಸಂದೇಹ", "ಏಕೆ", "ಮರು", "ನಂಬಿಕೆ",
        "संदेह", "निश्चित", "क्यों", "सटीकता", "पक्का", "भरोसा"
    ]

    status = last_scan.get("status", "healthy")
    crop = last_scan.get("crop", "Paddy")
    pred_cls = last_scan.get("pred_cls", "")
    display_name = last_scan.get("display_name", "Crop")
    organic_adv = last_scan.get("organic_advice", "")
    chemical_adv = last_scan.get("chemical_advice", "")
    soil_title = last_scan.get("soil_title", "")
    soil_desc = last_scan.get("soil_desc", "")
    weather_alert = last_scan.get("weather_alert_text", None)
    weather_res = last_scan.get("weather_alert_res", None)
    loc_weather = None
    if (crop, pred_cls) in WEATHER_ALERT_LOCALIZATION and (weather_alert or weather_res):
        loc_weather = WEATHER_ALERT_LOCALIZATION[(crop, pred_cls)].get(lang)
    elif weather_res and isinstance(weather_res.get("voice_alerts"), dict):
        loc_weather = weather_res["voice_alerts"].get(lang, weather_alert)
    elif isinstance(weather_alert, dict):
        loc_weather = weather_alert.get(lang, weather_alert.get("en"))
    elif weather_alert:
        loc_weather = weather_alert
    tungro_note = last_scan.get("tungro_note", None)
    cedar_note = last_scan.get("cedar_note", None)
    is_ood = last_scan.get("is_ood", False)
    is_abstain = last_scan.get("is_abstain", False)
    is_cross_crop = last_scan.get("is_cross_crop", False)
    conf = last_scan.get("confidence", 0.0)
    grade = last_scan.get("grade", 1)

    # Localized lookup from dictionaries
    tx_meta = TREATMENT_LOCALIZATION.get(crop, {}).get(pred_cls, {})
    if tx_meta:
        status = tx_meta.get("status", status)
        loc_dis_name = DISEASE_LOCALIZATION.get(crop, {}).get(pred_cls, {}).get(lang, display_name)
        loc_org = tx_meta.get("organic_advice", {}).get(lang, organic_adv)
        loc_chem = tx_meta.get("chemical_advice", {}).get(lang, chemical_adv)
    else:
        loc_dis_name = display_name
        loc_org = organic_adv
        loc_chem = chemical_adv

    # Ensure Tungro warning is present for Paddy hispa / downy_mildew
    if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
        if lang == "kn":
            tungro_note = "ಗಿಡ್ಡವಾದ ಸಸಿಗಳು ಹಾಗೂ ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣದ ಎಲೆಗಳು ಕಂಡುಬಂದರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಆಗಿರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದಲ್ಲಿ, ನಿಮ್ಮ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
        elif lang == "hi":
            tungro_note = "बौने पौधे और पीली-नारंगी पत्तियां तुंग्रो वायरस (जिसका कोई इलाज नहीं है) भी हो सकती हैं। यदि ऐसा है, तो अपने KVK से सलाह लें।"
        else:
            tungro_note = "Stunted plants with yellow-orange leaves can also be Tungro virus (no cure). If so, consult your KVK."

    # Ensure Cedar Apple Rust notice
    if crop == "Apple" and pred_cls == "cedar_apple_rust":
        if lang == "kn":
            cedar_note = "ಯುಎಸ್ ಸಾರ್ವಜನಿಕ ಡೇಟಾದ ವರ್ಗ; ಸ್ಥಳೀಯವಾಗಿ ನಿಮ್ಮ KVK ಜೊತೆ ದೃಢೀಕರಿಸಿ."
        elif lang == "hi":
            cedar_note = "अमेरिकी सार्वजनिक डेटा से वर्ग; स्थानीय KVK से पुष्टि करें।"
        else:
            cedar_note = "Class from US public data; confirm locally with your KVK."

    response = ""
    is_matched = True

    # 1. WHAT SHOULD I DO FIRST?
    if any(k in q for k in first_keywords) or q == get_ui_string("chip_first", lang).lower():
        if is_cross_crop:
            response = last_scan.get("organic_advice", get_ui_string("cross_crop_refusal_msg", lang))
        elif is_ood or is_abstain:
            response = get_ui_string("refusal_abstain_msg" if is_abstain else "refusal_ood_msg", lang)
        elif status == "healthy":
            if lang == "kn":
                response = "ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ತೋಟವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ."
            elif lang == "hi":
                response = "आपकी फसल स्वस्थ है। किसी उपचार की आवश्यकता नहीं है। खेत को साफ रखें।"
            else:
                response = "Your crop is healthy. No treatment is needed. Maintain regular field sanitation."
        elif status == "no_cure":
            if lang == "kn":
                response = f"ಎಚ್ಚರಿಕೆ: {loc_dis_name} ಕಂಡುಬಂದಿದೆ. ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ, ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
            elif lang == "hi":
                response = f"चेतावनी: {loc_dis_name} का संक्रमण पाया गया है। इसका कोई रासायनिक उपचार उपलब्ध नहीं है। रसचूसक कीटों को नियंत्रित करें, रोगी पौधों को हटाएं और तुरंत KVK से संपर्क करें।"
            else:
                response = f"Warning: {display_name} detected. {organic_adv}"
        else:
            if lang == "kn":
                response = f"ಮೊದಲು ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ: {loc_org}"
            elif lang == "hi":
                response = f"पहले जैविक और पारंपरिक उपाय अपनाएं: {loc_org}"
            else:
                response = f"Start with organic and cultural steps first: {organic_adv}"

    # 2. IS THERE A CURE?
    elif any(k in q for k in cure_keywords) or q == get_ui_string("chip_cure", lang).lower():
        if is_cross_crop:
            response = last_scan.get("organic_advice", get_ui_string("cross_crop_refusal_msg", lang))
        elif is_ood or is_abstain:
            response = get_ui_string("refusal_abstain_msg" if is_abstain else "refusal_ood_msg", lang)
        elif status == "healthy":
            if lang == "kn":
                response = "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ; ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."
            elif lang == "hi":
                response = "फसल स्वस्थ है; किसी उपचार की आवश्यकता नहीं है।"
            else:
                response = "Your foliage is healthy; no cure is needed."
        elif status == "no_cure":
            if lang == "kn":
                response = f"ಇಲ್ಲ, ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. {loc_dis_name} ಗುಣಪಡಿಸಲಾಗದ ರೋಗ; ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ ಮತ್ತು ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ."
            elif lang == "hi":
                response = f"नहीं, इसका कोई रासायनिक इलाज नहीं है। {loc_dis_name} एक लाइलाज बीमारी है; रोगी पौधों को हटाएं और रसचूसक कीटों को नियंत्रित करें।"
            else:
                response = f"No chemical cure exists for {display_name}. It is an incurable systemic infection. Remove infected plants and control insect vectors with your KVK."
        else:
            if lang == "kn":
                response = f"ಹೌದು, {loc_dis_name} ರೋಗವನ್ನು ನಿಯಂತ್ರಿಸಬಹುದು. ಮೊದಲು ಸಾವಯವ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ: {loc_org} ಹೆಚ್ಚಿನ ತೀವ್ರತೆಗೆ KVK ಶಿಫಾರಸು ಮಾಡಿದ ರಾಸಾಯನಿಕ ಆಯ್ಕೆ: {loc_chem}"
            elif lang == "hi":
                response = f"हाँ, {loc_dis_name} का प्रबंधन किया जा सकता है। पहले जैविक उपाय अपनाएं: {loc_org} अधिक संक्रमण पर KVK अनुशंसित रासायनिक विकल्प: {loc_chem}"
            else:
                response = f"Yes, {display_name} can be managed. Start with cultural/organic steps: {organic_adv} Registered chemical option: {chemical_adv}"

    # 3. WHEN SHOULD I SPRAY?
    elif any(k in q for k in spray_keywords) or q == get_ui_string("chip_spray", lang).lower():
        if is_cross_crop:
            response = last_scan.get("organic_advice", get_ui_string("cross_crop_refusal_msg", lang))
        elif is_ood or is_abstain:
            response = get_ui_string("refusal_abstain_msg" if is_abstain else "refusal_ood_msg", lang)
        elif status == "healthy":
            if lang == "kn":
                response = "ಆರೋಗ್ಯಕರ ಬೆಳೆಗೆ ಯಾವುದೇ ಸಿಂಪರಣೆ ಅಗತ್ಯವಿಲ್ಲ."
            elif lang == "hi":
                response = "स्वस्थ फसल पर किसी छिड़काव की आवश्यकता नहीं है।"
            else:
                response = "No spray is needed for healthy foliage."
        elif status == "no_cure":
            if lang == "kn":
                response = "ಈ ರೋಗಕ್ಕೆ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ; ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ನಿಯಂತ್ರಣಕ್ಕಾಗಿ ಮಾತ್ರ ಸ್ಥಳೀಯ KVK ಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
            elif lang == "hi":
                response = "इस स्थिति के लिए फफूंदनाशी का छिड़काव न करें; कोई रासायनिक इलाज नहीं है। केवल कीट नियंत्रण के लिए स्थानीय KVK से सलाह लें।"
            else:
                response = "Do not spray fungicides for this condition; no chemical cure exists. Only manage insect vectors under local KVK guidance."
        else:
            # Treatable
            if lang == "kn":
                parts = []
                if loc_weather:
                    parts.append(f"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ (ನಿಯಮ-ಆಧಾರಿತ ಅಂದಾಜು, ಸಿಂಪರಣಾ ವೇಳಾಪಟ್ಟಿಯಲ್ಲ): {loc_weather} ಇಲ್ಲಿ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಎಂದರೆ ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಸೂಚಿಸಿದ ಶಿಫಾರಸು.")
                if grade < 2:
                    parts.append(f"ಲಘು ತೀವ್ರತೆಗೆ, ಮೊದಲು ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ ({loc_org}) ಕ್ಯಾಲೆಂಡರ್ ಸಿಂಪರಣೆ ಮಾಡಬೇಡಿ; ರಾಸಾಯನಿಕವು ತೀವ್ರ ಹಂತಕ್ಕೆ ಮಾತ್ರ.")
                else:
                    parts.append(f"ಹೆಚ್ಚಿನ ತೀವ್ರತೆಗೆ (ಗ್ರೇಡ್ 2+), KVK ನೋಂದಾಯಿತ ರಾಸಾಯನಿಕ ಆಯ್ಕೆ: {loc_chem}")
                parts.append(get_ui_string("footer_disclaimer_tx", lang))
                response = " ".join(parts)
            elif lang == "hi":
                parts = []
                if loc_weather:
                    parts.append(f"मौसम चेतावनी (नियम-आधारित अनुमान, छिड़काव समय सारणी नहीं): {loc_weather} यहाँ बचाव के उपाय का अर्थ आपके स्थानीय KVK द्वारा बताई गई सुरक्षात्मक सलाह है।")
                if grade < 2:
                    parts.append(f"शुरुआती लक्षणों के लिए, पहले जैविक और पारंपरिक उपाय अपनाएं ({loc_org}) कैलेंडर के अनुसार छिड़काव न करें; रासायनिक दवा केवल गंभीर प्रकोप के लिए है।")
                else:
                    parts.append(f"गंभीर संक्रमण के लिए (ग्रेड 2+), KVK अनुमोदित रासायनिक विकल्प: {loc_chem}")
                parts.append(get_ui_string("footer_disclaimer_tx", lang))
                response = " ".join(parts)
            else:
                parts = []
                if loc_weather:
                    parts.append(
                        f"Weather alert (rule-based estimate, not a spraying schedule): {loc_weather} "
                        "Here, 'protective measure' means the preventive option advised by your local KVK."
                    )
                if grade < 2:
                    parts.append(
                        f"For mild symptoms, start with cultural and biological steps ({organic_adv}). Do not schedule calendar sprays; chemicals are strictly for severe outbreaks under KVK guidance."
                    )
                else:
                    parts.append(
                        f"For moderate to severe infection (Grade 2+), KVK-registered chemical option: {chemical_adv}"
                    )
                parts.append(get_ui_string("footer_disclaimer_tx", lang))
                response = " ".join(parts)

    # 4. SOIL TIPS
    elif any(k in q for k in soil_keywords) or q == get_ui_string("chip_soil", lang).lower():
        if soil_title and soil_desc:
            response = f"{soil_title}: {soil_desc} ({get_ui_string('footer_disclaimer_soil', lang)})"
        else:
            response = "Conduct a soil test before applying nutrients. Follow local KVK guidance."

    # 5. WEATHER RISK
    elif any(k in q for k in weather_keywords) or q == get_ui_string("chip_weather", lang).lower():
        if loc_weather:
            response = f"{loc_weather} ({get_ui_string('rule_based_tag', lang)})"
        else:
            response = get_ui_string("no_rule_for_crop", lang)

    # 6. WHY ARE YOU UNSURE?
    elif any(k in q for k in unsure_keywords) or q == get_ui_string("chip_unsure", lang).lower():
        if is_cross_crop:
            response = last_scan.get("organic_advice", get_ui_string("cross_crop_refusal_msg", lang))
        elif is_ood:
            response = get_ui_string("refusal_ood_msg", lang)
        elif is_abstain:
            if lang == "kn":
                response = f"ವಿಶ್ವಾಸಾರ್ಹತೆ ({conf:.1%}) 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದ ಕಾರಣ ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ. 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಿದ್ದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ."
            elif lang == "hi":
                response = f"सटीकता ({conf:.1%}) 70% से कम होने के कारण निदान रोक दिया गया। 70% से कम होने पर ऐप निदान करने से मना करता है।"
            else:
                response = f"Diagnosis was withheld because confidence ({conf:.1%}) fell below 70%. The app declines to diagnose below 70%."
        else:
            if lang == "kn":
                response = f"ಮಾದರಿಯು ಈ ಚಿತ್ರವನ್ನು {conf:.1%} ವಿಶ್ವಾಸಾರ್ಹತೆಯೊಂದಿಗೆ ಗುರುತಿಸಿದೆ. ವಿಶ್ವಾಸಾರ್ಹತೆಯು 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ."
            elif lang == "hi":
                response = f"मॉडल ने इस तस्वीर को {conf:.1%} सटीकता के साथ पहचाना है। सटीकता 70% से कम होने पर ऐप निदान करने से मना करता है।"
            else:
                response = f"The model diagnosed this image with {conf:.1%} confidence. The app declines to diagnose below 70%."

        if cedar_note and cedar_note not in response:
            if lang == "kn":
                response = f"{response} ಸೂಚನೆ: {cedar_note}"
            elif lang == "hi":
                response = f"{response} सूचना: {cedar_note}"
            else:
                response = f"{response} Notice: {cedar_note}"

    # UNMATCHED QUERY
    else:
        response = get_ui_string("chat_unmatched", lang)
        is_matched = False

    # ── ATTACH TUNGRO NOTE TO EVERY CHAT ANSWER FOR HISPA / DOWNY MILDEW ──────
    if (crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]) or (tungro_note and is_matched):
        if tungro_note and tungro_note not in response:
            if lang == "kn":
                response = f"{response} {tungro_note}"
            elif lang == "hi":
                response = f"{response} {tungro_note}"
            else:
                response = f"{response} Tungro Warning: {tungro_note}"

    return (response, is_matched)

