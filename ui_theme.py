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
    "apple_data_origin": "Apple models trained on public lab data (not field-validated)",
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
        "apple_public_banner": "⚠️ Trained on public lab data, not field-validated in orchards.",
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
        "tip_line": "ಹತ್ತಿರದಿಂದ, ಬೆಳಕಿನಲ್ಲಿ, ಒಂದು ಎಲೆ ಅಥವಾ ಅಡಿಕೆ",
        "choose_crop": "ಬೆಳೆ ಆಯ್ಕೆಮಾಡಿ",
        "crop_paddy": "ಭತ್ತ",
        "crop_arecanut": "ಅಡಿಕೆ",
        "crop_apple": "ಸೇಬು",
        "take_photo": "ಕ್ಯಾಮರಾ ಬಳಸಿ ಫೋಟೋ ತೆಗೆಯಿರಿ",
        "upload_file": "ಅಥವಾ ಮೊಬೈಲ್‌ನಿಂದ ಫೋಟೋ ಆಯ್ಕೆಮಾಡಿ",
        # Last result card
        "last_result_title": "ನಿಮ್ಮ ಹಿಂದಿನ ಫಲಿತಾಂಶ",
        "no_scans_yet": "ಇನ್ನೂ ಯಾವುದೇ ಸ್ಕ್ಯಾನ್ ಮಾಡಿಲ್ಲ.",
        "crop_label": "ಬೆಳೆ",
        "date_label": "ದಿನಾಂಕ",
        "confidence_label": "ವಿಶ್ವಾಸಾರ್ಹತೆ",
        "recommendation_label": "ಶಿಫಾರಸು",
        # Result Screen
        "back_btn": "← ಹಿಂದಕ್ಕೆ",
        "status_healthy": "ಆರೋಗ್ಯಕರ",
        "status_treat": "ಚಿಕಿತ್ಸೆ ಸಾಧ್ಯ",
        "status_no_cure": "ಪರಿಹಾರವಿಲ್ಲ / ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ",
        "status_unsure": "ಖಚಿತವಿಲ್ಲ",
        "what_to_do": "ಮಾಡಬೇಕಾದ ಕ್ರಮಗಳು",
        "organic_title": "ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು",
        "severe_expander": "ಸಮಸ್ಯೆ ಗಂಭೀರವಾಗಿದ್ದರೆ (ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ)",
        "weather_alert_title": "48 ಗಂಟೆಗಳ ಮುಂಜಾಗ್ರತಾ ಎಚ್ಚರಿಕೆ",
        "soil_card_title": "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್",
        "listen_audio": "ಧ್ವನಿ ಸಲಹೆ",
        "experts_expander": "ತಜ್ಞರು ಮತ್ತು ತೀರ್ಪುಗಾರರಿಗೆ",
        "trust_audit_title": "ವಿಶ್ವಾಸಾರ್ಹತೆ ಮತ್ತು ಆಡಿಟ್ ಅಂಕಿಅಂಶಗಳು",
        "apple_public_banner": "⚠️ ಸಾರ್ವಜನಿಕ ಲ್ಯಾಬ್ ಡೇಟಾದಲ್ಲಿ ತರಬೇತಿ ಪಡೆದಿದೆ, ತೋಟಗಳಲ್ಲಿ ಕ್ಷೇತ್ರ-ಪರೀಕ್ಷೆ ಮಾಡಲಾಗಿಲ್ಲ.",
        "areca_provisional": "ತಾತ್ಕಾಲಿಕ: ಸ್ವತಂತ್ರವಾಗಿ ಮೌಲ್ಯೀಕರಿಸಲಾಗಿಲ್ಲ.",
        "paddy_severity_qwk": "ಸ್ವತಂತ್ರ ಪರೀಕ್ಷಾ QWK ಸುಮಾರು 0.67 (ಅವಳಿ-ಮುಕ್ತ, ಗಾಯದ ತರಗತಿಗಳು); ಮುಖಪುಟದ 0.94 ಉಬ್ಬಿಸಲಾಗಿದೆ.",
        "paddy_severity_fixed": "ಈ ರೋಗಕ್ಕೆ ಸ್ಥಿರ ನಿಯಮದ ಮೂಲಕ ತೀವ್ರತೆಯನ್ನು ನಿಗದಿಪಡಿಸಲಾಗಿದೆ.",
        "tungro_warning_title": "ತುಂಗ್ರೋ ಎಚ್ಚರಿಕೆ",
        "cedar_rust_caption": "⚠️ ಯುಎಸ್ ಸಾರ್ವಜನಿಕ ಡೇಟಾದ ವರ್ಗ; ಸ್ಥಳೀಯವಾಗಿ ನಿಮ್ಮ ಕೆವಿಕೆ ಜೊತೆ ದೃಢೀಕರಿಸಿ.",
        "areca_bud_borer_warning": "⚠️ ಕಡಿಮೆ ವಿಶ್ವಾಸಾರ್ಹತೆಯ ಸೂಚನೆ: ಮಾದರಿಯು ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು ಮತ್ತು ಹಳದಿ ಎಲೆ ರೋಗವನ್ನು ಗೊಂದಲಗೊಳಿಸುತ್ತದೆ. ಚಿಕಿತ್ಸೆಗೆ ಮುನ್ನ ಕೆವಿಕೆ/ಸಿಪಿಸಿಆರ್‌ಐ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        "footer_disclaimer_tx": "⚠️ ಬಳಸುವ ಮುನ್ನ ICAR / CPCRI / CIB&RC ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        "footer_disclaimer_soil": "⚠️ ನಿಮ್ಮ ಕೆವಿಕೆ / ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        # Refusal states
        "refusal_ood_title": "ಚಿತ್ರವು ನಿಗದಿತ ಬೆಳೆಯ ವ್ಯಾಪ್ತಿಯಲ್ಲಿಲ್ಲ",
        "refusal_ood_msg": "ಇದು ಆಯ್ಕೆಮಾಡಿದ ಬೆಳೆಯ ಹತ್ತಿರದ ಫೋಟೋದಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಬಾಧಿತ ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಅಡಿಕೆಯ ಹತ್ತಿರದ ಫೋಟೋವನ್ನು ಬೆಳಕಿನಲ್ಲಿ ತೆಗೆಯಿರಿ.",
        "refusal_abstain_title": "ಸುರಕ್ಷತೆಗಾಗಿ ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ",
        "refusal_abstain_msg": "ಈ ಫೋಟೋದ ಬಗ್ಗೆ ಖಚಿತವಿಲ್ಲ. ಹಗಲಿನಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ಮರು-ಫೋಟೋ ತೆಗೆಯಿರಿ ಅಥವಾ ನಿಮ್ಮ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        # Navigation
        "tab_home": "ಮುಖಪುಟ",
        "tab_chat": "ಎಐ ಸಂಭಾಷಣೆ",
        "tab_weather": "ಹವಾಮಾನ",
        # AI Chat
        "chat_title": "ಕೃಷಿ ಸಹಾಯಕ ಸಂಭಾಷಣೆ",
        "chat_no_context": "ಮೊದಲು ಎಲೆಯ ಫೋಟೋ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ, ನಂತರ ನಾನು ಫಲಿತಾಂಶವನ್ನು ವಿವರಿಸಬಲ್ಲೆ.",
        "chat_disclaimer": "ಉತ್ತರಗಳು ನಿಮ್ಮ ಕೊನೆಯ ಸ್ಕ್ಯಾನ್‌ನಲ್ಲಿ ತೋರಿಸಲಾದ ಸಲಹೆಯಿಂದ ಬಂದಿವೆ. ಕೆವಿಕೆ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.",
        "chat_clear_btn": "ಸಂಭಾಷಣೆ ಅಳಿಸಿ",
        "chat_placeholder": "ನಿಮ್ಮ ಸ್ಕ್ಯಾನ್ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ...",
        "chip_first": "ಮೊದಲು ನಾನು ಏನು ಮಾಡಬೇಕು?",
        "chip_cure": "ಇದಕ್ಕೆ ಪರಿಹಾರವಿದೆಯೇ?",
        "chip_spray": "ಔಷಧಿ ಯಾವಾಗ ಸಿಂಪಡಿಸಬೇಕು?",
        "chip_soil": "ಮಣ್ಣಿನ ಸಲಹೆಗಳು",
        "chip_weather": "ಹವಾಮಾನ ಅಪಾಯ",
        "chip_unsure": "ಖಚಿತವಾಗಿಲ್ಲದಿರಲು ಕಾರಣವೇನು?",
        "chat_unmatched": "ನಾನು ನಿಮ್ಮ ಕೊನೆಯ ಫಲಿತಾಂಶವನ್ನು ಮಾತ್ರ ವಿವರಿಸಬಲ್ಲೆ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        # Weather View
        "weather_view_title": "48 ಗಂಟೆಗಳ ನೇರ ಹವಾಮಾನ ಮತ್ತು ರೋಗದ ಒತ್ತಡ",
        "weather_6h_blocks": "48 ಗಂಟೆಗಳ ಹವಾಮಾನ ಸಾರಾಂಶ (6 ಗಂಟೆಗಳ ವಿವರ)",
        "crop_weather_risk": "ಬೆಳೆ ಹವಾಮಾನ ಅಪಾಯದ ವಿಶ್ಲೇಷಣೆ",
        "rule_based_tag": "ನಿಯಮ-ಆಧಾರಿತ ಅಂದಾಜು, ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಯಲ್ಲ",
        "no_rule_for_crop": "ℹ️ ಮುಂದಿನ 48 ಗಂಟೆಗಳಲ್ಲಿ ಈ ಸ್ಥಿತಿಗೆ ಯಾವುದೇ ಹವಾಮಾನ ನಿಯಮ ಅನ್ವಯಿಸುವುದಿಲ್ಲ.",
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
        "tip_line": "पास से, दिन के उजाले में, एक पत्ती या एक फल",
        "choose_crop": "फसल चुनें",
        "crop_paddy": "धान",
        "crop_arecanut": "सुपारी",
        "crop_apple": "सेब",
        "take_photo": "कैमरे से फोटो लें",
        "upload_file": "या डिवाइस से फोटो चुनें",
        # Last result card
        "last_result_title": "आपका पिछला परिणाम",
        "no_scans_yet": "अभी तक कोई स्कैन नहीं हुआ।",
        "crop_label": "फसल",
        "date_label": "दिनांक",
        "confidence_label": "सटीकता",
        "recommendation_label": "अनुशंसा",
        # Result Screen
        "back_btn": "← वापस",
        "status_healthy": "स्वस्थ",
        "status_treat": "उपचार संभव",
        "status_no_cure": "इलाज नहीं / विशेषज्ञ से मिलें",
        "status_unsure": "संदेहास्पद",
        "what_to_do": "क्या करें",
        "organic_title": "जैविक और प्राकृतिक उपाय",
        "severe_expander": "यदि समस्या गंभीर हो (रासायनिक उपचार)",
        "weather_alert_title": "48 घंटे की निवारक चेतावनी",
        "soil_card_title": "मृदा स्वास्थ्य कार्ड",
        "listen_audio": "ध्वनि सलाह",
        "experts_expander": "विशेषज्ञों और परीक्षकों के लिए",
        "trust_audit_title": "विश्वसनीयता एवं ऑडिट रिपोर्ट",
        "apple_public_banner": "⚠️ सार्वजनिक लैब डेटा पर प्रशिक्षित, बगीचों में फील्ड-परीक्षण नहीं किया गया।",
        "areca_provisional": "अनंतिम: स्वतंत्र रूप से मान्य नहीं किया गया।",
        "paddy_severity_qwk": "स्वतंत्र परीक्षण QWK लगभग 0.67 (ट्विन-मुक्त, घाव वर्ग); मुख्य 0.94 बढ़ा-चढ़ाकर है।",
        "paddy_severity_fixed": "इस वर्ग के लिए गंभीरता एक निश्चित नियम द्वारा निर्धारित की गई है।",
        "tungro_warning_title": "तुंग्रो चेतावनी",
        "cedar_rust_caption": "⚠️ अमेरिकी सार्वजनिक डेटा से वर्ग; स्थानीय केवीके से पुष्टि करें।",
        "areca_bud_borer_warning": "⚠️ कम विश्वसनीयता सूचना: मॉडल कली छेदक और पीली पत्ती रोग में भ्रमित होता है। उपचार से पहले केवीके से पुष्टि करें।",
        "footer_disclaimer_tx": "⚠️ उपयोग से पहले ICAR / CPCRI / CIB&RC से पुष्टि करें।",
        "footer_disclaimer_soil": "⚠️ अपने केवीके / मृदा स्वास्थ्य कार्ड से जांच करें।",
        # Refusal states
        "refusal_ood_title": "तस्वीर फसल के दायरे से बाहर है",
        "refusal_ood_msg": "यह चुनी गई फसल की करीबी तस्वीर नहीं लगती। कृपया प्रभावित पत्ती या फल की साफ फोटो लें।",
        "refusal_abstain_title": "सुरक्षा कारणों से निदान रोक दिया गया",
        "refusal_abstain_msg": "इस फोटो पर निश्चित नहीं हैं। दिन के उजाले में दोबारा फोटो लें या अपने कृषि अधिकारी से सलाह लें।",
        # Navigation
        "tab_home": "होम",
        "tab_chat": "एआई चैट",
        "tab_weather": "मौसम",
        # AI Chat
        "chat_title": "कृषि सहायक चैट",
        "chat_no_context": "पहले एक पत्ती स्कैन करें, फिर मैं परिणाम समझा सकता हूँ।",
        "chat_disclaimer": "उत्तर आपके पिछले स्कैन में दिखाई गई सलाह पर आधारित हैं। अपने केवीके से पुष्टि करें।",
        "chat_clear_btn": "चैट साफ करें",
        "chat_placeholder": "अपने स्कैन के बारे में सवाल पूछें...",
        "chip_first": "मुझे सबसे पहले क्या करना चाहिए?",
        "chip_cure": "क्या इसका कोई इलाज है?",
        "chip_spray": "दवा कब छिड़कनी चाहिए?",
        "chip_soil": "मिट्टी की सलाह",
        "chip_weather": "मौसम का जोखिम",
        "chip_unsure": "आप निश्चित क्यों नहीं हैं?",
        "chat_unmatched": "मैं केवल आपके पिछले परिणाम की व्याख्या कर सकता हूँ। कृपया अपने नजदीकी केवीके या कृषि अधिकारी से संपर्क करें।",
        # Weather View
        "weather_view_title": "48 घंटे का मौसम और रोग का दबाव",
        "weather_6h_blocks": "48 घंटे का मौसम सारांश (6 घंटे के अंतराल पर)",
        "crop_weather_risk": "फसल मौसम जोखिम विश्लेषण",
        "rule_based_tag": "नियम-आधारित अनुमान, मौसम पूर्वानुमान नहीं",
        "no_rule_for_crop": "ℹ️ अगले 48 घंटों में इस रोग के लिए कोई मौसम नियम सक्रिय नहीं है।",
    }
}

def get_ui_string(key: str, lang: str = "en") -> str:
    lang_dict = UI_STRINGS.get(lang, UI_STRINGS["en"])
    return lang_dict.get(key, UI_STRINGS["en"].get(key, key))

# ─── MINIMAL THEME CSS (ALL IN ONE BLOCK, NO EXTERNAL CSS/JS) ────────────────
CSS_STYLES = f"""
<style>
    /* App Canvas & Font Stack */
    .stApp {{
        background-color: {THEME_CONFIG['bg']} !important;
        font-family: {THEME_CONFIG['font_stack']} !important;
        font-size: {THEME_CONFIG['base_font_size']} !important;
        color: {THEME_CONFIG['text_dark']} !important;
    }}

    /* Mobile First Centered Container (Max Width 760px) */
    .block-container {{
        max-width: {THEME_CONFIG['max_width']} !important;
        margin: 0 auto !important;
        padding-top: 1rem !important;
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
    button, .stButton > button, select, input {{
        min-height: {THEME_CONFIG['min_tap_target']} !important;
        font-family: {THEME_CONFIG['font_stack']} !important;
        border-radius: 10px !important;
    }}

    .stButton > button {{
        background-color: {THEME_CONFIG['card_bg']} !important;
        color: {THEME_CONFIG['primary']} !important;
        border: 1px solid {THEME_CONFIG['sage']} !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{
        background-color: #F1F5EE !important;
        border-color: {THEME_CONFIG['primary']} !important;
        color: {THEME_CONFIG['primary']} !important;
    }}
    .stButton > button:active, .stButton > button[kind="primary"] {{
        background-color: {THEME_CONFIG['primary']} !important;
        color: #FFFFFF !important;
        border-color: {THEME_CONFIG['primary']} !important;
    }}

    /* Top Tabs Styling */
    button[data-baseweb="tab"] {{
        min-height: 48px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: {THEME_CONFIG['text_muted']} !important;
        padding: 10px 18px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {THEME_CONFIG['primary']} !important;
        border-bottom-color: {THEME_CONFIG['primary']} !important;
        border-bottom-width: 3px !important;
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
        "ಮೊದಲು", "ಪ್ರಾರಂಭ", "ಕ್ರಮ", "ಸಾವಯವ",
        "पहले", "शुरुआत", "उपाय", "जैविक", "कदम"
    ]
    # Intent 2: "Is there a cure?"
    cure_keywords = [
        "cure", "heal", "treatable", "curable", "fix",
        "ಪರಿಹಾರ", "ಗುಣ", "ವಾಸಿಯಾಗ", "ಚಿಕಿತ್ಸೆ",
        "इलाज", "उपचार", "ठीक", "संभव"
    ]
    # Intent 3: "When should I spray?" / Spraying advice
    spray_keywords = [
        "spray", "chemical", "pesticide", "fungicide", "medicine", "when should i spray",
        "ಸಿಂಪರಣೆ", "ಸಿಂಪಡಿಸ", "ಔಷಧಿ", "ರಾಸಾಯನಿಕ",
        "छिड़काव", "दवा", "कीटनाशक", "फफूंदनाशी", "रासायनिक"
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
        "ಖಚಿತ", "ಸಂದೇಹ", "ಏಕೆ", "ಮರು",
        "संदेह", "निश्चित", "क्यों", "सटीकता"
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
    tungro_note = last_scan.get("tungro_note", None)
    cedar_note = last_scan.get("cedar_note", None)
    is_ood = last_scan.get("is_ood", False)
    is_abstain = last_scan.get("is_abstain", False)
    conf = last_scan.get("confidence", 0.0)
    grade = last_scan.get("grade", 1)

    # Ensure Tungro warning is present for Paddy hispa / downy_mildew
    if (crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]):
        if lang == "kn":
            tungro_note = "ಗಿಡ್ಡವಾದ ಸಸಿಗಳು ಹಾಗೂ ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣದ ಎಲೆಗಳು ಕಂಡುಬಂದರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಆಗಿರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದಲ್ಲಿ, ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
        elif lang == "hi":
            tungro_note = "बौने पौधे और पीली-नारंगी पत्तियां तुंग्रो वायरस (जिसका कोई इलाज नहीं है) भी हो सकती हैं। यदि ऐसा है, तो अपने केवीके से सलाह लें।"
        else:
            tungro_note = "Stunted plants with yellow-orange leaves can also be Tungro virus (no cure). If so, consult your KVK."
    elif tungro_note:
        if lang == "kn" and not any("\u0C80" <= c <= "\u0CFF" for c in tungro_note):
            tungro_note = "ಗಿಡ್ಡವಾದ ಸಸಿಗಳು ಹಾಗೂ ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣದ ಎಲೆಗಳು ಕಂಡುಬಂದರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಆಗಿರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದಲ್ಲಿ, ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
        elif lang == "hi" and not any("\u0900" <= c <= "\u097F" for c in tungro_note):
            tungro_note = "बौने पौधे और पीली-नारंगी पत्तियां तुंग्रो वायरस (जिसका कोई इलाज नहीं है) भी हो सकती हैं। यदि ऐसा है, तो अपने केवीके से सलाह लें।"


    response = ""
    is_matched = True

    # 1. WHAT SHOULD I DO FIRST?
    if any(k in q for k in first_keywords) or q == get_ui_string("chip_first", lang).lower():
        if is_ood or is_abstain:
            response = get_ui_string("refusal_abstain_msg" if is_abstain else "refusal_ood_msg", lang)
        elif status == "healthy":
            if lang == "kn":
                response = "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಯಾವುದೇ ಚಿಕಿತ್ಸೆಯ ಅಗತ್ಯವಿಲ್ಲ. ನಿಯಮಿತ ತೋಟ ನೈರ್ಮಲ್ಯವನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ."
            elif lang == "hi":
                response = "फसल स्वस्थ है। किसी उपचार की आवश्यकता नहीं है। खेत की सामान्य स्वच्छता बनाए रखें।"
            else:
                response = "Your crop is healthy. No treatment is needed. Maintain regular field sanitation."
        elif status == "no_cure":
            if lang == "kn":
                response = f"ಎಚ್ಚರಿಕೆ: {display_name} ಕಂಡುಬಂದಿದೆ. ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ, ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
            elif lang == "hi":
                response = f"चेतावनी: {display_name} का संक्रमण पाया गया है। इसका कोई रासायनिक उपचार उपलब्ध नहीं है। रसचूसक कीटों को नियंत्रित करें, रोगी पौधों को हटाएं और तुरंत कृषि विज्ञान केंद्र से संपर्क करें।"
            else:
                response = f"Warning: {display_name} detected. No chemical cure exists. Manage insect vectors, remove infected plants, and consult your local KVK."
        else:
            if lang == "kn":
                response = f"ಮೊದಲು ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ: {organic_adv}"
            elif lang == "hi":
                response = f"पहले जैविक और प्राकृतिक उपाय अपनाएं: {organic_adv}"
            else:
                response = f"Start with organic and cultural steps first: {organic_adv}"

    # 2. IS THERE A CURE?
    elif any(k in q for k in cure_keywords) or q == get_ui_string("chip_cure", lang).lower():
        if is_ood or is_abstain:
            response = get_ui_string("refusal_abstain_msg" if is_abstain else "refusal_ood_msg", lang)
        elif status == "healthy":
            if lang == "kn":
                response = "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ; ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಥವಾ ಪರಿಹಾರದ ಅಗತ್ಯವಿಲ್ಲ."
            elif lang == "hi":
                response = "फसल स्वस्थ है; किसी इलाज की आवश्यकता नहीं है।"
            else:
                response = "Your foliage is healthy; no cure is needed."
        elif status == "no_cure":
            if lang == "kn":
                response = f"ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. {display_name} ವೈರಸ್ / ಫೈಟೋಪ್ಲಾಸ್ಮಾ ರೋಗವಾಗಿದ್ದು, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತು ನಾಶಪಡಿಸುವುದು ಮತ್ತು ವಾಹಕ ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸುವುದು ಮುಖ್ಯ."
            elif lang == "hi":
                response = f"इसका कोई रासायनिक उपचार उपलब्ध नहीं है। {display_name} एक लाइलाज रोग है; संक्रमित पौधों को नष्ट करें और कीट वाहकों को नियंत्रित करें।"
            else:
                response = f"No chemical cure exists for {display_name}. It is an incurable systemic infection. Remove infected plants and control insect vectors with your KVK."
        else:
            if lang == "kn":
                response = f"ಹೌದು, {display_name} ನಿಯಂತ್ರಿಸಬಹುದಾದ ರೋಗ. ಆರಂಭದಲ್ಲಿ ಸಾವಯವ ಕ್ರಮಗಳನ್ನು ಬಳಸಿ, ಗಂಭೀರವಾಗಿದ್ದರೆ ಶಿಫಾರಸು ಮಾಡಿದ ರಾಸಾಯನಿಕ ಕ್ರಮಗಳನ್ನು ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಕೈಗೊಳ್ಳಬಹುದು."
            elif lang == "hi":
                response = f"हाँ, {display_name} का प्रबंधन संभव है। शुरुआती अवस्था में जैविक उपाय करें, गंभीर होने पर केवीके की सलाह अनुसार रासायनिक छिड़काव करें।"
            else:
                response = f"Yes, {display_name} can be managed. Start with cultural/organic steps. If severe, registered chemical options are available from your KVK."

    # 3. WHEN SHOULD I SPRAY?
    elif any(k in q for k in spray_keywords) or q == get_ui_string("chip_spray", lang).lower():
        if is_ood or is_abstain:
            response = get_ui_string("refusal_abstain_msg" if is_abstain else "refusal_ood_msg", lang)
        elif status == "healthy":
            if lang == "kn":
                response = "ಆರೋಗ್ಯಕರ ಬೆಳೆಗೆ ಯಾವುದೇ ಔಷಧ ಸಿಂಪರಣೆಯ ಅಗತ್ಯವಿಲ್ಲ."
            elif lang == "hi":
                response = "स्वस्थ फसल पर किसी दवा के छिड़काव की आवश्यकता नहीं है।"
            else:
                response = "No spray is needed for healthy foliage."
        elif status == "no_cure":
            if lang == "kn":
                response = "ಇದಕ್ಕೆ ಯಾವುದೇ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ; ರೋಗ ಗುಣವಾಗುವುದಿಲ್ಲ. ವಾಹಕ ಕೀಟ ನಿಯಂತ್ರಣಕ್ಕಾಗಿ ಮಾತ್ರ ಕೆವಿಕೆ ಸಲಹೆ ಪಡೆಯಿರಿ."
            elif lang == "hi":
                response = "इस पर कोई फफूंदनाशक न छिड़कें; इसका इलाज नहीं होगा। केवल कीट नियंत्रण के लिए केवीके से सलाह लें।"
            else:
                response = "Do not spray fungicides for this condition; no chemical cure exists. Only manage insect vectors under local KVK guidance."
        else:
            # Treatable
            if lang == "kn":
                parts = []
                if weather_alert:
                    parts.append(f"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ (ನಿಯಮ-ಆಧಾರಿತ ಅಂದಾಜು — ಔಷಧ ಸಿಂಪರಣಾ ವೇಳಾಪಟ್ಟಿಯಲ್ಲ): {weather_alert} ಇಲ್ಲಿ 'ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ' ಎಂದರೆ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಶಿಫಾರಸು ಮಾಡಿದ ಮುನ್ನೆಚ್ಚರಿಕೆಯ ಕ್ರಮ.")
                if grade < 2:
                    parts.append(f"ಲಘು ಸೋಂಕಿಗೆ, ಸಾವಯವ ಮತ್ತು ಜೈವಿಕ ಕ್ರಮಗಳಿಂದ ಪ್ರಾರಂಭಿಸಿ ({organic_adv}). ನಿಯಮಿತ ವೇಳಾಪಟ್ಟಿ ಪ್ರಕಾರ ರಾಸಾಯನಿಕ ಸಿಂಪಡಿಸಬೇಡಿ; ತೀವ್ರವಾದಾಗ ಮಾತ್ರ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಬಳಸಿ.")
                else:
                    parts.append(f"ಸೋಂಕು ತೀವ್ರವಾಗಿದ್ದರೆ (ಗ್ರೇಡ್ 2+), ಕೆವಿಕೆ ನೋಂದಾಯಿತ ರಾಸಾಯನಿಕ ಆಯ್ಕೆ: {chemical_adv}")
                parts.append(get_ui_string("footer_disclaimer_tx", lang))
                response = " ".join(parts)
            elif lang == "hi":
                parts = []
                if weather_alert:
                    parts.append(f"मौसम चेतावनी (नियम-आधारित अनुमान — छिड़काव की समय-सारणी नहीं): {weather_alert} यहाँ 'बचाव के उपाय' का अर्थ स्थानीय केवीके द्वारा अनुशंसित उपाय है।")
                if grade < 2:
                    parts.append(f"हल्के संक्रमण के लिए, पहले जैविक और पारंपरिक उपाय करें ({organic_adv})। कैलेंडर के अनुसार छिड़काव न करें; रासायनिक दवा केवल गंभीर प्रकोप के लिए है।")
                else:
                    parts.append(f"गंभीर संक्रमण के लिए (ग्रेड 2+), केवीके अनुमोदित रासायनिक विकल्प: {chemical_adv}")
                parts.append(get_ui_string("footer_disclaimer_tx", lang))
                response = " ".join(parts)
            else:
                parts = []
                if weather_alert:
                    parts.append(
                        f"Weather alert (rule-based estimate, not a spraying schedule): {weather_alert} "
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
        if weather_alert:
            response = f"{weather_alert} ({get_ui_string('rule_based_tag', lang)})"
        else:
            response = get_ui_string("no_rule_for_crop", lang)

    # 6. WHY ARE YOU UNSURE?
    elif any(k in q for k in unsure_keywords) or q == get_ui_string("chip_unsure", lang).lower():
        if is_ood:
            response = get_ui_string("refusal_ood_msg", lang)
        elif is_abstain:
            if lang == "kn":
                response = f"ಮಾದರಿಯ ವಿಶ್ವಾಸಾರ್ಹತೆ ({conf:.1%}) 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾಗಿದೆ. ವಿಶ್ವಾಸಾರ್ಹತೆಯು 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ."
                if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    response += " ಹಿಸ್ಪಾ ಮತ್ತು ಬೂದಿ ರೋಗಕ್ಕೆ, ತುಂಗ್ರೋ ವೈರಸ್ ಕೂಡ ಇದೇ ರೀತಿ ಕಾಣಿಸಬಹುದು."
            elif lang == "hi":
                response = f"मॉडल की सटीकता ({conf:.1%}) 70% से कम रही। सटीकता 70% से कम होने पर ऐप निदान करने से मना करता है।"
                if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    response += " हिस्पा और डाउनी फफूंदी के लिए, तुंग्रो वायरस भी इसी तरह का दिख सकता है।"
            else:
                response = f"Diagnosis was withheld because confidence ({conf:.1%}) fell below 70%. The app declines to diagnose below 70%."
                if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    response += " For Hispa and Downy Mildew, Tungro virus can look similar in the field."
        else:
            if lang == "kn":
                base_unsure = f"ಮಾದರಿಯು ಈ ಚಿತ್ರವನ್ನು {conf:.1%} ವಿಶ್ವಾಸಾರ್ಹತೆಯೊಂದಿಗೆ ಗುರುತಿಸಿದೆ. ವಿಶ್ವಾಸಾರ್ಹತೆಯು 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ."
                if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    base_unsure += " ಹಿಸ್ಪಾ ಮತ್ತು ಬೂದಿ ರೋಗಕ್ಕೆ, ತುಂಗ್ರೋ ವೈರಸ್ ಕೂಡ ಇದೇ ರೀತಿ ಕಾಣಿಸಬಹುದು."
            elif lang == "hi":
                base_unsure = f"मॉडल ने इस तस्वीर को {conf:.1%} सटीकता के साथ पहचाना है। सटीकता 70% से कम होने पर ऐप निदान करने से मना करता है।"
                if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    base_unsure += " हिस्पा और डाउनी फफूंदी के लिए, तुंग्रो वायरस भी इसी तरह का दिख सकता है।"
            else:
                base_unsure = f"The model diagnosed this image with {conf:.1%} confidence. The app declines to diagnose below 70%."
                if crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
                    base_unsure += " For Hispa and Downy Mildew, Tungro can look similar."
            response = base_unsure

        if cedar_note and cedar_note not in response:
            response = f"{response} Notice: {cedar_note}"

    # UNMATCHED QUERY
    else:
        response = get_ui_string("chat_unmatched", lang)
        is_matched = False

    # ── ATTACH TUNGRO NOTE TO EVERY CHAT ANSWER FOR HISPA / DOWNY MILDEW ──────
    if (crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]) or (tungro_note and is_matched):
        if tungro_note and tungro_note not in response:
            if lang == "kn":
                response = f"{response} ತುಂಗ್ರೋ ಎಚ್ಚರಿಕೆ: {tungro_note}"
            elif lang == "hi":
                response = f"{response} तुंग्रो चेतावनी: {tungro_note}"
            else:
                response = f"{response} Tungro Warning: {tungro_note}"

    return (response, is_matched)
