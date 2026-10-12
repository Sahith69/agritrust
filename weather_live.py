"""
weather_live.py — Live Weather Forecast and 48-Hour Disease Risk Engine.
Uses Open-Meteo API (free, no key) with caching, timeout, and robust offline fallback.
"""

import time
import requests
from datetime import datetime

# ─── DISTRICT GPS COORDINATES ─────────────────────────────────────────────────
DISTRICT_COORDINATES = {
    "Mangaluru":  {"lat": 12.9141, "lon": 74.8560},
    "Udupi":      {"lat": 13.3409, "lon": 74.7421},
    "Shivamogga": {"lat": 13.9299, "lon": 75.5681},
    "Mysuru":     {"lat": 12.2958, "lon": 76.6394},
    "Srinagar":   {"lat": 34.0837, "lon": 74.7973},
    "Shimla":     {"lat": 31.1048, "lon": 77.1734},
    "Kullu":      {"lat": 31.9579, "lon": 77.1095},
    "Dehradun":   {"lat": 30.3165, "lon": 78.0322},
}

# ─── IN-MEMORY CACHE (30-MINUTE TTL) ──────────────────────────────────────────
_CACHE = {}  # key: (lat_round, lon_round) -> (timestamp, data_dict)
CACHE_TTL_SECONDS = 1800  # 30 minutes


def clear_cache():
    """Clears in-memory forecast cache (useful for testing)."""
    global _CACHE
    _CACHE.clear()


def fetch_forecast(lat: float, lon: float, timeout: float = 8.0) -> dict:
    """
    Fetches hourly weather forecast from Open-Meteo for the next 48 hours.
    Fields requested: hourly temperature_2m, relative_humidity_2m, precipitation, soil_moisture_0_to_1cm.
    Timeout: 8.0 seconds. Results cached for 30 minutes.
    """
    key = (round(lat, 4), round(lon, 4))
    now = time.time()
    
    if key in _CACHE:
        cached_ts, cached_data = _CACHE[key]
        if (now - cached_ts) < CACHE_TTL_SECONDS:
            return cached_data

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,soil_moisture_0_to_1cm"
        f"&forecast_days=2&timezone=auto"
    )

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Append fetch metadata
    data["_fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["_fetched_timestamp"] = now
    
    _CACHE[key] = (now, data)
    return data


# ─── 48-HOUR ALERT GATING MAPPING (EXACTLY 3 ALLOWED RULES) ───────────────────
ALLOWED_WEATHER_RULES = {
    ("Apple", "apple_scab"): "Scab",
    ("Paddy", "blast"): "Blast",
    ("Arecanut", "mahali_koleroga"): "Koleroga",
}


def compute_48h_disease_risk(forecast_data: dict, crop: str, disease_name: str = "", pred_cls: str = "") -> dict:
    """
    Computes hours in the next 48 h that meet established epidemiological rules.
    STRICT ALERT GATING:
      - Apple:    only 'apple_scab' (Scab rule: rain > 0.1 mm/h, 6.0 <= temp <= 24.0 °C, RH >= 85%)
      - Paddy:    only 'blast' (Blast rule: RH >= 90%, 20.0 <= temp <= 30.0 °C)
      - Arecanut: only 'mahali_koleroga' (Koleroga rule: rain > 0.5 mm/h, RH >= 90%)
      
    Returns None for every other class (hispa, tungro, black_rot, cedar_apple_rust, bud_borer, YLD, healthy, etc.).
    """
    # Normalize class key
    cls_key = pred_cls.strip() if pred_cls else ""
    if not cls_key and disease_name:
        # Fallback mapping from common display names if pred_cls not passed
        dn = disease_name.lower()
        if "scab" in dn: cls_key = "apple_scab"
        elif "blast" in dn: cls_key = "blast"
        elif "koleroga" in dn or "mahali" in dn: cls_key = "mahali_koleroga"

    rule_key = (crop, cls_key)
    if rule_key not in ALLOWED_WEATHER_RULES:
        return None

    rule_name = ALLOWED_WEATHER_RULES[rule_key]

    hourly = forecast_data.get("hourly", {})
    temps = hourly.get("temperature_2m", [])
    rhs = hourly.get("relative_humidity_2m", [])
    precips = hourly.get("precipitation", [])
    
    n_hours = min(len(temps), len(rhs), len(precips), 48)
    matched_hours = []

    for i in range(n_hours):
        t = temps[i]
        h = rhs[i]
        p = precips[i]
        match = False

        if crop == "Apple" and cls_key == "apple_scab":
            if p > 0.1 and 6.0 <= t <= 24.0 and h >= 85.0:
                match = True
        elif crop == "Paddy" and cls_key == "blast":
            if h >= 90.0 and 20.0 <= t <= 30.0:
                match = True
        elif crop == "Arecanut" and cls_key == "mahali_koleroga":
            if p > 0.5 and h >= 90.0:
                match = True

        if match:
            matched_hours.append(i)

    n_matched = len(matched_hours)
    first_hour = matched_hours[0] if n_matched > 0 else None

    # Categorize risk based on 48h duration
    if n_matched >= 12:
        risk_cat = "High"
    elif n_matched >= 4:
        risk_cat = "Moderate"
    elif n_matched >= 1:
        risk_cat = "Low"
    else:
        risk_cat = "Low"

    # Format UI alert text & voice strings
    # For Paddy blast: use "before conditions worsen"
    # For Scab & Koleroga: use "before the wet spell"
    is_blast = (crop == "Paddy" and cls_key == "blast")

    if n_matched > 0:
        if first_hour == 0:
            timing_str = "active now"
        else:
            timing_str = f"expected in about {first_hour} hours"
        
        if is_blast:
            advice_en = "Apply a protective measure before conditions worsen, as advised by your KVK."
        else:
            advice_en = "Apply a protective measure before the wet spell, as advised by your KVK."

        alert_text = (
            f"{rule_name}-favourable conditions {timing_str} "
            f"({n_matched} hours of favourable weather in the next 48 h). "
            f"{advice_en}"
        )
    else:
        alert_text = (
            f"Low risk: No {rule_name.lower()}-favourable weather conditions expected in the next 48 hours."
        )

    # Voice advisory spoken alerts (no numbers other than hours)
    if n_matched > 0:
        if is_blast:
            if first_hour == 0:
                voice_en = f"Weather alert: Blast-favourable weather active now for {n_matched} hours. Apply protective measures before conditions worsen as advised by your KVK."
                voice_kn = f"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಈಗ ಸಕ್ರಿಯವಾಗಿದೆ ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."  # REVIEW (Kannada)
                voice_hi = f"मौसम चेतावनी: रोग के अनुकूल मौसम अभी सक्रिय है ({n_matched} घंटे तक)। स्थिति बिगड़ने से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"  # REVIEW (Hindi)
            else:
                voice_en = f"Weather alert: Blast-favourable weather expected in {first_hour} hours for {n_matched} hours. Apply protective measures before conditions worsen as advised by your KVK."
                voice_kn = f"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಸುಮಾರು {first_hour} ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."  # REVIEW (Kannada)
                voice_hi = f"मौसम चेतावनी: लगभग {first_hour} घंटों में रोग के अनुकूल मौसम की संभावना है ({n_matched} घंटे तक)। स्थिति बिगड़ने से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"  # REVIEW (Hindi)
        else:
            if first_hour == 0:
                voice_en = f"Weather alert: {rule_name}-favourable weather active now for {n_matched} hours. Apply protective measures before the wet spell as advised by your KVK."
                voice_kn = f"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಈಗ ಸಕ್ರಿಯವಾಗಿದೆ ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."
                voice_hi = f"मौसम चेतावनी: रोग के अनुकूल मौसम अभी सक्रिय है ({n_matched} घंटे तक)। बारिश से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"
            else:
                voice_en = f"Weather alert: {rule_name}-favourable weather expected in {first_hour} hours for {n_matched} hours. Apply protective measures before the wet spell as advised by your KVK."
                voice_kn = f"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಸುಮಾರು {first_hour} ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."
                voice_hi = f"मौसम चेतावनी: लगभग {first_hour} घंटों में रोग के अनुकूल मौसम की संभावना है ({n_matched} घंटे तक)। बारिश से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"
    else:
        voice_en = "Weather alert: low disease risk from weather in the next forty-eight hours."
        voice_kn = "ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಮುಂದಿನ ನಲವತ್ತೆಂಟು ಗಂಟೆಗಳಲ್ಲಿ ಹವಾಮಾನದಿಂದ ರೋಗದ ಅಪಾಯ ಕಡಿಮೆ ಇದೆ."
        voice_hi = "मौसम चेतावनी: अगले अड़तालीस घंटों में मौसम से रोग का जोखिम कम है।"

    return {
        "crop": crop,
        "rule_name": rule_name,
        "matched_count": n_matched,
        "first_hour": first_hour,
        "matched_indices": matched_hours,
        "risk_cat": risk_cat,
        "alert_text": alert_text,
        "voice_alerts": {
            "en": voice_en,
            "kn": voice_kn,
            "hi": voice_hi,
        }
    }
