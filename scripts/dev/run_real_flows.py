#!/usr/bin/env python3
"""
run_real_flows.py — Executes the exact production app.py inference pipeline
on the required verification cases:
1. Tungro Paddy (4 images misclassified as hispa)
2. Paddy Blast in Mangaluru
3. Apple Scab in Srinagar
4. Healthy Paddy
5. Blurry Photo
6. Desk in Paddy mode

Prints exactly what is rendered (HTML/UI components) and spoken (voice text) for each case.
"""

import os
import sys
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
from weather_live import DISTRICT_COORDINATES, fetch_forecast, compute_48h_disease_risk

def run_app_pipeline(image_path: str, crop: str, district: str, use_live_weather: bool = True, lang: str = "en"):
    print(f"\n" + "=" * 95)
    print(f"CASE RUN: Crop={crop} | District={district} | Lang={lang} | LiveWeather={use_live_weather}")
    print(f"Image: {image_path}")
    print("=" * 95)

    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    active_crop = crop
    lang_code = lang

    # 1. Weather fetching (matching sidebar)
    forecast_data = None
    fetch_success = False
    temp_c = 26.0
    soil_moist_pct = 72.0
    soil_moist_display = "not available"
    soil_moist_available = False
    humidity = 75.0

    if use_live_weather:
        coords = DISTRICT_COORDINATES[district]
        try:
            forecast_data = fetch_forecast(coords["lat"], coords["lon"], timeout=8.0)
            fetch_success = True
            fetch_time_str = forecast_data.get("_fetched_at", "Recent")
            hourly = forecast_data["hourly"]
            temp_c = float(hourly["temperature_2m"][0])
            humidity = float(hourly["relative_humidity_2m"][0])
            precip = float(hourly["precipitation"][0])

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
            print(f"[Sidebar Weather] Live forecast: {district}, fetched {fetch_time_str}")
            print(f"[Sidebar Weather] Current: Temp={temp_c:.1f}°C | Topsoil moisture={soil_moist_display} | RH={humidity:.0f}% (Precip: {precip:.1f} mm/h)")
        except Exception as e:
            fetch_success = False
            forecast_data = None
            print(f"[Sidebar Weather] Live fetch failed: {e}. Falling back to default manual values.")
    else:
        print("[Sidebar Weather] Offline: using entered values (Temp=26°C, Soil=72%, RH=75%).")
        soil_moist_display = f"{soil_moist_pct:.0f}%"
        soil_moist_available = True

    # 2. OOD Guard
    is_ood, ood_score, ood_thresh = app.check_ood(img, active_crop)
    print(f"[OOD Check] Score={ood_score:.4f} vs Threshold={ood_thresh:.4f} -> is_ood={is_ood}")

    if is_ood:
        print("\n--- RENDERED UI (OOD REJECTION) ---")
        print(f"⚠️ Error: This does not look like a close-up {active_crop.lower()} photo. Retake a well-lit close-up of the affected leaf, trunk or nut.")
        print("Info: The image features deviate significantly from close-up crop pathology training distributions.")
        print("Withheld: Diagnosis, severity grading, advisory, Grad-CAM, and voice are suppressed for safety.")
        print("Voice: [DELETED / WITHHELD - No audio called]")
        return

    # 3. Disease Classification
    pred_cls, dis_conf, dis_probs, gradcam_img, display_name, clf_available = \
        app.run_disease_classification(img, active_crop)
    print(f"[Classifier] Predicted={pred_cls} ('{display_name}') | Confidence={dis_conf:.1%}")

    # 4. Abstain check
    abstain_tau = app.ABSTAIN_THRESHOLDS.get(active_crop, 0.70)
    if dis_conf < abstain_tau:
        print("\n--- RENDERED UI (ABSTAIN SAFETY REJECTION) ---")
        print("⚠️ Error: Not sure about this photo. Retake it in daylight or consult your KVK / agriculture officer.")
        print("Info: The classifier confidence fell below the safety validation threshold. Disease diagnosis, pesticide prescription, Grad-CAM heatmap, and voice advisory are withheld.")
        print("Voice: [DELETED / WITHHELD - No audio called]")
        return

    # 5. Severity grading
    is_healthy = (pred_cls in ["normal", "healthy", "healthy_foot", "healthy_leaf", "healthy_nut", "healthy_trunk"]
                  or display_name.startswith("Healthy"))

    if active_crop != "Apple":
        if is_healthy:
            grade = 0
            sev_conf = 1.0
            cv_ratio = 0.0
            sev_card_rendered = False
        elif active_crop == "Paddy" and pred_cls == "hispa":
            grade = 1
            sev_conf = 1.0
            cv_ratio = 0.0
            sev_card_rendered = True
        elif active_crop == "Paddy" and pred_cls == "dead_heart":
            grade = 3
            sev_conf = 1.0
            cv_ratio = 0.0
            sev_card_rendered = True
        else:
            grade, sev_conf, sev_probs, cv_ratio, lesion_overlay = \
                app.run_severity(img, active_crop, "Ordinal CORN")
            if grade == 0:
                grade = 1
            sev_card_rendered = True
    else:
        grade = 0
        sev_conf = 0.0
        cv_ratio = 0.0
        sev_card_rendered = False

    # 6. Environmental evaluation & Alert gating
    risk_score, risk_level, env_alert, action_modifier = app.evaluate_environmental_risk(
        crop=active_crop, disease_name=display_name, temp_c=temp_c, soil_moisture_pct=soil_moist_pct, humidity_pct=humidity
    )

    weather_alert_res = None
    weather_spoken_alert = ""
    if (not is_healthy) and use_live_weather and fetch_success and (forecast_data is not None):
        weather_alert_res = compute_48h_disease_risk(forecast_data, active_crop, display_name, pred_cls=pred_cls)
        if weather_alert_res is not None:
            weather_spoken_alert = weather_alert_res["voice_alerts"].get(lang_code, weather_alert_res["voice_alerts"]["en"])

    # 7. Management Plan
    crop_tx = app.TREATMENT_DATABASE.get(active_crop, {})
    dis_tx  = crop_tx.get(display_name, {})
    if active_crop == "Apple":
        if display_name == "Healthy (No Disease)":
            title_tx, desc_tx = dis_tx.get(0, ("Healthy Foliage", "No treatment needed."))
        else:
            title_tx, desc_tx = dis_tx.get("disease_advice", ("Disease Advisory", "Follow local KVK advice."))
    else:
        title_tx, desc_tx = dis_tx.get(grade, ("Action Plan", "Follow local KVK advice."))

    # 8. Soil Health Card
    soil_info = app.SOIL_HEALTH_ADVISORY.get(active_crop, app.SOIL_HEALTH_ADVISORY["Paddy"])
    soil_title, soil_desc = soil_info.get(lang_code, soil_info["en"])

    # 9. Voice Advisory
    spoken_voice = app.build_voice_advisory(active_crop, display_name, pred_cls, grade, lang_code, weather_spoken_alert)

    # ── OUTPUT DISPLAY ──
    print("\n--- RENDERED UI COMPONENTS ---")
    print(f"🦠 [Disease Banner] Detected Disease: {display_name} | Confidence: {dis_conf:.1%}")
    if active_crop == "Paddy" and pred_cls in ["hispa", "downy_mildew"]:
        tungro_text = app.TUNGRO_SAFETY_NOTE.get(lang_code, app.TUNGRO_SAFETY_NOTE["en"])
        print(f"  ℹ️ Tungro Warning Note: {tungro_text}")
    if active_crop == "Arecanut" and pred_cls in ["bud_borer", "yellow_leaf_disease"]:
        print("  ⚠️ Warning: Low Reliability Notice (Bud Borer / YLD confusion)")
    if active_crop == "Apple" and pred_cls == "cedar_apple_rust":
        print("  ⚠️ Caption: Class from US public data; confirm locally with your KVK.")

    print(f"\n🌦️ [Multimodal Card] Environmental Evaluation:")
    if use_live_weather and fetch_success:
        moist_text = f"Topsoil moisture (model estimate): {soil_moist_display}"
        print(f"  Conditions: Live forecast ({district}): Temp={temp_c:.1f}°C | {moist_text} | RH={humidity:.0f}%")
        if weather_alert_res is not None:
            print(f"  ⏱️ 48-Hour Alert ({weather_alert_res['risk_cat']} Risk): {weather_alert_res['alert_text']}")
            print(f"     (Rule matched {weather_alert_res['matched_count']} hours in next 48 h)")
        else:
            print("  ⏱️ 48-Hour Alert: ℹ️ No weather rule for this disease.")
    else:
        print(f"  Conditions: Field conditions (entered by user): Temp={temp_c:.1f}°C | Soil Moisture: {soil_moist_pct:.0f}% | RH={humidity:.0f}%")
        print(f"  Bioclimatic Outbreak Risk: {risk_score:.0f}% ({risk_level})")
        print(f"  Agronomic Assessment: {env_alert}")
        print("  ⏱️ 48-Hour Alert: ℹ️ 48-hour alert needs the live forecast.")

    print(f"\n📊 [Severity Card]:")
    if active_crop == "Apple":
        print("  🍎 Severity grading is withheld for Apple (model trained on public lab data).")
    elif is_healthy:
        print("  🌿 Foliage Healthy: No disease lesions detected. Severity grading is withheld.")
    else:
        print(f"  Grade {grade}: Mild/Moderate/Severe (Confidence: {sev_conf:.1%}, Lesion area: {cv_ratio:.1%})")
        if active_crop == "Arecanut":
            print("  Caption: 'Provisional: not independently validated.'")
        elif active_crop == "Paddy":
            if pred_cls in ["hispa", "dead_heart"]:
                print("  Caption: 'Severity is assigned by a fixed rule for this class.'")
            elif pred_cls not in ["normal"]:
                print("  Caption: 'Independent-test QWK about 0.67 (twin-free, lesion classes); headline 0.94 is inflated.'")

    print(f"\n💊 [Management Plan]:")
    print(f"  Title: {title_tx}")
    print(f"  Plan: {desc_tx}")
    print(f"  Modifier: {action_modifier}")
    print("  Disclaimer: ⚠️ Verify against ICAR / CPCRI / CIB&RC before use.")

    print(f"\n🌱 [Soil Health Card]:")
    print(f"  Title: {soil_title}")
    print(f"  Guidance: {soil_desc}")
    print("  Footer: ⚠️ Verify with your KVK / soil health card.")

    print(f"\n🎙️ [Voice Advisory Spoken Text] ({lang}):")
    print(f"  \"{spoken_voice}\"")


def main():
    print("=" * 95)
    print("RUNNING REAL APP FLOWS (VERIFICATION SUITE)")
    print("=" * 95)

    samples_dir = os.path.join(PROJECT_ROOT, "tests", "samples")

    # Flow 1: 4 Tungro images misclassified as Hispa
    tungro_hispa_images = [
        os.path.join(samples_dir, "tungro_102991.jpg"),
        os.path.join(samples_dir, "tungro_103609.jpg"),
        os.path.join(samples_dir, "tungro_109629.jpg"),
        os.path.join(samples_dir, "tungro_109527.jpg"),
    ]
    print("\n" + "#" * 95)
    print("FLOW 1: TUNGRO PADDY (4 IMAGES MISCLASSIFIED AS HISPA)")
    print("#" * 95)
    for p in tungro_hispa_images:
        run_app_pipeline(p, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="en")

    # Flow 2: Paddy Blast in Mangaluru
    print("\n" + "#" * 95)
    print("FLOW 2: PADDY BLAST IN MANGALURU")
    print("#" * 95)
    paddy_blast = os.path.join(samples_dir, "paddy_blast_102374.jpg")
    run_app_pipeline(paddy_blast, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="en")
    print("\n--- Also testing Kannada & Hindi speech for Paddy Blast in Mangaluru ---")
    run_app_pipeline(paddy_blast, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="kn")
    run_app_pipeline(paddy_blast, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="hi")

    # Flow 3: Apple Scab in Srinagar
    print("\n" + "#" * 95)
    print("FLOW 3: APPLE SCAB IN SRINAGAR")
    print("#" * 95)
    apple_scab = os.path.join(samples_dir, "apple_scab.jpg")
    run_app_pipeline(apple_scab, crop="Apple", district="Srinagar", use_live_weather=True, lang="en")

    # Flow 4: Healthy Paddy
    print("\n" + "#" * 95)
    print("FLOW 4: HEALTHY PADDY")
    print("#" * 95)
    healthy_paddy = os.path.join(samples_dir, "healthy_paddy_101753.jpg")
    run_app_pipeline(healthy_paddy, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="en")

    # Flow 5: Blurry Photo
    print("\n" + "#" * 95)
    print("FLOW 5: BLURRY PHOTO (IN PADDY MODE)")
    print("#" * 95)
    blurry_photo = os.path.join(samples_dir, "blurry_leaf.jpg")
    run_app_pipeline(blurry_photo, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="en")

    # Flow 6: Desk in Paddy Mode
    print("\n" + "#" * 95)
    print("FLOW 6: DESK IN PADDY MODE")
    print("#" * 95)
    desk_photo = os.path.join(samples_dir, "desk.jpg")
    run_app_pipeline(desk_photo, crop="Paddy", district="Mangaluru", use_live_weather=True, lang="en")

    # Flow 7: Live Weather Off (Fallback test)
    print("\n" + "#" * 95)
    print("FLOW 7: LIVE WEATHER OFF (PADDY BLAST OFFLINE)")
    print("#" * 95)
    run_app_pipeline(paddy_blast, crop="Paddy", district="Mangaluru", use_live_weather=False, lang="en")

if __name__ == "__main__":
    main()
