# AgriTrust AI — Day-to-Day Conversational Translations (Hindi & Kannada)
## बोलचाल की हिंदी एवं ಆಡುಮಾತಿನ ಕನ್ನಡ (Farmer-First Localized Guide)

This document provides a comprehensive, field-ready translation of all user interface texts, safety guardrails, grounded chat responses, disease terminology, and voice advisory scripts from formal bookish/Sanskritized phrasing into **natural, day-to-day spoken language** used by farmers and Krishi Vigyan Kendra (KVK) field extension workers.

---

## 📌 Guiding Principles for Conversational Tone

1. **Clarity Over Jargon:** Replace academic botanical nouns with the everyday terms farmers actually use in village mandis and farm gates.
2. **Action-Oriented Warnings:** Instead of abstract legal disclaimers ("ऐप निदान करने से मना करता है" / "ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ"), state clearly what happened in plain terms ("फोटो साफ नहीं है, इसलिए ऐप गलत अंदाज़ा नहीं लगा रहा").
3. **Conversational Pronouns & Flow:** Use natural phrasing that sounds like a helpful agricultural officer talking face-to-face with a grower.
4. **Preserve Regulated Chemical Names:** Keep exact pesticide formulations (e.g., *Tricyclazole 75 WP*, *Chlorpyriphos 20 EC*, *Mancozeb*) and KVK advisories intact so safety is never compromised.

---

## 1. 🔘 Navigation, Core Actions & Quick Reply Chips

| Key | English Source | Old Formal Hindi | 🗣️ Spoken Day-to-Day Hindi | Old Formal Kannada | 🗣️ Spoken Day-to-Day Kannada |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tab_home` | Home | होम / मुख्य पृष्ठ | **होम** | ಮುಖಪುಟ | **ಮುಖಪುಟ (ಹೋಮ್)** |
| `tab_chat` | AI Chat | एआई चैट | **सवाल-जवाब** | ಎಐ ಸಂಭಾಷಣೆ | **ಕೃಷಿ ಚಾಟ್ / ಪ್ರಶ್ನೋತ್ತರ** |
| `tab_weather` | Weather | मौसम | **मौसम** | ಹವಾಮಾನ | **ಹವಾಮಾನ** |
| `take_photo` | Take photo with camera | कैमरे से फोटो लें | **कैमरे से फोटो खींचें** | ಕ್ಯಾಮರಾ ಬಳಸಿ ಫೋಟೋ ತೆಗೆಯಿರಿ | **ಕ್ಯಾಮರಾದಿಂದ ಫೋಟೋ ತೆಗೆಯಿರಿ** |
| `upload_file` | Or upload photo from device | या डिवाइस से फोटो चुनें | **या फोन से फोटो अपलोड करें** | ಅಥವಾ ಮೊಬೈಲ್‌ನಿಂದ ಫೋಟೋ ಆಯ್ಕೆಮಾಡಿ | **ಅಥವಾ ಮೊಬೈಲ್‌ನಿಂದ ಫೋಟೋ ಹಾಕಿ** |
| `tip_line` | Close-up, daylight, one leaf or one nut | पास से, दिन के उजाले में, एक पत्ती या एक फल | **दिन की रोशनी में, पास से एक पत्ती या फल की फोटो लें** | ಹತ್ತಿರದಿಂದ, ಬೆಳಕಿನಲ್ಲಿ, ಒಂದು ಎಲೆ ಅಥವಾ ಅಡಿಕೆ | **ಬೆಳಕಿನಲ್ಲಿ, ಹತ್ತಿರದಿಂದ ಒಂದೇ ಎಲೆ ಅಥವಾ ಅಡಿಕೆಯ ಫೋಟೋ ತೆಗೆಯಿರಿ** |
| `chip_first` | What should I do first? | मुझे सबसे पहले क्या करना चाहिए? | **मुझे पहले क्या करना चाहिए?** | ನಾನು ಮೊದಲು ಏನು ಮಾಡಬೇಕು? | **ನಾನು ಮೊದಲು ಏನು ಮಾಡಬೇಕು?** |
| `chip_cure` | Is there a cure? | क्या इसका कोई इलाज है? | **क्या यह बीमारी ठीक हो सकती है?** | ಇದಕ್ಕೆ ಪರಿಹಾರವಿದೆಯೇ? | **ಈ ರೋಗ ವಾಸಿಯಾಗುತ್ತಾ?** |
| `chip_spray` | When should I spray? | मुझे कब छिड़काव करना चाहिए? | **स्प्रे कब और कैसे करना चाहिए?** | ನಾನು ಯಾವಾಗ ಸಿಂಪಡಿಸಬೇಕು? | **ಮದ್ದು ಯಾವಾಗ ಸಿಂಪಡಿಸಬೇಕು?** |
| `chip_unsure` | Why are you unsure? | आपको संदेह क्यों है? | **ऐप को पक्का भरोसा क्यों नहीं है?** | ನಿಮಗೆ ಏಕೆ ಅನುಮಾನವಿದೆ? | **ಆ್ಯಪ್‌ಗೆ ಯಾಕೆ ಪಕ್ಕಾ ಖಚಿತವಿಲ್ಲ?** |
| `back_btn` | ← Back to Photo Upload | ← फोटो अपलोड पर वापस जाएं | **← वापस जाएं (नई फोटो लें)** | ← ಫೋಟೋ ಅಪ್‌ಲೋಡ್‌ಗೆ ಹಿಂತಿರುಗಿ | **← ವಾಪಸ್ ಹೋಗಿ (ಹೊಸ ಫೋಟೋ)** |
| `listen_audio` | Listen to Voice Advisory | वॉइस एडवाइजरी सुनें | **बोलकर सुनें (आवाज में सलाह)** | ಧ್ವನಿ ಸಲಹೆಯನ್ನು ಆಲಿಸಿ | **ಧ್ವನಿಯಲ್ಲಿ ಕೇಳಿ (ಆಡಿಯೋ)** |

---

## 2. 🛡️ Guardrails, Rejection Cards & Safety Disclaimers

### A. Out-of-Distribution (OOD) Guard (Blurry Photo, Wall, Non-Crop Image)

* **English:** `"This does not look like a close-up paddy, arecanut or apple photo. Retake a well-lit close-up of the affected leaf, trunk or nut."`
* **Old Formal Hindi:** `"यह धान, सुपारी या सेब की क्लोज-अप तस्वीर नहीं लग रही है। प्रभावित पत्ती, तने या फल की अच्छी रोशनी में क्लोज-अप फोटो पुनः लें।"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"यह फोटो धान, सुपारी या सेब की साफ और पास से ली गई फोटो नहीं लग रही है। कृपया अच्छी रोशनी में बीमारी वाली पत्ती, तने या फल के बिल्कुल पास जाकर दोबारा फोटो खींचें।"**
* **Old Formal Kannada:** `"ಇದು ಭತ್ತ, ಅಡಿಕೆ ಅಥವಾ ಸೇಬಿನ ಹತ್ತಿರದ ಫೋಟೋದಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ಬಾಧಿತ ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಕಾಯಿಯ ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಹತ್ತಿರದಿಂದ ಮತ್ತೊಮ್ಮೆ ಫೋಟೋ ತೆಗೆಯಿರಿ."`
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಇದು ಭತ್ತ, ಅಡಿಕೆ ಅಥವಾ ಸೇಬಿನ ಸ್ಪಷ್ಟವಾದ ಫೋಟೋದಂತೆ ಕಾಣ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಒಳ್ಳೆ ಬೆಳಕಿನಲ್ಲಿ, ರೋಗ ಬಂದಿರೋ ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಅಡಿಕೆಯ ಹತ್ತಿರಕ್ಕೆ ಹೋಗಿ ಇನ್ನೊಮ್ಮೆ ಸ್ಪಷ್ಟವಾಗಿ ಫೋಟೋ ತೆಗೆಯಿರಿ."**

---

### B. Conservative Abstain Guard (Confidence Below 70%)

* **English:** `"Diagnosis withheld (Low Confidence). The image is ambiguous or lighting is poor. Retake a clear close-up or consult your local KVK extension officer."`
* **Old Formal Hindi:** `"निदान रोक दिया गया (कम विश्वास)। तस्वीर अस्पष्ट है या रोशनी खराब है। स्पष्ट क्लोज-अप पुनः लें या स्थानीय केवीके अधिकारी से परामर्श करें।"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"बीमारी का पक्का पता नहीं चल पाया (भरोसा 70% से कम है)। फोटो धुंधली है या रोशनी कम है। कृपया साफ फोटो दोबारा खींचें या अपनी नजदीकी कृषि विज्ञान केंद्र (केवीके) से सलाह लें।"**
* **Old Formal Kannada:** `"ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ (ಕಡಿಮೆ ವಿಶ್ವಾಸಾರ್ಹತೆ). ಚಿತ್ರವು ಅಸ್ಪಷ್ಟವಾಗಿದೆ ಅಥವಾ ಬೆಳಕು ಸರಿಯಾಗಿಲ್ಲ. ಸ್ಪಷ್ಟ ಫೋಟೋವನ್ನು ಮರುತೆಗೆಯಿರಿ ಅಥವಾ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ರೋಗ ಖಚಿತವಾಗಿ ಪತ್ತೆಯಾಗಿಲ್ಲ (70% ಗಿಂತ ಕಡಿಮೆ ನಂಬಿಕೆ ಇದೆ). ಫೋಟೋ ಮಬ್ಬಾಗಿದೆ ಅಥವಾ ಬೆಳಕು ಸಾಲುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾಗಿ ಮತ್ತೊಂದು ಫೋಟೋ ತೆಗೆಯಿರಿ ಅಥವಾ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."**

---

### C. Tungro Virus High-Stakes Warning (For Hispa & Downy Mildew Diagnoses)

* **English:** `"Stunted plants with yellow-orange leaves can also be Tungro virus (no cure). If so, consult your KVK."`
* **Old Formal Hindi:** `"बौने पौधे और पीली-नारंगी पत्तियां तुंग्रो वायरस (जिसका कोई इलाज नहीं है) भी हो सकती हैं। यदि ऐसा है, तो अपने केवीके से सलाह लें।"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"सावधानी: अगर पौधे छोटे (बौने) रह गए हैं और पत्तियां पीली या नारंगी पड़ रही हैं, तो यह तुंग्रो वायरस हो सकता है (इसका कोई इलाज नहीं है)। ऐसा दिखे तो तुरंत अपने केवीके से संपर्क करें।"**
* **Old Formal Kannada:** `"ಗಿಡ್ಡವಾದ ಸಸಿಗಳು ಹಾಗೂ ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣದ ಎಲೆಗಳು ಕಂಡುಬಂದರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಆಗಿರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದಲ್ಲಿ, ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಎಚ್ಚರಿಕೆ: ಗಿಡಗಳು ಗಿಡ್ಡವಾಗಿ ನಿಂತು, ಎಲೆಗಳು ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದ್ರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಇರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದರೆ ಕೂಡಲೇ ನಿಮ್ಮ ಕೆವಿಕೆ ಅಧಿಕಾರಿಯನ್ನು ಭೇಟಿ ಮಾಡಿ."**

---

### D. Audit Benchmark Warning (Paddy 0.67 vs 0.94 QWK)

* **English:** `"Independent-test QWK about 0.67 (twin-free, lesion classes); headline 0.94 is inflated."`
* **Old Formal Hindi:** `"स्वतंत्र परीक्षण QWK लगभग 0.67 (ट्विन-मुक्त, घाव वर्ग); मुख्य 0.94 बढ़ा-चढ़ाकर है।"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"ईमानदार जांच स्कोर: असली खेत के टेस्ट में स्कोर लगभग 0.67 है; विज्ञापनों में दिखाया जाने वाला 0.94 का दावा बढ़ा-चढ़ाकर पेश किया गया है।"**
* **Old Formal Kannada:** `"ಸ್ವತಂತ್ರ ಪರೀಕ್ಷೆಯ QWK ಸುಮಾರು 0.67 (ಡ್ಯೂಪ್ಲಿಕೇಟ್-ಮುಕ್ತ, ಗಾಯದ ವರ್ಗಗಳು); ಜಾಹೀರಾತಿನ 0.94 ಉತ್ಪ್ರೇಕ್ಷಿತವಾಗಿದೆ."`
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ನೈಜ ತಪಾಸಣಾ ವರದಿ: ನಿಜವಾದ ಹೊಲದ ಪರೀಕ್ಷೆಯಲ್ಲಿ ನಿಖರತೆ ಸುಮಾರು 0.67 ಇದೆ; ಹೊರಗೆ ಹೇಳಲಾಗುವ 0.94 ರ ಅಂಕ ಉತ್ಪ್ರೇಕ್ಷೆಯದ್ದಾಗಿದೆ."**

---

## 3. 💬 Grounded Chat: Conversational Question & Answer Responses

### Q1: "What should I do first?" (मुझे पहले क्या करना चाहिए? / ನಾನು ಮೊದಲು ಏನು ಮಾಡಬೇಕು?)

#### Case A: Treatable Disease
* **English:** `"Start with organic and cultural steps first: {organic_adv}"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"सबसे पहले जैविक और घरेलू उपाय अपनाएं: {organic_adv} (अगर लक्षण गंभीर हों, तभी रासायनिक दवा डालें)।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಮೊದಲು ನೈಸರ್ಗಿಕ ಮತ್ತು ಸಾವಯವ ಕ್ರಮಗಳನ್ನು ಮಾಡಿ: {organic_adv} (ರೋಗ ತುಂಬಾ ಜಾಸ್ತಿಯಾದ್ರೆ ಮಾತ್ರ ಕೆವಿಕೆ ಹೇಳಿದ ಮದ್ದು ಸಿಂಪಡಿಸಿ)."**

#### Case B: Incurable Viral / Phytoplasma Disease (Tungro, Yellow Leaf Disease)
* **English:** `"Warning: {display_name} detected. No chemical cure exists. Manage insect vectors, remove infected plants, and consult your local KVK."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"चेतावनी: फसल में {display_name} पाया गया है। किसी भी केमिकल या दवा से यह ठीक नहीं हो सकता। तुरंत रस चूसने वाले कीड़ों को रोकें, बीमार पौधों को उखाड़कर खेत से बाहर फेंकें, और अपने नजदीकी केवीके से सलाह लें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಎಚ್ಚರಿಕೆ: ಬೆಳೆಗೆ {display_name} ಬಂದಿದೆ. ಇದಕ್ಕೆ ಯಾವುದೇ ಕೆಮಿಕಲ್ ಅಥವಾ ಔಷಧಿಯಿಂದ ಪರಿಹಾರವಿಲ್ಲ. ರೋಗ ಹರಡುವ ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ರೋಗ ಬಂದಿರೋ ಗಿಡಗಳನ್ನು ತಕ್ಷಣ ಕಿತ್ತು ನಾಶಮಾಡಿ ಮತ್ತು ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."**

---

### Q2: "Is there a cure?" (क्या यह बीमारी ठीक हो सकती है? / ಈ ರೋಗ ವಾಸಿಯಾಗುತ್ತಾ?)

#### Case A: Treatable Disease
* **English:** `"Yes, {display_name} can be managed. Start with cultural/organic steps. If severe, registered chemical options are available from your KVK."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"हाँ, {display_name} को पूरी तरह रोका जा सकता है। शुरुआत में जैविक और साफ-सफाई के उपाय करें। अगर बीमारी बहुत ज्यादा फैल गई है, तो केवीके द्वारा बताई गई सही दवा का इस्तेमाल करें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಹೌದು, {display_name} ರೋಗವನ್ನು ತಡೆಗಟ್ಟಬಹುದು. ಮೊದಲಿಗೆ ಸಾವಯವ ಹಾಗೂ ತೋಟದ ನೈರ್ಮಲ್ಯ ಕಾಪಾಡಿಕೊಳ್ಳಿ. ಸೋಂಕು ತೀವ್ರವಾಗಿದ್ದರೆ ಕೆವಿಕೆ ಶಿಫಾರಸು ಮಾಡಿದ ಸರಿಯಾದ ಮದ್ದು ಬಳಸಿ."**

#### Case B: Incurable Viral / Phytoplasma Disease
* **English:** `"No chemical cure exists. {display_name} is incurable; remove infected plants and manage insect vectors."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"नहीं, इसका कोई रासायनिक इलाज उपलब्ध नहीं है। {display_name} एक लाइलाज बीमारी है; रोग लगे पौधों को खेत से उखाड़कर नष्ट करें और कीड़ों को फैलने से रोकें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಇಲ್ಲ, ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. {display_name} ವಾಸಿಯಾಗದ ರೋಗ; ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತು ಹಾಕಿ ಮತ್ತು ಹರಡುವ ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ."**

---

### Q3: "When should I spray?" (स्प्रे कब और कैसे करना चाहिए? / ಮದ್ದು ಯಾವಾಗ ಸಿಂಪಡಿಸಬೇಕು?)

* **English:** `"Weather alert (rule-based estimate, not a spraying schedule): {weather_alert}. Here, 'protective measure' means the preventive option advised by your local KVK. For mild symptoms, start with cultural and biological steps ({organic_adv}). Do not schedule calendar sprays; chemicals are strictly for severe outbreaks under KVK guidance."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"मौसम की चेतावनी (अनुमान के आधार पर — कोई तय टाइमटेबल नहीं): {weather_alert}। यहाँ 'बचाव' का मतलब आपके नजदीकी केवीके द्वारा बताई गई एहतियात है। हल्के लक्षणों में पहले जैविक उपाय अपनाएं ({organic_adv})। बिना सोचे-समझे हर हफ्ते स्प्रे न करें; तेज रासायनिक दवा केवल तभी डालें जब बीमारी बहुत बढ़ जाए और केवीके ने अनुमति दी हो।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ (ನಿಯಮಿತ ಅಂದಾಜು — ಔಷಧಿಯ ನಿಶ್ಚಿತ ವೇಳಾಪಟ್ಟಿಯಲ್ಲ): {weather_alert}। ಇಲ್ಲಿ 'ಮುಂಜಾಗ್ರತೆ' ಎಂದರೆ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸೂಚಿಸಿದ ಎಚ್ಚರಿಕೆಯ ಕ್ರಮ. ಸಣ್ಣ ಪ್ರಮಾಣದ ಲಕ್ಷಣಗಳಿದ್ದರೆ ಮೊದಲು ಸಾವಯವ ಕ್ರಮಗಳನ್ನು ಮಾಡಿ ({organic_adv})। ಸುಖಾಸುಮ್ಮನೆ ದಿನಾ ಮದ್ದು ಸಿಂಪಡಿಸಬೇಡಿ; ಸೋಂಕು ತೀವ್ರವಾದಾಗ ಮಾತ್ರ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ರಾಸಾಯನಿಕ ಬಳಸಿ."**

---

### Q4: "Why are you unsure?" (ऐप को पक्का भरोसा क्यों नहीं है? / ಆ್ಯಪ್‌ಗೆ ಯಾಕೆ ಪಕ್ಕಾ ಖಚಿತವಿಲ್ಲ?)

#### Confident Diagnosis (Confidence $\ge 70\%$)
* **English:** `"The model diagnosed this image with {conf:.1%} confidence. The app declines to diagnose below 70%. For Hispa and Downy Mildew, Tungro can look similar."`
* **Old Formal Hindi:** `"मॉडल ने इस तस्वीर को {conf:.1%} सटीकता के साथ पहचाना है। सटीकता 70% से कम होने पर ऐप निदान करने से मना करता है। हिस्पा और डाउनी फफूंदी के लिए, तुंग्रो वायरस भी इसी तरह का दिख सकता है।"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"ऐप ने इस फोटो को {conf:.1%} भरोसे के साथ पहचाना है। जब भरोसा 70% से कम होता है, तो ऐप तुक्का लगाने के बजाय रुक जाता है। ध्यान रखें: हिस्पा और डाउनी फफूंदी के लक्षण कभी-कभी तुंग्रो वायरस जैसे भी दिखते हैं।"**
* **Old Formal Kannada:** `"ಮಾದರಿಯು ಈ ಚಿತ್ರವನ್ನು {conf:.1%} ವಿಶ್ವಾಸಾರ್ಹತೆಯೊಂದಿಗೆ ಗುರುತಿಸಿದೆ. ವಿಶ್ವಾಸಾರ್ಹತೆಯು 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ. ಹಿಸ್ಪಾ ಮತ್ತು ಬೂದಿ ರೋಗಕ್ಕೆ, ತುಂಗ್ರೋ ವೈರಸ್ ಕೂಡ ಇದೇ ರೀತಿ ಕಾಣಿಸಬಹುದು."`
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಆ್ಯಪ್ ಈ ಫೋಟೋವನ್ನು {conf:.1%} ನಂಬಿಕೆಯಿಂದ ಗುರುತಿಸಿದೆ. 70% ಕ್ಕಿಂತ ಕಡಿಮೆ ಖಚಿತತೆ ಇದ್ದಾಗ ಆ್ಯಪ್ ತಪ್ಪು ಊಹೆ ಮಾಡದೆ ರೋಗ ಪತ್ತೆ ನಿಲ್ಲಿಸುತ್ತದೆ. ನೆನಪಿರಲಿ: ಹಿಸ್ಪಾ ಮತ್ತು ಬೂದಿ ರೋಗದ ಲಕ್ಷಣಗಳು ಕೆಲವೊಮ್ಮೆ ತುಂಗ್ರೋ ವೈರಸ್ ತರಹವೇ ಕಾಣಿಸಬಹುದು."**

#### Abstained Diagnosis (Confidence $< 70\%$)
* **English:** `"Diagnosis was withheld because confidence ({conf:.1%}) fell below 70%. The app declines to diagnose below 70%. For Hispa and Downy Mildew, Tungro virus can look similar in the field."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"बीमारी का नाम नहीं बताया गया क्योंकि ऐप का भरोसा ({conf:.1%}) 70% से कम रह गया। गलत जानकारी से बचाने के लिए 70% से कम होने पर ऐप कोई नतीजा नहीं देता। खेत में हिस्पा या डाउनी फफूंदी कभी-कभी तुंग्रो वायरस जैसी दिख सकती है, इसलिए केवीके से पुष्टि करें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ರೋಗ ಖಚಿತವಾಗಿಲ್ಲ, ಏಕೆಂದರೆ ಆ್ಯಪ್‌ನ ವಿಶ್ವಾಸ ({conf:.1%}) 70% ಗಿಂತ ಕಡಿಮೆಯಾಗಿದೆ. ರೈತರಿಗೆ ತಪ್ಪು ಮಾಹಿತಿ ತಪ್ಪಿಸಲು 70% ಗಿಂತ ಕಡಿಮೆ ಇದ್ದಾಗ ಆ್ಯಪ್ ರೋಗ ಹೇಳುವುದಿಲ್ಲ. ಹೊಲದಲ್ಲಿ ಹಿಸ್ಪಾ ಅಥವಾ ಬೂದಿ ರೋಗವು ತುಂಗ್ರೋ ವೈರಸ್‌ನಂತೆಯೇ ಕಾಣಿಸಬಹುದಾಗಿದ್ದರಿಂದ ಕೆವಿಕೆಯಲ್ಲಿ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."**

---

## 4. 🌦️ 48-Hour Live Weather Alerts (Spoken In Voice & Screen)

### A. Paddy Blast Alert
* **English:** `"Blast-favourable weather expected in {hours} hours for {dur} hours. Apply protective measures before conditions worsen as advised by your KVK."`
* **Old Formal Hindi:** `"ब्लास्ट-अनुकूल मौसम {hours} घंटों में {dur} घंटों के लिए अपेक्षित है। स्थिति बिगड़ने से पहले केवीके द्वारा अनुशंसित सुरक्षात्मक उपाय लागू करें।"`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"मौसम चेतावनी: अगले {hours} घंटों में धान के ब्लास्ट रोग के अनुकूल मौसम बन रहा है जो {dur} घंटे रहेगा। हालात बिगड़ने से पहले अपने केवीके की सलाह अनुसार बचाव का छिड़काव कर लें।"**
* **Old Formal Kannada:** `"ಬ್ಲಾಸ್ಟ್-ಅನುಕೂಲಕರ ಹವಾಮಾನವು {hours} ಗಂಟೆಗಳಲ್ಲಿ {dur} ಗಂಟೆಗಳ ಕಾಲ ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ. ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮೊದಲು ನಿಮ್ಮ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ರಕ್ಷಣಾತ್ಮಕ ಕ್ರಮಗಳನ್ನು ಅಳವಡಿಸಿ."`
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ (ಬೆಂಕಿ ರೋಗ) ಹೆಚ್ಚಾಗುವ ಹವಾಮಾನವಿದ್ದು, {dur} ಗಂಟೆಗಳ ಕಾಲ ಇರಲಿದೆ. ಹವಾಮಾನ ಕೆಡುವ ಮುನ್ನವೇ ನಿಮ್ಮ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಔಷಧ ಸಿಂಪಡಿಸಿ."**

### B. Apple Scab Alert
* **English:** `"Scab-favourable weather expected in {hours} hours for {dur} hours. Apply protective measures before the wet spell as advised by your KVK."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"मौसम चेतावनी: अगले {hours} घंटों में सेब के स्कैब रोग का खतरा बन रहा है जो {dur} घंटे रहेगा। बारिश शुरू होने से पहले अपने केवीके द्वारा बताई गई दवा का छिड़काव कर लें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಸೇಬಿನ ಸ್ಕ್ಯಾಬ್ ರೋಗಕ್ಕೆ ಪೂರಕವಾದ ಹವಾಮಾನ {dur} ಗಂಟೆಗಳ ಕಾಲ ಇರಲಿದೆ. ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ ಕೆವಿಕೆ ಸೂಚಿಸಿದ ರಕ್ಷಣಾತ್ಮಕ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."**

### C. Arecanut Mahali / Koleroga Alert
* **English:** `"Koleroga-favourable continuous rain expected in {hours} hours for {dur} hours. Ensure bunch spray coverage before the wet spell under KVK guidance."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"मौसम चेतावनी: अगले {hours} घंटों में कोलेलेरोगा (महाली सड़ांध) बढ़ाने वाली लगातार बारिश की संभावना है जो {dur} घंटे रहेगी। भारी बारिश से पहले केवीके की सलाह अनुसार अड़की के गुच्छों पर बोर्डो मिश्रण या दवा का छिड़काव पूरा करें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಕೊಳೆರೋಗ (ಮಹಾಲಿ) ಉಲ್ಬಣಿಸುವ ಸತತ ಮಳೆಯ ಲಕ್ಷಣಗಳಿದ್ದು, {dur} ಗಂಟೆಗಳ ಕಾಲ ಇರಲಿದೆ. ಧಾರಾಕಾರ ಮಳೆ ಆರಂಭವಾಗುವ ಮುನ್ನವೇ ಕೆವಿಕೆ ಮಾರ್ಗದರ್ಶನದಂತೆ ಅಡಿಕೆ ಗೊನೆಗಳಿಗೆ ಬೋರ್ಡೋ ಸಿಂಪರಣೆ ಮುಗಿಸಿ."**

---

## 5. 🌾 Crop Names & Field Disease Common Terminology

### A. Crop Labels
| Key | English | Old Formal Hindi | 🗣️ Spoken Hindi | Old Formal Kannada | 🗣️ Spoken Kannada |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `crop_paddy` | Paddy | धान | **धान (चावल)** | ಭತ್ತ | **ಭತ್ತ** |
| `crop_arecanut` | Arecanut | सुपारी | **सुपारी (अड़की)** | ಅಡಿಕೆ | **ಅಡಿಕೆ** |
| `crop_apple` | Apple | सेब | **सेब** | ಸೇಬು | **ಸೇಬು** |

### B. Paddy Diseases
| Class Key | English Botanical | Old Hindi | 🗣️ Spoken Hindi | Old Kannada | 🗣️ Spoken Kannada |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bacterial_leaf_blight` | Bacterial Leaf Blight | जीवाणु पत्ती झुलसा | **पत्ती झुलसा (बैक्टीरियल ब्लाइट)** | ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಕರಕು ರೋಗ | **ಎಲೆ ಕರಕು ರೋಗ (ಬ್ಯಾಕ್ಟೀರಿಯಲ್)** |
| `bacterial_leaf_streak` | Bacterial Leaf Streak | जीवाणु पत्ती धारी | **पत्तियों पर लंबी धारियां (पत्ती धारी)** | ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಪಟ್ಟೆ ರೋಗ | **ಎಲೆ ಪಟ್ಟೆ ರೋಗ** |
| `bacterial_panicle_blight` | Bacterial Panicle Blight | जीवाणु बाली झुलसा | **बाली का झुलसा रोग** | ತೆನೆ ಕರಕು ರೋಗ | **ತೆನೆ ಕರಕು ರೋಗ** |
| `blast` | Rice Blast | झुलसा रोग (ब्लास्ट) | **धान का ब्लास्ट (झोंका रोग)** | ಬೆಂಕಿ ರೋಗ (ಬ್ಲಾಸ್ಟ್) | **ಬೆಂಕಿ ರೋಗ (ಬ್ಲಾಸ್ಟ್)** |
| `brown_spot` | Brown Spot | भूरा धब्बा रोग | **भूरे धब्बे की बीमारी** | ಕಂದು ಚುಕ್ಕೆ ರೋಗ | **ಕಂದು ಚುಕ್ಕೆ ರೋಗ** |
| `dead_heart` | Dead Heart | डेड हार्ट (तना छेदक प्रकोप) | **तना छेदक (डेड हार्ट - तना सूखना)** | ಸುಳಿ ಒಣಗು ರೋಗ (ಡೆಡ್ ಹಾರ್ಟ್) | **ಸುಳಿ ಒಣಗು ರೋಗ (ಕಾಂಡ ಕೊರೆಯುವ ಕೀಟ)** |
| `downy_mildew` | Downy Mildew | डाउनी फफूंदी | **डाउनी फफूंदी (पत्तियों के नीचे फफूंद)** | ಬೂದಿ ರೋಗ | **ಡೌನಿ ಮಿಲ್ಡ್ಯೂ (ತಳ ಬೂದಿ ರೋಗ)** |
| `hispa` | Hispa | हिस्पा कीट प्रकोप | **धान का हिस्पा (कांटेदार कीड़ा)** | ಹಿಸ್ಪಾ ಕೀಟ ಬಾಧೆ | **ಹಿಸ್ಪಾ ಕೀಟ ಬಾಧೆ** |
| `normal` | Healthy (No Disease) | स्वस्थ (कोई बीमारी नहीं) | **स्वस्थ फसल (कोई बीमारी नहीं)** | ಆರೋಗ್ಯಕರ (ಯಾವುದೇ ರೋಗವಿಲ್ಲ) | **ಆರೋಗ್ಯಕರ ಬೆಳೆ (ಯಾವುದೇ ರೋಗವಿಲ್ಲ)** |
| `tungro` | Tungro Virus | तुंग्रो वायरस | **तुंग्रो वायरस (लाइलाज विषाणु)** | ತುಂಗ್ರೋ ವೈರಸ್ | **ತುಂಗ್ರೋ ವೈರಸ್ (ಔಷಧಿಯಿಲ್ಲದ ರೋಗ)** |

### C. Arecanut Diseases
| Class Key | English Botanical | Old Hindi | 🗣️ Spoken Hindi | Old Kannada | 🗣️ Spoken Kannada |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bud_borer` | Bud Borer | कली छेदक कीट | **अड़की का कली छेदक कीड़ा** | ಸುಳಿ ಕೊರೆಯುವ ಹುಳು (ಮೊಗ್ಗು ಕೊರೆಯುವ ಕೀಟ) | **ಸುಳಿ ಕೊರೆಯುವ ಹುಳು** |
| `mahali_koleroga` | Mahali / Koleroga | महाली / कोलेलेरोगा (फल सड़न) | **महाली / कोलेलेरोगा (अड़की फल सड़न)** | ಮಹಾಲಿ / ಕೊಳೆರೋಗ (ಕಾಯಿ ಕೊಳೆ) | **ಮಹಾಲಿ / ಕೊಳೆರೋಗ (ಅಡಿಕೆ ಉದುರುವುದು)** |
| `stem_bleeding` | Stem Bleeding | तना स्राव (स्टेम ब्लीडिंग) | **तने से रस बहना (स्टेम ब्लीडिंग)** | ಕಾಂಡ ಸೋರುವಿಕೆ (ತೊಗಟೆ ಸ್ರಾವ) | **ಕಾಂಡ ಸೋರುವಿಕೆ (ಕಾಂಡದಲ್ಲಿ ರಸ ಸೋರುವುದು)** |
| `stem_cracking` | Stem Cracking | तना फटना (स्टेम क्रैकिंग) | **तने का फटना (दरारें पड़ना)** | ಕಾಂಡ ಬಿರುಕು ರೋಗ | **ಕಾಂಡ ಬಿರುಕು ರೋಗ** |
| `yellow_leaf_disease` | Yellow Leaf Disease | पीली पत्ती रोग (वाईएलडी) | **पीली पत्ती रोग (लाइलाज वाईएलडी)** | ಹಳದಿ ಎಲೆ ರೋಗ (ವೈಎಲ್‌ಡಿ) | **ಹಳದಿ ಎಲೆ ರೋಗ (ವೈಎಲ್‌ಡಿ)** |

### D. Apple Diseases
| Class Key | English Botanical | Old Hindi | 🗣️ Spoken Hindi | Old Kannada | 🗣️ Spoken Kannada |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `apple_scab` | Apple Scab | सेब का स्कैब | **सेब का स्कैब (काले-भूरे धब्बे)** | ಸೇಬಿನ ಸ್ಕ್ಯಾಬ್ ರೋಗ | **ಸೇಬಿನ ಸ್ಕ್ಯಾಬ್ ರೋಗ** |
| `black_rot` | Black Rot | ब्लैक रॉट (काला सड़न) | **काला सड़न रोग (ब्लैक रॉट)** | ಕಪ್ಪು ಕೊಳೆ ರೋಗ | **ಕಪ್ಪು ಕೊಳೆ ರೋಗ** |
| `cedar_apple_rust` | Cedar Apple Rust | सीडर एप्पल रस्ट (गेरुआ) | **सीडर एप्पल रस्ट (जंग रोग)** | ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ) | **ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ)** |
| `healthy` | Healthy (No Disease) | स्वस्थ (कोई बीमारी नहीं) | **स्वस्थ सेब (कोई बीमारी नहीं)** | ಆರೋಗ್ಯಕರ (ಯಾವುದೇ ರೋಗವಿಲ್ಲ) | **ಆರೋಗ್ಯಕರ ಸೇಬು (ಯಾವುದೇ ರೋಗವಿಲ್ಲ)** |

---

## 6. 🎙️ Spoken Voice Advisory Scripts (gTTS Audio Engine)

### A. Healthy Crop Voice Script
* **English:** `"Crop: {crop}. Foliage is healthy with no disease symptoms. No treatment needed."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"फसल: {crop}। आपकी फसल बिल्कुल स्वस्थ है और पत्तों पर कोई बीमारी नहीं है। किसी भी दवा या छिड़काव की जरूरत नहीं है।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಬೆಳೆ: {crop}। ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ ಮತ್ತು ಎಲೆಗಳಲ್ಲಿ ಯಾವುದೇ ರೋಗವಿಲ್ಲ. ಯಾವುದೇ ಔಷಧ ಸಿಂಪರಣೆಯ ಅಗತ್ಯವಿಲ್ಲ."**

---

### B. Mild Infection Voice Script (Organic-First)
* **English:** `"Crop: {crop}. Disease: {disease}. Mild severity. Start with organic and cultural steps: remove infected leaves and maintain field sanitation. Consult your local KVK before using chemicals."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"फसल: {crop}। बीमारी: {disease}। संक्रमण अभी शुरुआती अवस्था में है। पहले जैविक उपाय करें: खराब पत्तियों को तोड़कर हटा दें और खेत में साफ-सफाई रखें। तेज रासायनिक दवा डालने से पहले अपने केवीके से जरूर पूछें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಬೆಳೆ: {crop}। ರೋಗ: {disease}। ರೋಗದ ಲಕ್ಷಣಗಳು ಈಗಷ್ಟೇ ಆರಂಭವಾಗಿವೆ. ಮೊದಲು ಸಾವಯವ ಕ್ರಮಗಳನ್ನು ಮಾಡಿ: ರೋಗ ಬಂದಿರೋ ಎಲೆಗಳನ್ನು ಕಿತ್ತು ಹಾಕಿ ಮತ್ತು ತೋಟವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ. ರಾಸಾಯನಿಕ ಮದ್ದು ಬಳಸುವ ಮುನ್ನ ಕೆವಿಕೆಯಿಂದ ಸಲಹೆ ಪಡೆಯಿರಿ."**

---

### C. Moderate to Severe Infection Voice Script
* **English:** `"Crop: {crop}. Disease: {disease}. Moderate to severe infection. Cultural measures alone are insufficient. Consult your local KVK for registered chemical treatments and follow product label advice."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"फसल: {crop}। बीमारी: {disease}। बीमारी काफी बढ़ चुकी है और केवल जैविक उपाय काफी नहीं होंगे। अपनी नजदीकी कृषि विज्ञान केंद्र (केवीके) से संपर्क करें और उनके द्वारा बताई गई सही दवा को डिब्बे पर लिखे निर्देशानुसार ही छिड़कें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಬೆಳೆ: {crop}। ರೋಗ: {disease}। ರೋಗವು ತೀವ್ರವಾಗಿ ಉಲ್ಬಣಿಸಿದೆ ಮತ್ತು ಬರೀ ಸಾವಯವ ಕ್ರಮ ಸಾಕಾಗುವುದಿಲ್ಲ. ತಕ್ಷಣ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಕೃಷಿ ಅಧಿಕಾರಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ, ಅವರು ಶಿಫಾರಸು ಮಾಡಿದ ಸರಿಯಾದ ಔಷಧಿಯನ್ನು ಲೇಬಲ್ ಸೂಚನೆಯಂತೆ ಸಿಂಪಡಿಸಿ."**

---

## 7. 🌱 Soil Health & Nutrient Management Cards

### A. Paddy Soil Guidance
* **English:** `"Incorporate organic compost or green manure (Dhaincha). Inoculate with Azolla or biofertilisers to reduce synthetic urea reliance. Conduct a soil test before adding nutrients."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"खेत में गोबर की सड़ी खाद या हरी खाद (ढैंचा) मिलाएं। यूरिया पर निर्भरता कम करने के लिए अजोला या बायो-फर्टिलाइजर का इस्तेमाल करें। खाद डालने से पहले मिट्टी की जांच जरूर करवाएं।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಗದ್ದೆಗೆ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಅಥವಾ ಹಸಿರೆಲೆ ಗೊಬ್ಬರ (ಡೈಂಚಾ) ಹಾಕಿ. ಯೂರಿಯಾ ಖರ್ಚು ಕಡಿಮೆ ಮಾಡಲು ಅಜೋಲ್ಲಾ ಅಥವಾ ಜೈವಿಕ ಗೊಬ್ಬರ ಸೇರಿಸಿ. ಗೊಬ್ಬರ ಹಾಕುವ ಮುನ್ನ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ."**

### B. Arecanut Soil Guidance
* **English:** `"Apply well-decomposed farmyard manure and green leaf mulching in the basin before the monsoon. Inoculate with mycorrhiza (VAM) and Trichoderma-enriched compost to suppress soil-borne foot rot."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"मानसून शुरू होने से पहले अड़की के थाले में अच्छी तरह सड़ी गोबर की खाद और हरी पत्तियों की मल्चिंग करें। जड़ों की सड़न रोकने के लिए ट्राइकोडर्मा और माइकोराइजा मिली हुई खाद डालें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಮಳೆಗಾಲ ಆರಂಭವಾಗುವ ಮುನ್ನವೇ ಅಡಿಕೆ ಬುಡಕ್ಕೆ ಚೆನ್ನಾಗಿ ಕೊಳೆತ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಮತ್ತು ಹಸಿರೆಲೆ ಹೊದಿಕೆ ಹಾಕಿ. ಬೇರು ಕೊಳೆ ರೋಗ ತಡೆಯಲು ಟ್ರೈಕೋಡರ್ಮ ಮಿಶ್ರಿತ ಸಾವಯವ ಗೊಬ್ಬರವನ್ನು ಬುಡಕ್ಕೆ ಕೊಡಿ."**

### C. Apple Soil Guidance
* **English:** `"Apply well-rotted farmyard manure and maintain organic tree-basin mulching during dormancy. Conduct a soil test before applying calcium or boron amendments."`
* **🗣️ Spoken Day-to-Day Hindi:**
  > **"सर्दियों में पेड़ के थाले में सड़ी गोबर की खाद डालें और घास-फूस की मल्चिंग बनाए रखें। कैल्शियम या बोरॉन डालने से पहले बगीचे की मिट्टी की जांच जरूर करवा लें।"**
* **🗣️ Spoken Day-to-Day Kannada:**
  > **"ಗಿಡಗಳು ಸುಪ್ತಾವಸ್ಥೆಯಲ್ಲಿದ್ದಾಗ (ಚಳಿಗಾಲದಲ್ಲಿ) ಬುಡಕ್ಕೆ ಕಳಿತ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಹಾಕಿ ಎಲೆಗಳ ಹೊದಿಕೆ ಮಾಡಿ. ಕ್ಯಾಲ್ಸಿಯಂ ಅಥವಾ ಬೋರಾನ್ ಕೊಡುವ ಮುನ್ನ ತೋಟದ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."**

---

## 8. Summary of Replaced Bookish Terms

| Bookish Sanskritized Hindi | Natural Spoken Hindi | Bookish Formal Kannada | Natural Spoken Kannada |
| :--- | :--- | :--- | :--- |
| **निदान** (Nidaan) | **पहचान / जांच** (Pehchan / Jaanch) | **ರೋಗನಿರ್ಣಯ** (Roganirnaya) | **ರೋಗ ಪತ್ತೆ** (Roga Patte) |
| **सटीकता / विश्वासार्ಹತೆ** | **भरोसा / यकीन** (Bharosa / Yakeen) | **ವಿಶ್ವಾಸಾರ್ಹತೆ** (Vishvasarhate) | **ಖಚಿತತೆ / ನಂಬಿಕೆ** (Khachitate / Nambike) |
| **रोकथाम / नियंत्रण** | **रोकना / बचाव** (Rokna / Bachaav) | **ನಿಯಂತ್ರಣ** (Niyantrana) | **ತಡೆಗಟ್ಟುವಿಕೆ** (Tadegattuvike) |
| **नष्ट करें** | **उखाड़कर फेंक दें** (Ukhaad kar phenk dein) | **ನಾಶಪಡಿಸಿ** (Nashapadisi) | **ಕಿತ್ತು ಬಿಸಾಕಿ / ನಾಶಮಾಡಿ** |
| **छिड़काव अनुसूची** | **स्प्रे का टाइमटेबल** (Spray ka timetable) | **ಸಿಂಪರಣಾ ವೇಳಾಪಟ್ಟಿ** | **ಮದ್ದು ಸಿಂಪಡಿಸುವ ಸಮಯ** |
| **जीवाणु** | **बैक्टीरिया** (Bacteria) | **ಬ್ಯಾಕ್ಟೀರಿಯಲ್** | **ಬ್ಯಾಕ್ಟೀರಿಯಾ ರೋಗ** |
| **प्रकोप** | **फैलाव / असर** (Phailav / Asar) | **ಬಾಧೆ / ಉಲ್ಬಣ** | **ಕಾಟ / ರೋಗ ಹೆಚ್ಚಾಗುವುದು** |
