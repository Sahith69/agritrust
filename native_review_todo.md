# Native-Speaker Review TODO List (Kannada & Hindi)

This document tracks all Kannada (`kn`) and Hindi (`hi`) localized agricultural disease terminology and voice advisory strings used in AgriTrust AI for validation by native-speaking agricultural extension officers / agronomists before field deployment.

---

## 🚨 Priority Items Flagged for Immediate Review

1. **Hindi: "Rice Blast" (`blast`)**
   - **Current string:** `"झुलसा रोग (ब्लास्ट)"` / `"झुलसा रोग"`
   - **Reason for Review:** In Hindi agronomic terminology, *"झुलसा"* typically denotes Blight (*Bacterial Leaf Blight* or *Phytophthora blight*), whereas Rice Blast (*Magnaporthe oryzae*) is standardly referenced as *"झोंका रोग"* or simply *"ब्लास्ट रोग"* in ICAR / KVK publications.
   - **Proposed native alternatives to verify:** `"झोंका रोग (ब्लास्ट)"` vs `"धान का ब्लास्ट रोग"`.

2. **Kannada: "Downy Mildew" (`downy_mildew`)**
   - **Current string:** `"ಬೂದಿ ರೋಗ"`
   - **Reason for Review:** In Kannada agronomic usage, *"ಬೂದಿ ರೋಗ"* (Boodi Roga) traditionally and universally denotes **Powdery Mildew** (*Erysiphe* / *Oidium* powdery spore coating on leaf surfaces). Downy Mildew (*Sclerospora* / *Peronospora*) is an oomycete infection affecting underside leaf tissue and is standardly referred to in UAS Dharwad / GKVK Bangalore literature as *"ಡೌನಿ ಮಿಲ್ಡ್ಯೂ"* (Downy Mildew) or *"ತಳ ಬೂದಿ ರೋಗ"*.
   - **Proposed native alternatives to verify:** `"ಡೌನಿ ಮಿಲ್ಡ್ಯೂ (ತಳ ಬೂದಿ ರೋಗ)"` vs `"ಡೌನಿ ಮಿಲ್ಡ್ಯೂ ರೋಗ"`.

3. **Kannada: Garbled Cedar Apple Rust Advisory Sentence (`cedar_apple_rust`)**
   - **Current string:** `"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಹತ್ತಿರದಲ್ಲಿ ಜೂನಿಪರ್ ಅಥವಾ ಸೀಡರ್ ಮರಗಳಿದ್ದರೆ ಅವುಗಳಲ್ಲಿನ ಗಂಟುಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬೆಳೆಯಿರಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
   - **Reason for Review:** The sentence *"ಹತ್ತಿರದಲ್ಲಿ ಜೂನಿಪರ್ ಅಥವಾ ಸೀಡರ್ ಮರಗಳಿದ್ದರೆ ಅವುಗಳಲ್ಲಿನ ಗಂಟುಗಳನ್ನು ತೆಗೆಯಿರಿ"* (removing rust galls if juniper or cedar trees are present nearby) reads awkwardly/garbled in Kannada. Eastern red cedar and juniper are non-native temperate trees unfamiliar to Kannada farmers, and *"ಗಂಟುಗಳು"* literally means knots or physical joints rather than fungal galls (*ರಸ್ಟ್ ಗಂಟು / ಶಿಲೀಂಧ್ರ ಗಂಟು*). Additionally, Cedar Apple Rust (*Gymnosporangium juniperi-virginianae*) is a North American disease class from PlantVillage lab data not established in Indian apple orchards.
   - **Proposed native alternatives to verify:** Rephrase with natural agronomic phrasing: *"ತೋಟದ ಆಸುಪಾಸಿನಲ್ಲಿ ಜೂನಿಪರ್ ಅಥವಾ ಸೀಡರ್ ಮರಗಳಿದ್ದರೆ ಅವುಗಳ ಶಿಲೀಂಧ್ರ ಗಂಟುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬೆಳೆಯಿರಿ."*

4. **Hindi: Verification of Dropped Chemical Advice Line across Templates**
   - **Current strings:**
     - Mild treatable diseases: `"रसायन छिड़कने से पहले केवीके से सलाह लें।"`
     - Moderate / severe diseases: `"अनुशंसित दवाओं के लिए अपने नजदीकी कृषि विज्ञान केंद्र से संपर्क करें और लेबल के अनुसार छिड़काव करें।"`
     - Apple diseases: `"रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"`
   - **Reason for Review:** In earlier prototype iterations, the chemical consultation line was inadvertently dropped or truncated in Hindi voice templates for mild infections and apple advisories. While reinstated, native speakers should review that the chemical guidance line is grammatically natural and uniformly appended across all disease severity levels, ensuring no condition branch drops the mandatory KVK chemical consultation directive.
   - **Proposed native alternatives to verify:** Ensure consistent phrasing such as *"रासायनिक उपचार के लिए अनुमोदित कीटनाशक/फफूंदनाशक हेतु अपने नजदीकी केवीके से सलाह लें और उत्पाद लेबल के निर्देशानुसार ही उपयोग करें।"*

---

## 🌾 Paddy (ಭತ್ತ / धान) — Disease Class Names

| Class Key | English Name | Kannada String | Hindi String |
| :--- | :--- | :--- | :--- |
| `bacterial_leaf_blight` | Bacterial Leaf Blight | `ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಕರಕು ರೋಗ` | `जीवाणु पत्ती झुलसा` |
| `bacterial_leaf_streak` | Bacterial Leaf Streak | `ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಪಟ್ಟೆ ರೋಗ` | `जीवाणु पत्ती धारी` |
| `bacterial_panicle_blight` | Bacterial Panicle Blight | `ತೆನೆ ಕರಕು ರೋಗ` | `जीवाणु बाली झुलसा` |
| `blast` | Rice Blast | `ಬೆಂಕಿ ರೋಗ (ಬ್ಲಾಸ್ಟ್)` | `झुलसा रोग (ब्लास्ट)` *(Flagged above)* |
| `brown_spot` | Brown Spot | `ಕಂದು ಚುಕ್ಕೆ ರೋಗ` | `भूरा धब्बा रोग` |
| `dead_heart` | Dead Heart | `ಸುಳಿ ಒಣಗು ರೋಗ (ಡೆಡ್ ಹಾರ್ಟ್)` | `डेड हार्ट (तना छेदक प्रकोप)` |
| `downy_mildew` | Downy Mildew | `ಬೂದಿ ರೋಗ` *(Flagged above)* | `डाउनी फफूंदी` |
| `hispa` | Hispa | `ಹಿಸ್ಪಾ ಕೀಟ ಬಾಧೆ` | `हिस्पा कीट प्रकोप` |
| `normal` | Healthy (No Disease) | `ಆರೋಗ್ಯಕರ ಬೆಳೆ` | `स्वस्थ फसल` |
| `tungro` | Tungro Virus | `ತುಂಗ್ರೋ ವೈರಸ್ ರೋಗ` | `तुंग्रो वायरस` |

---

## 🌴 Arecanut (ಅಡಿಕೆ / सुपारी) — Condition Class Names

| Class Key | English Name | Kannada String | Hindi String |
| :--- | :--- | :--- | :--- |
| `bud_borer` | Bud Borer | `ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು` | `कली छेदक कीट` |
| `healthy_foot` | Healthy (Foot) | `ಆರೋಗ್ಯಕರ ಬುಡ` | `स्वस्थ तना आधार` |
| `healthy_leaf` | Healthy (Leaf) | `ಆರೋಗ್ಯಕರ ಗರಿ` | `स्वस्थ पत्ती` |
| `healthy_nut` | Healthy (Nut) | `ಆರೋಗ್ಯಕರ ಅಡಿಕೆ ಕಾಯಿ` | `स्वस्थ सुपारी फल` |
| `healthy_trunk` | Healthy (Trunk) | `ಆರೋಗ್ಯಕರ ಮರ` | `स्वस्थ मुख्य तना` |
| `mahali_koleroga` | Mahali / Koleroga | `ಮಹಾಳಿ / ಕೊಳೆರೋಗ` | `महाली / कोलेरोगा सड़ांध` |
| `stem_bleeding` | Stem Bleeding | `ಕಾಂಡ ಸೋರುವಿಕೆ ರೋಗ` | `तना स्राव रोग` |
| `stem_cracking` | Stem Cracking | `ಕಾಂಡ ಬಿರುಕು ರೋಗ` | `तना फटना रोग` |
| `yellow_leaf_disease` | Yellow Leaf Disease | `ಹಳದಿ ಎಲೆ ರೋಗ` | `पीली पत्ती रोग` |

---

## 🍎 Apple (ಸೇಬು / सेब) — Disease Class Names

| Class Key | English Name | Kannada String | Hindi String |
| :--- | :--- | :--- | :--- |
| `apple_scab` | Apple Scab | `ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ರೋಗ` | `सेब का स्केब रोग` |
| `black_rot` | Black Rot | `ಕಪ್ಪು ಕೊಳೆರೋಗ` | `ब्लैक रॉट (काला सड़न रोग)` |
| `cedar_apple_rust` | Cedar Apple Rust | `ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ)` | `सीडर एप्पल रस्ट (गेरुआ रोग)` |
| `healthy` | Healthy (No Disease) | `ಆರೋಗ್ಯಕರ ಎಲೆ` | `स्वस्थ पत्तियां` |

---

## 🎙️ Multilingual Voice Advisory Templates

### 1. Healthy Foliage
- **Kannada:** `"ಬೆಳೆ: {crop_lbl}. ಎಲೆಗಳು ಆರೋಗ್ಯಕರವಾಗಿವೆ ಮತ್ತು ಯಾವುದೇ ರೋಗದ ಲಕ್ಷಣಗಳಿಲ್ಲ. ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."`
- **Hindi:** `"फसल: {crop_lbl}। पत्तियां स्वस्थ हैं और किसी रोग के लक्षण नहीं हैं। किसी उपचार की आवश्यकता नहीं है।"`

### 2. Apple Voice Advisories (Disease-specific cultural steps, no severity mention)
- **Apple Scab:**
  - *Kannada:* `"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಉದುರಿದ ಎಲೆಗಳನ್ನು ಹೆಕ್ಕಿ ಗೊಬ್ಬರ ಮಾಡಿ ಮತ್ತು ಎಲೆಗಳು ಬೇಗನೆ ಒಣಗಲು ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ ಗಾಳಿಯಾಡುವಂತೆ ಮಾಡಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
  - *Hindi:* `"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय: गिरी हुई पत्तियों को इकट्ठा करके नष्ट करें और हवा के आवागमन के लिए पेड़ों की छंटाई करें ताकि पत्तियां जल्दी सूखें। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"`
- **Black Rot:**
  - *Kannada:* `"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಒಣಗಿದ ಕೊಂಬೆಗಳು ಮತ್ತು ಒಣಗಿದ ಹಣ್ಣುಗಳನ್ನು ಮರ ಮತ್ತು ನೆಲದಿಂದ ತೆಗೆದುಹಾಕಿ ಶಿಲೀಂಧ್ರ ಮೂಲಗಳನ್ನು ನಾಶಪಡಿಸಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
  - *Hindi:* `"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय: सूखी टहनियों, घावों और सड़े-सूखे फलों को पेड़ों और बगीचे से हटाकर नष्ट करें ताकि फफूंद के स्रोत खत्म हों। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"`
- **Cedar Apple Rust:**
  - *Kannada:* `"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು: ಹತ್ತಿರದಲ್ಲಿ ಜೂನಿಪರ್ ಅಥವಾ ಸೀಡರ್ ಮರಗಳಿದ್ದರೆ ಅವುಗಳಲ್ಲಿನ ಗಂಟುಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬೆಳೆಯಿರಿ. ರಾಸಾಯನಿಕ ಬಳಕೆಗಾಗಿ ನೋಂದಾಯಿತ ಔಷಧಿಗೆ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
  - *Hindi:* `"फसल: {crop_lbl}। रोग: {dis_lbl}। जैविक और पारंपरिक उपाय: यदि आसपास जुनिपर या सीडर के पेड़ मौजूद हों तो रस्ट की गांठों को हटाएं और प्रतिरोधी किस्में लगाएं। रासायनिक दवाओं के लिए अनुमोदित उत्पाद हेतु अपने केवीके से संपर्क करें।"`

### 3. Incurable Diseases (Paddy Tungro & Arecanut YLD)
- **Kannada:** `"ಬೆಳೆ: {crop_lbl}. ಎಚ್ಚರಿಕೆ: {dis_lbl} ಕಂಡುಬಂದಿದೆ. ಇದಕ್ಕೆ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಪರಿಹಾರವಿಲ್ಲ. ಕೀಟ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಕಿತ್ತುಹಾಕಿ, ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
- **Hindi:** `"फसल: {crop_lbl}। चेतावनी: {dis_lbl} का संक्रमण पाया गया है। इसका कोई रासायनिक उपचार उपलब्ध नहीं है। रसचूसक कीटों को नियंत्रित करें, रोगी पौधों को हटाएं और तुरंत कृषि विज्ञान केंद्र से संपर्क करें।"`

### 4. Treatable Diseases (Paddy & Arecanut)
- **Mild (Grade < 2):**
  - *Kannada:* `"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ಲಘು ತೀವ್ರತೆ. ಮೊದಲು ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ: ರೋಗಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ತೋಟವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ. ರಾಸಾಯನಿಕ ಬಳಸುವ ಮುನ್ನ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`
  - *Hindi:* `"फसल: {crop_lbl}। रोग: {dis_lbl}। शुरुआती अवस्था। पहले जैविक और पारंपरिक उपाय अपनाएं: संक्रमित पत्तियों को हटाएं और खेत साफ रखें। रसायन छिड़कने से पहले केवीके से सलाह लें।"`
- **Moderate / Severe (Grade >= 2):**
  - *Kannada:* `"ಬೆಳೆ: {crop_lbl}. ರೋಗ: {dis_lbl}. ತೀವ್ರತೆಯು ಹೆಚ್ಚಾಗಿದೆ. ಸಾವಯವ ಕ್ರಮಗಳು ಮಾತ್ರ ಸಾಕಾಗುವುದಿಲ್ಲ. ಶಿಫಾರಸು ಮಾಡಿದ ಔಷಧಿಗಳಿಗಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಲೇಬಲ್ ಸೂಚನೆ ಪಾಲಿಸಿ."`
  - *Hindi:* `"फसल: {crop_lbl}। रोग: {dis_lbl}। संक्रमण गंभीर है। केवल जैविक उपाय पर्याप्त नहीं हैं। अनुशंसित दवाओं के लिए अपने नजदीकी कृषि विज्ञान केंद्र से संपर्क करें और लेबल के अनुसार छिड़काव करें।"`

---

## 5. 🌱 Soil Health Card Advisories (Kannada & Hindi)

### Title
- **Kannada:** `"ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಪೋಷಕಾಂಶ ನಿರ್ವಹಣೆ"`
- **Hindi:** `"मृदा स्वास्थ्य एवं पोषक तत्व प्रबंधन"`

### Paddy (ಭತ್ತ / धान)
- **Kannada:** `"ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಅಥವಾ ಹಸಿರೆಲೆ ಗೊಬ್ಬರ (ಡೈಂಚಾ) ಬಳಸಿ. ಯೂರಿಯಾ ಅವಲಂಬನೆ ಕಡಿಮೆ ಮಾಡಲು ಅಜೋಲ್ಲಾ ಅಥವಾ ಜೈವಿಕ ಗೊಬ್ಬರ ಸೇರಿಸಿ. ಪೋಷಕಾಂಶಗಳನ್ನು ನೀಡುವ ಮುನ್ನ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."`
- **Hindi:** `"सड़ा गोबर का खाद या हरी खाद (ढैंचा) मिलाएं। यूरिया पर निर्भरता घटाने के लिए अजोला या जैव उर्वरक का उपयोग करें। पोषक तत्व देने से पहले मिट्टी परीक्षण अवश्य करवाएं।"`

### Arecanut (ಅಡಿಕೆ / सुपारी)
- **Kannada:** `"ಅಡಿಕೆ ಬುಡಕ್ಕೆ ಒಣಗಿದ ಅಡಿಕೆ ಗರಿಗಳಿಂದ ಮಲ್ಚಿಂಗ್ ಮಾಡಿ. ಚೆನ್ನಾಗಿ ಕೊಳೆತ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಮತ್ತು ಟ್ರೈಕೋಡರ್ಮ ಮಿಶ್ರಿತ ಕಾಂಪೋಸ್ಟ್ ಹಾಕಿ. ಬೋರಾನ್ ಮತ್ತು ಪೊಟ್ಯಾಷ್ ಕೊರತೆಗಾಗಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."`
- **Hindi:** `"सुपारी के तने के चारों ओर पत्तियों से मल्चिंग करें। अच्छी सड़ी गोबर की खाद और ट्राइकोडर्मा युक्त कंपोस्ट डालें। बोरॉन और पोटाश के लिए मिट्टी परीक्षण करवाएं।"`

### Apple (ಸೇಬು / सेब)
- **Kannada:** `"ವಿಶ್ರಾಂತ ಕಾಲದಲ್ಲಿ ಮರದ ಬುಡಕ್ಕೆ ಚೆನ್ನಾಗಿ ಕೊಳೆತ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಹಾಕಿ ಮಲ್ಚಿಂಗ್ ಮಾಡಿ. ಕ್ಯಾಲ್ಸಿಯಂ ಅಥವಾ ಬೋರಾನ್ ನೀಡುವ ಮುನ್ನ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."`
- **Hindi:** `"सुप्त अवस्था के दौरान थालों में सड़ी गोबर की खाद डालें और मल्चिंग करें। कैल्शियम या बोरॉन डालने से पहले मिट्टी परीक्षण करवाएं।"`

---

## 6. 🌦️ 48-Hour Preventive Weather Alert Voice Templates

### A. General / Rain Diseases (Apple Scab & Arecanut Koleroga) — "Before the wet spell"
- **Disease-favourable conditions active now:**
  - *Kannada:* `"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಈಗ ಸಕ್ರಿಯವಾಗಿದೆ ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."`  # REVIEW
  - *Hindi:* `"मौसम चेतावनी: रोग के अनुकूल मौसम अभी सक्रिय है ({n_matched} घंटे तक)। बारिश से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"`  # REVIEW
- **Disease-favourable conditions expected in X hours:**
  - *Kannada:* `"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಸುಮಾರು {first_hour} ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಮಳೆ ಶುರುವಾಗುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."`  # REVIEW
  - *Hindi:* `"मौसम चेतावनी: लगभग {first_hour} घंटों में रोग के अनुकूल मौसम की संभावना है ({n_matched} घंटे तक)। बारिश से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"`  # REVIEW

### B. Paddy Blast (`blast`) — "Before conditions worsen" (Updated)
- **UI Advisory Text:**
  - *English:* `"Apply a protective measure before conditions worsen, as advised by your KVK."`
- **Disease-favourable conditions active now:**
  - *Kannada:* `"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಈಗ ಸಕ್ರಿಯವಾಗಿದೆ ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."`  # REVIEW (Kannada: ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ - before conditions worsen)
  - *Hindi:* `"मौसम चेतावनी: रोग के अनुकूल मौसम अभी सक्रिय है ({n_matched} घंटे तक)। स्थिति बिगड़ने से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"`  # REVIEW (Hindi: स्थिति बिगड़ने से पहले - before conditions worsen)
- **Disease-favourable conditions expected in X hours:**
  - *Kannada:* `"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಸುಮಾರು {first_hour} ಗಂಟೆಗಳಲ್ಲಿ ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರ ವಾತಾವರಣ ಉಂಟಾಗಬಹುದು ({n_matched} ಗಂಟೆಗಳ ಕಾಲ). ಪರಿಸ್ಥಿತಿ ಹದಗೆಡುವ ಮುನ್ನ ಕೆವಿಕೆ ಸಲಹೆಯಂತೆ ಮುಂಜಾಗ್ರತಾ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."`  # REVIEW
  - *Hindi:* `"मौसम चेतावनी: लगभग {first_hour} घंटों में रोग के अनुकूल मौसम की संभावना है ({n_matched} घंटे तक)। स्थिति बिगड़ने से पहले केवीके की सलाह अनुसार बचाव के उपाय करें।"`  # REVIEW

### C. Low disease weather risk (0 matched hours):
- *Kannada:* `"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಮುಂದಿನ ನಲವತ್ತೆಂಟು ಗಂಟೆಗಳಲ್ಲಿ ಹವಾಮಾನದಿಂದ ರೋಗದ ಅಪಾಯ ಕಡಿಮೆ ಇದೆ."`  # REVIEW
- *Hindi:* `"मौसम चेतावनी: अगले अड़तालीस घंटों में मौसम से रोग का जोखिम कम है।"`  # REVIEW

---

## 7. ⚠️ Tungro Safety Warning Note (Hispa & Downy Mildew Diagnoses)

- **English:** `"Stunted plants with yellow-orange leaves can also be Tungro virus (no cure). If so, consult your KVK."`
- **Kannada:** `"ಗಿಡ್ಡವಾದ ಸಸಿಗಳು ಹಾಗೂ ಹಳದಿ-ಕಿತ್ತಳೆ ಬಣ್ಣದ ಎಲೆಗಳು ಕಂಡುಬಂದರೆ ಅದು ತುಂಗ್ರೋ ವೈರಸ್ ಆಗಿರಬಹುದು (ಇದಕ್ಕೆ ಯಾವುದೇ ಔಷಧಿಯಿಲ್ಲ). ಹಾಗಿದ್ದಲ್ಲಿ, ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`  # REVIEW (Kannada)
- **Hindi:** `"बौने पौधे और पीली-नारंगी पत्तियां तुंग्रो वायरस (जिसका कोई इलाज नहीं है) भी हो सकती हैं। यदि ऐसा है, तो अपने केवीके से सलाह लें।"`  # REVIEW (Hindi)

---

## 8. 🗣️ Disease-Specific Voice Advisories for 7 Treatable Diseases (Card-Aligned)

Constructed directly from card text (Grade 1 organic steps; Grade $\ge 2$ chemical steps) with zero new agronomy.

### 1. Rice Blast (`blast`)
- **Grade 1 (Organic):**
  - *English:* `"Apply bio-fungicide Trichoderma viride or neem-based formulation; follow product label and local KVK advice."`
  - *Kannada:* `"ಟ್ರೈಕೋಡರ್ಮ ವಿರಿಡೆ ಜೈವಿಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ ಬೇವಿನ ಸೂತ್ರೀಕರಣವನ್ನು ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆಯನ್ನು ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"जैव कवकनाशी ट्राइकोडर्मा विरिडे या नीम आधारित कीटनाशक का छिड़काव करें; लेबल और स्थानीय केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Chemical):**
  - *English:* `"Apply systemic fungicide Tricyclazole 75 WP; follow product label and local KVK advice."`
  - *Kannada:* `"ಶಿಫಾರಸು ಮಾಡಿದ ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ 75 ಡಬ್ಲ್ಯೂಪಿ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"प्रणालीगत कवकनाशी ट्राइसाइक्लाजोल 75 डब्ल्यूपी का छिड़काव करें; लेबल निर्देश और केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)

### 2. Mahali / Koleroga (`mahali_koleroga`)
- **Grade 1 (Organic / Cultural):**
  - *English:* `"Tie polythene covers over developing bunches; apply preventive 1% Bordeaux Mixture before monsoon; follow product label and local KVK advice."`
  - *Kannada:* `"ಬೆಳೆಯುತ್ತಿರುವ ಅಡಿಕೆ ಗೊನೆಗಳಿಗೆ ಪಾಲಿಥಿನ್ ಹೊದಿಕೆ ಕಟ್ಟಿ; ಮುಂಗಾರಿಗೆ ಮುನ್ನ ಮುನ್ನೆಚ್ಚರಿಕೆಯಾಗಿ ಶೇಕಡಾ 1 ರ ಬೋರ್ಡೋ ದ್ರಾವಣ ಸಿಂಪಡಿಸಿ ಮತ್ತು ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"सुपारी के गुच्छों पर पॉलीथिन बांधें; मानसून से पहले बचाव के लिए 1% बोर्डो मिश्रण का छिड़काव करें और केवीके की सलाह मानें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Chemical):**
  - *English:* `"Spray 1% Bordeaux Mixture with adhesive sticker on all nut bunches; follow product label and local KVK advice."`
  - *Kannada:* `"ಎಲ್ಲಾ ಅಡಿಕೆ ಗೊನೆಗಳ ಮೇಲೆ ಅಂಟು ದ್ರವದೊಂದಿಗೆ ಶೇಕಡಾ 1 ರ ಬೋರ್ಡೋ ಮಿಶ್ರಣವನ್ನು ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"सभी गुच्छों पर चिपकाने वाले पदार्थ के साथ 1% बोर्डो मिश्रण का छिड़काव करें; लेबल और केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)

### 3. Bacterial Leaf Blight (`bacterial_leaf_blight`)
- **Grade 1 (Organic):**
  - *English:* `"Spray neem seed kernel extract. Avoid excess nitrogen top-dressing; follow product label and local KVK advice."`
  - *Kannada:* `"ಬೇವಿನ ಬೀಜದ ಕಷಾಯ ಸಿಂಪಡಿಸಿ. ಹೆಚ್ಚಿನ ಸಾರಜನಕ ಗೊಬ್ಬರ ನೀಡುವುದನ್ನು ತಪ್ಪಿಸಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"नीम के बीज का काढ़ा छिड़कें। यूरिया का अधिक उपयोग न करें; स्थानीय केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Chemical):**
  - *English:* `"Apply Copper Oxychloride + Streptocycline; follow product label and local KVK advice."`
  - *Kannada:* `"ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಮತ್ತು ಸ್ಟ್ರೆಪ್ಟೊಸೈಕ್ಲಿನ್ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಸೂಚನೆ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"कॉपर ऑक्सीक्लोराइड और स्ट्रेप्टोसाइक्लिन का छिड़काव करें; लेबल और केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)

### 4. Brown Spot (`brown_spot`)
- **Grade 1 (Organic / Cultural):**
  - *English:* `"Apply compost and correct potassium deficiency; follow local KVK advice."`
  - *Kannada:* `"ಕಾಂಪೋಸ್ಟ್ ಗೊಬ್ಬರ ಹಾಕಿ ಮತ್ತು ಪೊಟ್ಯಾಷ್ ಕೊರತೆಯನ್ನು ಸರಿಪಡಿಸಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"कम्पोस्ट खाद डालें और पोटाश की कमी को दूर करें; स्थानीय केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Chemical):**
  - *English:* `"Apply Mancozeb protective spray; follow product label and local KVK advice."`
  - *Kannada:* `"ಮ್ಯಾಂಕೋಜೆಬ್ ರಕ್ಷಣಾತ್ಮಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ; ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"मैंकोजेब कवकनाशी का सुरक्षात्मक छिड़काव करें; लेबल और केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)

### 5. Dead Heart (`dead_heart`)
- **Grade 1 (Cultural / Mechanical):**
  - *English:* `"Clip seedling tips before transplanting; install yellow stem borer pheromone traps; follow local KVK advice."`
  - *Kannada:* `"ನಾಟಿ ಮಾಡುವ ಮುನ್ನ ಸಸಿಗಳ ತುದಿಯನ್ನು ಕತ್ತರಿಸಿ; ಹಳದಿ ಕಾಂಡ ಕೊರೆಯುವ ಕೀಟದ ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"रोपाई से पहले पौध की नोक काटें; तना छेदक के लिए फेरोमोन ट्रैप लगाएं; केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Biological / Chemical):**
  - *English:* `"Release Trichogramma japonicum egg parasitoids; install pheromone traps; follow local KVK advice."`
  - *Kannada:* `"ಟ್ರೈಕೋಗ್ರಾಮಾ ಜಪಾನಿಕಂ ಮೊಟ್ಟೆ ಪರಾವಲಂಬಿಗಳನ್ನು ಬಿಡಿ; ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"ट्राइकोग्रामा जैपोनिकम परजीवी कीट छोड़ें; फेरोमोन ट्रैप लगाएं; स्थानीय केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)

### 6. Hispa (`hispa`)
- **Grade 1 (Mechanical):**
  - *English:* `"Manually clip infected leaf tips containing grubs; use sweep nets; follow local KVK advice."`
  - *Kannada:* `"ಮರಿ ಕೀಟಗಳಿರುವ ಎಲೆಗಳ ತುದಿಯನ್ನು ಕೈಯಿಂದ ಕತ್ತರಿಸಿ ನಾಶಪಡಿಸಿ; ಜಾಲರಿ ಬಳಸಿ ಕೀಟ ಹಿಡಿಯಿರಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"कीट युक्त पत्तियों की नोक को काटकर नष्ट करें; कीट पकड़ने वाले जाल का उपयोग करें; स्थानीय केवीके की सलाह मानें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Mechanical + Chemical consultation):**
  - *English:* `"Manually clip affected leaf tips containing grubs; consult your KVK for a registered product; follow product label and local KVK advice."`
  - *Kannada:* `"ಕೀಟವಿರುವ ಎಲೆಗಳ ತುದಿಯನ್ನು ಕತ್ತರಿಸಿ; ನೋಂದಾಯಿತ ಔಷಧಿಗಾಗಿ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಲೇಬಲ್ ಸೂಚನೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"संक्रमित पत्तियों की नोक काटें; अनुमोदित कीटनाशक के लिए केवीके से संपर्क करें और लेबल का पालन करें।"`  # REVIEW (Hindi)

### 7. Stem Bleeding (`stem_bleeding`)
- **Grade 1 (Cultural / Mechanical):**
  - *English:* `"Chisel out bleeding bark patch down to healthy tissue; apply Bordeaux paste; consult your KVK for a registered product; follow local KVK advice."`
  - *Kannada:* `"ರಸ ಸೋರುವ ತೊಗಟೆಯನ್ನು ಆರೋಗ್ಯಕರ ಭಾಗದವರೆಗೆ ಕೆತ್ತಿ ತೆಗೆಯಿರಿ; ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಹಚ್ಚಿ; ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಲಹೆ ಪಾಲಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"तने से रस बहने वाले हिस्से को स्वस्थ लकड़ी तक छीलें; बोर्डो पेस्ट लगाएं; स्थानीय केवीके की सलाह का पालन करें।"`  # REVIEW (Hindi)
- **Grade $\ge 2$ (Chemical / Surgical):**
  - *English:* `"Chisel out infected bark; apply Bordeaux paste; consult your KVK for a registered product (e.g. Tridemorph); follow product label and local KVK advice."`
  - *Kannada:* `"ಸೋಂಕಿತ ತೊಗಟೆಯನ್ನು ಕೆತ್ತಿ ತೆಗೆದು ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಹಚ್ಚಿ; ಶಿಫಾರಸು ಮಾಡಿದ ಔಷಧಿಗಾಗಿ ನಿಮ್ಮ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."`  # REVIEW (Kannada)
  - *Hindi:* `"संक्रमित छाल को छीलें और बोर्डो पेस्ट लगाएं; अनुशंसित दवा के लिए अपने केवीके से संपर्क करें।"`  # REVIEW (Hindi)

---

## 9. 📱 All 71 UI Localized Labels & Grounded Chat Strings (`ui_theme.py`)

All 71 UI buttons, cards, statuses, form labels, and grounded chat quick-reply chips localized in Kannada (`kn`) and Hindi (`hi`):

| Key | English String | Kannada String (`kn`) | Hindi String (`hi`) |
| :--- | :--- | :--- | :--- |
| `app_name` | `AgriTrust AI` | `AgriTrust AI` # REVIEW | `AgriTrust AI` # REVIEW |
| `apple_public_banner` | `⚠️ Trained on public lab data, not field-validated in orchards.` | `⚠️ ಸಾರ್ವಜನಿಕ ಲ್ಯಾಬ್ ಡೇಟಾದಲ್ಲಿ ತರಬೇತಿ ಪಡೆದಿದೆ, ತೋಟಗಳಲ್ಲಿ ಕ್ಷೇತ್ರ-ಪರೀಕ್ಷೆ ಮಾಡಲಾಗಿಲ್ಲ.` # REVIEW | `⚠️ सार्वजनिक लैब डेटा पर प्रशिक्षित, बगीचों में फील्ड-परीक्षण नहीं किया गया।` # REVIEW |
| `areca_bud_borer_warning` | `⚠️ Low Reliability Notice: The model frequently confuses Bud Borer and Yellow Leaf Disease (YLD). Confirm with KVK / CPCRI before treatment.` | `⚠️ ಕಡಿಮೆ ವಿಶ್ವಾಸಾರ್ಹತೆಯ ಸೂಚನೆ: ಮಾದರಿಯು ಮೊಗ್ಗು ಕೊರೆಯುವ ಹುಳು ಮತ್ತು ಹಳದಿ ಎಲೆ ರೋಗವನ್ನು ಗೊಂದಲಗೊಳಿಸುತ್ತದೆ. ಚಿಕಿತ್ಸೆಗೆ ಮುನ್ನ ಕೆವಿಕೆ/ಸಿಪಿಸಿಆರ್‌ಐ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.` # REVIEW | `⚠️ कम विश्वसनीयता सूचना: मॉडल कली छेदक और पीली पत्ती रोग में भ्रमित होता है। उपचार से पहले केवीके से पुष्टि करें।` # REVIEW |
| `areca_provisional` | `Provisional: not independently validated.` | `ತಾತ್ಕಾಲಿಕ: ಸ್ವತಂತ್ರವಾಗಿ ಮೌಲ್ಯೀಕರಿಸಲಾಗಿಲ್ಲ.` # REVIEW | `अनंतिम: स्वतंत्र रूप से मान्य नहीं किया गया।` # REVIEW |
| `back_btn` | `← Back` | `← ಹಿಂದಕ್ಕೆ` # REVIEW | `← वापस` # REVIEW |
| `cedar_rust_caption` | `⚠️ Class from US public data; confirm locally with your KVK.` | `⚠️ ಯುಎಸ್ ಸಾರ್ವಜನಿಕ ಡೇಟಾದ ವರ್ಗ; ಸ್ಥಳೀಯವಾಗಿ ನಿಮ್ಮ ಕೆವಿಕೆ ಜೊತೆ ದೃಢೀಕರಿಸಿ.` # REVIEW | `⚠️ अमेरिकी सार्वजनिक डेटा से वर्ग; स्थानीय केवीके से पुष्टि करें।` # REVIEW |
| `chat_clear_btn` | `Clear Chat` | `ಸಂಭಾಷಣೆ ಅಳಿಸಿ` # REVIEW | `चैट साफ करें` # REVIEW |
| `chat_disclaimer` | `Answers come from the advice shown for your last scan. Verify with your KVK.` | `ಉತ್ತರಗಳು ನಿಮ್ಮ ಕೊನೆಯ ಸ್ಕ್ಯಾನ್‌ನಲ್ಲಿ ತೋರಿಸಲಾದ ಸಲಹೆಯಿಂದ ಬಂದಿವೆ. ಕೆವಿಕೆ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.` # REVIEW | `उत्तर आपके पिछले स्कैन में दिखाई गई सलाह पर आधारित हैं। अपने केवीके से पुष्टि करें।` # REVIEW |
| `chat_no_context` | `Scan a leaf first and I can explain the result.` | `ಮೊದಲು ಎಲೆಯ ಫೋಟೋ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ, ನಂತರ ನಾನು ಫಲಿತಾಂಶವನ್ನು ವಿವರಿಸಬಲ್ಲೆ.` # REVIEW | `पहले एक पत्ती स्कैन करें, फिर मैं परिणाम समझा सकता हूँ।` # REVIEW |
| `chat_placeholder` | `Ask a question about your scan...` | `ನಿಮ್ಮ ಸ್ಕ್ಯಾನ್ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ...` # REVIEW | `अपने स्कैन के बारे में सवाल पूछें...` # REVIEW |
| `chat_title` | `Grounded Agricultural Assistant` | `ಕೃಷಿ ಸಹಾಯಕ ಸಂಭಾಷಣೆ` # REVIEW | `कृषि सहायक चैट` # REVIEW |
| `chat_unmatched` | `I can only explain your last result. Please ask your KVK / agriculture officer.` | `ನಾನು ನಿಮ್ಮ ಕೊನೆಯ ಫಲಿತಾಂಶವನ್ನು ಮಾತ್ರ ವಿವರಿಸಬಲ್ಲೆ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.` # REVIEW | `मैं केवल आपके पिछले परिणाम की व्याख्या कर सकता हूँ। कृपया अपने नजदीकी केवीके या कृषि अधिकारी से संपर्क करें।` # REVIEW |
| `chip_cure` | `Is there a cure?` | `ಇದಕ್ಕೆ ಪರಿಹಾರವಿದೆಯೇ?` # REVIEW | `क्या इसका कोई इलाज है?` # REVIEW |
| `chip_first` | `What should I do first?` | `ಮೊದಲು ನಾನು ಏನು ಮಾಡಬೇಕು?` # REVIEW | `मुझे सबसे पहले क्या करना चाहिए?` # REVIEW |
| `chip_soil` | `Soil tips` | `ಮಣ್ಣಿನ ಸಲಹೆಗಳು` # REVIEW | `मिट्टी की सलाह` # REVIEW |
| `chip_spray` | `When should I spray?` | `ಔಷಧಿ ಯಾವಾಗ ಸಿಂಪಡಿಸಬೇಕು?` # REVIEW | `दवा कब छिड़कनी चाहिए?` # REVIEW |
| `chip_unsure` | `Why are you unsure?` | `ಖಚಿತವಾಗಿಲ್ಲದಿರಲು ಕಾರಣವೇನು?` # REVIEW | `आप निश्चित क्यों नहीं हैं?` # REVIEW |
| `chip_weather` | `Weather risk` | `ಹವಾಮಾನ ಅಪಾಯ` # REVIEW | `मौसम का जोखिम` # REVIEW |
| `choose_crop` | `Choose Crop` | `ಬೆಳೆ ಆಯ್ಕೆಮಾಡಿ` # REVIEW | `फसल चुनें` # REVIEW |
| `confidence_label` | `Confidence` | `ವಿಶ್ವಾಸಾರ್ಹತೆ` # REVIEW | `सटीकता` # REVIEW |
| `crop_apple` | `Apple` | `ಸೇಬು` # REVIEW | `सेब` # REVIEW |
| `crop_arecanut` | `Arecanut` | `ಅಡಿಕೆ` # REVIEW | `सुपारी` # REVIEW |
| `crop_label` | `Crop` | `ಬೆಳೆ` # REVIEW | `फसल` # REVIEW |
| `crop_paddy` | `Paddy` | `ಭತ್ತ` # REVIEW | `धान` # REVIEW |
| `crop_weather_risk` | `Crop Weather Risk Analysis` | `ಬೆಳೆ ಹವಾಮಾನ ಅಪಾಯದ ವಿಶ್ಲೇಷಣೆ` # REVIEW | `फसल मौसम जोखिम विश्लेषण` # REVIEW |
| `date_label` | `Date` | `ದಿನಾಂಕ` # REVIEW | `दिनांक` # REVIEW |
| `district_label` | `District` | `ಜಿಲ್ಲೆ` # REVIEW | `ज़िला` # REVIEW |
| `experts_expander` | `For experts and judges` | `ತಜ್ಞರು ಮತ್ತು ತೀರ್ಪುಗಾರರಿಗೆ` # REVIEW | `विशेषज्ञों और परीक्षकों के लिए` # REVIEW |
| `footer_disclaimer_soil` | `⚠️ Verify with your KVK / soil health card.` | `⚠️ ನಿಮ್ಮ ಕೆವಿಕೆ / ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಜೊತೆ ಪರಿಶೀಲಿಸಿ.` # REVIEW | `⚠️ अपने केवीके / मृदा स्वास्थ्य कार्ड से जांच करें।` # REVIEW |
| `footer_disclaimer_tx` | `⚠️ Verify against ICAR / CPCRI / CIB&RC before use.` | `⚠️ ಬಳಸುವ ಮುನ್ನ ICAR / CPCRI / CIB&RC ಜೊತೆ ಪರಿಶೀಲಿಸಿ.` # REVIEW | `⚠️ उपयोग से पहले ICAR / CPCRI / CIB&RC से पुष्टि करें।` # REVIEW |
| `greeting` | `Hey, welcome farmer` | `ನಮಸ್ಕಾರ, ರೈತ ಮಿತ್ರರಿಗೆ ಸುಸ್ವಾಗತ` # REVIEW | `नमस्ते, किसान भाई आपका स्वागत है` # REVIEW |
| `humidity_label` | `Humidity` | `ಆರ್ದ್ರತೆ` # REVIEW | `नमी` # REVIEW |
| `last_result_title` | `Your Last Result` | `ನಿಮ್ಮ ಹಿಂದಿನ ಫಲಿತಾಂಶ` # REVIEW | `आपका पिछला परिणाम` # REVIEW |
| `listen_audio` | `Voice Advisory` | `ಧ್ವನಿ ಸಲಹೆ` # REVIEW | `ध्वनि सलाह` # REVIEW |
| `live_forecast_badge` | `Live Forecast` | `ನೇರ ಮುನ್ಸೂಚನೆ` # REVIEW | `लाइव पूर्वानुमान` # REVIEW |
| `no_rule_for_crop` | `ℹ️ No weather rule active for this condition in the next 48 hours.` | `ℹ️ ಮುಂದಿನ 48 ಗಂಟೆಗಳಲ್ಲಿ ಈ ಸ್ಥಿತಿಗೆ ಯಾವುದೇ ಹವಾಮಾನ ನಿಯಮ ಅನ್ವಯಿಸುವುದಿಲ್ಲ.` # REVIEW | `ℹ️ अगले 48 घंटों में इस रोग के लिए कोई मौसम नियम सक्रिय नहीं है।` # REVIEW |
| `no_scans_yet` | `No scans yet.` | `ಇನ್ನೂ ಯಾವುದೇ ಸ್ಕ್ಯಾನ್ ಮಾಡಿಲ್ಲ.` # REVIEW | `अभी तक कोई स्कैन नहीं हुआ।` # REVIEW |
| `offline_badge` | `Offline` | `ಆಫ್‌ಲೈನ್` # REVIEW | `ऑफलाइन` # REVIEW |
| `organic_title` | `Organic & Natural Steps` | `ಸಾವಯವ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕ್ರಮಗಳು` # REVIEW | `जैविक और प्राकृतिक उपाय` # REVIEW |
| `paddy_severity_fixed` | `Severity is assigned by a fixed rule for this class.` | `ಈ ರೋಗಕ್ಕೆ ಸ್ಥಿರ ನಿಯಮದ ಮೂಲಕ ತೀವ್ರತೆಯನ್ನು ನಿಗದಿಪಡಿಸಲಾಗಿದೆ.` # REVIEW | `इस वर्ग के लिए गंभीरता एक निश्चित नियम द्वारा निर्धारित की गई है।` # REVIEW |
| `paddy_severity_qwk` | `Independent-test QWK about 0.67 (twin-free, lesion classes); headline 0.94 is inflated.` | `ಸ್ವತಂತ್ರ ಪರೀಕ್ಷಾ QWK ಸುಮಾರು 0.67 (ಅವಳಿ-ಮುಕ್ತ, ಗಾಯದ ತರಗತಿಗಳು); ಮುಖಪುಟದ 0.94 ಉಬ್ಬಿಸಲಾಗಿದೆ.` # REVIEW | `स्वतंत्र परीक्षण QWK लगभग 0.67 (ट्विन-मुक्त, घाव वर्ग); मुख्य 0.94 बढ़ा-चढ़ाकर है।` # REVIEW |
| `rain_48h_label` | `Rain in 48h` | `48 ಗಂ ಮಳೆ` # REVIEW | `48 घंटे में बारिश` # REVIEW |
| `recommendation_label` | `Recommendation` | `ಶಿಫಾರಸು` # REVIEW | `अनुशंसा` # REVIEW |
| `refusal_abstain_msg` | `Not sure about this photo. Retake it in daylight or consult your KVK / agriculture officer.` | `ಈ ಫೋಟೋದ ಬಗ್ಗೆ ಖಚಿತವಿಲ್ಲ. ಹಗಲಿನಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ಮರು-ಫೋಟೋ ತೆಗೆಯಿರಿ ಅಥವಾ ನಿಮ್ಮ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.` # REVIEW | `इस फोटो पर निश्चित नहीं हैं। दिन के उजाले में दोबारा फोटो लें या अपने कृषि अधिकारी से सलाह लें।` # REVIEW |
| `refusal_abstain_title` | `Diagnosis Withheld for Safety` | `ಸುರಕ್ಷತೆಗಾಗಿ ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ` # REVIEW | `सुरक्षा कारणों से निदान रोक दिया गया` # REVIEW |
| `refusal_ood_msg` | `This does not look like a close-up photo of the selected crop. Retake a well-lit close-up of the affected leaf, trunk or nut.` | `ಇದು ಆಯ್ಕೆಮಾಡಿದ ಬೆಳೆಯ ಹತ್ತಿರದ ಫೋಟೋದಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಬಾಧಿತ ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಅಡಿಕೆಯ ಹತ್ತಿರದ ಫೋಟೋವನ್ನು ಬೆಳಕಿನಲ್ಲಿ ತೆಗೆಯಿರಿ.` # REVIEW | `यह चुनी गई फसल की करीबी तस्वीर नहीं लगती। कृपया प्रभावित पत्ती या फल की साफ फोटो लें।` # REVIEW |
| `refusal_ood_title` | `Image Outside Field Scope` | `ಚಿತ್ರವು ನಿಗದಿತ ಬೆಳೆಯ ವ್ಯಾಪ್ತಿಯಲ್ಲಿಲ್ಲ` # REVIEW | `तस्वीर फसल के दायरे से बाहर है` # REVIEW |
| `rule_based_tag` | `rule-based estimate, not a forecast` | `ನಿಯಮ-ಆಧಾರಿತ ಅಂದಾಜು, ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಯಲ್ಲ` # REVIEW | `नियम-आधारित अनुमान, मौसम पूर्वानुमान नहीं` # REVIEW |
| `scan_card_title` | `Scan / Upload Photo` | `ಫೋಟೋ ಸ್ಕ್ಯಾನ್ / ಅಪ್‌ಲೋಡ್ ಮಾಡಿ` # REVIEW | `फोटो स्कैन / अपलोड करें` # REVIEW |
| `severe_expander` | `If the problem is severe (Chemical Treatment)` | `ಸಮಸ್ಯೆ ಗಂಭೀರವಾಗಿದ್ದರೆ (ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ)` # REVIEW | `यदि समस्या गंभीर हो (रासायनिक उपचार)` # REVIEW |
| `soil_card_title` | `Soil Health Card` | `ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್` # REVIEW | `मृदा स्वास्थ्य कार्ड` # REVIEW |
| `status_healthy` | `Healthy` | `ಆರೋಗ್ಯಕರ` # REVIEW | `स्वस्थ` # REVIEW |
| `status_no_cure` | `No cure / See expert` | `ಪರಿಹಾರವಿಲ್ಲ / ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ` # REVIEW | `इलाज नहीं / विशेषज्ञ से मिलें` # REVIEW |
| `status_treat` | `Treatable` | `ಚಿಕಿತ್ಸೆ ಸಾಧ್ಯ` # REVIEW | `उपचार संभव` # REVIEW |
| `status_unsure` | `Not sure` | `ಖಚಿತವಿಲ್ಲ` # REVIEW | `संदेहास्पद` # REVIEW |
| `tab_chat` | `AI Chat` | `ಎಐ ಸಂಭಾಷಣೆ` # REVIEW | `एआई चैट` # REVIEW |
| `tab_home` | `Home` | `ಮುಖಪುಟ` # REVIEW | `होम` # REVIEW |
| `tab_weather` | `Weather` | `ಹವಾಮಾನ` # REVIEW | `मौसम` # REVIEW |
| `take_photo` | `Take photo with camera` | `ಕ್ಯಾಮರಾ ಬಳಸಿ ಫೋಟೋ ತೆಗೆಯಿರಿ` # REVIEW | `कैमरे से फोटो लें` # REVIEW |
| `temp_label` | `Temp` | `ತಾಪಮಾನ` # REVIEW | `तापमान` # REVIEW |
| `tip_line` | `Close-up, daylight, one leaf or one nut` | `ಹತ್ತಿರದಿಂದ, ಬೆಳಕಿನಲ್ಲಿ, ಒಂದು ಎಲೆ ಅಥವಾ ಅಡಿಕೆ` # REVIEW | `पास से, दिन के उजाले में, एक पत्ती या एक फल` # REVIEW |
| `trust_audit_title` | `Trust and Audit Benchmarks` | `ವಿಶ್ವಾಸಾರ್ಹತೆ ಮತ್ತು ಆಡಿಟ್ ಅಂಕಿಅಂಶಗಳು` # REVIEW | `विश्वसनीयता एवं ऑडिट रिपोर्ट` # REVIEW |
| `tungro_warning_title` | `Tungro Warning` | `ತುಂಗ್ರೋ ಎಚ್ಚರಿಕೆ` # REVIEW | `तुंग्रो चेतावनी` # REVIEW |
| `upload_file` | `Or upload photo from device` | `ಅಥವಾ ಮೊಬೈಲ್‌ನಿಂದ ಫೋಟೋ ಆಯ್ಕೆಮಾಡಿ` # REVIEW | `या डिवाइस से फोटो चुनें` # REVIEW |
| `weather_6h_blocks` | `48-Hour Forecast Summary (6-Hour Blocks)` | `48 ಗಂಟೆಗಳ ಹವಾಮಾನ ಸಾರಾಂಶ (6 ಗಂಟೆಗಳ ವಿವರ)` # REVIEW | `48 घंटे का मौसम सारांश (6 घंटे के अंतराल पर)` # REVIEW |
| `weather_alert_title` | `48-Hour Preventive Alert` | `48 ಗಂಟೆಗಳ ಮುಂಜಾಗ್ರತಾ ಎಚ್ಚರಿಕೆ` # REVIEW | `48 घंटे की निवारक चेतावनी` # REVIEW |
| `weather_status_dry` | `Dry` | `ಶುಷ್ಕ` # REVIEW | `शुष्क` # REVIEW |
| `weather_status_normal` | `Normal` | `ಸಾಮಾನ್ಯ` # REVIEW | `सामान्य` # REVIEW |
| `weather_status_rain` | `Rain Expected` | `ಮಳೆಯ ಸಾಧ್ಯತೆ` # REVIEW | `बारिश की संभावना` # REVIEW |
| `weather_view_title` | `48-Hour Live Weather & Disease Pressure` | `48 ಗಂಟೆಗಳ ನೇರ ಹವಾಮಾನ ಮತ್ತು ರೋಗದ ಒತ್ತಡ` # REVIEW | `48 घंटे का मौसम और रोग का दबाव` # REVIEW |
| `what_to_do` | `What to do` | `ಮಾಡಬೇಕಾದ ಕ್ರಮಗಳು` # REVIEW | `क्या करें` # REVIEW |

---

## 10. ❓ Grounded Chat: "Why are you unsure?" Localized Responses

Responses explaining confidence, the 70% decline safety threshold, and visual symptom overlap with Tungro virus:

### Normal Diagnosis (Confidence >= 70%)

- **English:** `"The model diagnosed this image with {conf:.1%} confidence. The app declines to diagnose below 70%. For Hispa and Downy Mildew, Tungro can look similar."`
- **Kannada (`kn`):** `"ಮಾದರಿಯು ಈ ಚಿತ್ರವನ್ನು {conf:.1%} ವಿಶ್ವಾಸಾರ್ಹತೆಯೊಂದಿಗೆ ಗುರುತಿಸಿದೆ. ವಿಶ್ವಾಸಾರ್ಹತೆಯು 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ. ಹಿಸ್ಪಾ ಮತ್ತು ಬೂದಿ ರೋಗಕ್ಕೆ, ತುಂಗ್ರೋ ವೈರಸ್ ಕೂಡ ಇದೇ ರೀತಿ ಕಾಣಿಸಬಹುದು."` # REVIEW
- **Hindi (`hi`):** `"मॉडल ने इस तस्वीर को {conf:.1%} सटीकता के साथ पहचाना है। सटीकता 70% से कम होने पर ऐप निदान करने से मना करता है। हिस्पा और डाउनी फफूंदी के लिए, तुंग्रो वायरस के लक्षण भी इसी तरह दिख सकते हैं।"` # REVIEW

### Abstained Diagnosis (Confidence < 70%)

- **English:** `"Diagnosis was withheld because confidence ({conf:.1%}) fell below 70%. The app declines to diagnose below 70%. For Hispa and Downy Mildew, Tungro virus can look similar in the field."`
- **Kannada (`kn`):** `"ಮಾದರಿಯ ವಿಶ್ವಾಸಾರ್ಹತೆ ({conf:.1%}) 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾಗಿದೆ. ವಿಶ್ವಾಸಾರ್ಹತೆಯು 70% ಕ್ಕಿಂತ ಕಡಿಮೆಯಾದಾಗ ಆ್ಯಪ್ ರೋಗನಿರ್ಣಯವನ್ನು ತಿರಸ್ಕರಿಸುತ್ತದೆ. ಹಿಸ್ಪಾ ಮತ್ತು ಬೂದಿ ರೋಗಕ್ಕೆ, ತುಂಗ್ರೋ ವೈರಸ್ ಕೂಡ ಇದೇ ರೀತಿ ಕಾಣಿಸಬಹುದು."` # REVIEW
- **Hindi (`hi`):** `"मॉडल की सटीकता ({conf:.1%}) 70% से कम रही। सटीकता 70% से कम होने पर ऐप निदान करने से मना करता है। हिस्पा और डाउनी फफूंदी के लिए, तुंग्रो वायरस भी इसी तरह का दिख सकता है।"` # REVIEW

