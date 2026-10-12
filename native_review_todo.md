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
- **Kannada:* `"ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ: ಮುಂದಿನ ನಲವತ್ತೆಂಟು ಗಂಟೆಗಳಲ್ಲಿ ಹವಾಮಾನದಿಂದ ರೋಗದ ಅಪಾಯ ಕಡಿಮೆ ಇದೆ."`  # REVIEW
- **Hindi:* `"मौसम चेतावनी: अगले अड़तालीस घंटों में मौसम से रोग का जोखिम कम है।"`  # REVIEW


