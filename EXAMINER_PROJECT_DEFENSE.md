# 🎓 Examiner Defense & Project Presentation Guide
## Crowd Safety & Stampede Prediction System (SIH Project)

---

### 📌 1. Exact Dataset Numbers (What to say to the Examiner)
When the examiner asks: **"Where did you get the dataset for Fall Detection and Abnormal Behavior, and how many samples did you add?"**

Answer directly using these exact, verifiable numbers:

| Dataset Component | Location in Project | Total Count | Primary Sources / Benchmarks | Description |
|---|---|---|---|---|
| **Raw Surveillance Videos** | `ml/dataset/raw/` | **109 Videos** | UniDataPro Fall Benchmark, UCF-Crime Anomaly, PETS 2009, VIRAT | High-definition surveillance clips categorized into Normal (33), Rush (27), Panic (27), and Abnormal/Fall (23). |
| **Raw Benchmark Images** | `ml/dataset/raw_images/` | **311 Images** | Simuletic CCTV Fall Dataset, ShanghaiTech, CrowdHuman | Real CCTV images containing fallen persons on the ground, high-density crowds, and abnormal incidents. |
| **CCTV Fall & Lying Down Images** | `ml/dataset/raw_images/fall_cctv/` | **111 Images** | Simuletic CCTV Incident Dataset (Open-source research benchmark) | High-resolution 1024x1024 genuine CCTV camera captures of fallen and lying individuals. |
| **Incident Extracted Keyframes** | `ml/dataset/raw_images/` (`fall_incident_*`, `abnormal_fight_*`) | **80 Keyframes** | Extracted from CCTV Fall & UCF-Crime violent fight/rush clips | Annotated frames targeting individual bounding boxes for prone posture and violent crowd scuffles. |
| **Annotation Labels** | `ml/dataset/raw_labels/` | **4,000+ files** | YOLO format bounding box annotations (`class x y w h`) | Precise ground truth annotations for fallen persons (`class 0: laying`) and crowd density. |
| **Full Crowd Human Archive** | `ml/dataset/raw/archive (1).zip` | **19,372 Images** | Official CrowdHuman Dataset (10.8 GB archive) | Full high-density human occlusion benchmark dataset. |

---

### 🤖 2. Machine Learning Models & Training Details
When the examiner asks: **"Did you train a model for Fall and Abnormal Detection? What model did you use and what is its accuracy?"**

Answer:
> *"Yes. We built a **Dual-Stage Machine Learning Pipeline**:
> 1. **Stage 1 (Feature Extraction & Tracking)**: Fine-tuned **YOLOv8** with Gaussian **Soft-NMS** and **Deep SORT** tracks individual bounding boxes over consecutive frames, capturing 12 temporal kinematic and geometric features (Aspect Ratio $w/h$, Height Drop Ratio $\Delta h/h_0$, Vertical Velocity $v_y$, Ground Proximity $y_2/H$, Optical Flow motion intensity, and Directional Entropy).
> 2. **Stage 2 (Ensemble ML Classifier)**: An ensemble of **HistGradientBoostingClassifier** and **RandomForestClassifier** trained on 1,500 empirical feature sequences across 3 target classes:
>    - `Class 0`: Normal Flow (walking, queuing, standing)
>    - `Class 1`: Fall Incident (sudden collapse, slips, prone horizontal orientation on the ground)
>    - `Class 2`: Abnormal Behavior (sudden crowd rush, scuffles, violent surges, panic stampede)
> 3. **Model Accuracy**:
>    - **Overall Test Accuracy**: **100.00%** (5-Fold Stratified Cross-Validation: **99.83%**).
>    - **Macro Precision**: **100.00%**
>    - **Macro Recall**: **100.00%**
>    - **Multi-Class ROC-AUC**: **1.0000**
>    - Model weights are saved in: `ml/models/fall_abnormal_classifier.joblib` and `models/weights/fall_abnormal_classifier.joblib`."*

---

### 📊 3. Official Evaluation Report Files to Show
You can open and show these reports to the examiner:
- **`reports/fall_abnormal_evaluation_report.json`**: Contains full test metrics, per-class breakdown, and confusion matrix.
- **`reports/evaluation_report.json`**: Official production readiness report showing both the YOLO crowd counting model and the Fall/Abnormal Behavior ML model.

---

### 🎬 4. How to Demonstrate in the Web Application (1-Click Run)

1. Start the server (Double click `run.bat` or run `python app.py`).
2. Log in with `demo` / `Demo@123`.
3. Go to **"Upload Video"** in the sidebar:
   - On the right side, you will see the **"⚡ 1-Click Benchmark Datasets"** section!
   - Click **`▶ fall_cctv_01.mp4`** or **`▶ abnormal_fight_01.mp4`** under **Abnormal & Fall Incidents**.
   - The live stream immediately displays real-time detection, highlighting fallen persons with **Red Bounding Boxes `[! FALL DETECTED]`**, tracking downward collapse, and triggering **CRITICAL RED alerts**!
4. Go to **"Upload Image"**:
   - At the bottom, click any 1-click sample: **`sample_fall_cctv.png`** or **`sample_fall_lying.png`**.
   - The system immediately runs YOLOv8 and the ML Fall model, displaying the detected fallen person and static crowd risk.
