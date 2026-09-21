# Real-Time Crowd Density Analysis and Stampede Risk Detection System Using Machine Learning

[![Smart India Hackathon](https://img.shields.io/badge/Hackathon-Smart%20India%20Hackathon-blue.svg)](https://sih.gov.in)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-brightgreen.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Deep SORT](https://img.shields.io/badge/Tracking-Deep%20SORT-purple.svg)]()
[![Flask](https://img.shields.io/badge/Backend-Flask%203.1-black.svg)](https://palletsprojects.com/p/flask/)

An intelligent, multi-stage computer vision surveillance and crowd safety monitoring application developed for the **Smart India Hackathon (SIH)**. The system analyzes video streams from uploaded files, laptop webcams, or live RTSP IP cameras to identify abnormal crowd behaviors, sudden rushes, panic stampede patterns, and fallen persons before catastrophic incidents occur.

---

## 🎯 1. Project Aim & Problem Statement

Public gatherings, religious festivals, transit hubs, and sports stadiums frequently experience dangerous crowd surges, localized stampedes, and crushing incidents. Early intervention is critical to save lives.

### Core Objectives:
- Detect people count and crowd density in real-time.
- Preserve overlapping detections in dense crowds using **Soft-NMS**.
- Track individuals across frames with persistent IDs using **Deep SORT**.
- Measure crowd movement and disorder using **Optical Flow** and **Motion Entropy**.
- Detect anomalies such as **sudden rushes, panic movement, and fallen persons**.
- Fuse multi-modal signals through an intelligent **Risk Engine** into actionable threat levels:
  - 🟢 **GREEN (NORMAL)**: Orderly crowd flow.
  - 🟡 **YELLOW (WARNING)**: Sudden rush surge, abnormal movement, elevated density.
  - 🔴 **RED (CRITICAL)**: Chaotic stampede movement or fallen person with trample hazard.
- Issue cooldown-throttled alerts and log time-series telemetry to SQLite database.

---

## 📐 2. End-to-End AI/ML Pipeline Architecture

```
                      INPUT STREAM
  [ Uploaded Video | Uploaded Image | Webcam | CCTV / IP Camera ]
                           │
                           ▼
                  FRAME PREPROCESSING
              (Resize 640x480, Normalization)
                           │
                           ▼
                YOLOv8 PERSON DETECTION
              (Ultralytics Nano, Class 0)
                           │
                           ▼
                       SOFT-NMS
      (Gaussian/Linear IoU Confidence Decay)
                           │
                           ▼
                   DEEP SORT TRACKING
         (Kalman Filter + Hungarian Algorithm)
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   CROWD DENSITY                     MOTION ANALYSIS
 (Normalized Score)           ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                        Optical Flow   Crowd Speed     Entropy
                        (Farneback)    (Trajectory)  (Directional)
                              └─────────────┬─────────────┘
                                            │
                                            ▼
                                  TEMPORAL PANIC & RUSH
                                            │
          ┌─────────────────────────────────┴───────────────────┐
          ▼                                                     ▼
   FALL DETECTION                                        RUSH / PANIC
(Aspect Ratio Change)                                (Temporal Window)
          └─────────────────────────────────┬───────────────────┘
                                            │
                                            ▼
                                   RISK ENGINE & FUSION
                     (Weighted Formula + Emergency Overrides)
                                            │
                           ┌────────────────┼────────────────┐
                           ▼                ▼                ▼
                         GREEN            YELLOW            RED
                        (Normal)        (Warning)       (Critical)
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                                 ALERT & COOLDOWN ENGINE
                                            │
                                            ▼
                            SURVEILLANCE DASHBOARD & DB LOGS
```

---

## 🔬 3. Mathematical & Algorithmic Formulations

### A. YOLOv8 Person Detection
Infers bounding box coordinates $[x_1, y_1, x_2, y_2]$, confidence score $s_i$, and class index (filtering COCO Class 0 = `person`).

### B. Soft-NMS (Gaussian & Linear)
Traditional Hard-NMS eliminates all overlapping bounding boxes whose $\text{IoU} \ge N_t$, creating severe detection blindspots in dense crowds. Soft-NMS decays confidence scores as a continuous function of overlap:

$$\text{Gaussian Decay: } s_i = s_i \cdot \exp\left(-\frac{\text{IoU}(M, b_i)^2}{\sigma}\right), \quad \forall b_i \notin D$$

$$\text{Linear Decay: } s_i = \begin{cases} s_i, & \text{IoU}(M, b_i) < N_t \\ s_i(1 - \text{IoU}(M, b_i)), & \text{IoU}(M, b_i) \ge N_t \end{cases}$$

### C. Deep SORT Tracking
Maintains an 8-dimensional state vector for each tracked person:

$$\mathbf{x} = [x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]^T$$

where $(x_c, y_c)$ is the bounding box center, $a = w/h$ is aspect ratio, $h$ is height, and derivatives represent respective velocities. Association is performed via the Hungarian Algorithm on IoU cost matrices.

### D. Farneback Optical Flow & Motion Entropy
Dense motion vectors $(u, v)$ are computed on consecutive grayscale frames. Direction angles $\theta = \arctan2(v, u) \in [0, 360^\circ)$ are quantized into $N=8$ angular bins. Directional Shannon Entropy measures disorder:

$$H = -\sum_{i=1}^{N} p_i \log_2(p_i), \quad H_{\text{norm}} = \frac{H}{\log_2(N)} \in [0.0, 1.0]$$

- **Normal unidirectional flow**: $H_{\text{norm}} < 0.35$ (Orderly).
- **Chaotic stampede scattering**: $H_{\text{norm}} > 0.65$ (High Disorder).

### E. Fall Detection
Evaluates rapid transition in person bounding box geometry:
1. Aspect ratio shift from upright ($a < 0.6$) to horizontal ($a > 1.15$).
2. Downward vertical velocity spike ($\dot{y}_c > 18\text{ px/frame}$) with subsequent immobility.

### F. Multi-Factor Risk Assessment Engine
Computes normalized composite risk score $[0.0, 1.0]$:

$$\text{Risk} = 0.20 \cdot D + 0.20 \cdot S + 0.20 \cdot M + 0.15 \cdot E + 0.10 \cdot P + 0.15 \cdot F$$

- $D$: Normalized Crowd Density ($[0.0, 1.0]$)
- $S$: Normalized Relative Speed ($[0.0, 1.0]$)
- $M$: Motion Intensity Score ($[0.0, 1.0]$)
- $E$: Directional Motion Entropy ($[0.0, 1.0]$)
- $P$: Temporal Panic Score ($[0.0, 1.0]$)
- $F$: Fall Anomaly Score ($[0.0, 1.0]$)

**Classification Boundaries:**
- $0.00 \le \text{Risk} \le 0.39$ ➔ **GREEN (NORMAL)**
- $0.40 \le \text{Risk} \le 0.69$ ➔ **YELLOW (WARNING)**
- $0.70 \le \text{Risk} \le 1.00$ ➔ **RED (CRITICAL)**

---

## 📁 4. Video Directory Organization (Datasets & Testing)

Place your video files in the following folders for automatic 1-click execution in the UI:

| Folder | Intended Footage Content | Expected Risk Level |
|---|---|---|
| `videos/normal/` | Calm, orderly crowd movement, queues, steady walking. | 🟢 GREEN |
| `videos/rush/` | Sudden rush surges, rapid acceleration towards exits. | 🟡 YELLOW |
| `videos/panic/` | Multi-directional chaotic stampedes, running, high disorder. | 🔴 RED |
| `videos/abnormal/` | Fallen persons, fainting, stumbling, fights. | 🟡 / 🔴 |

---

## 🚀 5. Quick Start (Windows)

### 1-Click Setup & Launch
1. **Setup:** Double-click `setup.bat` (creates venv, installs requirements, sets up database, and seeds demo account).
2. **Run:** Double-click `run.bat` (starts Flask server on `http://127.0.0.1:5000`).

### Manual Setup
```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Application
python app.py
```

### 🔑 Demo Account Credentials
- **Username / Email**: `demo` (or `demo@example.com`)
- **Password**: `Demo@123`

---

## 🧠 6. Machine Learning Pipeline & Training

### Zero Data Leakage Split
```powershell
# Extract frames from raw videos with sampling skip
python ml/preprocessing/video_to_frames.py --skip 5

# Validate label syntax and image readability
python ml/preprocessing/clean_data.py

# Video-level dataset split (70% Train / 20% Val / 10% Test)
python ml/preprocessing/split_dataset.py

# Augment training set
python ml/preprocessing/augmentation.py

# Fine-tune YOLOv8 on crowd dataset
python ml/training/train_yolo.py --epochs 30 --batch 4

# Authentic evaluation report
python ml/training/evaluate.py
```

---

## 🧪 7. Running Unit Tests

Execute the comprehensive Pytest verification suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 📋 8. Key Limitations & Considerations

1. **Camera Homography**: Physical speeds (km/h) and exact densities (people/m²) require scene-specific camera perspective calibration. Normalized units $[0.0, 1.0]$ are utilized without homography.
2. **Camera Angles**: Top-down or high angled views yield highest accuracy by minimizing occlusion.
3. **Low-Light / Night Surveillance**: Dense darkness requires infrared cameras or fine-tuning on nighttime datasets.
4. **Temporal Stability**: Panic and rush detection require observation over a 15-frame sliding window to prevent false triggers from momentary motion spikes.
