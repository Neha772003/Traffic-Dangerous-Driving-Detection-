# 🚦 TRAFFICSENSE AI

## Intelligent Traffic Dangerous-Driving Detection System

```{=html}
<p align="center">
```
`<b>`{=html}AI-powered traffic monitoring, vehicle tracking, motion
analysis, and dangerous-driving detection`</b>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/YOLO-Object%20Detection-green" alt="YOLO">`{=html}
`<img src="https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-orange" alt="ByteTrack">`{=html}
`<img src="https://img.shields.io/badge/Random%20Forest-Classification-red" alt="Random Forest">`{=html}
`<img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit" alt="Streamlit">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Project Overview

**TRAFFICSENSE AI** is an AI-based traffic monitoring and
dangerous-driving detection system designed to analyze traffic videos.

The system detects vehicles using **YOLO**, tracks vehicles with
**ByteTrack**, extracts vehicle motion and trajectory features, and uses
a **Random Forest classifier** to classify driving behavior as **SAFE**
or **DANGEROUS**.

The processed video displays vehicle-level intelligence directly on the
video, including:

-   Vehicle ID
-   Vehicle type
-   Estimated speed
-   Risk score
-   SAFE / DANGEROUS status
-   Sudden braking alerts
-   Wrong-way movement alerts

The project also provides a professional **Streamlit dashboard** for
vehicle analysis, traffic analytics, event monitoring, traffic-flow
visualization, and report generation.

> **Note:** The current Streamlit implementation uses a YOLO detector,
> ByteTrack tracking, StandardScaler preprocessing, Random Forest
> classification, and H.264 video rendering.
> fileciteturn28file3L374-L391

------------------------------------------------------------------------

## 🎯 Problem Statement

Traditional traffic monitoring systems often require continuous human
observation. This can make it difficult to identify risky driving
behavior quickly and consistently.

TRAFFICSENSE AI aims to automate traffic-video analysis by combining:

**Vehicle Detection → Vehicle Tracking → Motion Analysis → ML
Classification → Risk Visualization**

------------------------------------------------------------------------

## 🎯 Project Objectives

-   Detect vehicles from traffic surveillance videos.
-   Assign a unique tracking ID to each detected vehicle.
-   Track vehicle movement across video frames.
-   Extract motion and trajectory features.
-   Classify vehicle behavior using Random Forest.
-   Calculate a risk score from the model's dangerous-driving
    probability.
-   Highlight SAFE vehicles with **green** bounding boxes.
-   Highlight DANGEROUS vehicles with **red** bounding boxes.
-   Detect significant sudden braking using multi-frame evidence.
-   Detect sustained opposite-direction movement conservatively.
-   Provide traffic analytics through Streamlit.
-   Generate downloadable vehicle telemetry and summary reports.

------------------------------------------------------------------------

## ✨ Key Features

### 🚗 Vehicle Detection

Detects vehicles in traffic video using YOLO.

### 🆔 Vehicle Tracking

ByteTrack assigns unique IDs and follows vehicles across frames.

### 🧠 Dangerous-Driving Classification

A trained Random Forest model classifies vehicle behavior into:

-   🟢 **SAFE**
-   🔴 **DANGEROUS**

### 📊 Risk Score

Each vehicle receives a risk score from **0--100** based on the Random
Forest dangerous-driving probability.

### ⚡ Sudden Braking Detection

Sudden braking is reported only when a significant speed reduction is
supported by multiple frames. Slow or stationary vehicles are not
automatically treated as sudden-braking events.

### ⛔ Wrong-Way Detection

Wrong-way movement is detected conservatively using sustained movement
opposite to the dominant traffic-flow direction.

### 🎥 Annotated Video

The output video displays:

``` text
Vehicle ID
Vehicle Type
Status
Estimated Speed
Risk Score
Event Alerts
```

### 📈 Traffic Analytics

The Streamlit dashboard provides:

-   Vehicle statistics
-   Risk distribution
-   Speed analysis
-   Feature analysis
-   Traffic-flow trajectories
-   Event records

### 📑 Reports

The application can export:

-   Vehicle telemetry CSV
-   Processed MP4 video
-   Text summary report

The application includes dedicated download controls for these outputs.
fileciteturn28file2L237-L304

------------------------------------------------------------------------

# 🔄 How the System Works

``` text
Traffic Video
      │
      ▼
YOLO Vehicle Detection
      │
      ▼
ByteTrack Vehicle Tracking
      │
      ▼
Unique Vehicle IDs
      │
      ▼
Vehicle Trajectory Extraction
      │
      ▼
7 Motion Features
      │
      ▼
StandardScaler
      │
      ▼
Random Forest Classifier
      │
      ▼
Danger Probability
      │
      ▼
Risk Score (0–100)
      │
      ├───────────────┐
      ▼               ▼
   SAFE           DANGEROUS
  🟢 Green          🔴 Red
      │               │
      └───────┬───────┘
              ▼
       Streamlit Dashboard
              │
              ▼
      Analytics & Reports
```

------------------------------------------------------------------------

# 🤖 AI / ML Pipeline

## 1. Input Traffic Video

The application accepts traffic videos such as:

-   `.mp4`
-   `.avi`
-   `.mov`

The Streamlit interface currently exposes AVI, MP4, and MOV as supported
upload formats. fileciteturn28file4L458-L469

## 2. YOLO Vehicle Detection

YOLO detects vehicles frame-by-frame and provides:

-   Bounding boxes
-   Class IDs
-   Detection confidence

## 3. ByteTrack Tracking

ByteTrack associates detections between frames and provides a unique
tracking ID for each vehicle.

## 4. Trajectory Extraction

For every tracked vehicle, the system stores center-point movement:

``` text
(cx, cy)
```

These coordinates are used to calculate speed, acceleration,
deceleration, and direction changes.

## 5. Feature Extraction

The application calculates exactly seven features for the Random Forest
model. fileciteturn28file0L98-L112

## 6. StandardScaler

The seven features are transformed using the saved `final_scaler.pkl`.

## 7. Random Forest

The scaled features are passed to the saved Random Forest model:

``` text
final_random_forest.pkl
```

The model predicts SAFE or DANGEROUS behavior and provides a
dangerous-driving probability.

## 8. Risk Score

The dangerous probability is converted to a 0--100 risk score.

``` text
Risk Score = Dangerous Probability × 100
```

The implementation stores the dangerous probability directly as the risk
score. fileciteturn28file3L738-L750

------------------------------------------------------------------------

# 📐 Machine Learning Features

The Random Forest classifier uses exactly these **7 motion features**:

  Feature                  Description
  ------------------------ ----------------------------------------
  `avg_speed`              Average vehicle movement speed feature
  `max_speed`              Maximum observed vehicle speed feature
  `avg_acceleration`       Average positive acceleration
  `max_acceleration`       Maximum positive acceleration
  `max_deceleration`       Maximum observed deceleration
  `avg_direction_change`   Average change in movement direction
  `max_direction_change`   Maximum observed direction change

The implementation calculates these values from tracked vehicle
center-point movement. fileciteturn28file6L572-L633

------------------------------------------------------------------------

# 🚨 Dangerous Driving Detection

The Random Forest model evaluates the seven motion features and
predicts:

``` text
0 → SAFE
1 → DANGEROUS
```

The processed video uses:

-   🟢 **Green bounding box** → SAFE
-   🔴 **Red bounding box** → DANGEROUS

Each tracked vehicle can display its ID, type, status, estimated speed,
and risk score.

------------------------------------------------------------------------

# ⚡ Sudden Braking Detection

Sudden braking is treated as a **multi-frame event**, rather than simply
checking whether a vehicle is currently moving slowly.

The system:

1.  Collects vehicle speed history.
2.  Applies median smoothing to reduce noise.
3.  Checks the vehicle's previous speed.
4.  Measures the speed reduction.
5.  Requires a meaningful percentage drop.
6.  Requires decreasing-speed evidence across multiple frames.

This prevents every slow or stopped vehicle from being incorrectly
flagged as sudden braking. The current implementation explicitly
requires prior movement and multi-frame evidence.
fileciteturn28file3L336-L372

------------------------------------------------------------------------

# ⛔ Wrong-Way Detection

Wrong-way detection is based on sustained movement in the direction
opposite to the dominant traffic flow.

The system first estimates the dominant left/right movement direction
from sufficiently long vehicle tracks and then checks for sustained
opposite movement. fileciteturn28file9L829-L852

> **Important:** Wrong-way detection depends on the traffic scene and
> camera viewpoint. It should be interpreted as a conservative
> video-based indication, not as a universal road-rule determination.

------------------------------------------------------------------------

# 🖥️ Streamlit Dashboard

TRAFFICSENSE AI includes a professional Streamlit command-center
interface.

### Dashboard

Provides an overview of:

-   Total vehicles
-   Dangerous vehicles
-   Safe vehicles
-   AI confidence
-   Active processed video
-   Live AI events
-   Vehicle intelligence

### Live Detection

Allows users to:

-   Upload a traffic video
-   Inspect video metadata
-   Start AI analysis
-   Monitor processing progress
-   View the processed video

### Vehicle Analysis

Provides detailed vehicle-level inspection using the seven motion
features.

### Traffic Analytics

Provides graphical analysis of:

-   Vehicle risk
-   Speed
-   Driving behavior
-   Vehicle distribution

### Traffic Flow Analysis

Visualizes tracked vehicle trajectories using Plotly. The application
plots ByteTrack spatial vectors from the stored vehicle tracks.
fileciteturn28file8L749-L818

### Detected Events

Displays records for:

-   Dangerous driving
-   Sudden braking
-   Wrong-way movement

### Reports

Provides export options for:

-   CSV vehicle telemetry
-   Processed video
-   Text summary report

------------------------------------------------------------------------

# 🛠️ Technologies Used

  Technology     Purpose
  -------------- -----------------------------------------
  Python         Core programming language
  YOLO           Vehicle/object detection
  ByteTrack      Multi-object tracking
  OpenCV         Video processing and annotation
  NumPy          Numerical computation
  Pandas         Data processing and reporting
  Scikit-learn   Random Forest and feature scaling
  Joblib         Saving/loading ML models
  Streamlit      Interactive web dashboard
  Plotly         Interactive analytics and visualization
  FFmpeg         H.264 video encoding

------------------------------------------------------------------------

# 🏗️ Project Architecture

``` text
                    ┌──────────────────────┐
                    │    Traffic Video     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    YOLO Detector     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  ByteTrack Tracker   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Vehicle Trajectories │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 7 Motion Features   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    StandardScaler    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Random Forest      │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             ┌────────────┐        ┌──────────────┐
             │    SAFE    │        │  DANGEROUS   │
             │   🟢       │        │      🔴      │
             └────────────┘        └──────────────┘
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Dashboard │
                    └──────────────────────┘
```

------------------------------------------------------------------------

# 📁 Project Folder Structure

``` text
Traffic_Danger_Detection/
│
├── app.py
│
├── models/
│   ├── best.pt
│   ├── final_random_forest.pkl
│   └── final_scaler.pkl
│
├── outputs/
│
├── dataset/
│   └── videos/
│       └── traffic_video.avi
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# ⚙️ Installation

## 1. Clone the Repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Traffic_Danger_Detection
```

## 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

## 3. Place the Model Files

Place the required trained models inside:

``` text
models/
```

Required files:

``` text
models/
├── best.pt
├── final_random_forest.pkl
└── final_scaler.pkl
```

The application loads the Random Forest and scaler from their saved
model files. fileciteturn28file1L173-L191

------------------------------------------------------------------------

# 📦 Required Python Packages

A suitable `requirements.txt` should include the packages used by the
application:

``` text
streamlit
ultralytics
opencv-python
numpy
pandas
scikit-learn
joblib
plotly
imageio-ffmpeg
```

> Package versions can be pinned according to the Python environment
> used for deployment.

------------------------------------------------------------------------

# ▶️ How to Run the Project

From the project directory:

``` bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

------------------------------------------------------------------------

# 🤖 Model Files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `best.pt`                           YOLO vehicle detection model

  `final_random_forest.pkl`           Trained Random Forest
                                      dangerous-driving classifier

  `final_scaler.pkl`                  StandardScaler used before Random
                                      Forest inference
  -----------------------------------------------------------------------

The application uses the saved scaler to transform the seven feature
columns before calling the Random Forest model.
fileciteturn28file3L734-L750

------------------------------------------------------------------------

# 🎥 Input Video Format

Supported video formats:

``` text
MP4
AVI
MOV
```

Example:

``` text
dataset/
└── videos/
    └── traffic_video.avi
```

The Streamlit upload component currently accepts `.mp4`, `.avi`, and
`.mov`. fileciteturn28file4L460-L469

------------------------------------------------------------------------

# 📤 Output

After processing, TRAFFICSENSE AI produces:

### Annotated Video

The processed video contains:

``` text
Vehicle ID
Vehicle Type
SAFE / DANGEROUS
Estimated Speed
Risk Score
Event Alerts
```

### Vehicle Telemetry CSV

Contains vehicle-level analysis and extracted features.

### Summary Report

Contains information such as:

-   Total vehicles
-   Safe vehicles
-   Dangerous vehicles
-   Average risk
-   Average speed
-   Sudden braking incidents
-   Wrong-way incidents
-   AI pipeline configuration

------------------------------------------------------------------------

# ⚠️ Important Limitations

### 1. Estimated Speed

Vehicle speed is estimated from image/pixel movement and the configured
pixel-to-meter ratio.

Therefore:

> **Estimated Speed is not ground-truth real-world speed unless the
> camera scene has been properly calibrated.**

### 2. Camera Perspective

Perspective distortion can affect motion and speed estimates.

### 3. Tracking Quality

Occlusion, crowded traffic, poor video quality, and missed detections
can affect vehicle tracking.

### 4. Wrong-Way Detection

Wrong-way detection depends on the observed traffic-flow direction and
camera viewpoint.

### 5. ML Predictions

SAFE/DANGEROUS classification depends on the trained Random Forest model
and the quality/distribution of its training data.

### 6. Event Detection

Sudden braking and wrong-way movement are video-derived indicators and
should not automatically be treated as legal violations.

------------------------------------------------------------------------

# 🚀 Future Enhancements

Possible future improvements include:

-   Real-world camera calibration for more accurate speed estimation.
-   Lane-level wrong-way detection.
-   Traffic signal violation detection.
-   Automatic number plate recognition.
-   Real-time CCTV stream integration.
-   Cloud-based monitoring.
-   Multi-camera traffic monitoring.
-   Alert notifications.
-   Historical traffic dashboards.
-   Model retraining with larger and more diverse datasets.
-   Advanced deep-learning behavior classification.
-   Automatic incident snapshots and evidence generation.

------------------------------------------------------------------------

# 📸 Screenshots

Add your Streamlit screenshots here after uploading them to GitHub.

### Dashboard

``` text
![TRAFFICSENSE AI Dashboard](screenshots/dashboard.png)
```

### Live Detection

``` text
![Live Detection](screenshots/live_detection.png)
```

### Vehicle Analysis

``` text
![Vehicle Analysis](screenshots/vehicle_analysis.png)
```

### Traffic Analytics

``` text
![Traffic Analytics](screenshots/traffic_analytics.png)
```

### Detected Events

``` text
![Detected Events](screenshots/detected_events.png)
```

> Replace the image paths with your actual GitHub screenshot paths.

------------------------------------------------------------------------

# 🎬 Demo

Add your project demo video or GIF here:

``` text
[▶️ TRAFFICSENSE AI Demo](YOUR_DEMO_LINK)
```

You can also add a GitHub-hosted demo GIF:

``` text
![TRAFFICSENSE AI Demo](screenshots/demo.gif)
```

------------------------------------------------------------------------

# 📊 Project Highlights

``` text
✔ YOLO Vehicle Detection
✔ ByteTrack Multi-Object Tracking
✔ Unique Vehicle IDs
✔ 7 Motion Features
✔ StandardScaler Preprocessing
✔ Random Forest Classification
✔ Risk Score 0–100
✔ SAFE / DANGEROUS Visualization
✔ Sudden Braking Detection
✔ Wrong-Way Detection
✔ Streamlit Dashboard
✔ Plotly Analytics
✔ CSV & Text Reports
✔ H.264 Processed Video
```

------------------------------------------------------------------------

# 🧪 Example Vehicle Output

``` text
┌─────────────────────────────────────────┐
│ Vehicle ID : 12                         │
│ Type       : Car                        │
│ Status     : DANGEROUS 🔴              │
│ Est. Speed : 48.6 km/h                  │
│ Risk Score : 87%                        │
│ Sudden Braking : No                     │
│ Wrong Way      : No                     │
└─────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 🎓 Academic Use

This project can be used as an academic/final-year project demonstrating
the integration of:

-   Computer Vision
-   Object Detection
-   Multi-Object Tracking
-   Machine Learning
-   Feature Engineering
-   Video Analytics
-   Data Visualization
-   Streamlit Application Development

------------------------------------------------------------------------

# 👩‍💻 Author

**Neha Kamble**

MCA Student\
Savitribai Phule Pune University

------------------------------------------------------------------------

# ⭐ Acknowledgement

This project combines computer-vision detection and tracking with
machine-learning-based motion analysis to create an end-to-end traffic
monitoring workflow.

------------------------------------------------------------------------

## 📜 License

Add your preferred license here, for example:

``` text
MIT License
```

If you publish this project publicly, make sure the license matches the
datasets, model weights, and third-party components you use.

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<b>`{=html}🚦 TRAFFICSENSE AI --- Smarter Traffic Monitoring with
AI`</b>`{=html}
```{=html}
</p>
```
