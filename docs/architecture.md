# AI Exam Surveillance - System Architecture

## System Pipeline

CCTV Camera
    |
    v
OpenCV
    |
    v
YOLO11m Object Detection
    |
    +----------------------+
    |                      |
    v                      v
MediaPipe              Exam Objects
Face / Eye / Hand       Phone, Book,
Landmarks               Cheat Paper, etc.
    |
    v
Head Pose Estimation
    |
    v
GazeNet V2
    |
    v
Face Identity Module
    |
    v
37-Feature Extraction
    |
    v
XGBoost
    |
    +-------------------+
    |                   |
    v                   v
 Normal             Cheating
    |
    v
Firebase Logging
    |
    v
ESP32 Alert
    |
    v
Administrator Dashboard

## Components

### OpenCV
Handles camera capture, frame processing, buffering and video streaming.

### YOLO11m
Detects 9 examination-related objects:

- phone
- book
- cheat_paper
- calculator
- earphone
- sunglasses
- watch
- Answer_paper
- laptop

### MediaPipe
Provides face, eye and hand landmark information.

### Head Pose Estimation
Estimates:

- head pitch
- head yaw
- head roll

### GazeNet V2
Provides continuous gaze estimation and gaze-related features.

### Face Identity Module
Performs student identity verification.

### 37-Feature Extraction
Combines face, eye, hand, object, head-pose and gaze information into the behavioral feature representation.

### XGBoost
Classifies the behavioral state into:

- Normal
- Cheating

### Firebase
Used for cloud event logging.

### ESP32
Provides physical alert output such as buzzer and LED.

### Administrator Dashboard
Provides live monitoring, status, alerts and event information.

## Processing Flow

1. Capture camera frame.
2. Detect examination objects using YOLO11m.
3. Extract face, eye and hand landmarks.
4. Estimate head pose.
5. Estimate gaze using GazeNet V2.
6. Verify student identity.
7. Construct the 37-feature representation.
8. Run XGBoost classification.
9. Update system state.
10. Log configured events.
11. Trigger configured ESP32 alerts.
12. Display information on the administrator dashboard.

## Component Responsibility

| Component | Responsibility |
|---|---|
| OpenCV | Camera and video processing |
| YOLO11m | Object detection |
| MediaPipe | Face, eye and hand landmarks |
| Head Pose | Head orientation |
| GazeNet V2 | Gaze estimation |
| Face Identity | Student verification |
| Feature Extraction | 37-feature representation |
| XGBoost | Normal/Cheating classification |
| Firebase | Cloud logging |
| ESP32 | Physical alerts |
| Flask | Web application |

## Evaluation Note

The individual model metrics are component-level results. They should not be interpreted as a single end-to-end system accuracy.
