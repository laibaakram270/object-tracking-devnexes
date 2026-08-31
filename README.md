# DEV-NEXES Real-Time Object Tracking & Counting
 
It performs real-time object detection, tracking, and counting using YOLOv8 and SORT algorithm.

## 🎯 Features - 4 Week Implementation

### Week 1: Video Ingestion & YOLO Detection
- OpenCV video capture pipeline
- Real-time YOLOv8 object detection with bounding boxes

### Week 2: SORT Object Tracking Engine  
- Integrated SORT algorithm with Kalman filtering
- Assigns persistent unique IDs to objects across frames

### Week 3: Virtual Line Counting Trigger
- 2D vector line crossing logic
- On-screen HUD with IN, OUT, and TOTAL counters
- Triggers count when object centroid crosses yellow line

### Week 4: FPS Tuning & Demo
- FPS overlay for performance monitoring
- Optimized frame processing
- Final demo and documentation

 ## **Demo Videos** ##
 https://drive.google.com/file/d/1OHFF1iMkeuuoKdOAmrXTCLZ8lB4dL4A3/view?usp=drive_link

## 🛠️ Tech Stack
- **Python 3.10+**
- **OpenCV** - Video processing
- **Ultralytics YOLOv8** - Object Detection
- **SORT** - Object Tracking
- **NumPy, SciPy** - Math and geometry

## ⚡ Installation

1. Clone the repository
```bash
git clone https://https://github.com/laibaakram270/object-tracking-devnexes
cd object-tracking-devnexes
```
2. Install dependencies
 ```bash
pip install -r requirements.txt
```
3. Run the project
 ```bash
python main.py
```
📊 Output

The system displays:

Green bounding boxes with persistent IDs

Yellow virtual counting line

Real-time IN/OUT/

TOTAL counters

FPS counter
