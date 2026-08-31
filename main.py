import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time

# ====== CONFIG ======
VIDEO_PATH = 0 # 0 = webcam. Change to "videos/test.mp4" if you have a video file
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.4
LINE_Y = 300 # Counting line position
# ====================

# Global variables for counting
COUNTED_IDS = set()
IN_COUNT = 0
OUT_COUNT = 0

def main():
    global IN_COUNT, OUT_COUNT, COUNTED_IDS # <-- THIS LINE FIXES THE ERROR
    
    # MODULE 1: VIDEO CAPTURE & YOLOv8 DETECTION
    print("=== MODULE 1: LOADING YOLOv8 & VIDEO ===")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video/webcam")
        return

    # MODULE 2: SORT TRACKING ENGINE
    print("=== MODULE 2: INIT SORT TRACKER ===")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        
        # 1. YOLOv8 DETECTION - person, car, bus, truck, motorcycle
        results = model(frame, conf=CONFIDENCE, classes=[0, 2, 3, 5, 7])
        detections = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
        
        detections = np.array(detections) if len(detections) > 0 else np.empty((0,5))
        
        # 2. SORT TRACKING
        tracked_objects = tracker.update(detections)

        # 3. MODULE 3: VIRTUAL LINE COUNTING
        cv2.line(frame, (0, LINE_Y), (1280, LINE_Y), (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (500, LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Draw box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Check line crossing
            if cy > LINE_Y and obj_id not in COUNTED_IDS:
                COUNTED_IDS.add(obj_id)
                IN_COUNT += 1
        
        # MODULE 4: FPS TUNING & HUD
        frame_count += 1
        fps = frame_count / (time.time() - fps_start + 0.001)
        
        # HUD Overlay
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"IN: {IN_COUNT}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {OUT_COUNT}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"TOTAL: {IN_COUNT + OUT_COUNT}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("DEVNEXES Object Tracking & Counting", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"=== FINAL COUNT: IN={IN_COUNT}, OUT={OUT_COUNT} ===")

if __name__ == "__main__":
    main()