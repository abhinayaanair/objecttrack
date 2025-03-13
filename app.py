import cv2
import numpy as np
from ultralytics import YOLO
from sort.tracker import SortTracker  

model = YOLO("yolov8n.pt")  
tracker = SortTracker()  
video_path = 0 
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1  
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0]) 
            cls = int(box.cls[0])  

            if conf > 0.3:
                detections.append([x1, y1, x2, y2, conf, cls])  

    detections = np.array(detections, dtype=np.float32).reshape(-1, 6)  

    if detections.shape[0] == 0:
        detections = np.empty((0, 6), dtype=np.float32)  

    tracked_objects = tracker.update(detections, frame_id)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj[:5]) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("YOLO + SORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
