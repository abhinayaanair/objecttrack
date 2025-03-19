import cv2
import numpy as np
from ultralytics import YOLO
from motpy import MultiObjectTracker, Detection

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize motpy tracker (dt=1/30 assumes 30 FPS; adjust if needed)
tracker = MultiObjectTracker(dt=1/30)

# Set up video capture from default webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv8
    results = model(frame)
    motpy_detections = []

    # Extract and filter detections from YOLO results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])               # Confidence score
            cls = int(box.cls[0])                   # Class ID
            if conf > 0.3:                          # Confidence threshold
                motpy_detections.append(Detection(box=[x1, y1, x2, y2], score=conf, class_id=cls))

    # Update tracker with detections
    tracker.step(detections=motpy_detections)

    # Get active tracks
    tracked_objects = tracker.active_tracks()

    # Draw bounding boxes and IDs on the frame
    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, track.box)  # Extract bounding box
        obj_id = track.id                     # Unique track ID (string in motpy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("YOLO + motpy Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()