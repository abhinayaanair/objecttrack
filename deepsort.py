import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=5)

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

    
    if len(detections) == 0:
        detections_list = []
    else:
        detections_list = []
        for det in detections:
            bbox = list(det[:4].astype(int))
            conf = det[4]
            cls = det[5]
            detection = [bbox, conf, cls]
            detections_list.append(detection)

    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    tracks = tracker.update_tracks(detections_list, frame=frame_rgb)

    
    tracked_objects = []
    for track in tracks:
        if track.is_confirmed():
            x1, y1, x2, y2 = track.to_ltrb()
            obj_id = track.track_id
            tracked_objects.append([x1, y1, x2, y2, obj_id])

    
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    
    cv2.imshow("YOLO + Deep SORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()