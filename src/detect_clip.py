import cv2
from ultralytics import YOLO
import os

clip_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip.mp4'
output_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip_detectionsNEWNEW.mp4'

model = YOLO("runs/detect/models/player_detector_polished/weights/best.pt")

cap = cv2.VideoCapture(clip_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
detection_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.45, verbose=False)[0]

    
    kept = []
    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        if (y2 - y1) >= 150:
            kept.append(box)

    detection_counts.append(len(kept))

    for box in kept:   
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = float(box.conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    cv2.putText(frame, f"Frame: {frame_idx} | Detections: {len(kept)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        print(f"processed {frame_idx}/{total_frames} frames")

cap.release()
writer.release()

avg = sum(detection_counts) / len(detection_counts)
min_d = min(detection_counts)
max_d = max(detection_counts)

print(f"finished. output saved")
print(f"average detections per frame: {avg:.1f}")
print(f"min detections in a frame: {min_d}")
print(f"max detections in a frame: {max_d}")