import cv2
from ultralytics import YOLO
import os

#this is to find the worst frames to find the models weaknesses

clip_path = 'data/clips/test_clip.mp4'
output_path = 'data/frames/bad'
os.makedirs(output_path, exist_ok=True)

model = YOLO('runs/detect/models/player_detector_polished/weights/best.pt')
cap = cv2.VideoCapture(clip_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_idx = 0
detection_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)[0]

    kept = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        if (y2 - y1) >= 150 and y1 < height * 0.85:
            kept.append(box)

    detection_log.append((frame_idx, len(kept), frame.copy(), kept))
    frame_idx += 1

cap.release()

#get and save 5 worst frames
detection_log.sort(key=lambda x: x[1])
print('Worst detection frames:')
for frame_idx, count, frame, boxes in detection_log[:5]:
    print(f" for frame {frame_idx}: {count} detections")
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    path = f"{output_path}/worst_frame_{frame_idx}_count_{count}.jpg"
    cv2.imwrite(path, frame)
    print('bad frame saved')

#print best 5 frames
print("\nBest detection frames:")
detection_log.sort(key=lambda x: x[1], reverse=True)
for frame_idx, count, frame, boxes in detection_log[:5]:
    print(f"  Frame {frame_idx}: {count} detections")