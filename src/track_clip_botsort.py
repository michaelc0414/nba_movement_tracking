import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from boxmot import BotSort
from pathlib import Path

clip_path = 'data/clips/test_clip.mp4'
output_path = 'data/clips/test_clip_trackedNEW.mp4'

model = YOLO("runs/detect/models/player_detector5/weights/best.pt")

tracker = BotSort(
    reid_weights=Path("models/osnet_x0_25_msmt17.pt"),
    device=0,
    half=False,
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=100,
    match_thresh=0.8,
)

cap = cv2.VideoCapture(clip_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
all_track_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
        conf = float(box.conf)
        cls = float(box.cls)
        box_height = y2 - y1

        if box_height >= 150 and y1 < height * 0.85:
            detections.append([x1, y1, x2, y2, conf, cls])

    detections = np.array(detections) if detections else np.empty((0, 6))
    
    #running tracker
    tracks = tracker.update(detections, frame)

    for track in tracks:
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        track_id = int(track[4])
        conf = float(track[5])

        all_track_ids.add(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_idx} | Tracks: {len(tracks)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        print(f"processed {frame_idx}/{total_frames} frames")

cap.release()
writer.release()

print("Done")
print(f'total unique track ids: {len(all_track_ids)}')
print(f"track ids: {sorted(all_track_ids)}")