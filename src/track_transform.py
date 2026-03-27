import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.ocr_utils import extract_scoreboard

clip_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip.mp4'
output_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip_transformed.mp4'
homography_path = "models/homography_frame1.npy"

model = YOLO('models/yolov8x.pt')
H = np.load(homography_path)

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

def pixel_to_court(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])

cap = cv2.VideoCapture(clip_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
all_records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    game_clock, shot_clock, quarter = extract_scoreboard(frame)
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

        #bottom center of players bounding box
        foot_px = (x1 + x2) // 2
        foot_py = y2

        #transform to court coords
        court_x, court_y = pixel_to_court(foot_px, foot_py, H)

        #skip positions that are out of court bounds
        if not (0 <= court_x <= 50 and 0 <= court_y <= 47):
            continue

        all_records.append({
            "frame": frame_idx,
            "track_id": track_id,
            "pixel_x": foot_px,
            "pixel_y": foot_py,
            "court_x": round(court_x, 3),
            "court_y": round(court_y, 3),
            "confidence": round(conf, 3)
        })

        #draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (foot_px, foot_py), 4, (0, 0, 255), -1)

        #show track id and court coords
        label = f"ID {track_id} ({court_x:.1f}, {court_y:.1f})ft"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
    cv2.putText(frame, f"Frame: {frame_idx} | Tracks: {len(tracks)}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        print(f"processed {frame_idx}/{total_frames} frames")

cap.release()
writer.release()

df = pd.DataFrame(all_records)

df['game_id'] = 'GSW_SAC_2023_67'
df['clip_id'] = 'test_clip'
df['fps'] = fps

df = df[["game_id", "clip_id", "frame", "fps", "track_id",
         "pixel_x", "pixel_y", "court_x", "court_y", "confidence"]]

# save to parquet
parquet_path = "data/parquet/test_clip_tracking.parquet"
df.to_parquet(parquet_path, index=False)

print("Done")
print(f"total records : {len(all_records)}")
print(f"Parquet saved to: {parquet_path}")
print(f"Unique track IDs: {sorted(df['track_id'].unique())}")
print(f"Frames covered: {df['frame'].min()} to {df['frame'].max()}")
print(f"\nSample data:")
print(df.head(10).to_string(index=False))