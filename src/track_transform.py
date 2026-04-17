import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ocr_utils import extract_scoreboard

def merge_fragmented_tracks(df, max_frame_gap=30, max_pixel_dist=75):
    """Merge track fragments that likely belong to the same player.

    For each pair of tracks where A ends near where B begins (in both time and space),
    and A and B never coexist in the same frame, relabel B as A.
    """
    track_stats = df.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        last_px=("pixel_x", "last"),
        last_py=("pixel_y", "last"),
        first_px=("pixel_x", "first"),
        first_py=("pixel_y", "first"),
    )
    track_frames = df.groupby("track_id")["frame"].apply(set)
    track_stats["frames"] = track_frames

    # Build merge map: track_id -> canonical_id
    merge_map = {}
    sorted_tracks = track_stats.sort_values("first_frame")

    for tid_b, row_b in sorted_tracks.iterrows():
        best_match = None
        best_gap = float("inf")

        for tid_a, row_a in sorted_tracks.iterrows():
            if tid_a == tid_b:
                continue
            # A must end before B starts
            frame_gap = row_b["first_frame"] - row_a["last_frame"]
            if frame_gap < 1 or frame_gap > max_frame_gap:
                continue
            # No temporal overlap
            if row_a["frames"] & row_b["frames"]:
                continue
            # Spatial proximity: A's last position near B's first position
            dist = np.sqrt((row_a["last_px"] - row_b["first_px"])**2 +
                           (row_a["last_py"] - row_b["first_py"])**2)
            if dist > max_pixel_dist:
                continue
            if frame_gap < best_gap:
                best_gap = frame_gap
                best_match = tid_a

        if best_match is not None:
            # Follow transitive chain: if best_match was already merged, use its canonical
            canonical = best_match
            while canonical in merge_map:
                canonical = merge_map[canonical]
            merge_map[tid_b] = canonical

    if merge_map:
        before = df["track_id"].nunique()
        df["track_id"] = df["track_id"].map(lambda x: merge_map.get(x, x))
        after = df["track_id"].nunique()
        print(f"Track merge: {before} -> {after} unique IDs ({before - after} merged)")
    else:
        print("Track merge: no fragments to merge")

    return df

clip_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip.mp4'
output_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip_transformed.mp4'
homography_path = "models/homography_frame1.npy"

model = YOLO('runs/detect/models/player_detector5/weights/best.pt')
H = np.load(homography_path)

tracker = BotSort(
    reid_weights=Path("models/osnet_x1_0_msmt17.pt"),
    device=0,
    half=False,
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.55,
    track_buffer=90,
    match_thresh=0.7,
    proximity_thresh=0.6,
    appearance_thresh=0.45,
    cmc_method="ecc",
    frame_rate=30,
    with_reid=True,
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
        on_court = 0 <= court_x <= 50 and 0 <= court_y <= 47

        all_records.append({
            "frame": frame_idx,
            "track_id": track_id,
            "pixel_x": foot_px,
            "pixel_y": foot_py,
            "court_x": round(court_x, 3) if on_court else None,
            "court_y": round(court_y, 3) if on_court else None,
            "confidence": round(conf, 3),
            "on_court": on_court
        })

        #draw ALL tracked players regardless of court bounds
        color = (0, 255, 0) if on_court else (0, 165, 255)  # green=on court, orange=off court
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (foot_px, foot_py), 4, (0, 0, 255), -1)

        if on_court:
            label = f"ID {track_id} ({court_x:.1f}, {court_y:.1f})ft"
        else:
            label = f"ID {track_id} [off-court]"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
    cv2.putText(frame, f"Frame: {frame_idx} | Tracks: {len(tracks)}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        print(f"processed {frame_idx}/{total_frames} frames")

cap.release()
writer.release()

df = pd.DataFrame(all_records)
df = merge_fragmented_tracks(df)

df['game_id'] = 'GSW_SAC_2023_67'
df['clip_id'] = 'test_clip'
df['fps'] = fps

df = df[["game_id", "clip_id", "frame", "fps", "track_id",
         "pixel_x", "pixel_y", "court_x", "court_y", "confidence", "on_court"]]

# save to parquet
parquet_path = "data/parquet/test_clip_trackingNEW.parquet"
df.to_parquet(parquet_path, index=False)

print("Done")
print(f"Total records: {len(df)}")
print(f"On-court records: {df['on_court'].sum()}")
print(f"Off-court records: {(~df['on_court']).sum()}")
print(f"Parquet saved to: {parquet_path}")
print(f"Unique track IDs (all): {df['track_id'].nunique()}")
print(f"Unique track IDs (on-court only): {df[df['on_court']]['track_id'].nunique()}")
print(f"Frames covered: {df['frame'].min()} to {df['frame'].max()}")
print(f"\nSample data:")
print(df.head(10).to_string(index=False))