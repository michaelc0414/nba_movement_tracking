import cv2
from ultralytics import YOLO
import supervision as sv

clip_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip.mp4'
output_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip_tracked.mp4'

model = YOLO("C:/Users/Michael/Documents/nba_movement_tracking/src/models/yolov8x.pt")

tracker = sv.ByteTrack(
    track_activation_threshold=0.5,
    lost_track_buffer=60,
    minimum_matching_threshold=0.85,
    frame_rate=30
)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

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

    #converting to fit supervision datatype (Detections)
    detections = sv.Detections.from_ultralytics(results)

    mask = (
        (detections.xyxy[:, 3] - detections.xyxy[:, 1] >= 150) &
        (detections.xyxy[:, 1] < height * 0.85)
    )
    detections = detections[mask]

    #running tracker
    tracked = tracker.update_with_detections(detections)

    #collecting all track ids
    if len(tracked) > 0:
        for tid in tracked.tracker_id:
            all_track_ids.add(tid)

    #build labels
    labels = [f"ID {tid}" for tid in tracked.tracker_id]

    #annotate
    annotated = box_annotator.annotate(frame.copy(), tracked)
    annotated = label_annotator.annotate(annotated, tracked, labels)

    cv2.putText(frame, f"Frame: {frame_idx} | Tracks: {len(tracked)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    writer.write(annotated)
    frame_idx += 1

    if frame_idx % 30 == 0:
        print(f"processed {frame_idx}/{total_frames} frames")

cap.release()
writer.release()

print("Done")
print(f'total unique track ids: {len(all_track_ids)}')
print(f"track ids: {sorted(all_track_ids)}")