import cv2
from ultralytics import YOLO

frame_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/frames/frame_0001.jpg'
output_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/frames/frame_0001_detections.jpg'

#loading yolo model
model = YOLO("models/yolov8x.pt")
results = model(frame_path, conf=0.4, verbose=True)[0]

#printing detections
print(f"Total detections {len(results.boxes)}")
print(f"{'Index':<8}{'Class':<15}{'Confidence':<15}{'Bounding Box (x1,y1,x2,y2)'}")
print(f"--")

for i, box in  enumerate(results.boxes):
    class_id = int(box.cls)
    class_name = model.names[class_id]
    confidence = float(box.conf)
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
    print(f"{i:<8}{class_name:<15}{confidence:<15.2f}{x1}, {y1}, {x2}, {y2}")

#save marked image
annotated = results.plot()
cv2.imwrite(output_path, annotated)
print(f"frame saved")
