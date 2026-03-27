import easyocr
import cv2
import os

frame = cv2.imread("data/frames/frame_0001.jpg")
h, w = frame.shape[:2]

# crop bottom 15%
crop = frame[int(h * 0.85):h, :]
cv2.imwrite("data/frames/scoreboard_crop.jpg", crop)

# first run downloads models (~100MB), subsequent runs are instant
reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext(crop)

print("All detected text in scoreboard region:")
for (bbox, text, conf) in results:
    print(f"  '{text}' (conf: {conf:.2f})")