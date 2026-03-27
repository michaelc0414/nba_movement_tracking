import cv2
import os

clip_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/clips/test_clip.mp4'
output_path = 'C:/Users/Michael/Documents/nba_movement_tracking/data/frames/frame_0001.jpg'

cap = cv2.VideoCapture(clip_path)

#printing clip info
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"# of frames - {total_frames}")
print(f"fps - {fps}")
print(f"resolution - {width}x{height}")
print(f"duration - {total_frames / fps:.1f} seconds")

#get frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()

if ret:
    cv2.imwrite(output_path, frame)
    print(f"frame saved to {output_path}")
else:
    print("failed to read frame")

cap.release()