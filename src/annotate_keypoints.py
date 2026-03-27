import cv2
import numpy as np

frame_path = 'C:/Users/micha/Documents/nba_movement_tracking/nba_movement_tracking/data/frames/frame_1_test.png'
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{len(points)}: ({x},{y})", (x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Annotate", img)

img = cv2.imread(frame_path)
cv2.imshow("Annotate", img)
cv2.setMouseCallback("Annotate", click_event)

print("Click court keypoints in this order:")
print("1. Top-left paint corner (lane line meets free throw line, left side)")
print("2. Top-right paint corner (lane line meets free throw line, right side)")
print("3. Bottom-left paint corner (lane line meets baseline, left side)")
print("4. Bottom-right paint corner (lane line meets baseline, right side)")
print("5. Left sideline/baseline corner (if visible)")
print("6. Any other clear line intersection you can see")
print("\nPress 'q' when done, 'z' to undo last point")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('z') and points:
        points.pop()
        print(f"Undone. Points so far: {len(points)}")

cv2.destroyAllWindows()
print(f"\nFinal pixel points: {points}")