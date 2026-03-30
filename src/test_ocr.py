import easyocr
import cv2
from ocr_utils import extract_scoreboard
import os

#just testing the scoreboard extracter

frame = cv2.imread("data/frames/frame_0001.jpg")
game_clock, shot_clock, quarter = extract_scoreboard(frame)

print(f"game_clock: {game_clock}")
print(f"shot_clock: {shot_clock}")
print(f"quarter:    {quarter}")