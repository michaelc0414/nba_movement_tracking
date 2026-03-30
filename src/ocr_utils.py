import easyocr
import re
import cv2

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

def extract_scoreboard(frame):
    #this is to get the game clock, shot clock and quarter from the 
    #scoreboard
    h, w = frame.shape[:2]
    crop = frame[int(h * 0.85):h :] #bottom 15% of the screen

    results = reader.readtext(crop)

    game_clock = None
    shot_clock = None
    quarter = None

    for (_, text, conf) in results:
        text = text.strip().upper()

        if re.match(r'\d{1,2}:\{2}$', text) and conf > 0.4:
            game_clock = text

        if re.match(r'^\d{1,2}$', text) and conf > 0.8:
            val = int(text)
            if 1 <= val <= 24:
                shot_clock = text
        
        print(f"OCR text: '{text}' conf: {conf:.2f}")
        if re.match(r'^[1-4](ST|ND|RD|TH)$|^OT\d?$', text, re.IGNORECASE) and conf > 0.3:
            quarter = text

    return game_clock, shot_clock, quarter