import cv2
import numpy as np

video_path = "output.avi"
template_path = "pleskunov.png"

MATCH_THRESHOLD = 0.6
SCALE = 1.25

template_orig = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template_orig, (0, 0), fx=SCALE, fy=SCALE)
w, h = template.shape[::-1]

cap = cv2.VideoCapture(video_path)

match_count = 0
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray_frame.shape[0] < template.shape[0] or gray_frame.shape[1] < template.shape[1]:
        continue

    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= MATCH_THRESHOLD)

    if len(loc[0]) > 0:
        match_count += 1
        print(f"Найдено совпадение на кадре {frame_id}")

cap.release()
print(f"\n🔍 Общее количество совпадений: {match_count}")
