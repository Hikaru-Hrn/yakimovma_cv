import cv2

video_path = "output.avi"
reference_image_path = "14123/pleskunov.png"

ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1000)
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

GOOD_MATCHES_THRESHOLD = 20
FRAME_DIFF = 10
FRAME_STEP = 5

cap = cv2.VideoCapture(video_path)
frame_count = 0
match_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_STEP != 0:
        continue

    scale_factor = 640 / frame.shape[1] if frame.shape[1] > 640 else 1.0
    if scale_factor < 1.0:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is None:
        continue

    matches = bf.match(des_ref, des_frame)
    good_matches = sorted(matches, key=lambda x: x.distance)[:GOOD_MATCHES_THRESHOLD]

    if len(good_matches) >= GOOD_MATCHES_THRESHOLD:
        if not match_frames or frame_count - match_frames[-1] > FRAME_DIFF:
            match_frames.append(frame_count)

cap.release()

print(f"Количество появлений изображения: {len(match_frames)}")
print(f"Кадры с совпадениями: {match_frames}")
