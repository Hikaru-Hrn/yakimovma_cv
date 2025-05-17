import cv2

sunglasses = cv2.imread("deal-with-it.png", cv2.IMREAD_UNCHANGED)

if sunglasses.shape[2] == 4:
    sunglasses_rgb = sunglasses[:, :, :3]
    sunglasses_alpha = sunglasses[:, :, 3]
else:
    raise ValueError("Sunglasses image must have an alpha channel (transparency).")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade-eye.xml")

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    key = chr(cv2.waitKey(1) & 0xFF)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)

        if len(eyes) >= 2:
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            eye_center_x = (ex1 + ew1 // 2 + ex2 + ew2 // 2) // 2
            eye_center_y = (ey1 + eh1 // 2 + ey2 + eh2 // 2) // 2

            eye_distance = abs((ex1 + ew1 // 2) - (ex2 + ew2 // 2))
            if eye_distance < 10:
                continue

            glasses_width = int(eye_distance * 2.0)
            glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

            glasses_x = eye_center_x - glasses_width // 2
            glasses_y = eye_center_y - glasses_height // 2

            glasses_x = max(0, glasses_x)
            glasses_y = max(0, glasses_y)
            glasses_x2 = min(roi_color.shape[1], glasses_x + glasses_width)
            glasses_y2 = min(roi_color.shape[0], glasses_y + glasses_height)

            target_width = glasses_x2 - glasses_x
            target_height = glasses_y2 - glasses_y
            if target_width <= 0 or target_height <= 0:
                continue

            resized_glasses = cv2.resize(sunglasses_rgb, (target_width, target_height))
            resized_alpha = cv2.resize(sunglasses_alpha, (target_width, target_height))

            alpha = resized_alpha / 255.0

            for c in range(0, 3):
                roi_color[glasses_y:glasses_y2, glasses_x:glasses_x2, c] = (
                        alpha * resized_glasses[:, :, c] +
                        (1 - alpha) * roi_color[glasses_y:glasses_y2, glasses_x:glasses_x2, c]
                )

    if key == "q":
        break
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()