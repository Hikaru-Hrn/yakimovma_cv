import numpy as np
import cv2
from skimage.measure import regionprops as get_props
from skimage.morphology import label as label_regions

# Диапазон HSV-цветов (для маскировки объектов)
color_min = np.array([5, 120, 110])
color_max = np.array([120, 260, 220])

total_count = 0
image_indices = range(1, 13)

kernel = np.ones((9, 9), dtype=np.uint8)

for idx in image_indices:
    path = f"img/img ({idx}).jpg"
    original = cv2.imread(path)
    hsv_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    binary_mask = cv2.inRange(hsv_image, color_min, color_max)
    processed_mask = cv2.dilate(binary_mask, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    labeled_mask = label_regions(processed_mask)
    props = get_props(labeled_mask)

    found = [obj for obj in props if obj.area > 85000 and (1.0 - obj.eccentricity) < 0.02]
    found_count = len(found)
    total_count += found_count

    print(f"Image {idx} Pencils: {found_count}")

print(f"Total pencils in the pictures: {total_count}")
