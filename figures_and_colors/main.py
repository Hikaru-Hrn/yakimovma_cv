import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops

def classify_shade(hue_value):
    shade_names = ["Оттенок1", "Оттенок2", "Оттенок3", "Оттенок4", "Оттенок5", "Оттенок6"]
    thresholds = [0.192, 0.305, 0.415, 0.609, 0.833]
    for idx, threshold in enumerate(thresholds):
        if hue_value < threshold:
            return shade_names[idx]
    return shade_names[-1]

# Загрузка и подготовка изображения
img = plt.imread("balls_and_rects.png")
hsv_img = rgb2hsv(img)

# Бинаризация и разметка
gray_img = img.mean(axis=-1)
mask = gray_img > 0
labeled_img = label(mask)
props = regionprops(labeled_img)

ball_colors, rect_colors = [], []
ball_count, rect_count = 0, 0

# Классификация объектов
for prop in props:
    cy, cx = prop.centroid
    hue = hsv_img[int(cy), int(cx), 0]
    if prop.eccentricity == 0:
        ball_colors.append(hue)
        ball_count += 1
    else:
        rect_colors.append(hue)
        rect_count += 1

print(f"Общее количество объектов: {ball_count + rect_count}")

# Подсчет оттенков
ball_shades = {}
for hue in ball_colors:
    shade = classify_shade(hue)
    ball_shades[shade] = ball_shades.get(shade, 0) + 1

rect_shades = {}
for hue in rect_colors:
    shade = classify_shade(hue)
    rect_shades[shade] = rect_shades.get(shade, 0) + 1

print(f"Мячики по оттенкам: {sorted(ball_shades.items())}")
print(f"Прямоугольники по оттенкам: {sorted(rect_shades.items())}")

# Визуализация
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(sorted(ball_colors), marker='o')
plt.title("Оттенки мячиков")

plt.subplot(1, 2, 2)
plt.plot(sorted(rect_colors), marker='o')
plt.title("Оттенки прямоугольников")

plt.tight_layout()
plt.show()
