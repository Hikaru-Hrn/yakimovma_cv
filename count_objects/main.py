import numpy as np
import matplotlib.pyplot as plt

external = np.diag([1, 1, 1, 1]).reshape(4, 2, 2)
internal = np.logical_not(external)
cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])

def match(a, masks):
    for mask in masks:
        if np.all(a == mask):
            return True
    return False

#Немного доработанная функция подсчета фигур
def count_objects(image):
    padded = np.pad(image, ((1, 1), (1, 1)))
    E = 0
    for y in range(0, padded.shape[0] - 1):
        for x in range(0, padded.shape[1] - 1):
            sub = padded[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4


#Загружаем и обрабатываем

image = np.load('example1.npy')
if image.ndim == 3:
    image = image[:, :, 0] #Возьмем первый канал, если файл многоканальный
if not np.all(np.isin(image, [0, 1])):
    image = (image > 0).astype(np.uint8) #Приводим к бинарному виду
num_objects = count_objects(image)
print(f"Количество объектов: {num_objects}")

# Визуализация
plt.imshow(image, cmap='gray')
plt.title(f"Objects: {num_objects}")
plt.show()
