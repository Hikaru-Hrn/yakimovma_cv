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
    E = 0
    for y in range(0, image.shape[0] - 1):
         for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                 E += 1
            elif match(sub, internal):
                 E -= 1
            elif match(sub, cross):
                 E += 2
    return E / 4


#Загружаем и обрабатываем

image = np.load("example1.npy")
image[image > 0] = 1
count_objects_ex1 = count_objects(image)

plt.subplot(121)
plt.title(f"Objects: {count_objects_ex1}")
plt.imshow(image)

image = np.load("example2.npy")
count_objects_ex2 = 0
image[image > 0] = 1
for i in range(image.shape[-1]):
    count_objects_ex2 += count_objects(image[:,:,i])

plt.subplot(122)
plt.title(f"Objects: {count_objects_ex2}")
plt.imshow(image)

print(f"count objects in ex1: {count_objects_ex1}, in ex2: {count_objects_ex2}")
plt.show()