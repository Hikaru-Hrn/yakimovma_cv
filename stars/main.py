import numpy as np
import matplotlib.pyplot as plt


cross_array = np.array([[1,0,0,0,1],
                        [0,1,0,1,0],
                        [0,0,1,0,0],
                        [0,1,0,1,0],
                        [1,0,0,0,1]])

plus_array = np.array([[0,0,1,0,0],
                       [0,0,1,0,0],
                       [1,1,1,1,1],
                       [0,0,1,0,0],
                       [0,0,1,0,0]])

# Подсчет крестов и плюсов
def count_objects(image, array):
    cnt = 0
    for y in range(0, image.shape[0]-5):
        for x in range(0, image.shape[1]-5):
            sub = np.array(image[y:y+5, x:x+5], dtype="uint8")
            if np.array_equal(sub, array):
                cnt += 1
    return cnt

image = np.load('stars.npy')
count_of_cross = count_objects(image, cross_array)
count_of_plus = count_objects(image, plus_array)

print(f"Count of cross: {count_of_cross}")
print(f"Count of plus: {count_of_plus}")
print(f"Answer: {count_of_plus+count_of_cross}")

plt.imshow(image)
plt.show()