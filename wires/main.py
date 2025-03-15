import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (binary_closing, binary_opening, binary_erosion, binary_dilation)


data = np.load("wires3npy.txt")

labeled = label(data)

result = binary_erosion(data, np.ones(3).reshape(3, 1))

count_of_wires = np.max(labeled)
count_of_torn_wires = np.max(label(result))

print(f"Всего проводов: {count_of_wires}")

if (count_of_wires < count_of_torn_wires):
    print("Провод порван")
    for i in range(1, np.max(labeled) + 1):
        erosioned_wire = binary_erosion(labeled==i)
        count_of_one_wires = np.max(label(erosioned_wire))
        if count_of_one_wires == 1:
            print(f"Провод {i} не порван")
        else:
            print(f"Провод {i} порван на {count_of_one_wires} частей")
else:
    print("Провод не порван")



plt.subplot(121)
plt.imshow(data)
plt.subplot(122)
plt.imshow(result)
plt.show()