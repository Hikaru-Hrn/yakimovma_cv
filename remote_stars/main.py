import socket
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, center_of_mass

host = "84.237.21.36"
port = 5152

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def get_two_brightest_local_maxima(img, size=3):
    max_filt = maximum_filter(img, size=size)
    mask = (img == max_filt)  # локальные максимумы

    labeled, num_features = label(mask)
    if num_features < 2:
        raise ValueError("Меньше двух локальных максимумов")

    centers = center_of_mass(img, labeled, range(1, num_features + 1))

    brightness = [
        (i, img[labeled == (i + 1)].sum()) for i in range(num_features)
    ]
    brightness.sort(key=lambda x: x[1], reverse=True)

    top2 = [centers[brightness[0][0]], centers[brightness[1][0]]]
    return top2

    centers = center_of_mass(img, labeled, range(1, num_features + 1))

    brightness = [
        (i, img[labeled == (i + 1)].sum()) for i in range(num_features)
    ]
    brightness.sort(key=lambda x: x[1], reverse=True)

    top2 = [centers[brightness[0][0]], centers[brightness[1][0]]]
    return top2

def calc_distance(p1, p2):
    return round(np.linalg.norm(np.array(p1) - np.array(p2)), 1)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))

    plt.ion()
    plt.figure()

    for i in range(10):
        sock.send(b"get")
        bts = recvall(sock, 40002)
        if not bts:
            print("Ошибка получения изображения")
            break

        height, width = bts[0], bts[1]
        im1 = np.frombuffer(bts[2:40002], dtype="uint8").reshape(height, width)

        p1 = p2 = None

        p1, p2 = get_two_brightest_local_maxima(im1)
        distance = calc_distance(p1, p2)

        sock.send(str(distance).encode())
        response = sock.recv(10).decode()
        print(f"[{i + 1}/10] Расстояние: {distance}, Ответ: {response}")


        plt.clf()
        plt.imshow(im1, cmap='viridis')
        if p1 and p2:
            plt.scatter([p1[1], p2[1]], [p1[0], p2[0]], c='red', s=50)
        plt.title(f"Расстояние: {distance}")
        plt.pause(1)

        sock.send(b"beat")
        beat = sock.recv(10)
        if beat == b"yep":
            break
