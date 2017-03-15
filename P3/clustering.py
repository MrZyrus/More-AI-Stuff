import numpy as np
import cv2


def read_data(filename):
    f = open(filename, 'r')
    data = []
    classification = []
    for l in f:
        line = l.split(',')
        if not line[-1] in classification:
            classification.append(line[-1])
        line[-1] = classification.index(line[-1])
        data.append([float(i) for i in line])
    data = np.array(data)
    return data

def equal_centroids(a, b):
    return set([tuple(i) for i in np.float32(a)]) == set([tuple(i) for i in np.float32(b)])

def clustering(k, data):
    new_centroids = data[np.random.choice(data.shape[0], k)]
    old_centroids = data[np.random.choice(data.shape[0], k)]
    while not equal_centroids(new_centroids, old_centroids):
        old_centroids = new_centroids
        cluster = [[] for i in range(k)]
        for row in data:
            group = 0
            old_min = np.linalg.norm(old_centroids[0][:-1]-row[:-1])
            for i in enumerate(old_centroids):
                new_min = np.linalg.norm(old_centroids[i[0]][:-1]-row[:-1])
                if old_min > new_min:
                    group = i[0]
                    old_min = new_min
            cluster[group].append(row)
        new_centroids = []
        for i in cluster:
            new_centroids.append(np.mean(i, 0))
    return cluster

def img_quantization(k, img, iterations):
    new_centroids = img[np.random.choice(img.shape[0], k)]
    old_centroids = img[np.random.choice(img.shape[0], k)]
    it = 0
    while not equal_centroids(new_centroids, old_centroids) and it < iterations:
        old_centroids = new_centroids
        cluster = [[] for i in range(k)]
        label = []
        for row in img:
            group = 0
            old_min = np.linalg.norm(old_centroids[0]-row)
            for i in enumerate(old_centroids):
                new_min = np.linalg.norm(old_centroids[i[0]]-row)
                if old_min > new_min:
                    group = i[0]
                    old_min = new_min
            cluster[group].append(row)
            label.append(group)
        new_centroids = []
        for i in cluster:
            new_centroids.append(np.mean(i, 0))
        it += 1
    return cluster, label, new_centroids

data = read_data("iris.data.txt")

print("Using 2-means\n")
cluster = clustering(2, data)
for i in cluster:
    for row in i:
        print(row)
    print()

print("Using 3-means\n")
cluster = clustering(3, data)
for i in cluster:
    for row in i:
        print(row)
    print()

print("Using 4-means\n")
cluster = clustering(4, data)
for i in cluster:
    for row in i:
        print(row)
    print()

print("Using 5-means\n")
cluster = clustering(5, data)
for i in cluster:
    for row in i:
        print(row)
    print()

img = cv2.imread('Starry Night.jpg')
new_img = img.reshape((-1,3))
new_img = np.float32(new_img)

cluster, label, centroids = img_quantization(2, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K2.jpg', final_img)

cluster, label, centroids = img_quantization(4, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K4.jpg', final_img)

cluster, label, centroids = img_quantization(8, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K8.jpg', final_img)

cluster, label, centroids = img_quantization(16, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K16.jpg', final_img)

cluster, label, centroids = img_quantization(32, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K32.jpg', final_img)

cluster, label, centroids = img_quantization(64, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K64.jpg', final_img)

cluster, label, centroids = img_quantization(128, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Starry Night_K128.jpg', final_img)
