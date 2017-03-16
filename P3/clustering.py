import numpy as np
import cv2


def read_data(filename):    #To read the data from the file
    f = open(filename, 'r')
    data = []
    classification = []
    for l in f:
        line = l.split(',')
        if not line[-1] in classification:  #Swap every classification with a number
            classification.append(line[-1])
        line[-1] = classification.index(line[-1])
        data.append([float(i) for i in line])
    data = np.array(data)
    return data

def equal_centroids(a, b):  #To compare centroids
    return set([tuple(i) for i in np.float32(a)]) == set([tuple(i) for i in np.float32(b)])

def clustering(k, data):    #Clustering algorithm
    it = 0
    new_centroids = []  #Initialize 2 centroids as empty list
    old_centroids = []
    while it < k:   #Then picking k DIFFERENT centers
        sample = data[np.random.choice(data.shape[0], 1)]
        if not list(sample[0]) in new_centroids:    #That's why we check on the list if it's already there
            new_centroids.append(list(sample[0]))   #If it isn't, add it
            it +=1  #We keep doing this till we have k different centroids

    while it < k:   #We do this twice, go get two centroids to start the algorithm, this is for convergence purposes
        sample = data[np.random.choice(data.shape[0], 1)]
        if not list(sample[0]) in old_centroids:
            old_centroids.append(list(sample[0]))
            it +=1

    while not equal_centroids(new_centroids, old_centroids):    #Then we do k-means still convergence
        old_centroids = new_centroids
        cluster = [[] for i in range(k)]    #Initialize an empty cluster with k clusters
        for row in data:    #Then for every row in the data
            group = 0   #We assume it belongs on the first group
            old_min = np.linalg.norm(old_centroids[0][:-1]-row[:-1])    #Check the distance to the first centroid
            for i in enumerate(old_centroids):  #Then we check all the others
                new_min = np.linalg.norm(old_centroids[i[0]][:-1]-row[:-1]) #Notice how the last index of every row is technically the clasiffication, this is irrelevant for the k-means method
                if old_min > new_min:   #If it's closer
                    group = i[0]    #Update its group
                    old_min = new_min   #And its distance
            cluster[group].append(row)  #Finally add it to the cluster it belongs to
        new_centroids = []  #Lastly we update the centroids
        for i in cluster:
            new_centroids.append(np.mean(i, 0)) #With the mean of its cluster
    return cluster

def img_quantization(k, img, iterations):   #Due to the way the data had already classification attributes on the data that was worked with, we need a different k-means method for images
    it = 0
    new_centroids = []
    old_centroids = []
    while it < k:   #The principle is the same though
        sample = img[np.random.choice(img.shape[0], 1)]
        if not list(sample[0]) in new_centroids:
            new_centroids.append(list(sample[0]))
            it +=1

    it = 0
    while it < k:
        sample = img[np.random.choice(img.shape[0], 1)]
        if not list(sample[0]) in old_centroids:
            old_centroids.append(list(sample[0]))
            it +=1

    new_centroids = np.array(new_centroids)
    old_centroids = np.array(old_centroids)
    it = 0
    while not equal_centroids(new_centroids, old_centroids) and it < iterations:    #The only major difference is, we don't wait for convergence, we also have max iterations
        old_centroids = new_centroids
        cluster = [[] for i in range(k)]
        label = []
        for row in img:
            group = 0
            old_min = np.linalg.norm(old_centroids[0]-row)  #And also consider the whole row, not excluding the last attribute
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

img = cv2.imread('Wanderer above the Sea of Fog.jpg')
new_img = img.reshape((-1,3))
new_img = np.float32(new_img)

cluster, label, centroids = img_quantization(2, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K2.jpg', final_img)

cluster, label, centroids = img_quantization(4, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K4.jpg', final_img)

cluster, label, centroids = img_quantization(8, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K8.jpg', final_img)

cluster, label, centroids = img_quantization(16, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K16.jpg', final_img)

cluster, label, centroids = img_quantization(32, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K32.jpg', final_img)

cluster, label, centroids = img_quantization(64, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K64.jpg', final_img)

cluster, label, centroids = img_quantization(128, new_img, 10)
centroids = np.uint8(centroids)
final_img = centroids[label]
final_img = final_img.reshape((img.shape))
cv2.imwrite('Wanderer above the Sea of Fog_K128.jpg', final_img)
