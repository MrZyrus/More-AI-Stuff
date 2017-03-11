import numpy as np


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
    return set([tuple(i) for i in a]) == set([tuple(i) for i in b])

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

data = read_data("iris.data.txt")
cluster = clustering(2, data)

print("Using 2-means\n")
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
