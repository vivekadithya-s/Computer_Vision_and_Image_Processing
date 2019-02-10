UBIT = 'vivekadi'
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(sum([ord(c) for c in UBIT]))
MatchThreshold = 10
import matplotlib.pyplot as plt
from matplotlib import style

import matplotlib
import numpy as np
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
#% matplotlib
#inline
colors = ['r', 'g', 'b', 'y', 'c', 'm']

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')








class KMeans_custom():
    def __init__(self, num_clusters, tolerance=0.0001, epochs=30, centroids={}):
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.epochs = epochs
        self.centroids = centroids

    def get_centroid_dist(self, data):
        classification = list()
        for point in data:
            distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
            classification.append(distances.index(min(distances)))
        return np.array(classification)


    def fit(self, data):
        if self.centroids == {}:
            for i in range(self.num_clusters):
                self.centroids[i] = data[i]
                #print(data[i])
            # centroids = np.random.choice(data, self.num_clusters)
            # for i in range(len(centroids)):
            #     self.centroids[i] = centroids[i]

        for i in range(self.epochs):
            self.classifications = {}

            for cluster in range(self.num_clusters):
                self.classifications[cluster] = []

            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                #print(len(distances))
                #print(self.centroids[centroid])
                classification = distances.index(min(distances))
                '''if (classification == 2):
                    print(classification)'''

                self.classifications[classification].append(features)


            prev_centroid = dict(self.centroids)


            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for cent in self.centroids:
                original_centroid = prev_centroid[cent]
                current_centroid = self.centroids.get(cent)
                if np.sum(((original_centroid - current_centroid) * 100) * original_centroid) > self.tolerance:
                    optimized = False

            if optimized:
                break













def image_formation(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h, 3))
    image = np.array(image, dtype=np.float64)
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
           # image[i][j] = image[i][j] * 255
            label_idx += 1
    #print(image[30][30])
    print("Image created!")
    return image

def baboon_quantization(num_colors):
    image_matrixx = cv2.imread('data/baboon.jpg')
    image_matrix = cv2.cvtColor(image_matrixx, cv2.COLOR_BGR2RGB)
    #print(image_matrix[511][2][1])
    # Normalization
    image = np.array(image_matrix, dtype=np.float64) / 255
    #print(type(image))
    #print(image[511][2][1])
    #plt.imshow(image, interpolation='nearest')
    #plt.show()
    #plt.imshow(image_matrix, interpolation='nearest')
   # plt.show()

    #print(image)
    w, h, d = tuple(image.shape)
    image_array = np.reshape(image, (w * h, d))
    #print(image_array.shape)
    # Initialization K_Means class object
    image_quantization = KMeans_custom(num_clusters=num_colors, epochs=30)
    image_quantization.fit(image_array)
    labels = image_quantization.get_centroid_dist(image_array)
    print("\nImage quantization centroids are:")
    print(image_quantization.centroids)
    new_image = image_formation(image_quantization.centroids, labels, w, h)
    matplotlib.image.imsave('OutputTask3/task3_baboon_{}.png'.format(num_colors), new_image)













f1 = [5.9, 4.6, 6.2, 4.7, 5.5, 5.0, 4.9, 6.7, 5.1, 6.0]
f2 = [3.2, 2.9, 2.8, 3.2, 4.2, 3.0, 3.1, 3.1, 3.8, 3.0]
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# Number of clusters
k = 3
# X coordinates of random centroids
C_x = [6.2, 6.6, 6.5]

# Y coordinates of random centroids
C_y = [3.2, 3.7, 3.0]
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
# print(C)
# print(type(C))


# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', marker='^', s=150)
# plt.scatter(C_x, C_y, marker='o', s=200, c='b')
plt.scatter(C[0, 0], C[0, 1], marker='o', s=200, c=colors[0])
plt.scatter(C[1, 0], C[1, 1], marker='o', s=200, c=colors[1])
plt.scatter(C[2, 0], C[2, 1], marker='o', s=200, c=colors[2])
plt.savefig('OutputTask3/task3_iter1_a.png')

#plt.show()
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
colors = ['r', 'g', 'b', 'y', 'c', 'm']

count = 0
# Loop will run till the error becomes zero
while error != 0:
    count = count + 1
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = np.copy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    fig, axx = plt.subplots()
    axx.scatter(f1, f2, marker='^', s=150, edgecolor='black', facecolor='none')
    axx.scatter(C[0, 0], C[0, 1], marker='o', s=200, c=colors[0])
    axx.scatter(C[1, 0], C[1, 1], marker='o', s=200, c=colors[1])
    axx.scatter(C[2, 0], C[2, 1], marker='o', s=200, c=colors[2])
    plt.savefig('OutputTask3/task3_iter{}_b.png'.format(count))

    error = dist(C, C_old, None)
    fig, ax = plt.subplots()

    if (count > 1):
        break
    for i in range(3):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], marker='^', s=150, c=colors[i])
        ax.scatter(C[0, 0], C[0, 1], marker='o', s=200, c=colors[0])
        ax.scatter(C[1, 0], C[1, 1], marker='o', s=200, c=colors[1])
        ax.scatter(C[2, 0], C[2, 1], marker='o', s=200, c=colors[2])
    plt.savefig('OutputTask3/task3_iter{}_a.png'.format(count+1))

    if (error == 0):
        print("Final centroids are:")
        print(C)
        print("Please check OutputTask3/iters{}.png file to check the iterations")



baboon_quantization(5)




