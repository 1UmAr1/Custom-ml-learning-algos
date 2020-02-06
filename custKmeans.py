# Creating our own kmeans alfo

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5)
plt.show()

colors = 10 * ["g", "r", "c", "b", "k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classificaion = distances.index(min(distances))
                self.classifications[classificaion].append(featureset)

            prev_centroids = dict(self.centroids)

            for classificaion in self.classifications:
                self.centroids[classificaion] = np.average(self.classifications[classificaion], axis=0)

            optimezied = True
            for c in self.centroids:
                orignal_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-orignal_centroid)/ orignal_centroid * 100.0) > self.tol:
                    optimezied = False

            if optimezied:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classificaion = distances.index(min(distances))
        return classificaion


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, marker="x", s=150, linewidths=5)
unknowns = np.array([[1, 6],
                     [4, 3],
                     [9, 1],
                     [1, 1],
                     [5, 8],
                     [6, 7],
                     [8, 1],
                     [2, 2]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], color=colors[classification], marker="*", s=150, linewidths=5)

plt.show()

