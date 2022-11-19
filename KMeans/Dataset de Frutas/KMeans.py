import numpy as np
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, k, X, n_iterations):
        self.k = k
        self.X = X
        self.n_iterations = n_iterations

    def fit(self):

        idx = np.random.choice(len(self.X), self.k, replace=False)
        centroids = self.X[idx, :]

        distances = np.linalg.norm(self.X[:, None, :] - centroids[None, :, :], axis=-1)
        points = np.array([np.argmin(i) for i in distances])

        for _ in range(self.n_iterations): 
            centroids = []
            for idx in range(self.k):
                temp_centroid = self.X[points==idx].mean(axis=0) 
                centroids.append(temp_centroid)
 
        centroids = np.vstack(centroids)
         
        distances = np.linalg.norm(self.X[:, None, :] - centroids[None, :, :], axis=-1)
        points = np.array([np.argmin(i) for i in distances])
         
        return points, centroids

    def predict(self, X, centroids):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        points = np.array([np.argmin(i) for i in distances])
        return points

    def plot(self, label, data, centroids):
        u_labels_test = np.unique(label)
        for i in u_labels_test:
            plt.scatter(data[label == i , 0] , data[label == i , 1], label = i)
        plt.scatter(centroids[:, 0], centroids[:, 1], c = 'k', marker = 'x', s = 100, label = 'Centroids')
        plt.title('Classification', fontsize = 14)    
        plt.legend()
        plt.show()
        