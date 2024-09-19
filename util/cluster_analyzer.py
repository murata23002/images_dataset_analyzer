import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class ClusterAnalyzer:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def cluster_features(self, features):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=50)
        labels = kmeans.fit_predict(features)
        return labels, kmeans.cluster_centers_

    def find_representative_samples(
        self, features, labels, cluster_centers, image_paths, n_samples=5
    ):
        representative_samples = []
        representative_paths = []
        for i in range(cluster_centers.shape[0]):
            cluster_features = features[labels == i]
            cluster_image_paths = image_paths[labels == i]
            distances = cdist(cluster_features, cluster_centers[i].reshape(1, -1))
            closest_indices = np.argsort(distances.ravel())[:n_samples]
            representative_samples.append(cluster_features[closest_indices])
            representative_paths.append(cluster_image_paths[closest_indices])
        representative_samples = np.vstack(representative_samples)
        representative_paths = np.hstack(representative_paths)
        return representative_samples, representative_paths
