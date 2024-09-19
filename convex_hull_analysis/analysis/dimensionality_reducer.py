import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReducer:
    @staticmethod
    def reduce(method, features, n_components=2, n_neighbors=15):
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        elif method == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
        else:
            raise ValueError("Method should be 'pca', 'tsne', or 'umap'")
        return reducer.fit_transform(features)
