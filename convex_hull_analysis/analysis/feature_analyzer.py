import numpy as np

from .convex_hull_analyzer import ConvexHullAnalyzer
from .dimensionality_reducer import DimensionalityReducer


class FeatureAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.reducer = DimensionalityReducer()
        self.hull_analyzer = ConvexHullAnalyzer()

    def get_in_out_indices(self, reduced_new, hull):
        new_results = [
            self.hull_analyzer.is_outside_hull(point, hull) for point in reduced_new
        ]

        in_indices = [index for index, result in enumerate(new_results) if not result]
        out_indices = [index for index, result in enumerate(new_results) if result]

        return in_indices, out_indices

    def analyze_features(self, features, new_features, method, n_neighbors=15):
        all_features = np.vstack((features, new_features))
        reduced_features = self.reducer.reduce(
            method, all_features, n_neighbors=n_neighbors
        )

        reduced_original = reduced_features[: len(features)]
        reduced_new = reduced_features[len(features) :]  # noqa: E203

        hull = self.hull_analyzer.compute_hull(reduced_original)

        return reduced_original, reduced_new, hull
