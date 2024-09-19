import numpy as np
from scipy.spatial import ConvexHull


class ConvexHullAnalyzer:
    @staticmethod
    def compute_hull(reduced_features):
        return ConvexHull(reduced_features)

    @staticmethod
    def is_outside_hull(point, hull):
        return not np.all(
            np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0
        )
