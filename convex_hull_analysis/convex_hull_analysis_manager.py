import os

import numpy as np

from .analysis.feature_analyzer import FeatureAnalyzer
from .utils.feature_map_saver import FeatureMapSaver
from .utils.result_formatter import ResultFormatter
from .visualization.visualizer import Visualizer


class ConvexHullAnalysisManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.analyzer = FeatureAnalyzer(output_dir)
        self.visualizer = Visualizer()
        self.formatter = ResultFormatter()
        self.feature_map_saver = FeatureMapSaver()

    def run_analysis(
        self,
        features,
        new_features,
        image_paths_original,
        image_paths_new,
        method="pca",
        n_neighbors=15,
    ):
        reduced_original, reduced_new, hull = self.analyzer.analyze_features(
            features, new_features, method, n_neighbors
        )

        in_indices, out_indices = self.analyzer.get_in_out_indices(reduced_new, hull)

        formatted_results = self.formatter.format_results(
            in_indices, out_indices, image_paths_new
        )
        all_reduced = np.vstack((reduced_original, reduced_new))

        self.feature_map_saver.save_feature_map(
            all_reduced, os.path.join(self.output_dir, f"{method}_feature_map.csv")
        )

        self.visualizer.plot_convex_hull_without_images(
            reduced_original,
            reduced_new,
            os.path.join(
                self.output_dir, f"convex_hull_plot_without_images_{method}.png"
            ),
            hull,
        )

        self.visualizer.plot_features_with_images(
            reduced_original,
            reduced_new,
            image_paths_original,
            image_paths_new,
            os.path.join(self.output_dir, f"features_with_images_{method}.png"),
            hull,
            in_indices,
            out_indices,
        )

        return formatted_results
