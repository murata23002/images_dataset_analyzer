import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self):
        pass

    def plot_convex_hull_without_images(
        self, reduced_original, reduced_new, output_path, hull
    ):
        """
        Plot convex hull without images.

        :param reduced_original: The reduced original features to plot.
        :param reduced_new: The reduced new features to plot.
        :param output_path: The path to save the plot.
        :param hull: The convex hull object.
        """
        plt.figure(figsize=(10, 8))

        # Plot reduced original and new features
        plt.scatter(
            reduced_original[:, 0],
            reduced_original[:, 1],
            label="Original Features",
            color="blue",
            alpha=0.5,
        )
        plt.scatter(
            reduced_new[:, 0],
            reduced_new[:, 1],
            label="New Features",
            color="green",
            alpha=0.5,
        )

        # Plot the convex hull
        for simplex in hull.simplices:
            plt.plot(reduced_original[simplex, 0], reduced_original[simplex, 1], "k-")

        plt.title("Convex Hull without Images")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.show()

    def plot_features_with_images(
        self,
        reduced_original,
        reduced_new,
        original_image_paths,
        new_image_paths,
        output_path,
        hull,
        in_indices,
        out_indices,
    ):
        """
        Plot features with images.

        :param reduced_original: The reduced original features to plot.
        :param reduced_new: The reduced new features to plot.
        :param original_image_paths: The paths to the original images corresponding to the features.
        :param new_image_paths: The paths to the new images corresponding to the features.
        :param output_path: The path to save the plot.
        :param hull: The convex hull object.
        :param in_indices: Indices of points inside the convex hull.
        :param out_indices: Indices of points outside the convex hull.
        """
        all_features = np.vstack((reduced_original, reduced_new))
        all_image_paths = original_image_paths + new_image_paths
        plt.figure(figsize=(15, 10))

        # Plot reduced original and new features
        for i, (x, y) in enumerate(all_features):
            if i in in_indices:
                plt.scatter(
                    x,
                    y,
                    marker="o",
                    color="green",
                    label="Inliers" if i == in_indices[0] else "",
                    alpha=0.5,
                )
            elif i in out_indices:
                plt.scatter(
                    x,
                    y,
                    marker="x",
                    color="red",
                    label="Outliers" if i == out_indices[0] else "",
                    alpha=0.5,
                )
            else:
                plt.scatter(x, y, marker=".", color="blue", alpha=0.5)

        # Plot the convex hull
        for simplex in hull.simplices:
            plt.plot(reduced_original[simplex, 0], reduced_original[simplex, 1], "k-")

        plt.title("Features with Images")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True)

        # Annotate outliers with their image paths
        for idx in out_indices:
            x, y = all_features[idx]
            plt.annotate(
                all_image_paths[idx],
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="red",
            )

        plt.savefig(output_path)
        plt.show()
