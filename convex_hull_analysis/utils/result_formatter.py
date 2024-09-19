class ResultFormatter:
    @staticmethod
    def format_results(in_indices, out_indices, image_paths_new):
        results = []

        for i in in_indices:
            results.append((i, "inside", image_paths_new[i]))

        for i in out_indices:
            results.append((i, "outside", image_paths_new[i]))

        return results
