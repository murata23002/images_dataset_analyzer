import pandas as pd


class FeatureMapSaver:
    @staticmethod
    def save_feature_map(feature_map, output_csv):
        df = pd.DataFrame(feature_map, columns=["dim1", "dim2"])
        df.to_csv(output_csv, index=False)
