import os

import pandas as pd


def save_features_to_csv(features, image_paths, output_dir, output_csv):
    df = pd.DataFrame(features)
    df["image_path"] = image_paths
    output_path = os.path.join(output_dir, output_csv)
    df.to_csv(output_path, index=False)
