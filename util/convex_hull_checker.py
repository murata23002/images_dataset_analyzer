import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


class ConvexHullChecker:
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)
        self.hull = None

    def fit(self, features):
        reduced_features = self.pca.fit_transform(features)
        self.hull = ConvexHull(reduced_features)

    def check_outside(self, new_data):
        reduced_new_data = self.pca.transform(new_data)
        outside_flags = [
            not np.all(
                np.dot(self.hull.equations[:, :-1], point) + self.hull.equations[:, -1]
                <= 0
            )
            for point in reduced_new_data
        ]
        results = [
            (i, "outside" if flag else "inside") for i, flag in enumerate(outside_flags)
        ]
        return results


# 実行スクリプトとして使用する場合
if __name__ == "__main__":
    # ダミーデータの作成
    np.random.seed(42)  # 再現性のためのシード設定
    features = np.random.rand(100, 50)  # 既存の特徴量データ
    new_data = np.random.rand(10, 50)  # 新しいデータ

    checker = ConvexHullChecker(n_components=2)
    checker.fit(features)
    results = checker.check_outside(new_data)

    # 結果を表示
    for result in results:
        print(f"Data point {result[0]} is {result[1]}")
