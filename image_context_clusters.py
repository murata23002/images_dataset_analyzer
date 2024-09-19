import csv
import json

import matplotlib.pyplot as plt
from kneed import KneeLocator  # エルボー検出用ライブラリ
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# JSONデータを読み込む
with open("bow_output.json", "r") as f:
    data = json.load(f)

# Bag of Wordsを結合してテキスト形式に変換
corpus = [" ".join(item["bag_of_words"]) for item in data]

# TF-IDFベクトル化
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# エルボー法で最適なクラスタ数を見つける
distortions = []
K = range(1, 10)  # クラスタ数の範囲を指定
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_matrix)
    distortions.append(kmeans.inertia_)

# エルボーを検出
kneedle = KneeLocator(K, distortions, curve="convex", direction="decreasing")
optimal_clusters = kneedle.elbow

# エルボーが正しく検出されたかを確認
if optimal_clusters is None:
    print(
        "エルボーが検出されませんでした。クラスタ数の範囲を広げるか、手動でクラスタ数を設定してください。"
    )
    optimal_clusters = 20  # デフォルトのクラスタ数を設定（例: 3）

# 歪みをプロット
plt.figure(figsize=(10, 8))
plt.plot(K, distortions, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.title(f"Elbow Method For Optimal k (Optimal clusters = {optimal_clusters})")

if optimal_clusters is not None:
    plt.axvline(x=optimal_clusters, color="r", linestyle="--")

plt.savefig("elbow_method.png")
plt.show()

# 自動検出されたクラスタ数でK-Meansクラスタリングを実行
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# 各データのクラスターラベルを取得
labels = kmeans.labels_

# クラスタリング結果をCSVに出力
csv_file_path = "clustering_results.csv"
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Cluster"])
    for i, label in enumerate(labels):
        writer.writerow([data[i]["image_name"], label])

print(f"Clustering results have been saved to {csv_file_path}")

# クラスタリング結果の可視化（2次元プロット）
pca = PCA(n_components=2)
reduced_tfidf = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(10, 8))
for i in range(optimal_clusters):
    points = reduced_tfidf[labels == i]
    plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}")

plt.title("K-Means Clustering of Bag of Words")
plt.legend()
plt.savefig("kmeans_clustering.png")
plt.show()
