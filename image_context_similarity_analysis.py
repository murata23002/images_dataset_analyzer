import argparse
import json

from tqdm import tqdm

# 引数を解析するための準備
parser = argparse.ArgumentParser(
    description="Compare Bag of Words sets using Jaccard and Dice similarity."
)
parser.add_argument(
    "--json_path",
    type=str,
    default="bow_output.json",
    help="Path to the JSON file containing bag of words.",
)
args = parser.parse_args()

# JSONデータを読み込む
with open(args.json_path, "r") as f:
    data = json.load(f)


# 1. Jaccard類似度の計算
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# 2. Dice係数の計算
def dice_coefficient(set1, set2):
    intersection = len(set1.intersection(set2))
    return (
        2 * intersection / (len(set1) + len(set2))
        if (len(set1) + len(set2)) != 0
        else 0
    )


# 類似度を計算
print("Calculating set-based similarities...")
similarities = {}
for i, item1 in enumerate(tqdm(data)):
    set1 = set(item1["bag_of_words"])
    for item2 in data[i + 1 :]:
        set2 = set(item2["bag_of_words"])

        jaccard_sim = jaccard_similarity(set1, set2)
        dice_sim = dice_coefficient(set1, set2)

        similarities[(item1["image_name"], item2["image_name"])] = {
            "jaccard_similarity": jaccard_sim,
            "dice_coefficient": dice_sim,
        }

# 結果の出力
n_top = 10  # 上位n個のペアを表示
print(f"\nTop {n_top} pairs by Jaccard similarity:")
top_jaccard = sorted(
    similarities.items(), key=lambda x: x[1]["jaccard_similarity"], reverse=True
)[:n_top]
for pair, sim in top_jaccard:
    print(f"Pair: {pair}")
    print(f"Jaccard Similarity: {sim['jaccard_similarity']:.4f}")
    print(f"Dice Coefficient: {sim['dice_coefficient']:.4f}")
    print()

print(f"\nTop {n_top} pairs by Dice coefficient:")
top_dice = sorted(
    similarities.items(), key=lambda x: x[1]["dice_coefficient"], reverse=True
)[:n_top]
for pair, sim in top_dice:
    print(f"Pair: {pair}")
    print(f"Jaccard Similarity: {sim['jaccard_similarity']:.4f}")
    print(f"Dice Coefficient: {sim['dice_coefficient']:.4f}")
    print()
