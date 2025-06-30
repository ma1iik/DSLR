import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_toolkit import DataProcessor
import matplotlib.pyplot as plt

def manual_pair_plot(data_by_house, features, colors=None):
    num_features = len(features)
    houses = list(data_by_house.keys())
    if colors is None:
        colors = {
            house: color for house, color in zip(houses, ["red", "blue", "green", "orange"])
        }

    fig, axes = plt.subplots(num_features, num_features, figsize=(4*num_features, 4*num_features))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i][j]
            feat_x = features[j]
            feat_y = features[i]

            if i == j:
                for house in houses:
                    values = data_by_house[house][feat_x]
                    ax.hist(values, alpha=0.6, color=colors[house], label=house)
                if i == num_features - 1:
                    ax.set_xlabel(feat_x, fontsize=10)
                if j == 0:
                    ax.set_ylabel(feat_y, fontsize=10)
            else:
                for house in houses:
                    points = [
                        (x, y)
                        for x, y in zip(data_by_house[house][feat_x], data_by_house[house][feat_y])
                        if isinstance(x, float) and isinstance(y, float)
                    ]
                    if points:
                        xs, ys = zip(*points)
                        ax.scatter(xs, ys, alpha=0.6, color=colors[house], label=house)
                if i == num_features - 1:
                    ax.set_xlabel(feat_x, fontsize=10)
                if j == 0:
                    ax.set_ylabel(feat_y, fontsize=10)

            if i == 0 and j == num_features - 1:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='upper left', fontsize=8)


    plt.suptitle("Manual Pair Plot of Features Grouped by House", fontsize=16)
    plt.savefig("manual_pair_plot.png", dpi=300, bbox_inches='tight')
    # plt.show()


def extract_numeric_data_by_house(data, features):
    result = {}

    for row in data:
        house = row.get("Hogwarts House")
        if not house:
            continue

        if house not in result:
            result[house] = {feature: [] for feature in features}

        for feature in features:
            try:
                value = float(row.get(feature, ""))
                result[house][feature].append(value)
            except (ValueError, TypeError):
                continue  

    return result

def main():
    features = ["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes", "Arithmancy", "Divination" , "Muggle Studies", "History of Magic", "Transfiguration" , "Potions" , "Care of Magical Creatures", "Charms", "Flying"] 
    # data = DataProcessor.load_csv("../datasets/dataset_train.csv")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'datasets', 'dataset_train.csv')
    data = DataProcessor.load_csv(dataset_path)
    grouped = extract_numeric_data_by_house(data, features)
    manual_pair_plot(grouped, features)

if __name__ == "__main__":
    main()