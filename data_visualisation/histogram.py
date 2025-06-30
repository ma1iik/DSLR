import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_toolkit import Statistics, DataProcessor
import matplotlib.pyplot as plt
from collections import Counter

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'datasets', 'dataset_train.csv')
    data = DataProcessor.load_csv(dataset_path)
    # data = DataProcessor.load_csv("../datasets/dataset_train.csv")
    
    house_set = set()
    for row in data:
        house = row["Hogwarts House"]
        if house:
            house_set.add(house)
    house_list = list(house_set)

    names_by_house = {house: [] for house in house_list}
    for row in data:
        house = row["Hogwarts House"]
        if house in names_by_house:
            names_by_house[house].append(row)

    subjects = [
        "Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes",
        "Arithmancy", "Divination", "Muggle Studies", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    n_subjects = len(subjects)
    cols = 5
    rows = math.ceil(n_subjects / cols)

    plt.figure(figsize=(cols * 5, rows * 4))

    subject_meta_std = {}

    for idx, subject in enumerate(subjects, start=1):
        std_by_house = {}

        for house, students in names_by_house.items():
            scores = [
                float(student[subject])
                for student in students
                if DataProcessor.is_float(student[subject])
            ]
            if scores:
                std = Statistics.std(scores)
                std_by_house[house] = std

        # Calculate meta-std (std of stds across houses)
        if len(std_by_house) >= 2:
            meta_std = Statistics.std(list(std_by_house.values()))
            subject_meta_std[subject] = meta_std

        plt.subplot(rows, cols, idx)
        for house, students in names_by_house.items():
            scores = [
                float(student[subject])
                for student in students
                if DataProcessor.is_float(student[subject])
            ]
            if scores:
                plt.hist(scores, label=house, bins=5, alpha=0.5, edgecolor='black')

        plt.title(subject)
        plt.xlabel("Score")
        plt.ylabel("Students")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print("\n STD per subject (lower = more homogeneous between houses):")
    for subject, meta_std in sorted(subject_meta_std.items(), key=lambda x: x[1]):
        print(f"- {subject}: {meta_std:.3f}")

    most_homogeneous_subject = min(subject_meta_std, key=subject_meta_std.get)
    print(f"\n The most homogeneous subject across all houses is: **{most_homogeneous_subject}**")

if __name__ == "__main__":
    main()