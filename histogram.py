from ml_toolkit import Statistics, DataProcessor
import matplotlib.pyplot as plt

houses = {}
std_by_house = {}

def main():
    data = DataProcessor.load_csv("datasets/dataset_train.csv") # Load dataset from CSV
    houses = set()  # sets automatically avoid duplicates
    for row in data:
        house = row["Hogwarts House"]
        if house:  # skip empty values
            houses.add(house)    
    # Convert the set to a list for further use    
    house_list=list(houses)
    #  Group student rows by house into a dictionary
    # Now group by house {'Gryffindor': [row1, row2, ...], 'Slytherin': [rowX, rowY, ...], ...}
    names_by_house = {house: [] for house in house_list}  # dict with house name as key

    for row in data:
        house = row["Hogwarts House"]
        if house in names_by_house:
            names_by_house[house].append(row)
    # Define the subjects to analyze
    subjects = ["Astronomy", "Herbology", "Defense Against the Dark Arts"]    
    # Plot all houses' histograms on the same figure Create a single figure with subplots (1 row, N columns — one per subject)
    plt.figure(figsize=(15, 5))
    for idx, subject in enumerate(subjects, start=1):
        # plt.subplot(1, len(subjects), idx)  # 1 row, N columns: Create a subplot for each subject
        # Calculate standard deviation per house
        for house, students in names_by_house.items():
            scores = [
                float(student[subject])
                for student in students
                if DataProcessor.is_float(student[subject])    
            ]
            if scores:
                std = Statistics.std(scores)
                std_by_house[house] = std

    #Find the house with the lowest standard deviation
    most_homogeneous_house = min(std_by_house, key=std_by_house.get)
    for idx, subject in enumerate(subjects, start=1):
        plt.subplot(1, len(subjects), idx)  # 1 row, N columns: Create a subplot for each subject
        # Calculate standard deviation per house
        # Plot each house’s histogram
        for house, students in names_by_house.items():
            scores = [
                float(student[subject])
                for student in students
                if DataProcessor.is_float(student[subject])
            ]
            if scores:
                label = f"{house}* (most homog.)" if house == most_homogeneous_house else house
                plt.hist(scores, label=label, bins=5, alpha=0.5, edgecolor='black')

        # Add title and labels
        plt.title(f"{subject} (Most homogeneous: {most_homogeneous_house})")
        plt.xlabel("Score")
        plt.ylabel("Number of Students")
        plt.legend()
        plt.grid(True)
        plt.show()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()