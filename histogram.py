from ml_toolkit import DataProcessor
import matplotlib.pyplot as plt

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
        plt.subplot(1, len(subjects), idx)  # 1 row, N columns: Create a subplot for each subject
        # Plot histogram of scores for each house
        for house, students in names_by_house.items():
            # Extract float-convertible scores from student data
            scores = [
                float(student[subject])
                for student in students
                if DataProcessor.is_float(student[subject])  # Avoid empty or non-float values
            ]
                        # Plot histogram for this house in this subject
            plt.hist(scores, label=house, bins=5, alpha=0.5, edgecolor='black')

        # Add labels, legend, and show the plot
        plt.title(subject)
        plt.xlabel("Score")
        plt.ylabel("Number of Students")    
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()