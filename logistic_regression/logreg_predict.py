import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolkit import DataProcessor as dp

def load_weights_and_normalization(filename):
    weights_dict = {}
    
    with open(filename, 'r') as f:
        current_house = None
        for line in f:
            line = line.strip()
            if '_bias:' in line:
                house = line.split('_bias:')[0]
                bias = float(line.split(':')[1])
                current_house = house
                weights_dict[house] = {'bias': bias}
            elif '_weights:' in line:
                weights_str = line.split(':')[1]
                weights = [float(w) for w in weights_str.split(',')]
                weights_dict[current_house]['weights'] = weights
            elif '_means:' in line:
                means_str = line.split(':')[1]
                means = [float(m) for m in means_str.split(',')]
                weights_dict[current_house]['means'] = means
            elif '_stds:' in line:
                stds_str = line.split(':')[1]
                stds = [float(s) for s in stds_str.split(',')]
                weights_dict[current_house]['stds'] = stds
    
    return weights_dict

def stable_sigmoid(z):
    z = max(min(z, 500), -500)
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (ez + 1)

def prepare_student_grades(student):
    subjects = ['Astronomy', 'Herbology', 'Ancient Runes', 'Defense Against the Dark Arts']
    grades = []
    for subject in subjects:
        if dp.is_float(student[subject]):
            grade = float(student[subject])
        else:
            grade = 0.0
        grades.append(grade)
    
    return grades

def normalize_student_grades(grades, means, stds):
    normalized = []
    for i in range(len(grades)):
        if stds[i] != 0:
            normalized.append((grades[i] - means[i]) / stds[i])
        else:
            normalized.append(0.0)
    return normalized

def predict_house(student_grades, all_weights):
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    probabilities = []
    
    for house in houses:
        bias = all_weights[house]['bias']
        weights = all_weights[house]['weights']
        means = all_weights[house]['means']
        stds = all_weights[house]['stds']
        
        normalized_grades = normalize_student_grades(student_grades, means, stds)
        
        score = bias
        for i in range(len(weights)):
            score += weights[i] * normalized_grades[i]
        
        probability = stable_sigmoid(score)
        probabilities.append(probability)
    
    max_index = probabilities.index(max(probabilities))
    predicted_house = houses[max_index]
    
    return predicted_house, probabilities

def test_on_training_data():
    weights = load_weights_and_normalization('all_house_weights.txt')
    train_data = dp.load_csv('datasets/dataset_train.csv')
    
    y_true = []
    y_pred = []
    
    for i, student in enumerate(train_data):
        student_grades = prepare_student_grades(student)
        predicted_house, probabilities = predict_house(student_grades, weights)
        actual_house = student['Hogwarts House']
        
        y_true.append(actual_house)
        y_pred.append(predicted_house)
    
    try:
        from sklearn.metrics import accuracy_score
        sklearn_accuracy = accuracy_score(y_true, y_pred)
        sklearn_accuracy_percent = sklearn_accuracy * 100
        return sklearn_accuracy_percent
    except ImportError:
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        manual_accuracy = (correct / len(y_true)) * 100
        return manual_accuracy

def predict_test_data():
    weights = load_weights_and_normalization('all_house_weights.txt')
    test_data = dp.load_csv('datasets/dataset_test.csv')
    
    predictions = []
    
    for i, student in enumerate(test_data):
        student_grades = prepare_student_grades(student)
        predicted_house, probabilities = predict_house(student_grades, weights)
        predictions.append(predicted_house)
    
    with open('houses.csv', 'w') as f:
        f.write("Index,Hogwarts House\n")
        for i, house in enumerate(predictions):
            f.write(f"{i},{house}\n")

if __name__ == "__main__":
    training_accuracy = test_on_training_data()
    predict_test_data()