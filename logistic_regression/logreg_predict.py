import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolkit import DataProcessor as dp

def read_model_data(file):
	house_data = {}
	with open(file, 'r') as f:
		current = None
		for line in f:
			line = line.strip()
			if '_bias:' in line:
				house = line.split('_bias:')[0]
				bias = float(line.split(':')[1])
				current = house
				house_data[house] = {'bias': bias}
			elif '_weights:' in line:
				weights_str = line.split(':')[1]
				weights = [float(w) for w in weights_str.split(',')]
				house_data[current]['weights'] = weights
			elif '_means:' in line:
				means_str = line.split(':')[1]
				means = [float(m) for m in means_str.split(',')]
				house_data[current]['means'] = means
			elif '_stds:' in line:
				stds_str = line.split(':')[1]
				stds = [float(s) for s in stds_str.split(',')]
				house_data[current]['stds'] = stds
	return house_data

def sigmoid(x):
	x = max(min(x, 500), -500)
	if x >= 0:
		return 1 / (1 + math.exp(-x))
	else:
		exp_x = math.exp(x)
		return exp_x / (exp_x + 1)

def get_grades(student):
	subjects = ['Astronomy', 'Herbology', 'Ancient Runes', 'Defense Against the Dark Arts']
	grades = []
	for subject in subjects:
		if dp.is_float(student[subject]):
			grade = float(student[subject])
		else:
			grade = 0.0
		grades.append(grade)
	return grades

def normalize(grades, means, stds):
	result = []
	for i in range(len(grades)):
		if stds[i] != 0:
			result.append((grades[i] - means[i]) / stds[i])
		else:
			result.append(0.0)
	return result

def classify_student(grades, model_data):
	houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
	scores = []
	
	for house in houses:
		bias = model_data[house]['bias']
		weights = model_data[house]['weights']
		means = model_data[house]['means']
		stds = model_data[house]['stds']
		
		normalized = normalize(grades, means, stds)
		#  z = θ₀ + θ₁·x₁ + θ₂·x₂ + θ₃·x₃ + θ₄·x₄ ...
		score = bias
		for i in range(len(weights)):
			score += weights[i] * normalized[i] 
		
		prob = sigmoid(score) #  sigmoid(z) = 1 / (1 + e^(-z))
		scores.append(prob)
	
	best_match = scores.index(max(scores))
	return houses[best_match], scores

def run_predictions():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	dataset_path = os.path.join(project_root, 'datasets', 'dataset_test.csv')
	students = dp.load_csv(dataset_path)
	model = read_model_data(os.path.join(project_root, 'all_house_weights.txt'))
	# students = dp.load_csv('datasets/dataset_test.csv')
	
	results = []
	for i, student in enumerate(students):
		grades = get_grades(student)
		house, probs = classify_student(grades, model)
		results.append(house)
		output_path = os.path.join(project_root, 'houses.csv')
		with open(output_path, 'w') as f:
			f.write("Index,Hogwarts House\n")
			for i, house in enumerate(results):
				f.write(f"{i},{house}\n")

if __name__ == "__main__":
	run_predictions()