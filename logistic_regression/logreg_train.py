import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolkit import DataProcessor as dp
from ml_toolkit import Statistics as st

def normalize_features(all_stud_grades):
	if not all_stud_grades:
		return all_stud_grades, [], []
	num_feats = len(all_stud_grades[0])
	means = []
	stds = []
	for i in range(num_feats):
		feat_grades = [grades[i] for grades in all_stud_grades]
		mean_val = st.mean(feat_grades)
		std_val = st.std(feat_grades)
		means.append(mean_val)
		stds.append(std_val)
	normalized_d = []
	# x_normalized = (x - μ) / σ
	for grades in all_stud_grades:
		normalized = [(grades[i] - means[i]) / stds[i] for i in range(num_feats)]
		normalized_d.append(normalized)
	return normalized_d, means, stds

def load_and_prep_data(target_house):
	data = dp.load_csv('datasets/dataset_train.csv')
	subjects = ['Astronomy', 'Herbology', 'Ancient Runes', 'Defense Against the Dark Arts']
	all_labels = []
	all_stud_grades = []
	for student in data:
		if student['Hogwarts House'] == target_house:
			label = 1
		else:
			label = 0
		all_labels.append(label)
		stud_grades = []
		valid_st = True
		for subject in subjects:
			if dp.is_float(student[subject]):
				grade = float(student[subject])
				stud_grades.append(grade)
			else:
				valid_st = False
				break
		if valid_st:
			all_stud_grades.append(stud_grades)
		else:
			all_labels.pop()
	return all_stud_grades, all_labels

def stable_sigmoid(z):
	# σ(z) = 1 / (1 + e^(-z))
	z = max(min(z, 500), -500)
	if z >= 0:
		return 1 / (1 + math.exp(-z))
	else:
		ez = math.exp(z)
		return ez / (ez + 1)

def make_pred(stud_grades, weights, bias):
	score = bias  # θ₀
	for i in range(len(weights)):
		score += weights[i] * stud_grades[i]  # θᵢ * xᵢ 
	# Hypothesis: h_θ(x) = σ(z) = 1 / (1 + e^(-z))
	pred = stable_sigmoid(score)
	return pred

def train_house(target_house):
	all_stud_grades, all_labels = load_and_prep_data(target_house)
	all_stud_grades, means, stds = normalize_features(all_stud_grades)
	bias = 0.0
	weights = [0.0] * 4
	learning_rate = 0.1
	prev_error = float('inf')
	# Gradient Descent
	for epoch in range(2000):
		total_bias_gradient = 0
		total_weight_gradients = [0] * 4
		total_error = 0
		for student_num in range(len(all_stud_grades)):
			stud_grades = all_stud_grades[student_num]	# x^(i)
			actual_label = all_labels[student_num]			# y^(i)
			# Forward pass: h_θ(x^(i))
			pred = make_pred(stud_grades, weights, bias)
			# Error term: (h_θ(x^(i)) - y^(i))
			error = pred - actual_label
			# Cost function: J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))²
			total_error += error * error
			# Gradient computation:
			# ∂J/∂θ₀ = (1/m) * Σ(h_θ(x^(i)) - y^(i))
			total_bias_gradient += error
			# ∂J/∂θⱼ = (1/m) * Σ(h_θ(x^(i)) - y^(i)) * x_j^(i)
			for i in range(4):
				total_weight_gradients[i] += error * stud_grades[i]
		# Batch gradient descent update
		num_students = len(all_stud_grades)  # m
		# Update rule: θ := θ - α * ∂J/∂θ
		# θ₀ := θ₀ - α * (1/m) * Σ(h_θ(x^(i)) - y^(i))
		bias -= learning_rate * (total_bias_gradient / num_students)
		# θⱼ := θⱼ - α * (1/m) * Σ(h_θ(x^(i)) - y^(i)) * x_j^(i)
		for i in range(4):
			weights[i] -= learning_rate * (total_weight_gradients[i] / num_students)
		# Cost function value: J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))²
		avg_error = total_error / num_students
		# Convergence check: |J_prev - J_current| < ε
		if abs(prev_error - avg_error) < 0.000001:
			break
		prev_error = avg_error
		# Learning rate decay: α := α * decay_factor
		if epoch > 0 and epoch % 500 == 0:
			learning_rate *= 0.8
	return weights, bias, means, stds

def save_all_weights(all_trained_weights, filename):
	# Save trained parameters θ = {θ₀, θ₁, ..., θₙ} and normalization parameters
	with open(filename, 'w') as f:
		for house, (weights, bias, means, stds) in all_trained_weights.items():
			f.write(f"{house}_bias:{bias}\n")
			f.write(f"{house}_weights:{','.join(map(str, weights))}\n")
			f.write(f"{house}_means:{','.join(map(str, means))}\n")
			f.write(f"{house}_stds:{','.join(map(str, stds))}\n")

if __name__ == "__main__":
	houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
	all_trained_weights = {}
	
	for house in houses:
		weights, bias, means, stds = train_house(house)
		all_trained_weights[house] = (weights, bias, means, stds)
	
	save_all_weights(all_trained_weights, 'all_house_weights.txt')