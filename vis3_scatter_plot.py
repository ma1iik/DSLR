from ml_toolkit import Statistics as st 
from ml_toolkit import DataProcessor as dp
import matplotlib.pyplot as plt
import math

def calc_correlation(subject1, subject2, data):
	valid_pairs = dp.get_aligned_pairs(data, subject1, subject2)
	
	x_data = [pair[0] for pair in valid_pairs]
	y_data = [pair[1] for pair in valid_pairs]
	
	mean_x = st.mean(x_data)
	mean_y = st.mean(y_data)
	std_x = st.std(x_data)
	std_y = st.std(y_data)
	
	# prevention of division by zero further down in the correlation formula
	if std_x == 0 or std_y == 0:
		return 0
	#  Pearson Correlation Coefficient formula
	n = len(x_data)
	numerator = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
	correlation = numerator / ((n - 1) * std_x * std_y)
	# Pearson Correlation Coefficient lqys between -1 and 1, close to -1 is strong negative, close to zero is small to none, and close to 1 is strong positive
	return correlation

def plot_correlation(data, subject1, subject2, correlation):
	"""Create scatter plot for the most correlated pair"""
	
	house_colors = {
		'Gryffindor': 'red',
		'Slytherin': 'green', 
		'Ravenclaw': 'blue',
		'Hufflepuff': 'orange'
	}
	
	house_data = {'Gryffindor': [], 'Slytherin': [], 'Ravenclaw': [], 'Hufflepuff': []}
	
	for row in data:
		try:
			x_score = float(row[subject1])
			y_score = float(row[subject2])
			house = row["Hogwarts House"]
			if house in house_data:
				house_data[house].append((x_score, y_score))
		except (ValueError, TypeError, KeyError):
			continue

	plt.figure(figsize=(12, 8))
	for house, points in house_data.items():
		if points:
			x_vals = [p[0] for p in points]
			y_vals = [p[1] for p in points]
			plt.scatter(x_vals, y_vals, c=house_colors[house], label=house, alpha=0.7, s=40)
	plt.xlabel(subject1, fontsize=14)
	plt.ylabel(subject2, fontsize=14)
	plt.title(f'{subject1} vs {subject2}\nCorrelation: {correlation:.6f}', fontsize=16)
	plt.legend(fontsize=12)
	plt.grid(True, alpha=0.3)
	plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
	plt.show()

def find_most_correlated_features():
	data = dp.load_csv('datasets/dataset_train.csv')
	num_cols = dp.get_num_cols(data)
	subjects = [col for col in num_cols if col != "Index"]
	
	max_correlation = 0
	best_pair = None
	correlation_results = []
	
	for i in range(len(subjects)):
		for j in range(i + 1, len(subjects)):
			subject1 = subjects[i]
			subject2 = subjects[j]
			
			correlation = calc_correlation(subject1, subject2, data)
			
			valid_count = len(dp.get_aligned_pairs(data, subject1, subject2))
			
			correlation_results.append({
				'subject1': subject1,
				'subject2': subject2,
				'correlation': correlation,
				'abs_correlation': abs(correlation),
				'data_points': valid_count
			})
			
			# Track highest absolute correlation AND pair
			if abs(correlation) > abs(max_correlation):
				max_correlation = correlation
				best_pair = (subject1, subject2)

	# Sort by absolute correlation (highest first)
	correlation_results.sort(key=lambda x: x['abs_correlation'], reverse=True)

	print("Top 10 Most Correlated Feature Pairs:")
	print("-" * 95)
	print(f"{'Rank':<4} {'Subject 1':<30} {'Subject 2':<30} {'Correlation':<12} {'Points':<8}")
	print("-" * 95)

	for i, result in enumerate(correlation_results[:10]):
		subj1_display = result['subject1'][:28] + ".." if len(result['subject1']) > 30 else result['subject1']
		subj2_display = result['subject2'][:28] + ".." if len(result['subject2']) > 30 else result['subject2']
		
		print(f"{i+1:<4} {subj1_display:<30} {subj2_display:<30} "
			f"{result['correlation']:<12.6f} {result['data_points']:<8}")

	print("-" * 95)
	
	# Plot only the MOST correlated pair
	if best_pair:
		subject1, subject2 = best_pair
		if max_correlation > 0: 
			type = 'positive'
		else:
			type = 'negative'
		print(f'{subject1} and {subject2} have a {type} coorelation of: {max_correlation:.2f}.')
		if abs(max_correlation) > 0.8:
			print("This is a VERY STRONG correlation!")
		elif abs(max_correlation) > 0.6:
			print("This is a STRONG correlation!")
		elif abs(max_correlation) > 0.4:
			print("This is a MODERATE correlation!")
		else:
			print("This is a WEAK correlation.")

		plot_correlation(data, subject1, subject2, max_correlation)

		

if __name__ == '__main__':
	find_most_correlated_features()