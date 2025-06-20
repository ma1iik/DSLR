import math

class Statistics:
	@staticmethod
	def mean(data):
		return sum(data) / len(data)
	
	@staticmethod
	def std(data):
		return math.sqrt(Statistics.variance(data))
	
	@staticmethod
	def percentile(data, p):
		sorted_data = sorted(data)
		pos = (p / 100) * (len(data) - 1)
		if pos == int(pos):
			return sorted_data[int(pos)]
		lower_i = int(pos)
		upper_i = lower_i + 1
		fraction = pos - lower_i
		lower_val = sorted_data[lower_i]
		upper_val = sorted_data[upper_i]
		res = lower_val + fraction * (upper_val - lower_val)
		return res
	
	@staticmethod
	def min_max(data):
		return (min(data), max(data))
	
	@staticmethod
	def count_non_null(data):
		nn = [x for x in data  if x is not None and not math.isnan(x)]
		return nn
	
	@staticmethod
	def variance(data):
		total = sum((x - Statistics.mean(data)) ** 2 for x in data)
		variance = total / len(data)
		return variance
	
	@staticmethod
	def median(data):
		sorted_data = sorted(data)
		if len(data) % 2:
			return sorted_data[len(data) // 2]
		return ((sorted_data[len(data) // 2 - 1] + sorted_data[len(data) // 2]) / 2)
	
	@staticmethod
	def quartiles(data):
		sorted_data = sorted(data)
		middle_index = len(sorted_data) // 2 
		lower_half = sorted_data[:middle_index]
		upper_half = sorted_data[middle_index:]
		q1 = Statistics.median(lower_half)
		q3 = Statistics.median(upper_half)
		return (q1, q3)