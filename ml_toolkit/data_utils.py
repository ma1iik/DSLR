import csv
import math

class DataProcessor:
	@staticmethod
	def load_csv(filename):
		try:
			with open(filename, 'r') as file:
				reader = csv.DictReader(file)
				data = list(reader)
				return data
		except FileNotFoundError:
			print(f"Error: File '{filename}' not found")
			return None
	
	@staticmethod
	def get_num_cols(data):
		if not data:
			return []
		column_names = data[0].keys()
		sample = DataProcessor.sample_data(data)
		numeric_columns = []
		for column_name in column_names:
			col = DataProcessor.extract_column(sample, column_name)
			if DataProcessor.col_is_numeric(col):
				numeric_columns.append(column_name)
		return numeric_columns
			

	@staticmethod
	def col_is_numeric(col):
		num_count = 0
		valid = 0
		for x in col:
			if x is not None and x != '':
				valid += 1
				try:
					float_val = float(x)
					if not math.isnan(float_val):
						num_count += 1
				except (ValueError, TypeError):
					pass
		if valid == 0:
			return False
		return num_count == valid

	@staticmethod
	def extract_column(data, column_name):
		col = [row[column_name] for row in data]
		return col

	@staticmethod
	def clean_num_data(col_data):
		cleaned = []
		for val in col_data:
			if val is not None and val != '':
				try:
					float_val = float(val)
					if not math.isnan(float_val):
						cleaned.append(float_val)
				except (ValueError, TypeError):
					pass
		return cleaned

	@staticmethod
	def sample_data(data, sample_size=10):
		sample = data[:sample_size]
		return sample
	
	@staticmethod
	def is_float(value):
		try:
			float(value)
			return True
		except (ValueError, TypeError):
			return False
	
	@staticmethod
	def is_column_exist(value, check):
		if not value:
			return True
		try:
			for raw in value:
				if raw == check:
					return False
		except (ValueError, TypeError):
			return True	