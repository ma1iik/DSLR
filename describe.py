from ml_toolkit import Statistics, DataProcessor
import csv
import sys

def main():
	if len(sys.argv) != 2:
		print("Usage: python describe.py <csv_file>")
		return
	data = DataProcessor.load_csv(sys.argv[1])

if __name__ == '__main__':
	main()