from ml_toolkit import Statistics as st 
from ml_toolkit import DataProcessor as dp
import csv
import sys
import os

def find_fitting_columns(stats_dict):
	col_names = list(stats_dict.keys())
	max_terminal_width = os.get_terminal_size().columns
	print(f'TERMINAL: {max_terminal_width}')
	col_widths = [stats_dict[name]['max_width'] for name in col_names]
	fitting_cols = {}
	stats_width = 5

	if (sum(e['max_width'] + 2 for e in stats_dict.values()) + stats_width < max_terminal_width):
		return stats_dict
	else:
		max_edge_cols = len(col_names) // 2
		if 2 * max_edge_cols == len(col_names):
			max_edge_cols -= 1
		for num_edge_cols in range(max_edge_cols, 0, -1):
			left_widths = col_widths[:num_edge_cols]
			right_widths = col_widths[-num_edge_cols:]

			total_w = sum(left_widths) + sum(right_widths) + 6 + 3 + (num_edge_cols * 2)
			if total_w <= max_terminal_width:
				fitting_cols = {}
				for i in range(num_edge_cols):
					col_name = col_names[i]
					fitting_cols[col_name] = stats_dict[col_name]
				fitting_cols['...'] = {
				'count': '...',
				'mean': '...',
				'std': '...',
				'min': '...',
				'25%': '...',
				'50%': '...',
				'75%': '...',
				'max': '...',
				'max_width': 3
				}
				for i in range(-num_edge_cols, 0):
					col_name = col_names[i]
					fitting_cols[col_name] = stats_dict[col_name]
				return fitting_cols
	return {}
		



def print_describe(stats_dict):
	fitting_stats_dict = find_fitting_columns(stats_dict)
	stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
	headers = f"{'':5}"
	for col_name in fitting_stats_dict:
		width = fitting_stats_dict[col_name]['max_width'] + 2
		headers += f"{col_name:>{width}}"
	# last_spaces = (os.get_terminal_size().columns - len(headers)) * ''
	# headers	+= last_spaces
	print(headers)
	for stat in stats:
		line = f'{stat:<5}'
		for col_name, col_data in fitting_stats_dict.items():
			if col_name == '...':
				stat_value = '  ...'
			else:
				decimal_places = 6 if col_name != 'Index' else 5
				stat_value = f"{col_data[stat]:.{decimal_places}f}"
			line += f"{stat_value:>{col_data['max_width'] + 2}}"
		print(line)
		


def main():
	if len(sys.argv) != 2:
		print("Usage: python describe.py <csv_file>")
		return
	if os.path.exists(sys.argv[1]):
		data = dp.load_csv(sys.argv[1])
		num_cols = dp.get_num_cols(data)
		print(num_cols)
	else:
		print("Error: Path or dataset file name incorrect!")
		return
	print("Pandas describe() output:")
	stats_dict = {}
	for name in num_cols:
		max_len = len(name)
		raw_col = dp.extract_column(data, name)
		clean_col = dp.clean_num_data(raw_col)

		stats_dict[name] = {
			'count': len(clean_col),
			'mean': st.mean(clean_col),
			'std': st.std(clean_col),
			'min': st.min_max(clean_col)[0],
			'25%': st.percentile(clean_col, 25),
			'50%': st.percentile(clean_col, 50),
			'75%': st.percentile(clean_col, 75),
			'max': st.min_max(clean_col)[1],
		}

		for stat_val in stats_dict[name].values():
			precision = 5 if name == 'Index' else 6
			formatted = f"{stat_val:.{precision}f}"
			max_len = max(len(formatted), max_len)
		
		stats_dict[name]['max_width'] = max_len
	print_describe(stats_dict)


if __name__ == '__main__':
	main()