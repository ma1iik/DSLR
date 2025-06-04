import sys
from ml_toolkit import Statistics, DataProcessor

def print_summary_table(summary, max_features_per_page=5):
    headers = list(summary.keys())
    stats_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    min_col_width = 16  # Slightly smaller for a tighter fit
    max_feature_display_length = 15  # Shorter name length

    # Calculate dynamic column widths per feature
    col_widths = {}
    for h in headers:
        max_stat_width = max(
            len(f"{summary[h][s]:.6f}") if isinstance(summary[h][s], float) else len(str(summary[h][s]))
            for s in stats_names
        )
        col_widths[h] = max(max_feature_display_length, max_stat_width, min_col_width)

    stat_label_width = max(len("Statistic"), min_col_width)

    # Display in batches
    for i in range(0, len(headers), max_features_per_page):
        subset = headers[i:i + max_features_per_page]
        print("=" * 80)
        print(f"Showing features {i + 1} to {i + len(subset)} of {len(headers)}\n")

        # Header row
        header_row = "Statistic".ljust(stat_label_width)
        for h in subset:
            display_name = h if len(h) <= max_feature_display_length else h[:max_feature_display_length - 3] + "..."
            header_row += display_name.ljust(col_widths[h] + 2)
        print(header_row)

        # Statistic rows
        for stat_name in stats_names:
            row = stat_name.ljust(stat_label_width)
            for h in subset:
                val = summary[h][stat_name]
                if isinstance(val, float):
                    row += f"{val:.6f}".ljust(col_widths[h] + 2)
                else:
                    row += str(val).ljust(col_widths[h] + 2)
            print(row)

        print("\n")  # space between pages


def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py <csv_file>")
        return

    data = DataProcessor.load_csv(sys.argv[1])
    num_cols = DataProcessor.get_num_cols(data)

    summary = {}  # Dictionary to hold stats for each column

    for col_name in num_cols:
        col_data = DataProcessor.extract_column(data, col_name)
        clean_values = [float(val) for val in col_data if DataProcessor.is_float(val)]
        if not clean_values:
            continue
        clean_values.sort()
        stats = {
            "Count": len(clean_values),
            "Mean": Statistics.mean(clean_values),
            "Std": Statistics.std(clean_values),
            "Min": min(clean_values),
            "25%": Statistics.percentile(clean_values, 0.25),
            "50%": Statistics.percentile(clean_values, 0.50),
            "75%": Statistics.percentile(clean_values, 0.75),
            "Max": max(clean_values),
        }
        summary[col_name] = stats

    print_summary_table(summary)

if __name__ == "__main__":
    main()