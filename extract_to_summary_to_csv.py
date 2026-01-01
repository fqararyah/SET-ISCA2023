import os
import csv
import re

BASE_DIR = './out/polar/7nm'
OUTPUT_CSV = 'summary_results.csv'

def extract_values(filepath):
    values = {
        'Energy': None,
        'mac_energy': None,
        'buf_energy': None,
        'ubuf_energy': None,
        'noc_energy': None,
        'DRAM_energy': None
    }
    with open(filepath, 'r') as f:
        for line in f:
            for key in values:
                if line.startswith(f"{key}:"):
                    # Extract the value after the colon and strip whitespace
                    values[key] = float(line.split(':', 1)[1].strip())
    return values

def get_relative_parts(filepath):
    # Remove BASE_DIR and file name, split the rest
    rel_path = os.path.relpath(filepath, BASE_DIR)
    parts = rel_path.split(os.sep)  # Exclude the filename
    return parts

def main():
    rows = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('_summary.txt'):
                full_path = os.path.join(root, file)
                vals = extract_values(full_path)
                if None in vals.values():
                    continue  # Skip if any value is missing
                # Calculate ratios
                energy = vals['Energy']
                mac_ratio = "{:.2f}".format(vals['mac_energy'] / energy if energy else 0)
                buf_ratio = "{:.2f}".format(vals['buf_energy'] / energy if energy else 0)
                ubuf_ratio = "{:.2f}".format(vals['ubuf_energy'] / energy if energy else 0)
                noc_ratio = "{:.2f}".format(vals['noc_energy'] / energy if energy else 0)
                dram_ratio = "{:.2f}".format(vals['DRAM_energy'] / energy if energy else 0)
                # Get path parts for row index
                path_parts = get_relative_parts(full_path)
                row = path_parts + [
                    vals['Energy'],
                    vals['mac_energy'],
                    vals['buf_energy'],
                    vals['ubuf_energy'],
                    vals['noc_energy'],
                    vals['DRAM_energy'],
                    mac_ratio,
                    buf_ratio,
                    ubuf_ratio,
                    noc_ratio,
                    dram_ratio
                ]
                rows.append(row)

    # Determine max number of path columns
    max_path_cols = max(len(r) - 7 for r in rows) if rows else 0
    path_headers = [f'level_{i+1}' for i in range(max_path_cols)]
    headers = path_headers + [
        'Energy', 'mac_energy', 'buf_energy', 'ubuf_energy', 'noc_energy', 'DRAM_energy',
        'mac_energy/Energy', 'buf_energy/Energy', 'ubuf_energy/Energy', 'noc_energy/Energy', 'DRAM_energy/Energy'
    ]

    # Pad path columns for rows with fewer levels
    for row in rows:
        while len(row) < len(headers):
            row.insert(len(row)-7, '')

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == '__main__':
    main()