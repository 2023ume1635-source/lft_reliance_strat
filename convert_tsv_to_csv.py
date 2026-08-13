import csv
import os

# Variable for the input TSV filename - change this for other files
input_tsv = "reliance_depth_20260721_092030.tsv"

# Generate output CSV filename by replacing .tsv with .csv
output_csv = os.path.splitext(input_tsv)[0] + ".csv"

def convert_tsv_to_csv(tsv_file, csv_file):
    with open(tsv_file, 'r', newline='', encoding='utf-8') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            for row in tsv_reader:
                csv_writer.writerow(row)
    
    print(f"Successfully converted '{tsv_file}' to '{csv_file}'")

if __name__ == "__main__":
    convert_tsv_to_csv(input_tsv, output_csv)