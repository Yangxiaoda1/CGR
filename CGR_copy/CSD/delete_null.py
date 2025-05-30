
import csv
import os

# 定义CSV文件路径
input_csv_path = './data/wikiart.csv'
output_csv_path = './data/wikiart_new.csv'

# 检查文件是否存在
def file_exists(file_path):
    return os.path.exists(file_path)

# 读取CSV文件并过滤数据
with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # 检查文件路径是否存在
        if file_exists(row[4]):
            writer.writerow(row)
        else:
            print(f"File not found: {row[4]} - Skipping this row.")

print("Processing complete. Check the output CSV file for the filtered data.")