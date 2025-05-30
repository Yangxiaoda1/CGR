import os

folder_path = '/mnt/disk2/yangxiaoda/CGR/data/all_transfer'

file_count = 0
with os.scandir(folder_path) as entries:
    for entry in entries:
        if entry.is_file():
            file_count += 1

print(f"文件夹 {folder_path} 中共有 {file_count} 个文件")