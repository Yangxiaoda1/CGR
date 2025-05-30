import os
import shutil

# 源目录路径
source_dir = '/home/yangxiaoda/CGR/CSD/data/wikiart'
# 目标目录路径
target_dir = '/mnt/disk2/yangxiaoda/CGR/data/style'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源目录下的所有文件夹
for subdir, dirs, files in os.walk(source_dir):
    # 确保我们只在顶级文件夹中操作
    if subdir == source_dir:
        # 从每个文件夹中选择第一张.jpg文件
        for folder in dirs:
            folder_path = os.path.join(source_dir, folder)
            # 检查文件夹中是否有.jpg文件
            if any(f.lower().endswith('.jpg') for f in os.listdir(folder_path)):
                # 选择第一张.jpg文件
                for file in os.listdir(folder_path):
                    if file.lower().endswith('.jpg'):
                        # 构建源文件的完整路径
                        source_file_path = os.path.join(folder_path, file)
                        # 构建目标文件的名称和路径
                        target_file_name = f"{folder}.jpg"
                        target_file_path = os.path.join(target_dir, target_file_name)
                        # 复制文件
                        shutil.copy2(source_file_path, target_file_path)
                        print(f"Copied '{source_file_path}' to '{target_file_path}'")
                        break  # 跳出文件循环，因为我们只需要一张图片
            else:
                print(f"No .jpg files found in folder '{folder_path}'")
        break  # 跳出目录循环，因为我们只需要27个文件夹

print("Process completed.")