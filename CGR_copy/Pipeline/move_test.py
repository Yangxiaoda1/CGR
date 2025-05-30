# import os
# import shutil
# import numpy as np

# # 读取存有80个元素的npy文件
# npy_file_path = '/mnt/disk2/yangxiaoda/CGR/data/split_1img/name.npy'
# prefixes = np.load(npy_file_path)

# # 定义图片文件夹路径和目标文件夹路径
# source_folder = '/mnt/disk2/yangxiaoda/CGR/data/new_transfer'
# destination_folder = '/mnt/disk2/yangxiaoda/CGR/data/test_transfer'

# # 如果目标文件夹不存在，则创建它
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 遍历图片文件夹中的所有文件
# for filename in os.listdir(source_folder):
#     # 检查文件名是否以任意一个前缀开头
#     if any(filename.startswith(prefix) for prefix in prefixes):
#         # 构造源文件路径和目标文件路径
#         source_path = os.path.join(source_folder, filename)
#         destination_path = os.path.join(destination_folder, filename)
        
#         # 移动文件
#         shutil.move(source_path, destination_path)

# print("图片移动完成。")




import os
from PIL import Image

# 文件夹路径，根据你的实际情况修改
folder_path = '/mnt/disk2/yangxiaoda/CGR/data/new_transfer'

# 初始化计数器
image_count = 0

# 遍历文件夹中的每个文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        # 使用Pillow库打开文件
        with Image.open(file_path) as img:
            # 检查文件是否为图片格式
            img.verify()
            image_count += 1
    except (IOError, SyntaxError) as e:
        # 文件不是有效的图片文件
        continue

print(f"文件夹 {folder_path} 中的图片数量为: {image_count}")




# import os
# import shutil

# # 源文件夹路径，根据你的实际情况修改
# source_folder = '/mnt/disk2/yangxiaoda/CGR/data/test_transfer'

# # 目标文件夹路径，根据你的实际情况修改
# destination_folder = '/mnt/disk2/yangxiaoda/CGR/data/new_transfer'

# # 确保目标文件夹存在，如果不存在则创建
# os.makedirs(destination_folder, exist_ok=True)

# # 遍历源文件夹中的每个文件
# for filename in os.listdir(source_folder):
#     file_path = os.path.join(source_folder, filename)
#     if os.path.isfile(file_path):
#         # 检查文件是否为图片格式
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # 构造目标文件路径
#             destination_path = os.path.join(destination_folder, filename)
#             # 移动文件到目标文件夹
#             shutil.move(file_path, destination_path)
#             print(f"Moved {filename} to {destination_folder}")

# print("移动完成。")
