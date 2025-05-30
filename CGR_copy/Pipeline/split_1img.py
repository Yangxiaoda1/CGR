import os
import random
import numpy as np

# 主文件夹路径，根据你的实际情况修改
main_folder = '/mnt/disk2/yangxiaoda/CGR/data/10'

# 获取主文件夹下的所有子文件夹
subfolders = [os.path.join(main_folder, folder) for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

# 用于存放文件名的列表
file_names = []

# 遍历每个子文件夹
for folder in subfolders:
    # 找到该子文件夹下的所有.jpg文件
    jpg_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    
    # 如果有.jpg文件，则随机选择一个文件名，并将子文件夹名加入列表
    if jpg_files:
        random_file = random.choice(jpg_files)
        # 将子文件夹名和文件名组合，去掉斜杠'/'
        combined_name = os.path.join(os.path.basename(folder), random_file).replace('/', '').replace(".jpg","_")
        file_names.append(combined_name)

# 将文件名列表保存为.npy文件
np.save('/mnt/disk2/yangxiaoda/CGR/data/split_1img/name.npy', file_names)

file_name = '/mnt/disk2/yangxiaoda/CGR/data/split_1img/name.npy'  # 将文件名替换为你实际的.npy文件名

try:
    # 尝试从.npy文件中加载数据
    loaded_data = np.load(file_name)

    # 打印前50个元素
    print("前50个元素为：", loaded_data[:50])

except FileNotFoundError:
    print(f"文件 '{file_name}' 未找到，请检查文件路径。")
except Exception as e:
    print(f"读取文件 '{file_name}' 发生错误：", e)

print(loaded_data.shape)
print(loaded_data[66])