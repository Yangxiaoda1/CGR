import concurrent.futures
import numpy as np

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    image_names = []
    tensors = []

    num_lines = len(lines)
    for i in range(0, num_lines, 2):
        if i + 1 < num_lines:  # Ensure there are at least two lines left
            image_line = lines[i].strip()
            tensor_line = lines[i + 1].strip()

            tensor_line = tensor_line.replace("[","").replace("]","")

            parts = image_line.split('_')
            if len(parts) >= 4:
                processed_image_name = '_'.join(parts[3:])  # Join parts from the fourth part onwards
                image_names.append([processed_image_name])  # Wrap each name in a list to form (21600, 1) shape
            else:
                image_names.append([image_line])  # Wrap it in a list if less than 4 parts

            tensor_values = [float(x) for x in tensor_line.split(',') if x.strip()]
            tensors.append(tensor_values)  # Add each tensor value list to tensors list

    tensors_matrix = np.array(tensors)
    norms = np.linalg.norm(tensors_matrix, axis=1, keepdims=True)
    tensors_matrix = tensors_matrix / norms

    # Convert image_names_matrix to list
    image_names_list = [name[0] for name in image_names]

    # Save to .npy files
    folder = '/mnt/disk2/yangxiaoda/CGR/data/new_npy/CLIP(vitB-16)'
    np.save(f'{folder}/names.npy', image_names_list)
    np.save(f'{folder}/tensors.npy', tensors_matrix)
    print("OK")


# file_path='/mnt/disk2/yangxiaoda/CGR/data/contentdb_output.txt'
# file_path='/mnt/disk2/yangxiaoda/CGR/data/contentdb/new_output.txt'
file_path='/mnt/disk2/yangxiaoda/CGR/data/new_npy/CLIP(vitB-16)/db_out.txt'
process_file(file_path)
# 加载矩阵
image_names_matrix = np.load('/mnt/disk2/yangxiaoda/CGR/data/new_npy/CLIP(vitB-16)/names.npy')
tensors_matrix = np.load('/mnt/disk2/yangxiaoda/CGR/data/new_npy/CLIP(vitB-16)/tensors.npy')

# 打印 image_names_matrix 的前三个元素
print(image_names_matrix[:3])

# 打印 tensors_matrix 的前三行
print(tensors_matrix[:3])

# print(image_names_matrix.shape)
# print(tensors_matrix.shape)














# import torch
# def process_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     image_names = []
#     tensors = []

#     num_lines = len(lines)
#     for i in range(0, num_lines, 2):
#         if i + 1 < num_lines:  # Ensure there are at least two lines left
#             image_line = lines[i].strip()
#             tensor_line = lines[i + 1].strip()

#             # 只去掉一对 "[]"
#             if tensor_line.startswith('[') and tensor_line.endswith(']'):
#                 tensor_line = tensor_line[1:-1]  

#             # Process image name
#             parts = image_line.split('_')
#             if len(parts) >= 4:
#                 processed_image_name = '_'.join(parts[4:])  # Join parts from the fourth part onwards
#                 image_names.append([processed_image_name])  # 将每个名称包装在列表中以形成 (21600, 1) 形状
#             else:
#                 image_names.append([image_line])  # 如果有少于 4 个部分，也将其包装在列表中

#             tensor_values = [float(x) for x in tensor_line.split(',')]  # 将字符串转换为浮点数列表
#             tensors.append(tensor_values)  # 将每个张量值列表添加到 tensors 列表中

#     return np.array(image_names), np.array(tensors)  # 将列表转换为 NumPy 数组并返回

# file_path1 = '/mnt/disk2/yangxiaoda/CGR/data/styledb_output.txt'
# image_names1, tensors1 = process_file(file_path1)

# # 创建并填充矩阵
# name_matrix = np.array(image_names1)
# tensor_matrix = np.array(tensors1)

# # 保存到.npy 文件
# np.save('/mnt/disk2/yangxiaoda/CGR/data/combined_data.npy', np.array([name_matrix, tensor_matrix]))






# def cosine_similarity(a, b):
#     vec1 = np.array(eval(a)).flatten()
#     vec2 = np.array(eval(b)).flatten()
#     return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def calculate_similarities(index, tensor_matrix):
#     similarities = []
#     for i in range(len(tensor_matrix)):
#         if i!= index:
#             similarity = cosine_similarity(tensor_matrix[index], tensor_matrix[i])
#             similarities.append((i, similarity))
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return [i for i, _ in similarities[:10]]

# data = np.load('/mnt/disk2/yangxiaoda/CGR/data/combined_data.npy')
# name_matrix = data[0]
# tensor_matrix = data[1]

# # 创建一个线程池
# with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # 可以根据您的系统资源调整线程数量
#     # 提交任务到线程池
#     futures = [executor.submit(calculate_similarities, index, tensor_matrix) for index in range(len(tensor_matrix))]
#     # 等待所有任务完成并获取结果
#     results = [future.result() for future in concurrent.futures.as_completed(futures)]

# for i, result in enumerate(results):
#     print(f"For element at index {i}, the top 10 similar indices are: {result}")



















# # def process_file(file_path):
# #     with open(file_path, 'r') as file:
# #         lines = file.readlines()

# #     image_names = []
# #     tensors = []

# #     num_lines = len(lines)
# #     for i in range(0, num_lines, 2):
# #         if i + 1 < num_lines:  
# #             image_line = lines[i].strip()
# #             tensor_line = lines[i + 1].strip()

# #             # Process image name
# #             parts = image_line.split('_')
# #             if len(parts) >= 4:
# #                 processed_image_name = '_'.join(parts[4:])  
# #                 image_names.append(processed_image_name)
# #             else:
# #                 image_names.append(image_line)  

# #             tensors.append(tensor_line)

# #     return image_names, tensors



# data = np.load('/mnt/disk2/yangxiaoda/CGR/data/combined_data.npy')
# name_matrix = data[0]
# tensor_matrix = data[1]

# for idx in range(0,1):
#     print(name_matrix[idx])
#     print(tensor_matrix[idx])

# print(tensor_matrix.shape)
# print(tensor_matrix[0].shape)

# # 求两个矩阵的长度
# name_matrix_length = len(name_matrix)
# tensor_matrix_length = len(tensor_matrix)

# print(f"Name Matrix Length: {name_matrix_length}")
# print(f"Tensor Matrix Length: {tensor_matrix_length}")

# import numpy as np
# from scipy.spatial.distance import cosine

# def calculate_cosine_similarities(tensor_matrix):
#     similarities = []
#     for i, row in enumerate(tensor_matrix):
#         row_similarities = []
#         for j, other_row in enumerate(tensor_matrix):
#             if i!= j:
#                 similarity = 1 - cosine(row, other_row)
#                 row_similarities.append((j, similarity))
#         row_similarities.sort(key=lambda x: x[1], reverse=True)
#         similarities.append(row_similarities[:10])
#     return similarities

# data = np.load('/mnt/disk2/yangxiaoda/CGR/data/combined_data.npy')
# name_matrix = data[0]
# tensor_matrix = data[1]


# similarities = calculate_cosine_similarities(tensor_matrix)
# print(similarities.shape)
# print(similarities)
# flag = 0
# for idx, top_10 in enumerate(similarities):
#     top_10_names = [name_matrix[i] for i in [pair[0] for pair in top_10]]
#     if top_10_names.count(name_matrix[idx]) == 2:
#         flag += 1

# print("Flag:", flag)


# print(tensor_matrix.shape)
# print(tensor_matrix[:5])  # 打印前 5 行