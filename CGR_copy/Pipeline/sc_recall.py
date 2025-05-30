import numpy as np
import tqdm
import re
import argparse

def normalize_rows(matrix):
    # 计算每行向量的模长
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    # 将每行向量除以其模长
    normalized_matrix = matrix / row_norms

    return normalized_matrix

def calculate_recall(folder, rank):
    # 加载矩阵,矩阵每一行已经进行了单位化
    image_names_matrix = np.load(f'/mnt/disk2/yangxiaoda/CGR/data/new_npy/{folder}/names.npy')
    tensors_matrix = np.load(f'/mnt/disk2/yangxiaoda/CGR/data/new_npy/{folder}/tensors.npy')

    serch_names = np.load(f"/mnt/disk2/yangxiaoda/CGR/data/test_npy/{folder}/names.npy")
    serch_tensors = np.load(f"/mnt/disk2/yangxiaoda/CGR/data/test_npy/{folder}/tensors.npy")

    tensors_matrix = normalize_rows(tensors_matrix)
    serch_tensors = normalize_rows(serch_tensors)

    image_names_matrix = [s.replace('/mnt/disk2/yangxiaoda/CGR/data/new_transfer/', '') for s in image_names_matrix]
    serch_names = [s.replace('/mnt/disk2/yangxiaoda/CGR/data/test_transfer/', '') for s in serch_names]
    true_num = 0  # 可以搜索到的
    turns = 2160  # 搜索轮次

    contents = [''] * len(image_names_matrix)
    styles = [''] * len(image_names_matrix)
    search_contents = [''] * len(serch_names)
    search_styles = [''] * len(serch_names)

    for i in range(len(image_names_matrix)):
        styles[i] = image_names_matrix[i].split('_')[1]  # 截取 style
        contents[i] = re.sub(r'_(.*?)\.', '.', image_names_matrix[i])  # 截取内容 content
        contents[i] = re.sub(r'\d+', '', image_names_matrix[i])  # 去掉所有数字

    for i in range(len(serch_names)):
        search_styles[i] = serch_names[i].split('_')[1]  # 截取 style
        search_contents[i] = re.sub(r'_(.*?)\.', '.', serch_names[i])  # 截取内容 content
        search_contents[i] = re.sub(r'\d+', '', serch_names[i])  # 去掉所有数字

    for i in tqdm.tqdm(range(0, turns)):  # 一次查询
        result = np.dot(tensors_matrix, serch_tensors[i])
        sorted_indices = np.argsort(result)[::-1][:rank]  # 获取 rank 个最大相似度的角标
        for j in range(0, rank):
            if (styles[sorted_indices[j]] == search_styles[i]) and (contents[sorted_indices[j]] == search_contents[i]):
                true_num += 1
                break
    recall = round(true_num * 100 / turns, 4)
    print(f"recall@{rank}={recall}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate recall for image search.")
    parser.add_argument("--folder", type=str, required=True, help="Path of the folder.")
    parser.add_argument("--rank", type=int, required=True, help="Number of top results to consider.")

    args = parser.parse_args()
    print(args.folder)
    calculate_recall(args.folder, args.rank)