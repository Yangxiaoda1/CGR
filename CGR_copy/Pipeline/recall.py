# import numpy as np
# import tqdm
# import re
# def normalize_rows(matrix):
#     # 计算每行向量的模长
#     row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    
#     # 将每行向量除以其模长
#     normalized_matrix = matrix / row_norms
    
#     return normalized_matrix
# # 加载矩阵,矩阵每一行已经进行了单位化
# folder='InterVL'
# image_names_matrix = np.load(f'/mnt/disk2/yangxiaoda/CGR/data/new_npy/{folder}/names.npy')
# tensors_matrix = np.load(f'/mnt/disk2/yangxiaoda/CGR/data/new_npy/{folder}/tensors.npy')

# serch_names=np.load(f"/mnt/disk2/yangxiaoda/CGR/data/test_npy/{folder}/names.npy")
# serch_tensors=np.load(f"/mnt/disk2/yangxiaoda/CGR/data/test_npy/{folder}/tensors.npy")

# tensors_matrix = normalize_rows(tensors_matrix)
# serch_tensors=normalize_rows(serch_tensors)
# # image_names_matrix = [s.replace('/mnt/disk2/yangxiaoda/CGR/data/new_transfer/', '') for s in image_names_matrix]
# true_num=0     #可以搜索到的
# turns=2160   #搜索轮次
# rank = 1    #搜索的前多少名
# print(tensors_matrix[50])
# print(serch_names[56])
# op=""
# for i in range(len(image_names_matrix)):
#     if op=="style":
#         image_names_matrix[i] = image_names_matrix[i].split('_')[1]#截取style
#     elif op=="content":
#         image_names_matrix[i]=re.sub(r'_(.*?)\.', '.', image_names_matrix[i])#截取内容content
#         image_names_matrix[i] = re.sub(r'\d+', '', image_names_matrix[i])  # 去掉所有数字

# for i in range(len(serch_names)):
#     if op=="style":
#         serch_names[i] = serch_names[i].split('_')[1]#截取style
#     elif op=="content":
#         serch_names[i]=re.sub(r'_(.*?)\.', '.', serch_names[i])#截取内容content
#         serch_names[i] = re.sub(r'\d+', '', serch_names[i])  # 去掉所有数字








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

def calculate_mAp(folder, rank, op):
    image_names_matrix = np.load(f'/mnt/disk2/yangxiaoda/CGR/data/new_npy/{folder}/names.npy')
    tensors_matrix = np.load(f'/mnt/disk2/yangxiaoda/CGR/data/new_npy/{folder}/tensors.npy')

    serch_names = np.load(f"/mnt/disk2/yangxiaoda/CGR/data/test_npy/{folder}/names.npy")
    serch_tensors = np.load(f"/mnt/disk2/yangxiaoda/CGR/data/test_npy/{folder}/tensors.npy")

    tensors_matrix = normalize_rows(tensors_matrix)
    serch_tensors = normalize_rows(serch_tensors)

    turns = 2160  # 搜索轮次
    true_num = 0
    for i in range(len(image_names_matrix)):
        if op == "style":
            image_names_matrix[i] = image_names_matrix[i].split('_')[1]  # 截取 style
        elif op == "content":
            image_names_matrix[i] = re.sub(r'_(.*?)\.', '.', image_names_matrix[i])  # 截取内容 content
            image_names_matrix[i] = re.sub(r'\d+', '', image_names_matrix[i])  # 去掉所有数字

    for i in range(len(serch_names)):
        if op == "style":
            serch_names[i] = serch_names[i].split('_')[1]  # 截取 style
        elif op == "content":
            serch_names[i] = re.sub(r'_(.*?)\.', '.', serch_names[i])  # 截取内容 content
            serch_names[i] = re.sub(r'\d+', '', serch_names[i])  # 去掉所有数字

    for i in tqdm.tqdm(range(0, turns)):#一次查询
        result = np.dot(tensors_matrix, serch_tensors[i])
        sorted_indices = np.argsort(result)[::-1][:rank]#获取rank个最大相似度的角标
        for j in range(0, rank):
            if image_names_matrix[sorted_indices[j]] == serch_names[i]:
                true_num += 1
                break
    recall=round(true_num*100/turns,4)
    print(f"recall@{rank}={recall}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mAp for image search.")
    parser.add_argument("--folder", type=str, required=True, help="Path of the folder.")
    parser.add_argument("--rank", type=int, required=True, help="Number of top results to consider.")
    parser.add_argument("--op", type=str, required=True, choices=["style", "content"], help="Operation type: style or content")

    args = parser.parse_args()

    calculate_mAp(args.folder, args.rank, args.op)
