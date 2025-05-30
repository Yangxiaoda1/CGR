import numpy as np
import tqdm
import re
def normalize_rows(matrix):
    # 计算每行向量的模长
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    # 将每行向量除以其模长
    normalized_matrix = matrix / row_norms

    return normalized_matrix
# 加载矩阵,矩阵每一行已经进行了单位化
# folder='/mnt/disk2/yangxiaoda/CGR/new_data/content_'
folder='/mnt/disk2/yangxiaoda/CGR/miniwikiart_embedding'
image_names_matrix = np.load(f'{folder}/names.npy')
tensors_matrix = np.load(f'{folder}/tensors.npy')
tensors_matrix = normalize_rows(tensors_matrix)
print("content-mAp5")
# true_num=0     #可以搜索到的
turns=21600   #搜索轮次
rank = 5   #搜索的前多少名
for i in range(len(image_names_matrix)):
    num_index = next(i for i, c in enumerate(image_names_matrix[i]) if c.isdigit())
    image_names_matrix[i] = image_names_matrix[i][:num_index]#style
    # image_names_matrix[i] = re.sub(r'_(.*?)\.', '.', image_names_matrix[i])  # 截取内容 content
    # image_names_matrix[i] = re.sub(r'\d+', '', image_names_matrix[i])  # 去掉所有数字
print(image_names_matrix[:50]) 
# mAp=0
# for i in tqdm.tqdm(range(0, turns)):#一次查询
#     Ap=0
#     true_num=0
#     result = np.dot(tensors_matrix, tensors_matrix[i])
#     sorted_indices = np.argsort(result)[::-1][:rank+1]#获取rank个最大相似度的角标
#     for j in range(1, rank+1):
#         if image_names_matrix[sorted_indices[j]] == image_names_matrix[i]:
#             true_num += 1
#             Ap+=round(true_num/j,4)
#             # break
#     if true_num != 0:
#         Ap=Ap/true_num
#         mAp+=Ap

# # recall=round(true_num*100/(turns),2)
# # print(f"recall@{rank}={recall}%")
# mAp=round(mAp/turns,4)
# print(f"mAp@{rank}={mAp*100}%")

    

# print(image_names_matrix[:5])
# for i in range(0,5):
#     image_names_matrix[i][0]=re.sub(r'_(.*?)\.', '.', image_names_matrix[i][0])
#     print(image_names_matrix[i][0])
