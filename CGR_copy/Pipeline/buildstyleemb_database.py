import sys
import os
import torch
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

csd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSD'))
sys.path.append(csd_path)
from feature_encoder_test_csd import ImageEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageEncoder()
# img='/mnt/disk2/yangxiaoda/CGR/data/styletransfer/zebra10_2ActionPainting.jpg'
# features1 = model.encode(img)
# features1 = torch.tensor(features1, dtype=torch.float32)
# featurestot = [features1]
# featurestot = torch.stack(featurestot).squeeze(1)
# vec1 = np.array(eval(featurestot)).flatten()
# print(vec1)
folder = '/mnt/disk2/yangxiaoda/CGR/data/new_transfer'


with open('/mnt/disk2/yangxiaoda/CGR/data/new_npy/CSD(vitL-14)/db.txt', 'w') as f:
    total_files = sum(1 for _, _, files in os.walk(folder) for file in files if file.endswith('.jpg'))  # 计算总的 JPG 文件数量
    with tqdm(total=total_files, desc="Processing Images") as pbar:  # 创建进度条
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    features1 = model.encode(file_path)
                    features1 = torch.tensor(features1, dtype=torch.float32).to(device)
                    featurestot = []
                    featurestot.append(features1)
                    featurestot = torch.stack(featurestot).squeeze(1)
                    f.write(f"{file}: {features1}\n")  # 将结果写入文件
                    pbar.update(1)  # 每次处理完一个文件，更新进度条