# import torch
# import clip
# import os
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-L/14", device=device)
# folder = '/mnt/disk2/yangxiaoda/CGR/data/styletransfer'
# # image = preprocess(Image.open("/home/yangxiaoda/CGR/Test_pictures/CLIP.png")).unsqueeze(0).to(device)
# for root, dirs, files in os.walk(folder):
#     for file in files:
#         if file.endswith('.jpg'):
#             file_path = os.path.join(root, file)
#             with torch.no_grad():
#                 image_features = model.encode_image(file_path)
#                 featurestot = []
#                 featurestot.append(image_features)
#                 featurestot = torch.stack(featurestot).squeeze(1)
#                 print(file,":", image_features)


import torch
import clip
import os
from PIL import Image
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
folder = '/mnt/disk2/yangxiaoda/CGR/data/new_transfer'

with open('/mnt/disk2/yangxiaoda/CGR/data/new_npy/CLIP(vitB-16)/db.txt', 'w') as f:  # 打开或创建文件用于写入
    total_files = sum(1 for _, _, files in os.walk(folder) for file in files if file.endswith('.jpg'))  # 计算总的 JPG 文件数量
    with tqdm(total=total_files, desc="Processing Images") as pbar:  # 创建进度条
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    with torch.no_grad():
                        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
                        image_features = model.encode_image(image)
                        featurestot = []
                        featurestot.append(image_features)
                        featurestot = torch.stack(featurestot).squeeze(1)
                        f.write(f"{file}: {image_features}\n")  # 将结果写入文件
                    pbar.update(1)  # 每次处理完一个文件，更新进度条