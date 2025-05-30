import os
import random
import shutil

# 指定维基艺术下的27个风格的文件夹路径
wiki_art_path = '/home/yangxiaoda/CGR/CSD/data/wikiart'

# 指定保存提取图片的新文件夹路径
output_folder = '/mnt/disk2/yangxiaoda/CGR/tools/style'

# 创建保存提取图片的新文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取维基艺术下的所有风格文件夹列表
style_folders = os.listdir(wiki_art_path)

# 遍历每个风格文件夹
for style_folder in style_folders:
    style_folder_path = os.path.join(wiki_art_path, style_folder)
    
    # 检查路径是否为文件夹
    if os.path.isdir(style_folder_path):
        # 获取文件夹下所有图片文件
        image_files = [f for f in os.listdir(style_folder_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        
        # 确保文件夹中有足够的图片文件
        if len(image_files) >= 10:
            # 从图片文件中随机选择10个
            selected_images = random.sample(image_files, 20)
            
            # 遍历选中的图片并进行重命名并复制到新文件夹
            for i, image_name in enumerate(selected_images):
                original_path = os.path.join(style_folder_path, image_name)
                new_filename = f"{i + 1}___{style_folder}.jpg"  # 新文件名为<风格名>_1.jpg到<风格名>_10.jpg
                new_path = os.path.join(output_folder, new_filename)
                
                # 将文件复制并重命名为新文件名
                shutil.copyfile(original_path, new_path)
                
                print(f"Copied and renamed {image_name} to {new_filename} in {output_folder}")

        else:
            print(f"Not enough images in {style_folder}, skipping.")
