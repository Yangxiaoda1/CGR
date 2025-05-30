import sys
import os

# csd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSD'))
# sys.path.append(csd_path)
# from feature_encoder_test_csd import ImageEncoder

# 读取输入txt文件路径
input_file_path = '/mnt/disk2/yangxiaoda/CGR/data/test_npy/CLIP(vitB-16)/db.txt'  # 替换成你的styledatabase.txt文件路径
output_file_path = '/mnt/disk2/yangxiaoda/CGR/data/test_npy/CLIP(vitB-16)/db_out.txt'  # 替换成你的输出文件路径

# 打开输入文件并读取内容
with open(input_file_path, 'r') as f:
    lines = f.readlines()

# 打开输出文件准备写入内容
with open(output_file_path, 'w') as out_file:
    for line in lines:
        if 'jpg:' in line:
            split_index = line.index(':')
            image_name = line[:split_index].strip()  # 分离出图片名部分
            tensor_sequence = line[split_index + 1:].strip()  # 分离出tensor序列部分
        elif '   ' in line:
            str=line.strip()
            tensor_sequence+=str
            if 'cuda:' in line:
                tensor_sequence = tensor_sequence.replace(", device='cuda:0'","")
                tensor_sequence = tensor_sequence.replace(", device='cuda:0')", "")
                tensor_sequence = tensor_sequence.replace(",device='cuda:0')", "")
                tensor_sequence = tensor_sequence.replace("tensor(", "")
                tensor_sequence = tensor_sequence.replace(" ", "")
                tensor_sequence = tensor_sequence.replace("device='cuda:0',dtype=torch.float16)","")
                tensor_sequence = tensor_sequence.replace("dtype=torch.float16)","")
                tensor_sequence = tensor_sequence.replace(")","")
                # 写入到输出文件
                out_file.write(f"___{image_name}")
                out_file.write('\n')
                out_file.write(f"{tensor_sequence}")
                out_file.write('\n')

print("处理完成，并将结果写入文件:", output_file_path)