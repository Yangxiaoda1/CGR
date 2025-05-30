def encode_special_chars(input_str):
    """
    将字符串中的非ASCII字符转换为其对应的Unicode编码形式
    """
    encoded_str = ''
    for char in input_str:
        if ord(char) > 127:  # 只编码非ASCII字符
            encoded_str += '#U{:04x}'.format(ord(char))
        else:
            encoded_str += char
    return encoded_str

# 示例使用
original_filename = 'arnold-bã¶cklin_the-ride-of-death-the-fall-and-death.jpg'
encoded_filename = encode_special_chars(original_filename)
print(f"Original filename: {original_filename}")
print(f"Encoded filename: {encoded_filename}")

import os

dataset_path = "./data/wikiart"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        encoded_filename = encode_special_chars(file)
        new_file = encoded_filename
        if(new_file!=file):
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
            print(f"file {file} renamed {new_file}")
print('run over')
