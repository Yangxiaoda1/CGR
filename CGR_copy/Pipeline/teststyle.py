import sys
sys.path.append('/home/yangxiaoda/CGR')
from CSD.feature_encoder_test import ImageEncoder
import torch
model=ImageEncoder()

image1='/home/yangxiaoda/CGR/Test_pictures/abstract1.jpg'
image2='/home/yangxiaoda/CGR/Test_pictures/abstract2.jpg'
image3='/home/yangxiaoda/CGR/Test_pictures/abstract3.jpg'
image4='/home/yangxiaoda/CGR/Test_pictures/uk1.jpg'

features1=model.encode(image1)
print(features1[0][0])
features2=model.encode(image2)
features3=model.encode(image3)
features4=model.encode(image4)

device="cuda" if torch.cuda.is_available() else "cpu"

features1=torch.tensor(features1,dtype=torch.float32).to(device)
features2=torch.tensor(features2,dtype=torch.float32).to(device)
features3=torch.tensor(features3,dtype=torch.float32).to(device)
features4=torch.tensor(features4,dtype=torch.float32).to(device)