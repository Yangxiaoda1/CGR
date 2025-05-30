import sys
import os
import torch
import torch.nn.functional as F

# Add the CSD directory to sys.path
csd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSD'))
sys.path.append(csd_path)

from feature_encoder_test_csd import ImageEncoder

try:
    model = ImageEncoder()

    image1 = '/home/yangxiaoda/CGR/Test_pictures/abstract1.jpg'
    image2 = '/home/yangxiaoda/CGR/Test_pictures/abstract2.jpg'
    image3 = '/home/yangxiaoda/CGR/Test_pictures/abstract3.jpg'
    image4 = '/home/yangxiaoda/CGR/Test_pictures/uk1.jpg'
    image5='/home/yangxiaoda/CGR/Test_pictures/uk2.jpg'
    image6='/home/yangxiaoda/CGR/Test_pictures/popart1.jpg'
    image7='/home/yangxiaoda/CGR/Test_pictures/popart2.jpg'

    features1 = model.encode(image1)
    print("Features1:", features1[0][0])
    features2 = model.encode(image2)
    print("Features2 loaded")
    features3 = model.encode(image3)
    print("Features3 loaded")
    features4 = model.encode(image4)
    print("Features4 loaded")
    features5 = model.encode(image5)
    print("Features4 loaded")
    features6 = model.encode(image6)
    print("Features4 loaded")
    features7 = model.encode(image7)
    print("Features4 loaded")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    features1 = torch.tensor(features1, dtype=torch.float32).to(device)
    features2 = torch.tensor(features2, dtype=torch.float32).to(device)
    features3 = torch.tensor(features3, dtype=torch.float32).to(device)
    features4 = torch.tensor(features4, dtype=torch.float32).to(device)
    features5 = torch.tensor(features5, dtype=torch.float32).to(device)
    features6 = torch.tensor(features6, dtype=torch.float32).to(device)
    features7 = torch.tensor(features7, dtype=torch.float32).to(device)
    featurestot = []
    featurestot.append(features1)
    featurestot.append(features2)
    featurestot.append(features3)
    featurestot.append(features4)
    featurestot.append(features5)
    featurestot.append(features6)
    featurestot.append(features7)
    featurestot = torch.stack(featurestot).squeeze(1)

    print("featurestot shape:", featurestot.shape)

    similarity1 = torch.mm(features1, featurestot.t())
    print("similarity1:", similarity1)
    similarity2 = torch.mm(features2, featurestot.t())
    print("similarity2:", similarity2)
    similarity3 = torch.mm(features3, featurestot.t())
    print("similarity3:", similarity3)
    similarity4 = torch.mm(features4, featurestot.t())
    print("similarity4:", similarity4)
    similarity5 = torch.mm(features5, featurestot.t())
    print("similarity5:", similarity5)
    similarity6 = torch.mm(features6, featurestot.t())
    print("similarity6:", similarity6)
    similarity7 = torch.mm(features7, featurestot.t())
    print("similarity7:", similarity7)
except Exception as e:
    print("An error occurred:", str(e))