import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from CSD.model import CSD_CLIP
from CSD.utils import has_batchnorms, convert_state_dict, extract_features
from CSD.loss_utils import transforms_branch0

# Set default environment variables for single machine testing
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

# Initialize the distributed process group
dist.init_process_group(backend='nccl', init_method='env://')

class PNGDataset(Dataset):
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.image = self.image.convert("RGB")
        self.image = transforms_branch0(self.image)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image, idx

class Args:
    pt_style = 'csd'
    layer = 1
    arch = 'vit_base'
    model_path = "/home/yangxiaoda/CGR/CSD/pretrainedmodels/checkpoint.pth"
    eval_embed = 'head'

class ImageEncoder:
    def __init__(self, args=Args()):
        self.args = args
        assert args.model_path is not None, "Model path missing for CSD model"
        
        self.model = CSD_CLIP(args.arch, "default")
        
        if has_batchnorms(self.model):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        checkpoint = torch.load(args.model_path, map_location="cpu")
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"=> loaded checkpoint with msg {msg}")
        
        self.model.cuda()
        self.model.eval()
        self.preprocess = transforms_branch0

    def encode(self, image_path):
        dataset = PNGDataset(image_path)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            features = extract_features(model=self.model, data_loader=data_loader, use_cuda=True, eval_embed=self.args.eval_embed)
        
        return features.cpu().numpy()

def main():
    encoder = ImageEncoder()
    image_path = '/home/yangxiaoda/CGR/Test_pictures/abstract1.jpg'
    features = encoder.encode(image_path)
    print("Encoded features:", features.shape)

if __name__ == '__main__':
    main()
    dist.destroy_process_group()  # Clean up the process group
