import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class PNGDataset(Dataset):
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.image = self.image.convert("RGB")  # 转换为RGB
        self.image = self.image.resize((224, 224))  # 调整尺寸为模型输入要求的尺寸
        self.image = np.array(self.image).transpose((2, 0, 1))  # 转换为CHW格式
        self.image = torch.tensor(self.image, dtype=torch.float32) / 255.0  # 标准化

    def __len__(self):
        return 1  # 只有一个图像

    def __getitem__(self, idx):
        return self.image, idx

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.randn(x.size(0), 512)

class Args:
    pt_style = 'dino'
    layer = 1
    arch = 'vit_base'
    multilayer = [2, 4, 8]
    multiscale = False

def main_worker(rank, world_size, args, model, data_loader):
    dist.init_process_group(
        backend='nccl',  # or 'gloo' if you are not using GPUs
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )

    # 检查模型是否有任何需要梯度的参数
    if any(param.requires_grad for param in model.parameters()):
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])
    else:
        model = model.cuda(rank)

    # 提取特征
    from utils import extract_features
    features = extract_features(args, model, data_loader)
    print(f"Features shape: {features.shape}")

def main():
    image_path = '/home/yangxiaoda/CGR/CLIP.png'
    dataset = PNGDataset(image_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DummyModel()
    args = Args()

    world_size = 1  # 根据你的设置进行调整
    mp.spawn(main_worker, args=(world_size, args, model, data_loader), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
