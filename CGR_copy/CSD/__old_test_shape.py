import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.randn(3, 224, 224)
        index = idx
        return sample, index

class CUDADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, index = self.dataset[idx]
        return sample.cuda(non_blocking=True), index

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

    # 确保所有数据都在 CUDA 上
    for batch_idx, (samples, index) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        break  # 仅检查第一个批次

    # 提取特征
    from utils import extract_features
    features = extract_features(args, model, data_loader)
    print(f"Features shape: {features.shape}")

def main():
    world_size = 1  # 根据你的设置进行调整
    args = Args()

    model = DummyModel()
    dataset = DummyDataset(num_samples=320)
    cuda_dataset = CUDADataset(dataset)  # 包装数据集
    data_loader = DataLoader(cuda_dataset, batch_size=32, shuffle=False)

    mp.spawn(main_worker, args=(world_size, args, model, data_loader), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
