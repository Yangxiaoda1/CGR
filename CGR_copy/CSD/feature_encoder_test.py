import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
# from models import clip
import CSD.models.clip as clip
# import sys
# sys.path.append('/home/yangxiaoda/CGR')

class PNGDataset(Dataset):
    def __init__(self, image_path):
        """
        初始化数据集，读取并预处理PNG图像。

        参数:
        image_path (str): 图像文件的路径。
        """
        self.image = Image.open(image_path)
        self.image = self.image.convert("RGB")  # 转换为RGB格式

        # 检查并调整图像尺寸为224x224
        if self.image.size != (224, 224):
            self.image = self.image.resize((224, 224))

        # 将图像转换为CHW格式的numpy数组，并标准化到[0, 1]范围
        self.image = np.array(self.image).transpose((2, 0, 1))
        self.image = torch.tensor(self.image, dtype=torch.float32) / 255.0

    def __len__(self):
        return 1  # 数据集只有一个图像

    def __getitem__(self, idx):
        return self.image, idx

class Args:
    pt_style = 'clip'
    layer = 1
    arch = 'vit_base'
    multilayer = [2, 4, 8]
    multiscale = False

def main_worker(rank, world_size, args, model, data_loader, return_dict):
    """
    主工作进程函数，用于初始化分布式环境并提取特征。

    参数:
    rank (int): 当前进程的排名。
    world_size (int): 进程总数。
    args (Args): 提取特征所需的参数。
    model (torch.nn.Module): 需要提取特征的模型。
    data_loader (DataLoader): 数据加载器。
    return_dict (dict): 用于存储返回特征的共享字典。
    """
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )

    model = model.cuda(rank)
    if any(param.requires_grad for param in model.parameters()):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 打印模型和数据加载器的类型
    print(f"Model type: {type(model)}")
    print(f"Data loader type: {type(data_loader)}")

    from CSD.utils import extract_features
    features = extract_features(args, model, data_loader)
    if rank == 0:
        return_dict['features'] = features.cpu().numpy()


# import CLIP.clip as clip
class ImageEncoder:
    def __init__(self, args=Args()):
        model_tuple = clip.load('ViT-B/16')
        self.model, self.preprocess = model_tuple
        self.args = args

    def encode(self, image_path):
        """
        编码图像并提取特征。

        参数:
        image_path (str): 图像文件的路径。

        返回:
        numpy.ndarray: 提取的特征。
        """
        dataset = PNGDataset(image_path)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        world_size = 1
        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(main_worker, args=(world_size, self.args, self.model, data_loader, return_dict), nprocs=world_size, join=True)

        return return_dict['features']

def main():
    """
    主函数，初始化模型和编码器，并提取图像特征。
    """
    model = ImageEncoder()
    image_path = '/home/yangxiaoda/CGR/Test_pictures/abstract1.jpg'
    features = model.encode(image_path)
    print("Encoded features:", features.shape)

# if __name__ == '__main__':
#     main()
