import sys
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from feature_encoder_test_csd import ImageEncoder

# 环境变量设置
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'

# 初始化分布式环境
def init_distributed(backend='nccl', rank=0, world_size=1):
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Initialized distributed environment on rank {rank}.")

def qprint(var, str):
    print("\033[92m" + "{}:{}".format(str, var) + "\033[0m")

# 分布式函数
def main(rank, world_size):
    init_distributed(rank=rank, world_size=world_size)
    
    model = ImageEncoder()
    img = '/home/yangxiaoda/CGR/CSD/data/wikiart/Abstract_Expressionism/aaron-siskind_new-york-1951.jpg'
    features = model.encode(img)
    qprint(features.shape, 'features.shape')

# 启动分布式进程
if __name__ == "__main__":
    world_size = 2  # 假设你有两个GPU
    processes = []
    for rank in range(world_size):
        p = torch.multiprocessing.Process(target=main, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
