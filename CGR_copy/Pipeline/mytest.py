import sys
import os
import torch
import torch.nn.functional as F
# os.environ['MASTER_PORT'] = '29500'
csd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSD'))
sys.path.append(csd_path)
from feature_encoder_test_csd import ImageEncoder
# import torch.distributed as dist

# dist.init_process_group(backend='nccl', init_method='env://')

def qprint(var,str):
    print("\033[92m"+"{}:{}".format(str,var)+"\033[0m")
model = ImageEncoder()
img='/home/yangxiaoda/CGR/CSD/data/wikiart/Abstract_Expressionism/aaron-siskind_new-york-1951.jpg'
features = model.encode(img)
qprint(features.shape,'features.shape')
# styledatabase='/mnt/disk2/yangxiaoda/CGR/data/styledatabase.txt'
# with open(styledatabase,'r') as f:
#     lines=f.read().strip().split('\n')
#     lines=[item for item in lines]
# qprint(lines[1],'lines[1]')

# for it in lines:
#     category=it.split(':')[0]
#     emb=it.split(':')[1]
#     qprint(emb,'emb')
#     break
# qprint(lines[0],'lines[0]')