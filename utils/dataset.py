import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch
from PIL import Image
import random
from torchvision.transforms import functional as F
import numpy as np


class CDTXTDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        
        self.file_list = open(txt_path, 'r').readlines()
        
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pathA, pathB, pathLAB = self.file_list[idx].strip().split('  ')
        imgA = Image.open(pathA).convert('RGB')
        imgB = Image.open(pathB).convert('RGB')
        lab = Image.open(pathLAB).convert('L')

        imgA = np.asarray(imgA)  # H×W×3 的 numpy 数组
        imgB = np.asarray(imgB)
        lab = np.asarray(lab)//255    # H×W 的 numpy 数组
        if self.transform:
            augmented = self.transform(
                image=imgA,    # 第一时相 H×W×3 的 numpy 数组
                image1=imgB,   # 第二时相 H×W×3 的 numpy 数组
                mask=lab        # 标签 H×W 的 numpy 数组
            )
        inputA = augmented['image']   # Tensor, C×H×W
        inputB = augmented['image1']  # Tensor, C×H×W
        label = augmented['mask']    # Tensor, H×W
        concat = torch.cat((inputA, inputB), dim=0)

        return {'imgAB':concat, 'lab':label, 'pathA': pathA, 'pathB': pathB}


class  DGCDTXTDataset(Dataset):
    def __init__(self, train_domain=['WaterCDPNG', 'PX-CLCD', 'WHUCD', 'LEVIR-CD'], split='train', transform=None):
        self.train_domain_filelist = {}
        for domain in train_domain:
            domain_path = os.path.join(os.environ.get("CDPATH"), domain)
            self.train_domain_filelist[domain] = open(os.path.join(domain_path, f'{split}.txt'), 'r').readlines()

        if transform is not None:              
            self.transform = transform
        
    def __len__(self):
        return 4000

    def __getitem__(self, _):
        # 随机选择一个域
        domain = random.choice(list(self.train_domain_filelist.keys()))
        cur_file_list = self.train_domain_filelist[domain]
        
        # 从选定的域中随机选择一个样本
        idx = random.choice(range(len(cur_file_list)))

        # 获取样本路径
        pathA, pathB, pathLAB = cur_file_list[idx].strip().split('  ')
        imgA = Image.open(pathA).convert('RGB')
        imgB = Image.open(pathB).convert('RGB')
        lab = Image.open(pathLAB).convert('L')
        if lab.size[0]>256:
            transform = self.transform['crop']
        else:
            transform = self.transform['no_crop']

        imgA = np.asarray(imgA)  # H×W×3 的 numpy 数组
        imgB = np.asarray(imgB)
        lab = np.asarray(lab)//255    # H×W 的 numpy 数组

        augmented = transform(
            image=imgA,    # 第一时相 H×W×3 的 numpy 数组
            image1=imgB,   # 第二时相 H×W×3 的 numpy 数组
            mask=lab        # 标签 H×W 的 numpy 数组
        )
        inputA = augmented['image']   # Tensor, C×H×W
        inputB = augmented['image1']  # Tensor, C×H×W
        label = augmented['mask']    # Tensor, H×W
        concat = torch.cat((inputA, inputB), dim=0)

        return {'imgAB':concat, 'lab':label, 'pathA': pathA, 'pathB': pathB}