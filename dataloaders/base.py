import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import os
from natsort import natsorted

class BaseLoader:
    
    def __init__(self, root, dataset_name, transform, train_batch_size, test_batch_size):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))
        ])
        self.datasets = {
            "MNIST": 
            {
                "train": MNIST(os.path.join(root, "data"), train=True, download=True, transform=self.transform),
                "test": MNIST(os.path.join(root, "data"), train=False, download=True, transform=self.transform),
            },
            "CIFAR10": 
                {
                    "train": CIFAR10(os.path.join(root, "data"), train=True, download=True, transform=self.transform),
                    "test": CIFAR10(os.path.join(root, "data"), train=False, download=True, transform=self.transform),
                } 
        }
        self.root = os.path.join(root, "data", dataset_name)

        self.train_loader = None
        self.test_loader = None
        
    def get_loader(self):
        return self.train_loader, self.test_loader
  

from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import numpy as np
    
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.hushem_list = []
        
        for ix, cls in enumerate(natsorted(os.listdir(self.img_dir))):
            for img in os.listdir(os.path.join(self.img_dir, cls)):
                self.hushem_list.append({
                    "img_path": os.path.join(self.img_dir, cls, img),
                    "cls": ix
                })

    def __len__(self):
        return len(self.hushem_list)

    def __getitem__(self, idx):
        x_path, y = self.hushem_list[idx]["img_path"], self.hushem_list[idx]["cls"]
        x = np.array(Image.open(x_path))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y