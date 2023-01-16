from .base import BaseLoader
from torch.utils.data import DataLoader

class CIFAR10_loaders(BaseLoader):
    
    def __init__(self, root, transform, train_batch_size, test_batch_size, dataset_name="CIFAR10"):
        super().__init__(root, dataset_name, transform, train_batch_size, test_batch_size)
        self.train_loader = DataLoader(
            self.datasets[dataset_name]['train'],
            batch_size=train_batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.datasets[dataset_name]['test'],
            batch_size=test_batch_size,
            shuffle=False
        )