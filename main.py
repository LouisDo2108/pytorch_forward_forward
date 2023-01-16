import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

from dataloaders.base import CustomImageDataset
from dataloaders.mnist import MNIST_loaders
from dataloaders.cifar10 import CIFAR10_loaders
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda,Resize
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def overlay_y_on_x(x, y):
    x_ = x.clone()                            # Deep copy the input x
    x_[:, :10] *= 0.0                         # Make a zero matrix with the same shape as x
    x_[range(x.shape[0]), y] = x.max()        # Fill all columns with corresponding ground truth with the max of input x
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
            

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        )

    def train(self, x_pos, x_neg):
        
        # for i in tqdm(range(self.num_epochs)):
        for i in range(self.num_epochs):
            
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
            
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

def main(mode="HUSHEM"):
    torch.manual_seed(42)
    
    if mode == "MNIST":
        mnist = MNIST_loaders(
            root="/home/louis/code/ff_algo",
            transform=None,
            train_batch_size=5000,
            test_batch_size=5000
        )
        train_loader, test_loader = mnist.get_loader()
        net = Net([784, 500, 500])
    elif mode == "cifar10":
        cifar10 = CIFAR10_loaders(
            root="/home/louis/code/ff_algo",
            transform=None,
            train_batch_size=5000,
            test_batch_size=5000
        )
        train_loader, test_loader = cifar10.get_loader()
        net = Net([3072, 500, 500])
    else:
        transform = Compose([
            ToTensor(),
            Resize((64, 64)),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))
        ])
        # 17161
        custom_dataset = CustomImageDataset(
            img_dir="/home/louis/code/ff_algo/semen/HuSHem",
            transform=transform,
            target_transform=None,
        )
        
        random_seed = 42
        test_size = 0.2
        num_train = len(custom_dataset)
        indices = list(range(num_train))
        split = int(np.floor(test_size * num_train))

        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_loader = DataLoader(
            custom_dataset,
            batch_size=16,
            sampler=train_sampler,
            drop_last=True
        )
        
        test_loader = DataLoader(
            custom_dataset,
            batch_size=16,
            sampler=test_sampler,
            drop_last=True
        )

        net = Net([12288, 1000, 1000])
    
    for epoch in tqdm(range(10)):
        
        train_error = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_pos = overlay_y_on_x(x, y)

            rnd = torch.randperm(x.size(0))
            x_neg = overlay_y_on_x(x, y[rnd])
            net.train(x_pos, x_neg)
            
            train_error.append(1.0 - net.predict(x).eq(y).float().mean().item())
        print("train error:", np.array(np.mean(train_error)))

        test_error = []
        for x_te, y_te in test_loader:
            x_te, y_te = x_te.to(device), y_te.to(device)
            test_error.append(1.0 - net.predict(x_te).eq(y_te).float().mean().item())
        print('test error:', np.array(np.mean(test_error)))
            
        

# def test():
#     x = torch.randint(0, 10, (5, 10), dtype=torch.float32)
#     y = torch.randint(0, 10, (5, 1), dtype=torch.float32)
#     print(x)
#     print(y)
#     print(overlay_y_on_x(x, y))
    
    
if __name__ == "__main__":

    main()
    
    """
    MNIST_DATALOADER = [(X, Y), (X, Y), (X, Y)... ] 0->n (list, string => iterable)
    for x, y in MNIST_DATALOADER:
        img, label = x, y
    next(iter(MNIST_DATALOADER))
    
    MNIST_DATASET = ;
    len(MNIST_DATASET)
    MNIST_DATASET[0] -> (x0, y0)
    """

