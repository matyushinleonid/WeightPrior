from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm_notebook as tqdm

from models import *

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
                   train=True,
                   download=True,
                   transform=transforms.ToTensor()
                  ),
    batch_size=128
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, 
                   transform=transforms.ToTensor()
                  ),
    batch_size=128
)

model = Net()
model.cuda()

kernels_7x7 = []
kernels_5x5 = []

for n_model in tqdm(range(1,300)):
    
    torch.manual_seed(n_model)
    
    model = Net()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-2)
    
    for epoch in range(1,4):      
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

#         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             epoch, batch_idx * len(data), len(train_loader.dataset),
#             100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)

#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))

    if 100. * correct / len(test_loader.dataset) > 93:
        kernels_7x7.append(list(model.parameters())[0].cpu().data.numpy())
        kernels_5x5.append(list(model.parameters())[1].cpu().data.numpy())

        np.save('./kernels/kernels_7x7.npy', np.vstack(kernels_7x7))

        np.save('./kernels/kernels_5x5.npy', np.vstack(kernels_5x5))