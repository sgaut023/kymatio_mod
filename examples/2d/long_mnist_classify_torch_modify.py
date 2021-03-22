"""
Classification of Few Sample MNIST with Scattering
=====================================================================
Here we demonstrate a simple application of scattering on the MNIST dataset.
We use 5000 MNIST samples to train a linear classifier. Features are normalized by batch normalization.
Please also see more extensive classification examples/2d/cifar.py
that consider training CNNs on top of the scattering. These are
are included as  executable script with command line arguments
"""

###############################################################################
# If a GPU is available, let's use it!
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
import numpy as np
from numpy.random import RandomState
import torch
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets
from kymatio.scattering2d.core.scattering2d import scattering2d
import torch.nn as nn
from kymatio.torch import Scattering2D
import torch.optim
import math
import torch.nn.functional as F
from tqdm import tqdm


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

def construct_scattering(input, scattering, psi):
    if not torch.is_tensor(input):
        raise TypeError('The input should be a PyTorch Tensor.')

    if len(input.shape) < 2:
        raise RuntimeError('Input tensor must have at least two dimensions.')

    if not input.is_contiguous():
        raise RuntimeError('Tensor must be contiguous.')

    if (input.shape[-1] != scattering.N or input.shape[-2] != scattering.M) and not scattering.pre_pad:
        raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (scattering.M, scattering.N))

    if (input.shape[-1] != scattering.N_padded or input.shape[-2] != scattering_padded) and scattering.pre_pad:
        raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (scattering.M_padded, scattering.N_padded))

    if not scattering.out_type in ('array', 'list'):
        raise RuntimeError("The out_type must be one of 'array' or 'list'.")

    # phi, psi  = scattering.load_filters()
    # make_filters_diff(psi)
    # scattering.psi = psi

    batch_shape = input.shape[:-2]
    signal_shape = input.shape[-2:]

    input = input.reshape((-1,) + signal_shape)

    S = scattering2d(input, scattering.pad, scattering.unpad, scattering.backend, scattering.J,
                        scattering.L, scattering.phi, psi, scattering.max_order, scattering.out_type)

    if scattering.out_type == 'array':
        scattering_shape = S.shape[-3:]
        S = S.reshape(batch_shape + scattering_shape)
    # else:
    #     scattering_shape = S[0]['coef'].shape[-2:]
    #     new_shape = batch_shape + scattering_shape

    #     for x in S:
    #         x['coef'] = x['coef'].reshape(new_shape)

    return S

def make_filters_diff(psi):
    """ This function make the filters differentiable """
    # 
    filters = []
    for j in range(len(psi)):
        for k, v in psi[j].items():
            if not isinstance(k, int):
                continue
            v.requires_grad = True
            filters.append(v)
    return filters
        

def train(model, device, train_loader, optimizer, scattering, psi):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = construct_scattering(data, scattering, psi)
        #optimizer.add_param_group({'params': scattering.psi[6][0][0][0]})
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        #scattering.register_filters()


def test(model, device, test_loader, scattering, psi):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            data = construct_scattering(data, scattering, psi)
            output = model(data)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)
############################################################################
# Train a simple Hybrid Scattering + CNN model on MNIST.

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  
    torch.manual_seed(42)
    prng = RandomState(42)  
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    train_data = datasets.MNIST(
                    scattering_datasets.get_dataset_dir('MNIST'),
                    train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    # Extract a subset of 5000 samples from MNIST training
    random_permute=prng.permutation(np.arange(0,60000))[0:5000]
    train_data.data = train_data.data[random_permute]
    train_data.targets = train_data.targets[random_permute]
    train_loader = torch.utils.data.DataLoader(train_data,
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # Create the test loader on the full MNIST test set
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            scattering_datasets.get_dataset_dir('MNIST'),
            train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # Evaluate linear model on top of scattering
    scattering = Scattering2D(shape = (28, 28), J=2)
    K = 81 #Number of output coefficients for each spatial postiion

    if use_cuda:
        scattering = scattering.cuda()

    model = nn.Sequential(
        View(K, 7, 7),
        nn.BatchNorm2d(K),
        View(K * 7 * 7),
        nn.Linear(K * 7 * 7, 10)
    ).to(device)

    # Optimizer
    phi, psi  = scattering.load_filters()
    filters = make_filters_diff(psi)
    optimizer = torch.optim.SGD(filters + list(model.parameters()), lr=0.01, momentum=0.9,
                                weight_decay=0.0005)
    for epoch in range(0, 20):
        train( model, device, train_loader, optimizer, scattering, psi)
        print(f'EPOCH {epoch+1}/20 completed.')

    acc = test(model, device, test_loader, scattering, psi)
    print('Scattering order 2 linear model test accuracy: %.2f' % acc)

if __name__ == '__main__':
    main()