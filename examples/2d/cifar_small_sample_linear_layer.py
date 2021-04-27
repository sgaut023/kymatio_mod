"""
Classification on CIFAR10 (ResNet)
==================================

Based on pytorch example for CIFAR10
"""

import sys
from pathlib import Path 
sys.path.append(str(Path.cwd()))
import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import torch
import argparse
import kymatio.datasets as scattering_datasets
from kymatio.scattering2d.core.scattering2d import scattering2d
import torch.nn as nn
from numpy.random import RandomState
import numpy as np
import pickle
import time
import mlflow
import os
from examples.utils.context import get_context
from examples.utils.wavelet_visualization import get_filters_visualization



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
        

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_channels,  k=2, n=4, num_classes=10, standard=False):
        super().__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k * 3
        if standard:
            self.fc = nn.Linear(3*32*32, num_classes)
        else:
            self.fc = nn.Linear(3*64*81, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)



def train(model, device, train_loader, is_scattering_dif, optimizer, epoch, scattering, psi):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        if is_scattering_dif:
            data = construct_scattering(data, scattering, psi)
        else:
            data = scattering(data)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, is_scattering_dif, scattering, psi):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)
            if is_scattering_dif:
                data = construct_scattering(data, scattering, psi)
            else:
                data = scattering(data)
            output = model(data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def log_mlflow(params, model, test_acc, start_time,filters_plots_before, filters_plots_after ):
    duration = (time.time() - start_time)
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    with mlflow.start_run():
        mlflow.log_params(params['model'])   
        mlflow.log_params(params['scattering'])
        mlflow.log_params(params['preprocess']['dimension'])
        #mlflow.log_param('Duration', duration)
        mlflow.log_metric('Final Accuracy', test_acc[-1])
        mlflow.pytorch.log_model(model, artifact_path = 'model')

        #save filters 
        for key in filters_plots_before:
            mlflow.log_figure(filters_plots_before[key], f'filters_before/{key}.png')
            mlflow.log_figure(filters_plots_after[key], f'filters_after/{key}.png')
        
        # saving all accuracies
        np.savetxt('accuracy.csv', test_acc, delimiter=",")
        mlflow.log_artifact('accuracy.csv', 'accuracy_values')
        os.remove('accuracy.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, default='parameters.yml',
                        help="YML Parameter File Name")
    args = parser.parse_args()
    catalog, params = get_context(args.param_file)
    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    is_scattering_dif = False
    if params['model']['mode'] == 'scattering_dif' or params['model']['mode'] == 'scattering':
        J = params['scattering']['J']
        M, N= params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N']
        scattering = Scattering2D(J=J, shape=(M, N))
        K = 81*3
        model = LinearLayer(K, params['model']['width']).to(device)
        if use_cuda:
            scattering = scattering.cuda()
        
        phi, psi  = scattering.load_filters()
        # visualize wavlet filters before training
        filters_plots_before = {}
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, num_row = 2 , num_col =8 , mode =mode)
            filters_plots_before [mode]  = f  
        if params['model']['mode'] == 'scattering_dif' :
            lr_scattering = params['model']['lr_scattering']  
            filters = make_filters_diff(psi)
            is_scattering_dif = True

    else:
        model = LinearLayer(8, params['model']['width'], standard=True).to(device)
        scattering = Identity()
        psi = None
        filters_plots_before = {}


    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    #####cifar data
    cifar_data = datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    # Extract a subset of X samples per class
    prng = RandomState(params['model']['seed'])
    random_permute = prng.permutation(np.arange(0, 5000))[0:params['model']['num_samples']]
    indx = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute] for classe in range(0, 10)])

    cifar_data.data, cifar_data.targets = cifar_data.data[indx], list(np.array(cifar_data.targets)[indx])
    train_loader = torch.utils.data.DataLoader(cifar_data,
                                               batch_size=params['model']['batch_size'], shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    # Optimizer
    lr = params['model']['lr']
    M = params['model']['learning_schedule_multi']
    drops = [60*M,120*M,160*M]
    test_acc = []
    start_time = time.time()
    for epoch in range(0, params['model']['epoch']):
    #for epoch in range(0, 1):
        if epoch in drops or epoch==0:
            if is_scattering_dif:
                optimizer = torch.optim.SGD([{'params': filters, 'lr': lr_scattering}, 
                                        {'params': model.parameters()}], lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
                lr_scattering*=0.2
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2


        train(model, device, train_loader, is_scattering_dif, optimizer, epoch+1, scattering, psi)
        if epoch%10==0:
            test_acc.append(test(model, device, test_loader, is_scattering_dif, scattering, psi))

    #visualize filters
    filters_plots_after= {}
    if psi is not None:
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, num_row = 2 , num_col =8 , mode =mode)
            filters_plots_after[mode]  = f  
    
    log_mlflow(params, model, np.array(test_acc), start_time, filters_plots_before , filters_plots_after)

if __name__ == '__main__':
    main()
