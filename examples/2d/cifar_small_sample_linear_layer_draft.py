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
import copy
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


def test(model, device, test_loader, is_scattering_dif, scattering, psi, params_filters):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)
            if is_scattering_dif:
                wavelets  = morlets((scattering.M_padded, scattering.N_padded), params_filters[0], 
                                    params_filters[1], params_filters[2], params_filters[3], device=device )
                for i,d in enumerate(psi):
                    d[0]=wavelets[i].unsqueeze(2).real.contiguous().to(device)   
                data = construct_scattering(data, scattering, psi)
            else:
                data = scattering(data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def train(model, device, train_loader, is_scattering_dif, optimizer, epoch, scattering, psi, params_filters):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        if is_scattering_dif:
            wavelets  = morlets((scattering.M_padded, scattering.N_padded), params_filters[0], 
                                params_filters[1], params_filters[2], params_filters[3], device=device )
            for i,d in enumerate(psi):
                d[0]=wavelets[i].unsqueeze(2).real.contiguous().to(device)   
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

def create_filters_params(J, L):
    '''
        Create reusable filters parameters: orientations, xis, sigmas, sigmas
    '''
    orientations = []
    xis = []
    sigmas = []
    slants = []

    for j in range(J):
        for theta in range(L):
            sigmas.append(0.8 * 2**j)
            theta = ((int(L-L/2-1)-theta) * np.pi / L)
            xis.append(3.0 / 4.0 * np.pi /2**j)
            slants.append(4.0/L)
            orientations.append(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32))
            #R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
             
    xis = torch.FloatTensor(xis)
    sigmas = torch.FloatTensor(sigmas)
    slants = torch.FloatTensor(slants)
    orientations = torch.FloatTensor(orientations)   

    params = [orientations[:, 0], xis, sigmas, slants]
    # for param in params:
    #     param.requires_grad = True
    return  params

def raw_morlets(grid_or_shape, wave_vectors, gaussian_bases, morlet=True, ifftshift=True, fft=True):
    n_filters, n_dim = wave_vectors.shape
    assert gaussian_bases.shape == (n_filters, n_dim, n_dim)
    device = wave_vectors.device
    if isinstance(grid_or_shape, tuple):
        shape = grid_or_shape
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)
    else:
        shape = grid_or_shape.shape
        grid = grid
        _or_shape
    waves = torch.exp(1.0j * torch.matmul(grid.T, wave_vectors.T).T)
    gaussian_directions = torch.matmul(grid.T, gaussian_bases.T.reshape(n_dim, n_dim * n_filters)).T
    gaussian_directions = gaussian_directions.reshape((n_dim, n_filters) + shape)
    radii = torch.norm(gaussian_directions, dim=0)
    gaussians = torch.exp(-0.5 * radii ** 2)
    signal_dims = list(range(1, n_dim + 1))
    gabors = gaussians * waves
    if morlet:
        gaussian_sums = gaussians.sum(dim=signal_dims, keepdim=True)
        gabor_sums = gabors.sum(dim=signal_dims, keepdim=True).real
        morlets = gabors - gabor_sums / gaussian_sums * gaussians
        filters = morlets
    else:
        filters = gabors
    if ifftshift:
        filters = torch.fft.ifftshift(filters, dim=signal_dims)
    if fft:
        filters = torch.fft.fftn(filters, dim=signal_dims)
    return filters

def morlets(grid_or_shape, orientations, xis, sigmas, slants, device=None, morlet=True, ifftshift=True, fft=True):
    n_filters, ndim = orientations.shape
    if device is None:
        device = orientations.device
    orientations = orientations / (torch.norm(orientations, dim=1, keepdim=True) + 1e-19)
    wave_vectors = orientations * xis[:, np.newaxis]
    _, _, gauss_directions = torch.linalg.svd(orientations[:, np.newaxis])
    gauss_directions = gauss_directions / sigmas[:, np.newaxis, np.newaxis]
    indicator = torch.arange(ndim) < 1
    slant_modifications = (1.0 * indicator + slants[:, np.newaxis] * ~indicator).to(gauss_directions.device)
    gauss_directions = gauss_directions * slant_modifications[:, :, np.newaxis]
    wavelets = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft)
    return wavelets

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
        
        #build psi skeleton
        psi_skeleton = copy.deepcopy(psi)
        for i,d in enumerate(psi_skeleton):
            d[0]=None

        if params['model']['mode'] == 'scattering_dif' :
            lr_scattering = params['model']['lr_scattering']  
            params_filters = create_filters_params(J, scattering.L)
            wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                            params_filters[2], params_filters[3], device=device )
            for i,d in enumerate(psi):
                d[0]=wavelets[i] 
            is_scattering_dif = True

        # visualize wavlet filters before training
        filters_plots_before = {}
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, num_row = 2 , num_col =8 , mode =mode)
            filters_plots_before [mode]  = f  
        

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
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
                # optimizer = torch.optim.SGD([{'params': params_filters, 'lr': lr_scattering}, 
                #                         {'params': model.parameters()}], lr=lr, momentum=0.9,
                #                         weight_decay=0.0005)
                lr_scattering*=0.2
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2


        train(model, device, train_loader, is_scattering_dif, optimizer, epoch+1, scattering, psi_skeleton, params_filters )
        if epoch%10==0:
            test_acc.append(test(model, device, test_loader, is_scattering_dif, scattering, psi_skeleton, params_filters ))

    #visualize filters
    filters_plots_after= {}
    wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                            params_filters[2], params_filters[3], device=device )
    for i,d in enumerate(psi):
        d[0]=wavelets[i] 
    if psi is not None:
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, num_row = 2 , num_col =8 , mode =mode)
            filters_plots_after[mode]  = f  
    
    log_mlflow(params, model, np.array(test_acc), start_time, filters_plots_before , filters_plots_after)

if __name__ == '__main__':
    main()
