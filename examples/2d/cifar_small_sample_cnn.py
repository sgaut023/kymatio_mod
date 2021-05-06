"""
Classification on CIFAR10 (ResNet)
==================================

Based on pytorch example for CIFAR10
"""

import sys
from pathlib import Path 
import matplotlib.pyplot as plt
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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Scattering2dResNet(nn.Module):
    def __init__(self, in_channels,  k=2, n=4, num_classes=10,standard=False):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        if standard:

            self.init_conv = nn.Sequential(
                nn.Conv2d(3, self.ichannels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
                nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.standard = False

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            x = x.view(x.size(0), self.K, 8, 8)

        x = self.init_conv(x)

        if self.standard:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),accuracy ))
    return accuracy, test_loss

def train(model, device, train_loader, is_scattering_dif, optimizer, epoch, scattering, psi, params_filters):
    model.train()
    correct = 0
    train_loss = 0
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
        #print(f'orientations min:{params_filters[0].grad.min()}, max: {params_filters[0].grad.max()}')
        #print(f'slant min:{params_filters[3].grad.min()}, max: {params_filters[3].grad.max()}')
        #params_filters[0].grad = params_filters[0].grad * 10
        optimizer.step()

        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    #if batch_idx % 50 == 0:
    print('Train Epoch: {}\t Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%): '.format(
                epoch, train_loss, correct , len(train_loader.dataset),
                train_accuracy))
    return train_loss, train_accuracy

def visualize_loss(loss_train ,loss_test, step_test = 10):
    f = plt.figure (figsize=(7,7))
    plt.plot(np.arange(len(loss_test)*step_test, step=step_test), loss_test, label='Test Loss') 
    plt.plot(np.arange(len(loss_train)), loss_train, label= 'Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend() 
    return f     

def log_mlflow(params, model, test_acc, start_time,filters_plots_before, filters_plots_after , figures_plot):
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
        mlflow.log_figure(figures_plot[0], f'plot/train_test_loss.png')
        mlflow.log_figure(figures_plot[1], f'plot/train_test_accuracy.png')
        # saving all accuracies
        np.savetxt('accuracy.csv', test_acc, delimiter=",")
        mlflow.log_artifact('accuracy.csv', 'accuracy_values')
        os.remove('accuracy.csv')

# def create_filters_params(J, L, is_scattering_dif, ndim):
#     n_filters = J*L
#     sigmas = np.log(np.random.uniform(np.exp(0), np.exp(3), n_filters ))
#     # For the orientation, choose uniform on the circle 
#     #(can init some 2d gaussian values then divide by their norm 
#     # or take complex exponential/ cos & sin of uniform between 0 and 2pi).
#     orientations = np.random.normal(0,1,(n_filters,ndim)) 
#     norm = np.linalg.norm(orientations, axis=1).reshape(orientations.shape[0], 1)
#     orientations = orientations/norm
#     slants = np.random.uniform(0.5, 2,n_filters )# like uniform between 0.5 and 1.5.
#     xis = np.random.uniform(1, 2, n_filters )
    
#     xis = torch.FloatTensor(xis)
#     sigmas = torch.FloatTensor(sigmas)
#     slants = torch.FloatTensor(slants)
#     orientations = torch.FloatTensor(orientations) 
#     params = [orientations, xis, sigmas, slants]
#     if is_scattering_dif:
#         for param in params:
#             param.requires_grad = True
#     return  params

def create_filters_params(J, L, is_scattering_dif, ndim =2):
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
            #orientations.append(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32))
            R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
            orientations.append(R_inv)
    xis = torch.FloatTensor(xis)
    sigmas = torch.FloatTensor(sigmas)
    slants = torch.FloatTensor(slants)
    orientations = torch.FloatTensor(orientations)   

    params = [orientations[:, 0], xis, sigmas, slants]
    if is_scattering_dif:
        for param in params:
            param.requires_grad = True
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
    norm_factors = (2 * 3.1415 * sigmas * sigmas / slants).unsqueeze(1)
    norm_factors = norm_factors.expand([n_filters,grid_or_shape[0]]).unsqueeze(2).repeat(1,1,grid_or_shape[1])
    wavelets = wavelets / norm_factors

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
        model = Scattering2dResNet(K, params['model']['width']).to(device)
        if use_cuda:
            scattering = scattering.cuda()
        phi, psi  = scattering.load_filters()
        
        #build psi skeleton
        psi_skeleton = copy.deepcopy(psi)
        for i,d in enumerate(psi_skeleton):
            d[0]=None

        if params['model']['mode'] == 'scattering_dif' :
            lr_scattering = params['model']['lr_scattering']  
            params_filters = create_filters_params(J, scattering.L, True,  2)
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
    #drops= [200,400,800]
    test_acc = []
    start_time = time.time()
    train_losses, train_accuracies = [], []
    test_losses = []
    

    for epoch in range(0, params['model']['epoch']):
    #for epoch in range(0, 1):
        if epoch in drops or epoch==0:
            if is_scattering_dif:
                # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                #                         weight_decay=0.0005)
                optimizer = torch.optim.SGD([{'params': params_filters, 'lr': lr_scattering}, 
                                        {'params': model.parameters()}], lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
                lr_scattering*=0.2
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2


        train_loss, train_accuracy = train(model, device, train_loader, is_scattering_dif, optimizer, epoch+1, scattering, psi_skeleton, params_filters )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        if epoch%10==0:
            accuracy, test_loss = test(model, device, test_loader, is_scattering_dif, scattering, psi_skeleton, params_filters )
            test_losses.append(test_loss )
            test_acc.append(accuracy)

    # plot train and test loss
    f_loss = visualize_loss(train_losses ,test_losses, step_test = 10)
    f_accuracy = visualize_loss(train_accuracies ,test_acc, step_test = 10)
    
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
    
    log_mlflow(params, model, np.array(test_acc), start_time, filters_plots_before , filters_plots_after, [f_loss,f_accuracy] )

if __name__ == '__main__':
    main()
