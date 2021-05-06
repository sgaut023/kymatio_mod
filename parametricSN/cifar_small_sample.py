"""
Classification on CIFAR10 (ResNet)
==================================

Based on pytorch example for CIFAR10
"""

import sys
from pathlib import Path 
import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd()))
import numpy as np
import time
import copy
import os
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
import argparse
import kymatio.datasets as scattering_datasets
from kymatio import Scattering2D
from kymatio.scattering2d.core.scattering2d import scattering2d
import torch.nn as nn
from numpy.random import RandomState
from parametricSN.utils.context import get_context
from parametricSN.utils.wavelet_visualization import get_filters_visualization
from parametricSN.utils.create_filters import create_filters_params_random,  morlets
from parametricSN.utils.create_filters import create_filters_params, construct_scattering
from parametricSN.utils.log_mlflow import visualize_loss, visualize_learning_rates, log_mlflow
from parametricSN.utils.log_mlflow import log_mlflow


     
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
            self.fc1 = nn.Linear(3*32*32, 256)
            self.fc2 = nn.Linear(256, num_classes)
        else:
            self.fc1=  nn.Linear(3*64*81, 256)
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return self.fc2(x)

def get_lr_scheduler(optimizer, params, steps_per_epoch ):
    if params['model']['scheduler'] =='OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=steps_per_epoch, 
                                                    epochs= params['model']['epoch'] , three_phase=params['model']['three_phase'])
    elif params['model']['scheduler'] =='CosineAnnealingLR':
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = params['model']['T_max'], eta_min = 1e-8)
    elif params['model']['scheduler'] =='LambdaLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    elif params['model']['scheduler'] =='CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, 
                                            step_size_up=params['model']['T_max']*2,
                                             mode="triangular2")
    else:
        raise NotImplemented(f"Scheduler {params['model']['scheduler']} not implemented")
    return scheduler

def get_dataset(params, use_cuda):
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
    
    return train_loader,  test_loader 

def create_scattering(params, device, use_cuda):
    J = params['scattering']['J']
    M, N= params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N']
    scattering = Scattering2D(J=J, shape=(M, N))
    K = 81*3
    model = LinearLayer(K, params['model']['width']).to(device)
    if use_cuda:
        scattering = scattering.cuda()
    phi, psi  = scattering.load_filters()
    params_filters = []
    wavelets = None
    # if we want to optimize the parameters used to create the filters
    if params['model']['mode'] == 'scattering_dif' :      
        # We can initialize the parameters randomly or as the kymatio package does
        if params['model']['init_params'] =='Random':
            params_filters = create_filters_params_random( J* scattering.L , True,  2)
        else:
            n_filters = J*scattering.L
            params_filters = create_filters_params(J, scattering.L, True,  2)

        wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                        params_filters[2], params_filters[3], device=device )
        for i,d in enumerate(psi):
            d[0]=wavelets[i] 
           
    return  model, scattering, psi, wavelets, params_filters

def test(model, device, test_loader, is_scattering_dif, scattering, psi, params_filters):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        if is_scattering_dif:
                wavelets  = morlets((scattering.M_padded, scattering.N_padded), params_filters[0], 
                                    params_filters[1], params_filters[2], params_filters[3], device=device )
                for i,d in enumerate(psi):
                    d[0]=wavelets[i].unsqueeze(2).real.contiguous().to(device) 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)  
            if is_scattering_dif:
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

def train(model, device, train_loader, is_scattering_dif, scheduler, optimizer, epoch, scattering, psi, params_filters):
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
        optimizer.step()
        scheduler.step()

        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probabilityd
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    #if batch_idx % 50 == 0:
    print('Train Epoch: {}\t Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%): '.format(
                epoch, train_loss, correct , len(train_loader.dataset),
                train_accuracy))
    return train_loss, train_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, default='parameters.yml',
                        help="YML Parameter File Name")
    args = parser.parse_args()
    catalog, params = get_context(args.param_file)
    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    is_scattering_dif = False
    train_loader,  test_loader = get_dataset(params, use_cuda)
    
    # if the mode is 'scattering' or 'scattering_dif', then we need to construct the filters phi
    if params['model']['mode'] == 'scattering_dif' or params['model']['mode'] == 'scattering':
        
        model, scattering, psi, wavelets, params_filters = create_scattering(params, device, use_cuda)
        lr_scattering = params['model']['lr_scattering']  
        lr_orientation = params['model']['lr_orientation']  
        # to be able to use the kymatio code, we need 
        # to use the same data structure which is a list of dictionnary
        
        # visualize wavlet filters before training
        filters_plots_before = {}
       # num_row = int(n_filters  /8 )
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, num_row = 2 , num_col =8 , mode =mode)
            filters_plots_before [mode]  = f  
        
        #build psi skeleton
        #psi_skeleton = copy.deepcopy(psi)
        psi_skeleton = psi
        for i,d in enumerate(psi_skeleton):
            d[0]=None
   
    # here we raw images are directly fed to the linear layers
    elif params['model']['mode'] == 'standard':
        model = LinearLayer(8, params['model']['width'], standard=True).to(device)
        scattering = Identity()
        psi = None
        filters_plots_before = {}
        psi_skeleton = None
        params_filters =[] 


    # Optimizer
    lr = params['model']['lr']
    #M = params['model']['learning_schedule_multi']
    #drops = [60*M,120*M,160*M]
    test_acc = []
    start_time = time.time()
    train_losses, test_losses , train_accuracies = [], [], []
    lrs, lrs_scattering, lrs_orientation = [], [], []
    
    if params['model']['mode'] == 'scattering_dif':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, 
                                    {'params': params_filters[0], 'lr': lr_orientation },
                                    {'params': [params_filters[1], params_filters[2], params_filters[3]], 
                                    'lr': lr_scattering}],
                                    lr=lr, momentum=params['model']['momentum'],
                                    weight_decay=params['model']['weight_decay'])
    else: 
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=params['model']['momentum'],
                                                weight_decay=params['model']['weight_decay'])
    epochs  = params['model']['epoch']
    scheduler = get_lr_scheduler(optimizer, params, len(train_loader))

    for epoch in  range(0, epochs) :
        # save learning rates for mlflow
        lrs.append(optimizer.param_groups[0]['lr'])
        if params['model']['mode'] == 'scattering_dif':
            lrs_orientation.append(optimizer.param_groups[1]['lr'])
            lrs_scattering.append(optimizer.param_groups[2]['lr'])
        
        # training 
        train_loss, train_accuracy = train(model, device, train_loader, is_scattering_dif, scheduler, optimizer,  epoch+1, scattering, psi_skeleton, params_filters )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

       # at every 10 epochs, the test accuracy will be displayed
        if epoch%10==0:
            accuracy, test_loss = test(model, device, test_loader, is_scattering_dif, scattering, psi_skeleton, params_filters )
            test_losses.append(test_loss )
            test_acc.append(accuracy)

    # plot train and test loss
    f_loss = visualize_loss(train_losses ,test_losses, step_test = 10, y_label='loss')
    f_accuracy = visualize_loss(train_accuracies ,test_acc, step_test = 10, y_label='accuracy')
    
    #visualize learning rates
    f_lr = visualize_learning_rates(lrs, lrs_orientation, lrs_scattering)

    #visualize filters
    filters_plots_after= {}
    if psi is not None:
        if params['model']['mode'] == 'scattering_dif':
            wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                                params_filters[2], params_filters[3], device=device )
        
            for i,d in enumerate(psi):
                d[0]=wavelets[i] 
    
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, num_row = 2 , num_col =8 , mode =mode)
            filters_plots_after[mode]  = f  
    
    # save metrics and params in mlflow
    log_mlflow(params, model, np.array(test_acc), start_time, 
               filters_plots_before , filters_plots_after, 
               [f_loss,f_accuracy], f_lr )

if __name__ == '__main__':
    main()
