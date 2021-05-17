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
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
import argparse
import kymatio.datasets as scattering_datasets
from kymatio import Scattering2D
import torch.nn as nn
from numpy.random import RandomState
from parametricSN.utils.context import get_context
from parametricSN.utils.wavelet_visualization import get_filters_visualization
from parametricSN.utils.create_filters import create_filters_params_random,  morlets
from parametricSN.utils.create_filters import update_psi
from parametricSN.utils.create_filters import create_filters_params, construct_scattering
from parametricSN.utils.log_mlflow import visualize_loss, visualize_learning_rates, log_mlflow
from parametricSN.utils.log_mlflow import log_mlflow
from parametricSN.utils.auto_augment import AutoAugment, Cutout
from parametricSN.utils.cifar_loader import SmallSampleController
from parametricSN.utils.Scattering2dResNet import Scattering2dResNet
     
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    
class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self,num_classes=10,  standard=False, n_coefficients=81, 
                M_coefficient=8, N_coefficient=8):
    super().__init__()

    if standard:
        fc1 = nn.Linear(3*32*32, 512)
    else:
        fc1=  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients),  512)
    
    self.layers = nn.Sequential(
      fc1,
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, num_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    x = x.view(x.shape[0], -1)
    return self.layers(x)


class LinearLayer(nn.Module):
    def __init__(self, num_classes=10,  standard=False, n_coefficients=81, 
                M_coefficient=8, N_coefficient=8) :
        super().__init__()

        if standard:
            self.fc1 = nn.Linear(3*32*32, 256)
            self.fc2 = nn.Linear(256, num_classes)
        else:
            self.fc1=  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients), 1024)
            self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def get_lr_scheduler(optimizer, params, steps_per_epoch, epoch ):

    if params['model']['scheduler'] =='OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params['model']['max_lr'], 
                                                        steps_per_epoch=steps_per_epoch, 
                                                        epochs= epoch, 
                                                        three_phase=params['model']['three_phase'],
                                                        div_factor=params['model']['div_factor'])
    elif params['model']['scheduler'] =='CosineAnnealingLR':
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = params['model']['T_max'], eta_min = 1e-8)
    elif params['model']['scheduler'] =='LambdaLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    elif params['model']['scheduler'] =='CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, 
                                            step_size_up=params['model']['T_max']*2,
                                             mode="triangular2")
    elif params['model']['scheduler'] =='StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.2)
    elif params['model']['scheduler'] == 'NoScheduler':
        scheduler = None
    else:
        raise NotImplemented(f"Scheduler {params['model']['scheduler']} not implemented")
    return scheduler

def get_dataset(params, use_cuda):    
    NUM_CLASSES = params['model']['num_classes']
    TRAIN_SAMPLE_NUM = params['model']['train_sample_num']
    VAL_SAMPLE_NUM = params['model']['test_sample_num']
    TRAIN_BATCH_SIZE = params['model']['train_batch_size']
    VAL_BATCH_SIZE = params['model']['test_batch_size']
    VALIDATION_SET_NUM = 1
    AUGMENT = params['model']['augment']
    CIFAR_TRAIN = True
    SEED = params['model']['seed'] #None means a random seed 
    DATA_DIR = Path(params['model']['data_root'])/params['model']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')

    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
        
    #CIFAR-10 normalization    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                  std=[0.247, 0.243, 0.261])

    #default normalization
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params['model']['dataset'] == 'cifar':
        #DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

        if AUGMENT == 'autoaugment':
            print("\n[get_dataset(params, use_cuda)] Augmenting data with AutoAugment augmentation")
            trainTransform = [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                AutoAugment(),
                Cutout()
            ]
        elif AUGMENT == 'original-cifar':
            print("\n[get_dataset(params, use_cuda)] Augmenting data with original-cifar augmentation")
            trainTransform = [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
            ]
        elif AUGMENT == 'noaugment':
            print("\n[get_dataset(params, use_cuda)] No data augmentation")
            trainTransform = []

        elif AUGMENT == 'glico':
            NotImplemented(f"augment parameter {AUGMENT} not implemented")
        else: 
            NotImplemented(f"augment parameter {AUGMENT} not implemented")
 

        transform_train = transforms.Compose(trainTransform + [transforms.ToTensor(), normalize]) 
        transform_val = transforms.Compose([transforms.ToTensor(), normalize]) #careful to keep this one same

        dataset_train = datasets.CIFAR10(root=DATA_DIR,train=True, #use train dataset
                    transform=transform_train, download=True)

        dataset_val = datasets.CIFAR10(root=DATA_DIR,train=False, #use test dataset
                    transform=transform_val, download=True)
        ss = SmallSampleController(trainSampleNum=TRAIN_SAMPLE_NUM, valSampleNum=VAL_SAMPLE_NUM, 
        trainBatchSize=TRAIN_BATCH_SIZE,valBatchSize=VAL_BATCH_SIZE, multiplier=VALIDATION_SET_NUM, 
        trainDataset=dataset_train, valDataset=dataset_val )  

    
        train_loader_in_list, test_loader_in_list, seed = ss.generateNewSet(
        device,workers=num_workers,valMultiplier=VALIDATION_SET_NUM,seed=SEED) #Sample from datasets

        params['model']['seed'] = seed
        train_loader, test_loader = train_loader_in_list[0], test_loader_in_list[0] 
    
    elif params['model']['dataset'] == 'kth':
        #DATA_DIR = '/NOBACKUP/gauthiers/KTH/'

        if params['model']['seed'] == None:
            params['model']['seed'] = int(time.time()) #generate random seed
        dim_M = params['preprocess']['dimension']['M']
        dim_N = params['preprocess']['dimension']['N']
        trainTransform = [
            transforms.RandomCrop((dim_M,dim_N )),
            transforms.RandomHorizontalFlip(),
        ]
        valTransform = [
            transforms.CenterCrop((dim_M,dim_N)),
        ]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(trainTransform + [transforms.ToTensor(), normalize]) 
        transform_val = transforms.Compose(valTransform + [transforms.ToTensor(), normalize]) #careful to keep this one same

        datasets_val = []
        for sample in ['a', 'b', 'c', 'd']:
            if params['model']['sample_set'] == sample:
                dataset = datasets.ImageFolder(root=Path(DATA_DIR)/f'sample_{sample}', #use train dataset
                                            transform=transform_train)
                dataset_train = dataset
            else:
                dataset = datasets.ImageFolder(root=Path(DATA_DIR)/f'sample_{sample}', #use train dataset
                                            transform=transform_val)
                datasets_val.append(dataset)
        
        dataset_val = torch.utils.data.ConcatDataset(datasets_val)
                
        train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                           num_workers = num_workers, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(dataset_val,
                                           batch_size=VAL_BATCH_SIZE, shuffle=True,
                                           num_workers = num_workers, pin_memory = True)


    else: 
        NotImplemented(f"Dataset {params['model']['dataset']} not implemented")

    
    return train_loader,  test_loader 

def create_scattering(params, device, use_cuda, seed =0 ):
    J = params['scattering']['J']
    M, N= params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N']
    scattering = Scattering2D(J=J, shape=(M, N))
    n_coefficients= 1 + 8*J + 8*8*J*(J-1)//2
    M_coefficient = params['preprocess']['dimension']['M']/(2**J)
    N_coefficient = params['preprocess']['dimension']['N']/(2**J)
    K = n_coefficients*3

    if params['model']['architecture'] == 'linear_layer':
        model = LinearLayer(n_coefficients=n_coefficients, 
                num_classes = params['model']['num_classes'],
                M_coefficient=M_coefficient, 
                N_coefficient=N_coefficient).to(device)
    elif params['model']['architecture'] == 'mlp': 
        model = MLP(n_coefficients=n_coefficients, 
                    num_classes = params['model']['num_classes'],
                    M_coefficient=M_coefficient, 
                    N_coefficient=N_coefficient).to(device)
    elif params['model']['architecture'] == 'cnn': 
        model = Scattering2dResNet(K, params['model']['width'], 
                                 num_classes = params['model']['num_classes']).to(device)
    else:
        raise NotImplemented(f"Model {params['model']['architecture']} not implemented")

    if use_cuda:
        scattering = scattering.cuda()
    phi, psi  = scattering.load_filters()
    params_filters = []
       
    # if we want to optimize the parameters used to create the filters
    #if params['model']['mode'] == 'scattering_dif' :      
    # We can initialize the parameters randomly or as the kymatio package does
    requires_grad = True
    if params['model']['mode']== 'scattering':
        requires_grad = False
    if params['model']['init_params'] =='Random':
        params_filters = create_filters_params_random( J* scattering.L ,  requires_grad,  2, seed)
    else:
        n_filters = J*scattering.L
        params_filters = create_filters_params(J, scattering.L,  requires_grad,  2)

    wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device )
    
    psi = update_psi(J, psi, wavelets, device)           
    return  model, scattering, psi, wavelets, params_filters

def test(model, device, test_loader, is_scattering_dif, scattering, psi, params_filters):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        if is_scattering_dif:
                wavelets  = morlets((scattering.M_padded, scattering.N_padded), params_filters[0], 
                                    params_filters[1], params_filters[2], params_filters[3], device=device )
                psi = update_psi(scattering.J, psi, wavelets, device) 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)  
            if is_scattering_dif:
                data = construct_scattering(data, scattering, psi)
            elif psi != None:
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
    wavelets  = morlets((scattering.M_padded, scattering.N_padded), params_filters[0], 
                                params_filters[1], params_filters[2], params_filters[3], device=device )
    psi = update_psi(scattering.J, psi, wavelets, device) 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        if is_scattering_dif:
            wavelets = morlets((scattering.M_padded, scattering.N_padded), params_filters[0], 
                                params_filters[1], params_filters[2], params_filters[3], device=device )
            psi = update_psi(scattering.J, psi, wavelets, device) 
            data = construct_scattering(data, scattering, psi)
        elif psi != None:
            data = construct_scattering(data, scattering, psi)
        else:
            data = scattering(data)
            
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if scheduler != None:
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


def override_params(args,params):
    for k,v in args.__dict__.items():
        if v != None and k != "param_file":
            print(k,v)
            params["model"][k] = v


    return params



def run_train(args):
    catalog, params = get_context(args.param_file)
    params = override_params(args,params)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    is_scattering_dif = False
    train_loader,  test_loader = get_dataset(params, use_cuda)
    J = params['scattering']['J']
    epochs  = params['model']['epoch']

    # if the mode is 'scattering' or 'scattering_dif', then we need to construct the filters phi
    if params['model']['mode'] == 'scattering_dif':
        is_scattering_dif = True
        model, scattering, psi, wavelets, params_filters = create_scattering(params, device, use_cuda, seed = params['model']['seed'])
        lr_scattering = params['model']['lr_scattering']  
        lr_orientation = params['model']['lr_orientation']  
        
        filters_plots_before = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(psi, J, 8, mode =mode) 
            filters_plots_before [mode]  = f  
        
        psi_skeleton = psi #build psi skeleton (kymatio data structure)

        #set optimizer
        parameters = [
            {'params': model.parameters()},
            {'params': params_filters[0], 'lr': lr_orientation},
            {'params': [params_filters[1], params_filters[2],
             params_filters[3]],'lr': lr_scattering}
        ]

        if params['model']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(parameters,lr=params['model']['lr'], 
                betas=(0.9, 0.999), eps=1e-08, 
                weight_decay=params['model']['weight_decay'], amsgrad=False)

        elif params['model']['optimizer'] == 'sgd': 
            optimizer = torch.optim.SGD(parameters,lr=params['model']['lr'], 
            momentum=params['model']['momentum'],weight_decay=params['model']['weight_decay'])
        else:
            print("Invalid optimizer parameter passed")
        
        scheduler = get_lr_scheduler(optimizer, params, len(train_loader), 
                                     params['model']['epoch_scattering'])

    elif params['model']['mode'] == 'scattering':
        model, scattering, psi, wavelets, params_filters = create_scattering(params, device, use_cuda)
        #scattering.psi = psi
        lr_scattering = None 
        lr_orientation = None
        psi_skeleton = psi
        
        filters_plots_before = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(psi, J,8, mode =mode) 
            filters_plots_before [mode]  = f  
    
        optimizer = torch.optim.SGD(model.parameters(), lr=params['model']['lr'], 
        momentum=params['model']['momentum'], weight_decay=params['model']['weight_decay'])
        scheduler = get_lr_scheduler(optimizer, params, len(train_loader), 
                                     epochs)

    elif params['model']['mode'] == 'standard': #use the linear model only
        if params['model']['architecture'] == 'linear_layer':
            model = LinearLayer(8, params['model']['width'], standard=True).to(device)
        elif params['model']['architecture'] == 'cnn':
            model = Scattering2dResNet(8, params['model']['width'],standard=True).to(device)
        else:
            raise NotImplemented(f"Model {params['model']['architecture']} not implemented")

        
        scattering = Identity()
        psi = None
        filters_plots_before = {}
        psi_skeleton = None
        params_filters =[] 
        optimizer = torch.optim.SGD(model.parameters(), lr=params['model']['lr'], 
        momentum=params['model']['momentum'], weight_decay=params['model']['weight_decay'])
        scheduler = get_lr_scheduler(optimizer, params, len(train_loader), 
                                     epochs)


    #M = params['model']['learning_schedule_multi']
    #drops = [60*M,120*M,160*M] #eugene's scheduler

    test_acc = []
    start_time = time.time()
    train_losses, test_losses , train_accuracies = [], [], []
    lrs, lrs_scattering, lrs_orientation = [], [], []


    params['model']['trainable_parameters'] = '%.2fM' % (sum(p.numel() for p in optimizer.param_groups[0]["params"]) / 1000000.0)

    for epoch in  range(0, epochs) :
        # save learning rates for mlflow
        lrs.append(optimizer.param_groups[0]['lr'])
        if params['model']['mode'] == 'scattering_dif':
            if epoch < params['model']['epoch_scattering']:
                lrs_orientation.append(optimizer.param_groups[1]['lr'])
                lrs_scattering.append(optimizer.param_groups[2]['lr'])

            elif epoch == params['model']['epoch_scattering']:
                optimizer = torch.optim.SGD(model.parameters(), lr=params['model']['lr'], 
                                            momentum=params['model']['momentum'],
                                             weight_decay=params['model']['weight_decay'])
                params['model']['scheduler'] = 'StepLR'
                scheduler = get_lr_scheduler(optimizer, params, len(train_loader), epochs)
                lrs_orientation.append(0)     
                lrs_scattering.append(0)
            else:
                lrs_orientation.append(0)     
                lrs_scattering.append(0)

        # training 
        train_loss, train_accuracy = train(model, device, train_loader, is_scattering_dif, scheduler, optimizer,  epoch+1, scattering, psi_skeleton, params_filters )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

       # at every 10 epochs, the test accuracy will be displayed
        if epoch%params['model']['step_test']==0 or epoch == epochs -1:
            accuracy, test_loss = test(model, device, test_loader, is_scattering_dif, scattering, psi_skeleton, params_filters )
            test_losses.append(test_loss )
            test_acc.append(accuracy)


    # plot train and test loss
    f_loss = visualize_loss(train_losses ,test_losses, step_test = params['model']['step_test'], 
                            y_label='loss', num_samples =int(params['model']['train_sample_num']))
                             
    f_accuracy = visualize_loss(train_accuracies ,test_acc, step_test = params['model']['step_test'], 
                            y_label='accuracy', num_samples =int(params['model']['train_sample_num']))
                             
    f_accuracy_benchmark = visualize_loss(train_accuracies ,test_acc, step_test = params['model']['step_test'], 
                            y_label='accuracy', num_samples =int(params['model']['train_sample_num']),
                            benchmark =True)

    #visualize learning rates
    f_lr = visualize_learning_rates(lrs, lrs_orientation, lrs_scattering)

    #visualize filters
    filters_plots_after= {}
    if psi is not None:
        if params['model']['mode'] == 'scattering_dif':
            wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                                params_filters[2], params_filters[3], device=device )
        
            psi = update_psi(J, psi, wavelets, device)     
    
        for mode in ['fourier','real', 'imag' ]:
            f = get_filters_visualization(psi, J, 8, mode =mode)
            filters_plots_after[mode]  = f  
    
    # save metrics and params in mlflow
    log_mlflow(params, model, np.array(test_acc).round(2), np.array(test_losses).round(2), 
                np.array(train_accuracies).round(2),np.array(train_losses).round(2), start_time, 
               filters_plots_before , filters_plots_after, 
               [f_loss,f_accuracy, f_accuracy_benchmark ], f_lr )



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("run-train")
    subparser.set_defaults(callback=run_train)
    subparser.add_argument("--name", "-n")
    subparser.add_argument("--tester", "-tst", type=float)
    subparser.add_argument("--dataset", "-d", type=str, choices=['cifar', 'kth'])
    subparser.add_argument("--architecture", "-ar", type=str, choices=['cnn', 'linear_layer'])
    subparser.add_argument("--data-root", "-dr", type=str)
    subparser.add_argument("--data-folder", "-dfo", type=str)
    subparser.add_argument("--lr", "-lr", type=float)
    subparser.add_argument("--lr-scattering", "-lrs", type=float)
    subparser.add_argument("--lr-orientation", "-lro", type=float)
    subparser.add_argument("--max-lr", "-lrmax", type=float)
    subparser.add_argument("--div-factor", "-df", type=float)
    subparser.add_argument("--train-batch-size", "-tbs", type=int)
    subparser.add_argument("--test-batch-size", "-tstbs", type=int)
    subparser.add_argument("--weight-decay", "-wd", type=float)
    subparser.add_argument("--train-sample-num", "-tsn", type=int)
    subparser.add_argument("--test-sample-num", "-tstsn", type=int)
    subparser.add_argument("--width", "-width", type=int)
    subparser.add_argument("--momentum", "-mom", type=float)
    subparser.add_argument("--seed", "-s", type=int)
    subparser.add_argument("--mode", "-m", type=str, choices=['scattering_dif', 'scattering', 'standard'])
    subparser.add_argument("--epoch", "-e", type=int)
    subparser.add_argument("--epoch-scattering", "-es", type=int)
    subparser.add_argument("--optimizer", "-o", type=str)
    subparser.add_argument("--scheduler", "-sch", type=str, choices=['CosineAnnealingLR','OneCycleLR','LambdaLR','StepLR','NoScheduler'])
    subparser.add_argument("--init-params", "-ip", type=str,choices=['Random','Kymatio'])
    subparser.add_argument("--step-test", "-st", type=int)
    subparser.add_argument("--three_phase", "-tp", action="store_true",default=None)
    subparser.add_argument("--augment", "-a", type=str,choices=['autoaugment','original-cifar','noaugment','glico'])
    subparser.add_argument('--param_file', "-pf", type=str, default='parameters.yml',
                        help="YML Parameter File Name")

    args = parser.parse_args()
    args.callback(args)


if __name__ == '__main__':
    main()
