"""
Classification on CIFAR10 (ResNet)
==================================

Based on pytorch example for CIFAR10
"""

from parametricSN.utils.models import sn_HybridModel, sn_ScatteringBase
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
from parametricSN.utils.models import *

     
class Identity(nn.Module):
    """Identity nn.Module for ski"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    

def schedulerFactory(optimizer, params, steps_per_epoch, epoch ):
    """Factory for different schedulers"""

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
        dim_M = params['dimension']['M']
        dim_N = params['dimension']['N']
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


def test(model, device, test_loader):
    """test method"""

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)  
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),accuracy ))

    return accuracy, test_loss

def train(model, device, train_loader, scheduler, optimizer, epoch):
    """training method"""

    model.train()
    correct = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
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

    
    print('Train Epoch: {}\t Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%): '.format(
                epoch, train_loss, correct , len(train_loader.dataset),
                train_accuracy))

    return train_loss, train_accuracy


def override_params(args,params):
    """override passed params dict with CLI arguments"""

    for k,v in args.__dict__.items():
        if v != None and k != "param_file":
            print(k,v)
            params["model"][k] = v


    return params



def run_train(args):
    """Initializes and trains scattering models with different architectures
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    catalog, params = get_context(args.param_file) #parse params
    params = override_params(args,params) #override from CLI

    train_loader, test_loader = get_dataset(params, use_cuda) #load Dataset
    

    scatteringBase = sn_ScatteringBase( #create learnable of non-learnable scattering
        J=params['scattering']['J'],
        N=params['dimension']['N'],
        M=params['dimension']['M'],
        initialization=params['model']['init_params'],
        seed=params['model']['seed'],
        learnable=('scattering_dif' == params['model']['mode']),
        lr_orientation=params['model']['lr_orientation'],
        lr_scattering=params['model']['lr_scattering'],
        CUDA=use_cuda
    )

    top = modelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture=params['model']['architecture'],
        num_classes=len(train_loader.classes)
    )

    #use for cnn?
    # model = Scattering2dResNet(8, params['model']['width'],standard=True).to(device)

    hybridModel = sn_HybridModel(scatteringBase=scatteringBase,top=top)
    parameters = hybridModel.parameters()



    if params['model']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters,lr=params['model']['lr'], 
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=params['model']['weight_decay'], amsgrad=False)
    elif params['model']['optimizer'] == 'sgd': 
        optimizer = torch.optim.SGD(parameters,lr=params['model']['lr'], 
        momentum=params['model']['momentum'],weight_decay=params['model']['weight_decay'])
    else:
        print("Invalid optimizer parameter passed")


    scheduler = schedulerFactory(optimizer, params, len(train_loader), 
                                 params['model']['epoch'])


    #M = params['model']['learning_schedule_multi']
    #drops = [60*M,120*M,160*M] #eugene's scheduler

    test_acc = []
    start_time = time.time()
    train_losses, test_losses , train_accuracies = [], [], []
    lrs, lrs_scattering, lrs_orientation = [], [], []


    params['model']['trainable_parameters'] = '%.2fM' % (sum(p.numel() for p in optimizer.param_groups[0]["params"]) / 1000000.0)

    for epoch in  range(0, params['model']['epoch']) :
        lrs.append(optimizer.param_groups[0]['lr'])
        
        if params['model']['mode'] == 'scattering_dif':
            lrs_orientation.append(optimizer.param_groups[1]['lr'])
            lrs_scattering.append(optimizer.param_groups[2]['lr'])

        train_loss, train_accuracy = train(hybridModel, device, train_loader, scheduler, optimizer,  epoch+1)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        
        if epoch % params['model']['step_test'] == 0 or epoch == params['model']['epoch'] -1: #check test accuracy
            accuracy, test_loss = test(hybridModel, device, test_loader)
            test_losses.append(test_loss)
            test_acc.append(accuracy)



    #MLFLOW logging below



    # plot train and test loss
    f_loss = visualize_loss(
        train_losses, test_losses, step_test=params['model']['step_test'], 
        y_label='loss', num_samples=int(params['model']['train_sample_num'])
    )
                             
    f_accuracy = visualize_loss(
        train_accuracies ,test_acc, step_test=params['model']['step_test'], 
        y_label='accuracy', num_samples=int(params['model']['train_sample_num'])
    )
                             
    f_accuracy_benchmark = visualize_loss(
        train_accuracies, test_acc, step_test=params['model']['step_test'], 
        y_label='accuracy', num_samples=int(params['model']['train_sample_num']),
        benchmark =True
    )

    #visualize learning rates
    f_lr = visualize_learning_rates(lrs, lrs_orientation, lrs_scattering)

    #visualize filters
    filters_plots_before = hybridModel.scatteringBase.filters_plots_before
    hybridModel.scatteringBase.updateFilters() #update the filters based on the latest param update
    filters_plots_after = hybridModel.scatteringBase.getFilterViz() #get filter plots

       
    # save metrics and params in mlflow
    log_mlflow(params, hybridModel, np.array(test_acc).round(2), np.array(test_losses).round(2), 
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
