"""Main moodule for learnable Scattering Networks

Authors: Benjamin Therien, Shanel Gauthier

"""
import sys
from pathlib import Path 
sys.path.append(str(Path.cwd()))

import time
import argparse
import torch

import torch.nn.functional as F
import torch.nn as nn
import kymatio.datasets as scattering_datasets
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from numpy.random import RandomState
from parametricSN.utils.context import get_context
from parametricSN.utils.log_mlflow import visualize_loss, visualize_learning_rates, log_mlflow
from parametricSN.utils.log_mlflow import log_mlflow
from parametricSN.utils.auto_augment import AutoAugment, Cutout
from parametricSN.utils.cifar_loader import cifar_getDataloaders
from parametricSN.utils.kth_loader import kth_getDataloaders
from parametricSN.utils.Scattering2dResNet import Scattering2dResNet
from parametricSN.utils.models import *


class InvalidOptimizerError(Exception):
    """Error thrown when an invalid optimizer name is passed"""
    pass

def schedulerFactory(optimizer, params, steps_per_epoch, epoch ):
    """Factory for different schedulers"""

    if params['model']['scheduler'] =='OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=params['model']['max_lr'], 
            steps_per_epoch=steps_per_epoch, epochs=epoch, 
            three_phase=params['model']['three_phase'],
            div_factor=params['model']['div_factor']
        )
    elif params['model']['scheduler'] =='CosineAnnealingLR':
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = params['model']['T_max'], eta_min = 1e-8)
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

def optimizerFactory(parameters,params):
    """Factory for different optimizers"""

    if params['model']['optimizer'] == 'adam':
        return torch.optim.Adam(
            parameters,lr=params['model']['lr'], 
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=params['model']['weight_decay'], amsgrad=False
        )
    elif params['model']['optimizer'] == 'sgd': 
        return torch.optim.SGD(
            parameters, lr=params['model']['lr'], 
            momentum=params['model']['momentum'], weight_decay=params['model']['weight_decay']
        )
    else:
        print("Invalid optimizer parameter passed")
        raise InvalidOptimizerError

def datasetFactory(params,dataDir,use_cuda):

    if params['model']['dataset'] == "cifar":
        return cifar_getDataloaders(
                    trainSampleNum=params['model']['train_sample_num'], valSampleNum=params['model']['test_sample_num'], 
                    trainBatchSize=params['model']['train_batch_size'], valBatchSize=params['model']['test_batch_size'], 
                    multiplier=1, trainAugmentation=params['model']['augment'],
                    seed=params['model']['seed'], dataDir=dataDir, 
                    num_workers=params['processor']['cores'], use_cuda=use_cuda
                )
    elif params['model']['dataset'] == "kth":
        return kth_getDataloaders(
                    trainBatchSize=params['model']['train_batch_size'], valBatchSize=params['model']['test_batch_size'], 
                    trainAugmentation=params['model']['augment'], seed=params['model']['seed'], 
                    dataDir=dataDir, num_workers=4, 
                    use_cuda=use_cuda
                )
    elif params['model']['dataset'] == "x-ray":
        raise NotImplemented(f"Dataset {params['model']['dataset']} not implemented")
    else:
        raise NotImplemented(f"Dataset {params['model']['dataset']} not implemented")



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

    print("Overriding parameters:")
    for k,v in args.__dict__.items():
        if v != None and k != "param_file":
            print(k,v)
            params["model"][k] = v


    return params



def run_train(args):
    """Initializes and trains scattering models with different architectures
    """
    catalog, params = get_context(args.param_file) #parse params
    params = override_params(args,params) #override from CLI

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if params['model']['data_root'] != None:
        DATA_DIR = Path(params['model']['data_root'])/params['model']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

    train_loader, test_loader, params['model']['seed'] = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset
    

    scatteringBase = sn_ScatteringBase( #create learnable of non-learnable scattering
        J=params['scattering']['J'],
        N=params['dimension']['N'],
        M=params['dimension']['M'],
        initialization=params['model']['init_params'],
        seed=params['model']['seed'],
        learnable=('scattering_dif' == params['model']['mode']),
        lr_orientation=params['model']['lr_orientation'],
        lr_scattering=params['model']['lr_scattering'],
        use_cuda=use_cuda
    )

    top = modelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture=params['model']['architecture'],
        num_classes=params['model']['num_classes'], 
        use_cuda=use_cuda
    )

    #use for cnn?
    # model = Scattering2dResNet(8, params['model']['width'],standard=True).to(device)

    hybridModel = sn_HybridModel(scatteringBase=scatteringBase,top=top,use_cuda=use_cuda)

    optimizer = optimizerFactory(hybridModel.parameters(),params)

    scheduler = schedulerFactory(
        optimizer, params, 
        len(train_loader), params['model']['epoch']
    )


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
