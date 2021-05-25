"""Main module for learnable Scattering Networks

Authors: Benjamin Therien, Shanel Gauthier

Functions: 
    schedulerFactory -- get selected scheduler
    optimizerFactory -- get selected optimizer
    datasetFactory -- get selected dataset
    test -- test loop
    train -- train loop per epoch
    override_params -- override defaults from command line
    run_train -- callable functions for the program
    main -- parses arguments an calls specified callable

"""
import sys
from pathlib import Path 
sys.path.append(str(Path.cwd()))

import time
import argparse
import torch

import torch.nn.functional as F
import kymatio.datasets as scattering_datasets
import numpy as np

from numpy.random import RandomState
from parametricSN.utils.context import get_context
from parametricSN.utils.log_mlflow import visualize_loss, visualize_learning_rates, log_mlflow
from parametricSN.utils.log_mlflow import log_mlflow
from parametricSN.utils.cifar_loader import cifar_getDataloaders
from parametricSN.utils.kth_loader import kth_getDataloaders
from parametricSN.utils.xray_loader import xray_getDataloaders
from parametricSN.utils.models import *
from parametricSN.utils.optimizer_loader import *


def schedulerFactory(optimizer, params, steps_per_epoch):
    """Factory for different schedulers"""
    if params['optim']['alternating']: 
        return Scheduler(
                    optimizer, params['optim']['scheduler'], 
                    steps_per_epoch, optimizer.epoch_alternate[0], 
                    div_factor=params['optim']['div_factor'], max_lr=params['optim']['max_lr'], 
                    T_max = params['optim']['T_max'], num_step = 3
                )

    if params['optim']['scheduler'] =='OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=params['optim']['max_lr'], 
            steps_per_epoch=steps_per_epoch, epochs=params['model']['epoch'], 
            three_phase=params['optim']['three_phase'],
            div_factor=params['optim']['div_factor']
        )
    elif params['optim']['scheduler'] =='CosineAnnealingLR':
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = params['optim']['T_max'], eta_min = 1e-8)
    elif params['optim']['scheduler'] =='LambdaLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    elif params['optim']['scheduler'] =='CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, 
                                            step_size_up=params['optim']['T_max']*2,
                                             mode="triangular2")
    elif params['optim']['scheduler'] =='StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.2)
    elif params['optim']['scheduler'] == 'NoScheduler':
        scheduler = None
    else:
        raise NotImplemented(f"Scheduler {params['optim']['scheduler']} not implemented")
    return scheduler



def optimizerFactory(hybridModel,params):
    """Factory for different optimizers"""
    if params['optim']['alternating']:
        return Optimizer(
                    model=hybridModel.top, scatteringModel=hybridModel.scatteringBase, 
                    optimizer_name=params['optim']['name'], lr=params['optim']['lr'], 
                    weight_decay=params['optim']['weight_decay'], momentum=params['optim']['momentum'], 
                    epoch=params['model']['epoch'], num_phase=2
                )


    if params['optim']['name'] == 'adam':
        return torch.optim.Adam(
            hybridModel.parameters(),lr=params['optim']['lr'], 
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=params['optim']['weight_decay'], amsgrad=False
        )
    elif params['optim']['name'] == 'sgd': 
        return torch.optim.SGD(
            hybridModel.parameters(), lr=params['optim']['lr'], 
            momentum=params['optim']['momentum'], weight_decay=params['optim']['weight_decay']
        )
        
    else:
        raise NotImplemented(f"Optimizer {params['optim']['name']} not implemented")


def datasetFactory(params,dataDir,use_cuda):
    """
    returns:
        train_loader, test_loader, seed
    """

    if params['dataset']['name'].lower() == "cifar":
        return cifar_getDataloaders(
                    trainSampleNum=params['dataset']['train_sample_num'], valSampleNum=params['dataset']['test_sample_num'], 
                    trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
                    multiplier=1, trainAugmentation=params['dataset']['augment'],
                    seed=params['general']['seed'], dataDir=dataDir, 
                    num_workers=params['general']['cores'], use_cuda=use_cuda
                )
    elif params['dataset']['name'].lower() == "kth":
        return kth_getDataloaders(
                    trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
                    trainAugmentation=params['dataset']['augment'], height= params['dataset']['height'] , 
                    width = params['dataset']['width'], sample = params['dataset']['sample'] , 
                    seed=params['general']['seed'], dataDir=dataDir, num_workers=params['general']['cores'], 
                    use_cuda=use_cuda
                )
    elif params['dataset']['name'].lower() == "x-ray":
        return xray_getDataloaders(
            trainSampleNum=params['dataset']['train_sample_num'], valSampleNum=params['dataset']['test_sample_num'], 
            trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
            multiplier=1, trainAugmentation=params['dataset']['augment'],
            height= params['dataset']['height'] , width = params['dataset']['width'],
            seed=params['general']['seed'], dataDir=dataDir, 
            num_workers=params['general']['cores'], use_cuda=use_cuda
                )
    else:
        raise NotImplemented(f"Dataset {params['dataset']['name']} not implemented")



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

def train(model, device, train_loader, scheduler, optimizer, epoch, alternating=True):
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

        if alternating:
            optimizer.step(epoch)
        else:
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
            tempSplit = k.split('_')
            prefix = tempSplit[0]
            key = "_".join(tempSplit[1:])
            try:
                params[prefix][key] = v
                print("    ",k,v)
            except KeyError:
                # print("Invalid parameter {} skipped".format(prefix))
                pass

    return params


def run_train(args):
    """Initializes and trains scattering models with different architectures
    """
    catalog, params = get_context(args.param_file) #parse params
    params = override_params(args,params) #override from CLI

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

    train_loader, test_loader, params['general']['seed'] = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset
    

    scatteringBase = sn_ScatteringBase( #create learnable of non-learnable scattering
        J=params['scattering']['J'],
        N=params['dataset']['height'],
        M=params['dataset']['width'],
        initialization=params['scattering']['init_params'],
        seed=params['general']['seed'],
        learnable=params['scattering']['learnable'],
        lr_orientation=params['scattering']['lr_orientation'],
        lr_scattering=params['scattering']['lr_scattering'],
        use_cuda=use_cuda
    )

    top = modelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture=params['model']['name'],
        num_classes=params['dataset']['num_classes'], 
        width= params['model']['width'], 
        use_cuda=use_cuda
    )

    #use for cnn?
    # model = Scattering2dResNet(8, params['model']['width'],standard=True).to(device)

    hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda)

    optimizer = optimizerFactory(hybridModel=hybridModel, params=params)

    scheduler = schedulerFactory(optimizer, params, len(train_loader))

    if params['optim']['alternating']:
        optimizer.scheduler = scheduler


    #M = params['model']['learning_schedule_multi']
    #drops = [60*M,120*M,160*M] #eugene's scheduler

    test_acc = []
    start_time = time.time()
    train_losses, test_losses , train_accuracies = [], [], []
    lrs, lrs_scattering, lrs_orientation = [], [], []

    params['model']['trainable_parameters'] = "to be fixed"
    params['model']['trainable_parameters'] = '%.2fM' % (hybridModel.countLearnableParams() / 1000000.0)

    print("Starting train for hybridModel with {} parameters".format(params['model']['trainable_parameters']))

    for epoch in  range(0, params['model']['epoch']):

        if params['optim']['alternating']:
            if optimizer.phase % 2 == 0:
                lrs.append(optimizer.param_groups[0]['lr'])
                if params['scattering']['learnable']:
                    lrs_orientation.append(0)
                    lrs_scattering.append(0)
            else:
                lrs.append(0)
                if params['scattering']['learnable']:
                    lrs_orientation.append(optimizer.param_groups[0]['lr'])
                    lrs_scattering.append(optimizer.param_groups[1]['lr'])

        else:
            lrs.append(optimizer.param_groups[0]['lr'])
            if params['scattering']['learnable']:
                lrs_orientation.append(optimizer.param_groups[1]['lr'])
                lrs_scattering.append(optimizer.param_groups[2]['lr'])

        train_loss, train_accuracy = train(hybridModel, device, train_loader, scheduler, optimizer, epoch+1, alternating=params['optim']['alternating'])
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
        y_label='loss', num_samples=int(params['dataset']['train_sample_num'])
    )
                             
    f_accuracy = visualize_loss(
        train_accuracies ,test_acc, step_test=params['model']['step_test'], 
        y_label='accuracy', num_samples=int(params['dataset']['train_sample_num'])
    )
                             
    f_accuracy_benchmark = visualize_loss(
        train_accuracies, test_acc, step_test=params['model']['step_test'], 
        y_label='accuracy', num_samples=int(params['dataset']['train_sample_num']),
        benchmark =True
    )

    #visualize learning rates
    f_lr = visualize_learning_rates(lrs, lrs_orientation, lrs_scattering)

    #visualize filters
    filters_plots_before = hybridModel.scatteringBase.filters_plots_before
    hybridModel.scatteringBase.updateFilters() #update the filters based on the latest param update
    filters_plots_after = hybridModel.scatteringBase.getFilterViz() #get filter plots

       
    # save metrics and params in mlflow
    log_mlflow(
        params, hybridModel, np.array(test_acc).round(2), 
        np.array(test_losses).round(2), np.array(train_accuracies).round(2), np.array(train_losses).round(2), 
        start_time, filters_plots_before, filters_plots_after, 
        [f_loss,f_accuracy, f_accuracy_benchmark ], f_lr
    )



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("run-train")
    subparser.set_defaults(callback=run_train)
    #general
    subparser.add_argument("--general-cores", "-gc", type=int)
    subparser.add_argument("--general-seed", "-gseed", type=int)
    #mlflow 
    subparser.add_argument("--mlflow-tracking-uri", "-turi", type=str)
    subparser.add_argument("--mlflow-experiment-name", "-en", type=str)
    #dataset
    subparser.add_argument("--dataset-name", "-dname", type=str, choices=['cifar', 'kth', 'x-ray'])
    subparser.add_argument("--dataset-num-classes", "-dnc", type=int)
    subparser.add_argument("--dataset-train-batch-size", "-dtbs", type=int)
    subparser.add_argument("--dataset-test-batch-size", "-dtstbs", type=int)
    subparser.add_argument("--dataset-train-sample-num", "-dtsn", type=int)
    subparser.add_argument("--dataset-test-sample-num", "-dtstsn", type=int)
    subparser.add_argument("--dataset-data-root", "-ddr", type=str)
    subparser.add_argument("--dataset-data-folder", "-ddf", type=str)
    subparser.add_argument("--dataset-height", "-dh", type=int)
    subparser.add_argument("--dataset-width", "-dw", type=int)
    subparser.add_argument("--dataset-augment", "-daug", type=str, choices=['autoaugment','original-cifar','noaugment','glico'])
    subparser.add_argument("--dataset-sample", "-dsam", type=str, choices=['a','b','c','d'])
    #scattering
    subparser.add_argument("--scattering-j", "-sj", type=int)
    subparser.add_argument("--scattering-max-order", "-smo", type=int)
    subparser.add_argument("--scattering-lr-scattering", "-slrs", type=float)
    subparser.add_argument("--scattering-lr-orientation", "-slro", type=float)
    subparser.add_argument("--scattering-init-params", "-sip", type=str,choices=['Kymatio','Random'])
    subparser.add_argument("--scattering-learnable", "-sl", type=int, choices=[0,1])
    #optim
    subparser.add_argument("--optim-name", "-oname", type=str,choices=['adam', 'sgd', 'alternating'])
    subparser.add_argument("--optim-lr", "-olr", type=float)
    subparser.add_argument("--optim-weight-decay", "-owd", type=float)
    subparser.add_argument("--optim-momentum", "-omo", type=float)
    subparser.add_argument("--optim-max-lr", "-omaxlr", type=float)
    subparser.add_argument("--optim-scheduler", "-os", type=str, choices=['CosineAnnealingLR','OneCycleLR','LambdaLR','StepLR','NoScheduler'])
    subparser.add_argument("--optim-div-factor", "-odivf", type=int)
    subparser.add_argument("--optim-three-phase", "-otp", type=int, choices=[0,1])
    subparser.add_argument("--optim-alternating", "-oalt", type=int, choices=[0,1])
    subparser.add_argument("--optim-phase-num", "-opn", type=int)
    subparser.add_argument("--optim-T-max", "-otmax", type=int)
    #model 
    subparser.add_argument("--model-name", "-mname", type=str, choices=['cnn', 'mlp', 'linear_layer'])
    subparser.add_argument("--model-width", "-mw", type=int)
    subparser.add_argument("--model-epoch", "-me", type=int)
    subparser.add_argument("--model-step-test", "-mst", type=int)

    subparser.add_argument('--param_file', "-pf", type=str, default='parameters.yml',
                        help="YML Parameter File Name")

    args = parser.parse_args()

    for key in ['optim_alternating','optim_three_phase','scattering_learnable']:
        if args.__dict__[key] != None:
            args.__dict__[key] = bool(args.__dict__[key]) #make 0 and 1 arguments booleans

    args.callback(args)


if __name__ == '__main__':
    main()
