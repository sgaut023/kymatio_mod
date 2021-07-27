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
import random

import torch.nn.functional as F
import kymatio.datasets as scattering_datasets
import numpy as np

from numpy.random import RandomState
from parametricSN.utils.helpers import get_context
from parametricSN.utils.helpers import visualize_loss, visualize_learning_rates, log_mlflow, getSimplePlot
from parametricSN.data_loading.cifar_loader import cifar_getDataloaders
from parametricSN.data_loading.kth_loader import kth_getDataloaders
from parametricSN.data_loading.xray_loader import xray_getDataloaders

from parametricSN.utils import cosine_training, cross_entropy_training, cross_entropy_training_accumulation
from parametricSN.models.sn_top_models import topModelFactory
from parametricSN.models.sn_base_models import baseModelFactory
from parametricSN.models.sn_hybrid_models import sn_HybridModel
from parametricSN.utils.optimizer_loader import *
#from parametricSN.glico.glico_model.glico_frontend import GlicoController, trainGlico


def schedulerFactory(optimizer, params, steps_per_epoch):
    """Factory for OneCycle, CosineAnnealing, Lambda, Cyclic, and step schedulers

    parameters: 
    params -- dict of input parameters
    optimizer -- the optimizer paired with the scheduler
    steps_per_epoch -- number of steps the scheduler takes each epoch
    """

    if params['optim']['alternating']: 
        return Scheduler(
                    optimizer, params['optim']['scheduler'], 
                    steps_per_epoch, epochs=optimizer.epoch_alternate[0], 
                    div_factor=params['optim']['div_factor'], max_lr=params['optim']['max_lr'], 
                    T_max = params['optim']['T_max'], num_step = 2, three_phase=params['optim']['three_phase']
                )

    if params['optim']['scheduler'] =='OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=params['optim']['max_lr'], 
            steps_per_epoch=steps_per_epoch, epochs=params['model']['epoch'], 
            three_phase=params['optim']['three_phase'],
            div_factor=params['optim']['div_factor']
        )

        for group in optimizer.param_groups:
            if 'maxi_lr' in group .keys():
                group['max_lr'] = group['maxi_lr']

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch * int(params['model']['epoch']/2), 
                                                    gamma=0.5)

    elif params['optim']['scheduler'] == 'NoScheduler':
        scheduler = None

    else:
        raise NotImplemented(f"Scheduler {params['optim']['scheduler']} not implemented")

    return scheduler


def optimizerFactory(hybridModel, params):
    """Factory for adam, sgd, and a custom alternating optimizer

    parameters: 
    params -- dict of input parameters
    hybridModel -- the model used during training 
    """
    if params['optim']['alternating']:
        return Optimizer(
                    model=hybridModel.top, scatteringModel=hybridModel.scatteringBase, 
                    optimizer_name=params['optim']['name'], lr=params['optim']['lr'], 
                    weight_decay=params['optim']['weight_decay'], momentum=params['optim']['momentum'], 
                    epoch=params['model']['epoch'], num_phase=params['optim']['phase_num'],
                    phaseEnds=params['optim']['phase_ends'],scattering_max_lr=params['scattering']['max_lr'],
                    scattering_div_factor=params['scattering']['div_factor'],scattering_three_phase = params['scattering']['three_phase']
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


def datasetFactory(params, dataDir, use_cuda):
    """ Factory for Cifar-10, kth-tips2, and COVID-CRX2 datasets

    Creates and returns different dataloaders and datasets based on input

    parameters: 
    params -- dict of input parameters
    dataDir -- path to the dataset

    returns:
        train_loader, test_loader, seed
    """

    if params['dataset']['name'].lower() == "cifar":
        return cifar_getDataloaders(
                    trainSampleNum=params['dataset']['train_sample_num'], valSampleNum=params['dataset']['test_sample_num'], 
                    trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
                    multiplier=1, trainAugmentation=params['dataset']['augment'],
                    seed=params['general']['seed'], dataDir=dataDir, 
                    num_workers=params['general']['cores'], use_cuda=use_cuda,
                    glico=params['dataset']['glico']
                )
    elif params['dataset']['name'].lower() == "kth":
        return kth_getDataloaders(
                    trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
                    trainAugmentation=params['dataset']['augment'], height= params['dataset']['height'] , 
                    width = params['dataset']['width'], sample = params['dataset']['sample'] , 
                    seed=params['general']['seed'], dataDir=dataDir, num_workers=params['general']['cores'], 
                    use_cuda=use_cuda, glico=params['dataset']['glico']
                )
    elif params['dataset']['name'].lower() == "x-ray":
        return xray_getDataloaders(
            trainSampleNum=params['dataset']['train_sample_num'], valSampleNum=params['dataset']['test_sample_num'], 
            trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
            multiplier=1, trainAugmentation=params['dataset']['augment'],
            height= params['dataset']['height'] , width = params['dataset']['width'],
            seed=params['general']['seed'], dataDir=dataDir, 
            num_workers=params['general']['cores'], use_cuda=use_cuda, 
            glico=params['dataset']['glico']
        )
    else:
        raise NotImplemented(f"Dataset {params['dataset']['name']} not implemented")



def override_params(args, params):
    """override passed params dict with any CLI arguments
    
    parameters: 
    args -- namespace of arguments passed from CLI
    params -- dict of default arguments list    
    """

    # print("Overriding parameters:")
    for k,v in args.__dict__.items():
        if v != None and k != "param_file":
            tempSplit = k.split('_')
            prefix = tempSplit[0]
            key = "_".join(tempSplit[1:])
            try:
                params[prefix][key] = v
            except KeyError:
                pass

    return params

def get_train_test_functions(loss_name):
    """ Factory for different train and test Functions

    parameters: 
    loss_name -- the name of the loss function to use
    """
            
    if loss_name == 'cross-entropy':
        train = lambda *args, **kwargs : cross_entropy_training.train(*args,**kwargs)
        test = lambda *args : cross_entropy_training.test(*args)

    elif loss_name == 'cross-entropy-accum':
        train = lambda *args, **kwargs : cross_entropy_training_accumulation.train(*args,**kwargs)
        test = lambda *args : cross_entropy_training_accumulation.test(*args)    
    
    elif loss_name == 'cosine':
        train = lambda *args, **kwargs : cosine_training.train(*args, **kwargs)
        test = lambda *args : cosine_training.test(*args)

    else:
        raise NotImplemented(f"Loss {loss_name} not implemented")
    
    return train, test


def setAllSeeds(seed):
    """Helper for setting seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def estimateRemainingTime(trainTime,testTime,epochs,currentEpoch,testStep):
    meanTrain = np.mean(trainTime)
    meanTest = np.mean(testTime)

    remainingEpochs = epochs - currentEpoch

    remainingTrain = (meanTrain *  remainingEpochs) / 60
    remainingTest = (meanTest * (int(remainingEpochs / testStep) + 1)) / 60
    remainingTotal = remainingTest + remainingTrain

    print("[INFO] ~{:.2f} m remaining. Mean train epoch duration: {:.2f} s. Mean test epoch duration: {:.2f} s.".format(
        remainingTotal, meanTrain, meanTest
    ))

    return remainingTotal


def run_train(args):
    """Launches the training script 

    parameters:
    args -- namespace of arguments passed from CLI
    """
    torch.backends.cudnn.deterministic = True #Enable deterministic behaviour
    torch.backends.cudnn.benchmark = False #Enable deterministic behaviour

    catalog, params = get_context(args.param_file) #parse params
    params = override_params(args,params) #override from CLI

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

    if params['optim']['alternating'] and params['scattering']['learnable']:
        params['model']['epoch'] = params['model']['epoch'] + int(params['optim']['phase_ends'][-1])

    train_loader, test_loader, params['general']['seed'], glico_dataset = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset
    
    if glico_dataset != None:
        setAllSeeds(seed=params['general']['seed'])
        nag = trainGlico(glico_dataset,params['dataset']['num_classes'],epochs=750)
        glicoController = GlicoController(nag,steps=5,replaceProb=0.075)
    else:
        glicoController = None


    setAllSeeds(seed=params['general']['seed'])

    scatteringBase = baseModelFactory( #creat scattering base model
        architecture=params['scattering']['architecture'],
        J=params['scattering']['J'],
        N=params['dataset']['height'],
        M=params['dataset']['width'],
        second_order=params['scattering']['second_order'],
        initialization=params['scattering']['init_params'],
        seed=params['general']['seed'],
        learnable=params['scattering']['learnable'],
        lr_orientation=params['scattering']['lr_orientation'],
        lr_scattering=params['scattering']['lr_scattering'],
        device=device,
        use_cuda=use_cuda
    )

    setAllSeeds(seed=params['general']['seed'])
    
    top = topModelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture=params['model']['name'],
        num_classes=params['dataset']['num_classes'], 
        width= params['model']['width'], 
        average=params['model']['average'], 
        use_cuda=use_cuda
    )


    hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda)

    optimizer = optimizerFactory(hybridModel=hybridModel, params=params)

    if params['model']['loss'] == 'cross-entropy-accum':
        if params['dataset']['accum_step_multiple'] % params['dataset']['train_batch_size'] != 0:
            print("Incompatible batch size and accum step multiple")
            raise Exception
        else:
            steppingSize = int(params['dataset']['accum_step_multiple']/params['dataset']['train_batch_size'])

        params['dataset']['accum_step_multiple']
        scheduler = schedulerFactory(
            optimizer=optimizer, params=params, 
            steps_per_epoch=math.ceil(params['dataset']['train_sample_num']/params['dataset']['accum_step_multiple'])
        )
    else:
        scheduler = schedulerFactory(optimizer=optimizer, params=params, steps_per_epoch=len(train_loader))
        steppingSize = None

    
    if params['optim']['alternating']:
        optimizer.scheduler = scheduler




    test_acc = []
    start_time = time.time()
    train_losses, test_losses , train_accuracies = [], [], []
    lrs, lrs_scattering, lrs_orientation = [], [], []
    param_distance, wavelet_distance = [], []

    trainTime = []
    testTime = []

    # param_distance.append(hybridModel.scatteringBase.checkDistance(compared='params'))
    param_distance.append(hybridModel.scatteringBase.checkParamDistance())

    wavelet_distance.append(hybridModel.scatteringBase.checkDistance(compared='wavelets_complete'))
    
    params['model']['trainable_parameters'] = '%fM' % (hybridModel.countLearnableParams() / 1000000.0)
    print("Starting train for hybridModel with {} parameters".format(params['model']['trainable_parameters']))

    for epoch in  range(0, params['model']['epoch']):
        t1 = time.time()

        try:
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
        except Exception:
            pass

        train, test = get_train_test_functions(params['model']['loss'])
        train_loss, train_accuracy = train(hybridModel, device, train_loader, scheduler, optimizer, 
                                           epoch+1, alternating=params['optim']['alternating'],
                                           glicoController=glicoController, accum_step_multiple=steppingSize)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # param_distance.append(hybridModel.scatteringBase.checkDistance(compared='params'))
        param_distance.append(hybridModel.scatteringBase.checkParamDistance())

        wavelet_distance.append(hybridModel.scatteringBase.checkDistance(compared='wavelets_complete'))

        trainTime.append(time.time()-t1)
        if epoch % params['model']['step_test'] == 0 or epoch == params['model']['epoch'] -1: #check test accuracy
            t1 = time.time()
            accuracy, test_loss = test(hybridModel, device, test_loader)
            test_losses.append(test_loss)
            test_acc.append(accuracy)
            

            testTime.append(time.time()-t1)
            estimateRemainingTime(trainTime=trainTime,testTime=testTime,epochs= params['model']['epoch'],currentEpoch=epoch,testStep=params['model']['step_test'])





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

    paramDistancePlot = getSimplePlot(xlab='Epochs', ylab='Min Distance to TF params',
        title='Learnable parameters progress towards the TF initialization parameters', label='Dist to TF params',
        xvalues=[x+1 for x in range(len(param_distance))], yvalues=param_distance)

    waveletDistancePlot = getSimplePlot(xlab='Epochs', ylab='Min Distance to TF wavelets',
        title='Learnable wavelet filters progress towards the TF wavelet filters', label='Dist to TF wavelets',
        xvalues=[x+1 for x in range(len(wavelet_distance))], yvalues=wavelet_distance)

    if params['scattering']['architecture']  == 'scattering':
        #visualize filters
        filters_plots_before = hybridModel.scatteringBase.filters_plots_before
        hybridModel.scatteringBase.updateFilters() #update the filters based on the latest param update
        filters_plots_after = hybridModel.scatteringBase.getFilterViz() #get filter plots
        filters_values = hybridModel.scatteringBase.plotFilterValues()
        filters_grad = hybridModel.scatteringBase.plotFilterGrads()
        filters_parameters = hybridModel.scatteringBase.plotParameterValues()
    else:
        filters_plots_before = None
        filters_plots_after = None
        filters_values = None
        filters_grad = None

    log_mlflow(
        params=params, model=hybridModel, test_acc=np.array(test_acc).round(2), 
        test_loss=np.array(test_losses).round(2), train_acc=np.array(train_accuracies).round(2), 
        train_loss=np.array(train_losses).round(2), start_time=start_time, 
        filters_plots_before=filters_plots_before, filters_plots_after=filters_plots_after,
        misc_plots=[f_loss, f_accuracy, f_accuracy_benchmark, filters_grad, 
        filters_values, filters_parameters, f_lr, paramDistancePlot, waveletDistancePlot]
    )
    


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("run-train")
    subparser.set_defaults(callback=run_train)
    #general
    subparser.add_argument("--general-cores", "-gc", type=int)
    subparser.add_argument("--general-seed", "-gseed", type=int)
    subparser.add_argument("--general-save-metric", "-gsm", type=int)
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
    subparser.add_argument("--dataset-accum-step-multiple", "-dasm", type=int)
    subparser.add_argument("--dataset-data-root", "-ddr", type=str)
    subparser.add_argument("--dataset-data-folder", "-ddf", type=str)
    subparser.add_argument("--dataset-height", "-dh", type=int)
    subparser.add_argument("--dataset-width", "-dw", type=int)
    subparser.add_argument("--dataset-augment", "-daug", type=str, choices=['autoaugment','original-cifar','noaugment','glico'])
    subparser.add_argument("--dataset-glico", "-dg", type=int, choices=[0,1])
    subparser.add_argument("--dataset-sample", "-dsam", type=str, choices=['a','b','c','d'])
    #scattering
    subparser.add_argument("--scattering-J", "-sj", type=int)
    subparser.add_argument("--scattering-max-order", "-smo", type=int)
    subparser.add_argument("--scattering-lr-scattering", "-slrs", type=float)
    subparser.add_argument("--scattering-lr-orientation", "-slro", type=float)
    subparser.add_argument("--scattering-init-params", "-sip", type=str,choices=['Kymatio','Random'])
    subparser.add_argument("--scattering-learnable", "-sl", type=int, choices=[0,1])
    subparser.add_argument("--scattering-second-order", "-sso", type=int, choices=[0,1])
    subparser.add_argument("--scattering-max-lr", "-smaxlr", type=float)
    subparser.add_argument("--scattering-div-factor", "-sdivf", type=int)
    subparser.add_argument("--scattering-architecture", "-sa", type=str, choices=['scattering','identity'])
    subparser.add_argument("--scattering-three-phase", "-stp", type=int, choices=[0,1])
    #optim
    subparser.add_argument("--optim-name", "-oname", type=str,choices=['adam', 'sgd', 'alternating'])
    subparser.add_argument("--optim-lr", "-olr", type=float)
    subparser.add_argument("--optim-weight-decay", "-owd", type=float)
    subparser.add_argument("--optim-momentum", "-omo", type=float)
    subparser.add_argument("--optim-max-lr", "-omaxlr", type=float)
    subparser.add_argument("--optim-div-factor", "-odivf", type=int)
    subparser.add_argument("--optim-three-phase", "-otp", type=int, choices=[0,1])
    subparser.add_argument("--optim-scheduler", "-os", type=str, choices=['CosineAnnealingLR','OneCycleLR','LambdaLR','StepLR','NoScheduler'])    
    subparser.add_argument("--optim-alternating", "-oalt", type=int, choices=[0,1])
    subparser.add_argument("--optim-phase-num", "-opn", type=int)
    subparser.add_argument("--optim-phase-ends", "-ope", nargs="+", default=None)
    subparser.add_argument("--optim-T-max", "-otmax", type=int)

    #model 
    subparser.add_argument("--model-name", "-mname", type=str, choices=['cnn', 'mlp', 'linear_layer', 'resnet50'])
    subparser.add_argument("--model-width", "-mw", type=int)
    subparser.add_argument("--model-epoch", "-me", type=int)
    subparser.add_argument("--model-step-test", "-mst", type=int)
    subparser.add_argument("--model-loss", "-mloss", type=str, choices=['cosine', 'cross-entropy','cross-entropy-accum'])

    subparser.add_argument('--param_file', "-pf", type=str, default='parameters.yml',
                        help="YML Parameter File Name")

    args = parser.parse_args()

    for key in ['optim_alternating','optim_three_phase','scattering_learnable',
                'scattering_second_order','scattering_three_phase','dataset_glico']:
        if args.__dict__[key] != None:
            args.__dict__[key] = bool(args.__dict__[key]) #make 0 and 1 arguments booleans

    args.callback(args)


if __name__ == '__main__':
    main()
