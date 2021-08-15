"""Main module for Learnable Scattering Networks

Authors: Benjamin Therien, Shanel Gauthier

Functions: 
    run_train -- callable functions for the program
    main -- parses arguments an calls specified callable

"""
import sys
from pathlib import Path 
sys.path.append(str(Path.cwd()))

import time
import argparse
import torch
import math
import cv2

import kymatio.datasets as scattering_datasets
import numpy as np

from parametricSN.utils.helpers import get_context, visualize_loss, visualize_learning_rates, log_mlflow, getSimplePlot, override_params, setAllSeeds, estimateRemainingTime
from parametricSN.utils.optimizer_factory import optimizerFactory
from parametricSN.utils.scheduler_factory import schedulerFactory
from parametricSN.data_loading.dataset_factory import datasetFactory

from parametricSN.models.models_factory import topModelFactory, baseModelFactory
from parametricSN.models.sn_hybrid_models import sn_HybridModel
from parametricSN.training.training_factory import train_test_factory



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


    ssc = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset

    train_loader, test_loader, params['general']['seed'] = ssc.generateNewSet(#Sample from datasets
        device, workers=params['general']['cores'],
        seed=params['general']['seed'],
        load=False
    ) 

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
        filter_video=params['scattering']['filter_video'],
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

    hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda) #creat hybrid model

    optimizer = optimizerFactory(hybridModel=hybridModel, params=params)

    #use gradient accumulation if VRAM is constrained
    if params['model']['loss'] == 'cross-entropy-accum':
        if params['dataset']['accum_step_multiple'] % params['dataset']['train_batch_size'] != 0:
            print("Incompatible batch size and accum step multiple")
            raise Exception
        else:
            steppingSize = int(params['dataset']['accum_step_multiple']/params['dataset']['train_batch_size'])

        params['dataset']['accum_step_multiple']
        scheduler = schedulerFactory(
            optimizer=optimizer, params=params, 
            steps_per_epoch=math.ceil(ssc.trainSampleCount/params['dataset']['accum_step_multiple'])
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

    # if params['scattering']['filter_video']:

        # videoWriterReal = cv2.VideoWriter('videos/scatteringFilterProgressionReal{}epochs.avi'.format(params['model']['epoch']),cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        # videoWriterImag = cv2.VideoWriter('videos/scatteringFilterProgressionImag{}epochs.avi'.format(params['model']['epoch']),cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        # videoWriterFourier = cv2.VideoWriter('videos/scatteringFilterProgressionFourier{}epochs.avi'.format(params['model']['epoch']),cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)


    train, test = train_test_factory(params['model']['loss'])

    print(hybridModel.__dict__.keys())

    for epoch in  range(0, params['model']['epoch']):
        t1 = time.time()
        hybridModel.scatteringBase.setEpoch(epoch)

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

        
        train_loss, train_accuracy = train(hybridModel, device, train_loader, scheduler, optimizer, 
                                           epoch+1, alternating=params['optim']['alternating'],
                                           glicoController=None, accum_step_multiple=steppingSize)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # param_distance.append(hybridModel.scatteringBase.checkDistance(compared='params'))
        param_distance.append(hybridModel.scatteringBase.checkParamDistance())
        wavelet_distance.append(hybridModel.scatteringBase.checkDistance(compared='wavelets_complete'))

        # if params['scattering']['filter_video']:
        #     temp = cv2.applyColorMap(np.array(hybridModel.scatteringBase.getAllFilters(totalCount=16, scale=0, mode='real'),dtype=np.uint8),cv2.COLORMAP_TURBO)
        #     temp = cv2.putText(temp, "Epoch {}".format(epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     videoWriterReal.write(temp)

        #     temp = cv2.applyColorMap(np.array(hybridModel.scatteringBase.getAllFilters(totalCount=16, scale=0, mode='imag'),dtype=np.uint8),cv2.COLORMAP_TURBO)
        #     temp = cv2.putText(temp, "Epoch {}".format(epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     videoWriterImag.write(temp)

        #     temp = cv2.applyColorMap(np.array(hybridModel.scatteringBase.getAllFilters(totalCount=16, scale=0, mode='fourier'),dtype=np.uint8),cv2.COLORMAP_TURBO)
        #     temp = cv2.putText(temp, "Epoch {}".format(epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     videoWriterFourier.write(temp)

        trainTime.append(time.time()-t1)
        if epoch % params['model']['step_test'] == 0 or epoch == params['model']['epoch'] -1: #check test accuracy
            t1 = time.time()
            accuracy, test_loss = test(hybridModel, device, test_loader)
            test_losses.append(test_loss)
            test_acc.append(accuracy)
            

            testTime.append(time.time()-t1)
            estimateRemainingTime(trainTime=trainTime,testTime=testTime,epochs= params['model']['epoch'],currentEpoch=epoch,testStep=params['model']['step_test'])

    if params['scattering']['filter_video']:
        hybridModel.scatteringBase.releaseVideoWriters()
        # videoWriter.release()


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
    subparser.add_argument("--scattering-filter-video", "-sfv", type=int, choices=[0,1])

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
                'scattering_second_order','scattering_three_phase','dataset_glico',
                'scattering_filter_video']:
        if args.__dict__[key] != None:
            args.__dict__[key] = bool(args.__dict__[key]) #make 0 and 1 arguments booleans

    args.callback(args)


if __name__ == '__main__':
    main()
