"""Helpers for the main

Authors: Benjamin Therien, Shanel Gauthier

Functions:
    get_context -- TODO Shanel
    visualize_loss --
    visualize_learning_rates --
    getSimplePlot -- 
    log_csv_file -- 
    log_mlflow --
    override_params -- 
    setAllSeeds --
    estimateRemainingTime --
"""

import random
import torch
import time
import mlflow
import os
import yaml
import sys

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path 

sys.path.append(str(Path.cwd()))

def get_context(parameters_file, full_path = False):
    """ TODO Shanel """
    # Get the current project path (where you open the notebook)
    # and go up two levels to get the project path
    current_dir = Path.cwd()
    #proj_path = current_dir.parent
    proj_path = current_dir
    # make the code in src available to import in this notebook
    sys.path.append(os.path.join(proj_path, 'kymatio'))

    #Catalog contains all the paths related to datasets
    if full_path:
        params_path = parameters_file
    else:
        params_path = os.path.join(proj_path, f'conf/{parameters_file}')
    
    catalog_path = os.path.join(proj_path, 'conf/data_catalog.yml')
    with open(catalog_path, "r") as f:
        catalog = yaml.safe_load(f)
    # Params contains all of the dataset creation parameters and model parameters
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

        
    return catalog, params


def visualize_loss(loss_train ,loss_test, step_test = 10, y_label='loss', num_samples =100, benchmark =False):
    """Plot Loss/accuracy"""
    f = plt.figure (figsize=(7,7))
    plt.plot(np.arange(len(loss_test)*step_test, step=step_test), loss_test, label=f'Test {y_label}') 

    if y_label == 'accuracy' and benchmark:
        if num_samples == 100:
            benchmark = 38.9
        elif num_samples == 500:
            benchmark = 54.7
        elif num_samples == 1000:
            benchmark = 62.0
        else: 
            benchmark = None

        if benchmark is not None:
            plt.axhline(y=benchmark, color='r', linestyle='-')
    plt.plot(np.arange(len(loss_train)), loss_train, label= f'Train {y_label}')
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.legend() 
    return f  

def visualize_learning_rates(lrs, lrs_orientation, lrs_scattering):
    """Plot learning rates"""
    f = plt.figure (figsize=(7,7))
    epochs = np.arange(len(lrs))
    plt.plot(epochs, lrs, label='Linear Layer LR') 

    if len(lrs_orientation) > 0:
        plt.plot(epochs, lrs_orientation, label='Orientation LR')

    if len(lrs_scattering) > 0:
        plt.plot(epochs, lrs_scattering, label='Scattering LR')

    plt.ylabel('LR')
    plt.xlabel('Epoch')
    plt.legend() 
    return f  

def getSimplePlot(xlab,ylab,title,label,xvalues,yvalues,figsize=(7,7)):
    plot = plt.figure(figsize=figsize)
    plt.title(title)
    plt.plot(xvalues, yvalues, label=label) 
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.legend() 
    return plot

def log_csv_file(name, file):
    """ TODO Shanel"""
    np.savetxt(name,  file, delimiter=",")
    mlflow.log_artifact(name, 'metrics')
    os.remove(name)

def rename_params(prefix, params):
    return {f'{prefix}-' + str(key): val for key, val in  params.items()}

def log_mlflow(params, model, test_acc, test_loss, train_acc, 
               train_loss, start_time, filters_plots_before, 
               filters_plots_after, misc_plots):
    """Log stats in mlflow
    
    parameters: 
        params -- the parameters passed to the program
        model -- the hybrid model used during training 
        test_acc -- list of test accuracies over epochs
        test_loss --  list of test losses over epochs
        train_acc --  list of train accuracies over epochs
        train_loss --  list of train losses over epochs
        start_time -- the time at which the current run was started 
        filters_plots_before -- plots of scattering filter values before training 
        filters_plots_after -- plots of scattering filter values after training 
        misc_plots -- a list of miscelaneous plots to log in mlflow
    """

    duration = (time.time() - start_time)
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])

    with mlflow.start_run():
        #metrics = {'AVG- ' + str(key): val for key, val in metrics.items()}
        mlflow.log_params(rename_params('model', params['model']))   
        mlflow.log_params(rename_params('scattering', params['scattering']))
        mlflow.log_params(rename_params('dataset', params['dataset']))
        mlflow.log_params(rename_params('optim', params['optim']))
        mlflow.log_params(params['general'])
        mlflow.log_param('Duration', duration)
        mlflow.log_metric('Final Accuracy', test_acc[-1])
        if params['model']['save']:
            mlflow.pytorch.log_model(model, artifact_path = 'model')
            mlflow.log_dict(params, "model/parameters.yml")
        #save filters 
        try:
            for key in filters_plots_before:
                
                    mlflow.log_figure(filters_plots_before[key], f'filters_before/{key}.pdf')
                    mlflow.log_figure(filters_plots_after[key], f'filters_after/{key}.pdf')
        except:
            pass

        mlflow.log_figure(misc_plots[0], f'plot/train_test_loss.pdf')
        mlflow.log_figure(misc_plots[1], f'plot/train_test_accuracy.pdf')
        mlflow.log_figure(misc_plots[2], f'plot/train_test_accuracy_2.pdf')
        
        try:
            mlflow.log_figure(misc_plots[3], f'learnable_parameters/filters_grad.pdf')
            mlflow.log_figure(misc_plots[4], f'learnable_parameters/filter_values.pdf')
            mlflow.log_figure(misc_plots[5], f'learnable_parameters/filter_parameters.pdf')
        except:
            pass

        mlflow.log_figure(misc_plots[6], f'plot/lr.pdf')
        mlflow.log_figure(misc_plots[7], f'learnable_parameters/param_distance.pdf')
        mlflow.log_figure(misc_plots[8], f'learnable_parameters/wavelet_distance.pdf')

        # saving all accuracies
        log_csv_file('test_acc.csv', test_acc)
        log_csv_file('train_acc.csv', train_acc)
        log_csv_file('test_loss.csv', test_loss)
        log_csv_file('train_loss.csv', train_loss)
        print(f"finish logging{params['mlflow']['tracking_uri']}")




def override_params(args, params):
    """override passed params dict with any CLI arguments
    
    parameters: 
        args -- namespace of arguments passed from CLI
        params -- dict of default arguments list    
    """
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


def setAllSeeds(seed):
    """Helper for setting seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def estimateRemainingTime(trainTime, testTime, epochs, currentEpoch, testStep):
    """Estimates the remaining training time based on imput
    
    Estimates remaining training time by using averages of the 
    each training and test epoch computed. Displays a message 
    indicating averages expected remaining time.

    parameters:
        trainTime -- list of time elapsed for each training epoch
        testTime -- list of time elapsed for each testing epoch
        epochs -- the total number of epochs specified
        currentEpoch -- the current epoch 
        testStep -- epoch multiple for validation set verfification 
    """
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