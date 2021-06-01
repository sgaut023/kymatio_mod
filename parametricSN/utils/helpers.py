
import time
import mlflow
import os
import yaml
import sys

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path 

sys.path.append(str(Path.cwd()))

def get_context(parameters_file):
    # Get the current project path (where you open the notebook)
    # and go up two levels to get the project path
    current_dir = Path.cwd()
    #proj_path = current_dir.parent
    proj_path = current_dir
    # make the code in src available to import in this notebook
    sys.path.append(os.path.join(proj_path, 'kymatio'))

    #Catalog contains all the paths related to datasets
    with open(os.path.join(proj_path, 'conf/data_catalog.yml'), "r") as f:
        catalog = yaml.safe_load(f)
        
    # Params contains all of the dataset creation parameters and model parameters
    with open(os.path.join(proj_path, f'conf/{parameters_file}'), "r") as f:
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

def log_csv_file(name, file):
    np.savetxt(name,  file, delimiter=",")
    mlflow.log_artifact(name, 'metrics')
    os.remove(name)

def rename_params(prefix, params):
    return {f'{prefix}-' + str(key): val for key, val in  params.items()}

def log_mlflow(params, model, test_acc, test_loss, train_acc, 
               train_loss, start_time, filters_plots_before, 
               filters_plots_after, figures_plot, f_lr):
    """Log stats in mlflow"""

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
        #mlflow.pytorch.log_model(model, artifact_path = 'model')

        #save filters 
        for key in filters_plots_before:
            mlflow.log_figure(filters_plots_before[key], f'filters_before/{key}.pdf')
            mlflow.log_figure(filters_plots_after[key], f'filters_after/{key}.pdf')

        mlflow.log_figure(figures_plot[0], f'plot/train_test_loss.pdf')
        mlflow.log_figure(figures_plot[1], f'plot/train_test_accuracy.pdf')
        mlflow.log_figure(figures_plot[2], f'plot/train_test_accuracy_2.pdf')
        mlflow.log_figure(f_lr, f'plot/lr.pdf')

        # saving all accuracies
        log_csv_file('test_acc.csv', test_acc)
        log_csv_file('train_acc.csv', train_acc)
        log_csv_file('test_loss.csv', test_loss)
        log_csv_file('train_loss.csv', train_loss)
        print(f"finish logging{params['mlflow']['tracking_uri']}")

