import sys
from pathlib import Path 
import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd()))
import time
import numpy as np
import mlflow
import os


def visualize_loss(loss_train ,loss_test, step_test = 10, y_label='loss'):
    f = plt.figure (figsize=(7,7))
    plt.plot(np.arange(len(loss_test)*step_test, step=step_test), loss_test, label='Test Loss') 
    plt.plot(np.arange(len(loss_train)), loss_train, label= 'Train Loss')
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.legend() 
    return f  

def visualize_learning_rates(lrs, lrs_orientation, lrs_scattering):
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

def log_mlflow(params, model, test_acc, start_time,filters_plots_before, filters_plots_after , 
                figures_plot, f_lr):
    duration = (time.time() - start_time)
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    with mlflow.start_run():
        mlflow.log_params(params['model'])   
        mlflow.log_params(params['scattering'])
        mlflow.log_params(params['preprocess']['dimension'])
        mlflow.log_param('Duration', duration)
        mlflow.log_metric('Final Accuracy', test_acc[-1])
        mlflow.pytorch.log_model(model, artifact_path = 'model')

        #save filters 
        for key in filters_plots_before:
            mlflow.log_figure(filters_plots_before[key], f'filters_before/{key}.png')
            mlflow.log_figure(filters_plots_after[key], f'filters_after/{key}.png')
        mlflow.log_figure(figures_plot[0], f'plot/train_test_loss.png')
        mlflow.log_figure(figures_plot[1], f'plot/train_test_accuracy.png')
        mlflow.log_figure(f_lr, f'plot/lr.png')
        # saving all accuracies
        np.savetxt('accuracy.csv', test_acc, delimiter=",")
        mlflow.log_artifact('accuracy.csv', 'accuracy_values')
        os.remove('accuracy.csv')
