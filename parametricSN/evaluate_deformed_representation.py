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

import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd()))

import argparse
import torch
import mlflow
import torchvision

import torch.nn.functional as F
import kymatio.datasets as scattering_datasets
import numpy as np
import os

from parametricSN.utils.helpers import get_context
from parametricSN.cifar_small_sample import datasetFactory
from parametricSN.utils.helpers import rename_params, log_csv_file


def evaluate_deformed_repressentation(args):
    """Initializes and trains scattering models with different architectures
    """
    model_path = args.model_path
    model_path_learnable = args.model_path_learnable
    _, params = get_context(os.path.join(model_path,'parameters.yml'), True) #parse params
    params['dataset']['test_batch_size'] =1
    params['dataset']['train_batch_size'] =1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader= get_test_loader(params, use_cuda)
    
    # load weights of the classical scattering network


    deformations, l2_norm= [], []
    num_data = 15

    hybridModel, hybridModelLearnable = load_models_weights(model_path, model_path_learnable,device)

    print("Starting evaluate representationfor hybridModel with {} parameters".format(params['model']['trainable_parameters']))

    it = (iter(test_loader))  
    img, _ = next(it)
    img.to(device)  
    x, y, x_labels, titles = [], [], [], []
    x_learnable, y_learnable = [], []
    
    # rotation
    angles = torch.arange(0,30, 30/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees=[0,0])
    l2_norm, deformations, l2_norm_learnable, deformations_learnable = compute_l2norm(hybridModel,hybridModelLearnable,  'rotation',img, angles, transforms, device)
    x_labels.append('Degree')
    titles.append('Transformation: Rotation')
    x.append(deformations)
    y.append(l2_norm)
    x_learnable.append(deformations_learnable)
    y_learnable.append(l2_norm_learnable)

    # distortion
    distortion_levels = torch.arange(0,1, 1/ num_data ).to(device)
    transforms = torchvision.transforms.RandomPerspective(distortion_scale=0, p=1)
    l2_norm, deformations, l2_norm_learnable, deformations_learnable = compute_l2norm(hybridModel, hybridModelLearnable, 'distortion', img, distortion_levels, transforms, device)
    x_labels.append('Distortion Level')
    titles.append('Transformation: Distortion')
    append_to_list(x, y,x_learnable,y_learnable, deformations, l2_norm, deformations_learnable, l2_norm_learnable)

    # shear
    shears = torch.arange(0,40, 40/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees = 0, shear= [0, 0])
    l2_norm, deformations, l2_norm_learnable, deformations_learnable = compute_l2norm(hybridModel,hybridModelLearnable, 'shear', img, shears, transforms, device)
    x_labels.append('Degree')
    titles.append('Transformation: Shear')
    append_to_list(x, y,x_learnable,y_learnable, deformations, l2_norm, deformations_learnable, l2_norm_learnable)

    # Sharpness
    # deformations_list = torch.arange(0,100, 1, dtype=int ).to(device)
    # transforms = torchvision.transforms.RandomAdjustSharpness(sharpness_factor=100, p=1)
    # l2_norm, deformations, l2_norm_learnable, deformations_learnable = compute_l2norm(hybridModel, hybridModelLearnable, 'sharpness', img, deformations_list , transforms, device )
    # x_labels.append('Sharpness Factor')
    # titles.append('Transformation: sharpness')
    # append_to_list(x, y,x_learnable,y_learnable, deformations, l2_norm, deformations_learnable, l2_norm_learnable)
    
    # Horizontal translation
    height = params['dataset']['height'] 
    max_translate = int(height* 0.07)
    deformations_list = torch.arange(0,max_translate , max_translate /num_data, dtype=int ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees = 0, translate=[0,0])
    l2_norm, deformations, l2_norm_learnable, deformations_learnable = compute_l2norm(hybridModel, hybridModelLearnable, 'translate', img, deformations_list , transforms, device )
    x_labels.append('Horizontal Translation ratio')
    titles.append('Transformation: Horizontal Translation')
    append_to_list(x, y,x_learnable,y_learnable, deformations, l2_norm, deformations_learnable, l2_norm_learnable)
    

    print("Done evaluate representationfor hybridModel with {} parameters".format(params['model']['trainable_parameters']))
    distance, distance_learnable = get_baseline(img, (iter(test_loader))  , hybridModel, hybridModelLearnable,  device)
    f = visualize_l2norm(x, y,x_learnable,y_learnable,titles, x_labels, [distance, distance_learnable])
    log_mlflow(params, deformations, l2_norm, f, img)

def load_models_weights(model_path, model_path_learnable,device ):
    hybridModel = mlflow.pytorch.load_model(model_path)
    hybridModel.to(device)
    hybridModel.eval()

    # load weights of the classical scattering network
    hybridModelLearnable = mlflow.pytorch.load_model(model_path_learnable)
    hybridModelLearnable.to(device)
    hybridModelLearnable.eval()
    return hybridModel, hybridModelLearnable

def get_test_loader(params, use_cuda):

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

  
    train_loader, test_loader, params['general']['seed'], _ = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset

    return train_loader, test_loader


def append_to_list(x, y,x_learnable,y_learnable, deformations, l2_norm, deformations_learnable, l2_norm_learnable):
    x.append(deformations)
    y.append(l2_norm)
    x_learnable.append(deformations_learnable)
    y_learnable.append(l2_norm_learnable)

def compute_l2norm(hybridModel, hybridModelLearnable, deformation, img, deformation_list, transforms, device = None):
    deformations, deformations_learnable= [], []
    l2_norm, l2_norm_learnable = [], []
    first_transformation = True
    with torch.no_grad():
        for v in deformation_list:
            if first_transformation:
                representation = hybridModel.scatteringBase(img.to(device))
                representation_0 = representation
                first_transformation= False
            else:
                if deformation == 'rotation':
                    transforms.degrees = [v.item(), v.item()]
                    img_deformed = transforms(img).to(device)
                elif deformation == 'shear':
                    transforms.shear = [v.item(), v.item()]
                    img_deformed = transforms(img).to(device)
                elif deformation == 'distortion':
                    (v.item() % 2) == 0
                    transforms.distortion_scale = v.item()
                    img_deformed = transforms(img).to(device)
                elif deformation == 'sharpness':
                    transforms.sharpness_factor = v.item()
                    img_deformed = transforms(img).to(device)
                elif deformation == 'translate':
                    ret = transforms.get_params(transforms.degrees, transforms.translate, transforms.scale, transforms.shear, img.shape)
                    ret = list(ret)
                    ret[1] = (v.item(),0)
                    ret= tuple(ret)
                    img_deformed  = torchvision.transforms.functional.affine(img.to(device), *ret, interpolation=transforms.interpolation, fill=False)
                
                representation = hybridModel.scatteringBase(img_deformed)
                representation_learnable = hybridModelLearnable.scatteringBase(img_deformed)
                deformations.append(v.item())
                deformations_learnable.append(v.item())
                l2_norm.append(torch.dist(representation_0, representation ).item())
                l2_norm_learnable.append(torch.dist(representation_0, representation_learnable ).item())
    l2_norm = np.array(l2_norm)
    deformations = np.array(deformations)
    l2_norm_learnable = np.array(l2_norm_learnable)
    deformations_learnable = np.array(deformations_learnable)
    return l2_norm, deformations, l2_norm_learnable, deformations_learnable


def get_baseline(img, it, hybridModel, hybridModel_learnable,  device, num_images=50):
    '''
        In this function, we construct the image scattering coefficients of 
            1- classical scattering network (representation_0)
            2- leanrnable scattering network (representation_0_learnable)
        Then, we do the same thing for 10 (default_value, but can be changed) differents images.
        One we have all the representations, we compute the average euclidien distance 
        bewteen representation_0 and representation_0 and all the representations.
        Thus, the outputs of the function are the 2 average distances computed. 
    '''
    distances, distances_learnable= [], []
    first_transformation = True
    with torch.no_grad():
        for i in range(num_images):
            if first_transformation:
                representation_0 = hybridModel.scatteringBase(img.to(device))
                representation_0_learnbable = hybridModel_learnable.scatteringBase(img.to(device))
                first_transformation= False
            else:
                img2, _ = next(it)  
                representation = hybridModel.scatteringBase(img2.to(device) )
                representation_learnbable = hybridModel_learnable.scatteringBase(img2.to(device))
                distances.append(torch.dist(representation_0, representation ).item())
                distances_learnable.append(torch.dist(representation_0_learnbable,representation_learnbable).item())
    return np.array(distances).mean(), np.array(distances_learnable).mean(),

    
def visualize_l2norm(x, y, x_learnable,y_learnable,titles, x_labels, baselines):
    col = 2
    row = 2
    size = (35, 10*row,)

    f, axarr = plt.subplots(row, col, figsize=size) # create plots
    count = 0
    for i in range(row):
        for p in range(col):
            axarr[i, p].scatter(x_learnable[count], y_learnable[count], label='learnable scattering', color ='blue')
            axarr[i, p].scatter(x[count], y[count], label='scattering', color ='orange')
            axarr[i, p].axhline(y=baselines[0], color='blue', linestyle='-')
            axarr[i, p].axhline(y=baselines[1], color='orange', linestyle='-')
            axarr[i, p].set_xlabel(x_labels[count], fontsize=20)
            axarr[i, p].set_ylabel('||S(x_tild) - S(x)||', fontsize=20)
            axarr[i, p].set_title(titles[count], fontsize=20)
            axarr[i, p].legend()
            count+=1 
    return f

def log_mlflow(params, deformations, l2_norm, figure, img):
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment('Exp Deformation')

    with mlflow.start_run():
        #metrics = {'AVG- ' + str(key): val for key, val in metrics.items()}
        mlflow.log_params(rename_params('model', params['model']))   
        mlflow.log_params(rename_params('scattering', params['scattering']))
        mlflow.log_params(rename_params('dataset', params['dataset']))
        mlflow.log_params(rename_params('optim', params['optim']))
        mlflow.log_params(params['general'])
        mlflow.log_dict(params, "model/parameters.yml")
        mlflow.log_figure(figure, f'plot/deformation.pdf')
        img = torchvision.transforms.functional.to_pil_image(img.cpu()[0,:, :, :])
        mlflow.log_image(img, "image.png")
        log_csv_file('deformations.csv', deformations)
        log_csv_file('l2norm.csv', l2_norm)
        print(f"finish logging{params['mlflow']['tracking_uri']}")



if __name__ == '__main__':
    # We need the paths to 2 differents models
    # The first Path is the path to the classical scattering network
    # The second Path is the path to the learnable scattering network
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--model-path-learnable", "-ml", type=str, required=True)
    args = parser.parse_args()
    evaluate_deformed_repressentation(args)
