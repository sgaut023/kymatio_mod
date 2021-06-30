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
from numpy.testing._private.utils import print_assert_equal

import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd()))

import argparse
import torch
import math
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
    _, params = get_context(os.path.join(model_path,'parameters.yml'), True) #parse params
    params['dataset']['test_batch_size'] =1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

  
    _, test_loader, params['general']['seed'], _ = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset
    
    # load weights
    hybridModel = mlflow.pytorch.load_model(model_path)
    hybridModel.to(device)
    hybridModel.eval()

    deformations= []
    l2_norm = []
    num_data = 100 

    print("Starting evaluate representationfor hybridModel with {} parameters".format(params['model']['trainable_parameters']))

    img, _ = next(iter(test_loader))  
    img.to(device)  
    x, y, x_labels, titles = [], [], [], []
    
    # rotation
    angles = torch.arange(0,180, 180/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees=0)
    l2_norm, deformations = compute_l2norm(hybridModel, 'rotation',img, angles, transforms, device)
    deformations = deformations * torch.norm(img)
    x_labels.append('degree')
    titles.append('Transformation: Rotation')
    x.append(deformations)
    y.append(l2_norm)

    # distortion
    distortion_levels = torch.arange(0,1, 1/ num_data ).to(device)
    transforms = torchvision.transforms.RandomPerspective(distortion_scale=0)
    l2_norm, deformations = compute_l2norm(hybridModel, 'distortion', img, distortion_levels, transforms, device)
    deformations = deformations * torch.norm(img)
    x_labels.append('Distortion Level')
    titles.append('Transformation: Distortion')
    x.append(deformations)
    y.append(l2_norm)

    # shear
    shears = torch.arange(0,90, 90/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees = 0, shear=0)
    l2_norm, deformations = compute_l2norm(hybridModel,'shear', img, shears, transforms, device)
    deformations = deformations * torch.norm(img)
    x_labels.append('degree')
    titles.append('Transformation: Shear')
    x.append(deformations)
    y.append(l2_norm)

    # shear
    shears = torch.arange(0,180, 180/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees = 0, shear=0)
    l2_norm, deformations = compute_l2norm(hybridModel,'shear', img, shears, transforms, device)
    deformations = deformations * torch.norm(img)
    x_labels.append('degree')
    titles.append('Transformation: Shear')
    x.append(deformations)
    y.append(l2_norm)

    # gaussian blur
    # L = min(params['dataset']['height'], params['dataset']['width'])
    # deformations_list = torch.arange(0,L, L/ num_data, dtype=int ).to(device)
    # transforms = torchvision.transforms.GaussianBlur(kernel_size = 1)
    # # set_value = lambda transforms, v: transforms.kernel_size = (v.item(),v.item())
    # l2_norm, deformations = compute_l2norm(hybridModel, 'gaussian', img, deformations_list , transforms, device )
    # x_labels.append('Kernel Size')
    # titles.append('Transformation: Gaussian Blur')
    # x.append(deformations)
    # y.append(l2_norm)

 
    f = visualize_l2norm(x, y, titles, x_labels)
    log_mlflow(params, deformations, l2_norm, f, img)

def compute_l2norm(hybridModel, deformation, img, deformation_list, transforms, device = None):
    deformations= []
    l2_norm = []
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
                elif deformation == 'shear':
                    transforms.shear = [v.item(), v.item()]
                elif deformation == 'gaussian':
                        transforms.kernel_size = (v.item(), v.item())
                elif deformation == 'distortion':
                    transforms.distortion_scale = v.item()
                img_deformed = transforms(img).to(device)
                representation = hybridModel.scatteringBase(img_deformed)
                deformations.append(v.item())
                l2_norm.append(torch.dist(representation_0, representation ).item())
    l2_norm = np.array(l2_norm)
    deformations = np.array(deformations)
    return l2_norm, deformations

def visualize_l2norm(x, y, titles, x_labels):
    col = 2
    row = 2
    size = (35, 10*row,)

    f, axarr = plt.subplots(row, col, figsize=size) # create plots
    count = 0
    for i in range(row):
        for p in range(col):
            axarr[i, p].scatter(x[count], y[count])
            axarr[i, p].set_xlabel(x_labels[count], fontsize=20)
            axarr[i, p].set_ylabel('||S(x_tild) - S(x)||', fontsize=20)
            axarr[i, p].set_title(titles[count], fontsize=20)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str)
    args = parser.parse_args()
    evaluate_deformed_repressentation(args)
