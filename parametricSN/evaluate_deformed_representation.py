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
import sys
import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd()))

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

def get_l2norm_deformation( model_path,  test_loader, img, device = None, num_data = 15):
    
    deformations, l2_norm= [], []
    hybridModel= load_models_weights(model_path,device)
    _, params = get_context(os.path.join(model_path,'parameters.yml'), True) 
    
    print("Starting evaluate representationfor hybridModel".format(params['model']['trainable_parameters']))

    x, y, x_labels, titles = [], [], [], []
    
    # rotation
    angles = torch.arange(0,30, 30/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees=[0,0])
    l2_norm, deformations= compute_l2norm(hybridModel, 'rotation',img, angles, transforms, device)
    x_labels.append('Degree')
    titles.append('Transformation: Rotation')
    append_to_list(x, y, l2_norm, deformations)

    # distortion
    distortion_levels = torch.arange(0,1, 1/ num_data ).to(device)
    transforms = torchvision.transforms.RandomPerspective(distortion_scale=0, p=1)
    l2_norm, deformations = compute_l2norm(hybridModel,  'distortion', img, distortion_levels, transforms, device)
    x_labels.append('Distortion Level')
    titles.append('Transformation: Distortion')
    append_to_list(x, y, l2_norm, deformations)

    # shear
    shears = torch.arange(0,40, 40/ num_data ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees = 0, shear= [0, 0])
    l2_norm, deformations = compute_l2norm(hybridModel, 'shear', img, shears, transforms, device)
    x_labels.append('Degree')
    titles.append('Transformation: Shear')
    append_to_list(x, y, l2_norm, deformations)

    # Sharpness
    # deformations_list = torch.arange(0,100, 1, dtype=int ).to(device)
    # transforms = torchvision.transforms.RandomAdjustSharpness(sharpness_factor=100, p=1)
    # l2_norm, deformations, l2_norm_learnable, deformations_learnable = compute_l2norm(hybridModel, hybridModelLearnable, 'sharpness', img, deformations_list , transforms, device )
    # x_labels.append('Sharpness Factor')
    # titles.append('Transformation: sharpness')
    # append_to_list(x, y,x_learnable,y_learnable, deformations, l2_norm, deformations_learnable, l2_norm_learnable)
    
    # Horizontal translation
    height = params['dataset']['height'] 
    max_translate = int(height* 0.1)
    deformations_list = torch.arange(0,max_translate , max_translate /num_data, dtype=int ).to(device)
    transforms = torchvision.transforms.RandomAffine(degrees = 0, translate=[0,0])
    l2_norm, deformations= compute_l2norm(hybridModel,  'translate', img, deformations_list , transforms, device )
    x_labels.append('Horizontal Translation ratio')
    titles.append('Transformation: Horizontal Translation')
    append_to_list(x, y, l2_norm, deformations)

    distance  = get_baseline(img, (iter(test_loader)), hybridModel,  device)

    values = {"model_path": model_path,"distance": distance,  "x":x, "y": y, 
                "x_labels": x_labels, "titles":titles, "params": params}
    print("Done evaluating representationfor hybridModel: {}".format(model_path))
    return values
    

def evaluate_deformed_repressentation(models):
    """Initializes and trains scattering models with different architectures
    """

    # we'll use the parameters.yml of the first model to generate our dataloader
    _, params = get_context(os.path.join(models[0],'parameters.yml'), True) #parse params
    params['dataset']['test_batch_size'] =1
    params['dataset']['train_batch_size'] =1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader= get_test_loader(params, use_cuda)

    it = (iter(test_loader))  
    img, _ = next(it)
    img.to(device)  
    
    model_values = []
    for model in models:
        model_values.append(get_l2norm_deformation( model, test_loader, img, device, num_data = 15))
    
    f = visualize_l2norm(model_values)
    log_mlflow(params, model_values, f, img)

def load_models_weights(model_path, device ):
    hybridModel = mlflow.pytorch.load_model(model_path)
    hybridModel.to(device)
    hybridModel.eval()
    return hybridModel

def get_test_loader(params, use_cuda):

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')
    train_loader, test_loader, params['general']['seed'], _ = datasetFactory(params,DATA_DIR,use_cuda) 
    return train_loader, test_loader


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
                deformations.append(v.item())
                l2_norm.append(torch.dist(representation_0, representation ).item())

    l2_norm = np.array(l2_norm)
    deformations = np.array(deformations)
    return l2_norm, deformations

def append_to_list(x, y, l2_norm, deformations):
    x.append(deformations)
    y.append(l2_norm)


def get_baseline(img, it, hybridModel,  device, num_images=50):
    '''
        In this function, we construct the image scattering coefficients of 
            1- classical scattering network (representation_0)
            2- leanrnable scattering network (representation_0_learnable)
        Then, we do the same thing for 10 (default_value, but can be changed) differents images.
        One we have all the representations, we compute the average euclidien distance 
        bewteen representation_0 and representation_0 and all the representations.
        Thus, the outputs of the function are the 2 average distances computed. 
    '''
    distances= []
    first_transformation = True
    with torch.no_grad():
        for i in range(num_images):
            if first_transformation:
                representation_0 = hybridModel.scatteringBase(img.to(device))
                first_transformation= False
            else:
                img2, _ = next(it)  
                representation = hybridModel.scatteringBase(img2.to(device) )
                distances.append(torch.dist(representation_0, representation ).item())
    return np.array(distances).mean()

    
def visualize_l2norm(model_values):
    plt.rcParams.update({'font.size': 20})
    row = 2
    col = 2
    size = (35, 10*row,)
    colors = ['#ff0000','#0000ff', '#008000','#ffd700', '#800000', '#ff00ff' ]
    f, axarr = plt.subplots(row, col, figsize=size) # create plots
    for c, model_value in enumerate(model_values):
        count = 0
        for i in range(row):
            for p in range(col):
                axarr[i, p].scatter(x= model_value["x"][count], y= model_value["y"][count], 
                                    label= f'{model_value["params"]["scattering"]["init_params"]} + {model_value["params"]["scattering"]["learnable"]}',
                                    color =colors[c])
                axarr[i, p].axhline(y= model_value['distance'], color=colors[c], linestyle='-')
                axarr[i, p].set_xlabel(model_value["x_labels"][count], fontsize=20)
                axarr[i, p].set_title(model_value['titles'][count],    fontsize=20)
                axarr[i, p].set_ylabel('||S(x_tild) - S(x)||',   fontsize=20)
                axarr[i, p].legend(fontsize = 20)
                count+=1 
    return f

def log_mlflow(params, model_values, f, img):
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
        mlflow.log_figure(f, f'plot/deformation.pdf')
        img = torchvision.transforms.functional.to_pil_image(img.cpu()[0,:, :, :])
        mlflow.log_image(img, "image.png")
        # for value in model_values:
        #     learnable = value['params']['scattering']['learnable']
        #     init= value['params']['scattering']['init_params']
        #     log_csv_file(f"{learnable}_{init}_deformations.csv", value['x'])
        #     log_csv_file(f"{learnable}_{init}_l2norm.csv", value['y'])
        print(f"finish logging{params['mlflow']['tracking_uri']}")

if __name__ == '__main__':
    n = len(sys.argv)
    print("Total paths passed:", n-1)
    models = []
    for i in range(1, n):
        models.append(sys.argv[i])
    evaluate_deformed_repressentation(models)
