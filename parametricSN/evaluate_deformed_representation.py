"""
TODO Shanel 
TODO Laurent
"""
import sys
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import ToPILImage
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

def apply_transformation(max_value, name, hybridModel, img,x,y,titles,transformation_names,imgs_deformed, transform = None, device = None, num_data = 15):
    """
    TODO Shanel 
    TODO Laurent
    """
    if max_value > num_data:
        deformation_levels = torch.arange(0,max_value, max_value/ num_data,dtype = int)#.to(device)
    else:
        deformation_levels = torch.arange(0,max_value, max_value/ num_data )#.to(device)
    l2_norm, deformations, img_deformed = compute_l2norm(hybridModel, name,img, deformation_levels, transform, device)
    titles.append(f'Transformation: {name}')
    imgs_deformed.append(img_deformed)
    transformation_names.append(name)

    append_to_list(x, y, l2_norm, deformations)

def get_l2norm_deformation( model_path,  test_loader, img, device = None, num_data = 15):
    """
    TODO Shanel 
    TODO Laurent
    """
    hybridModel= load_models_weights(model_path,device)
    _, params = get_context(os.path.join(model_path,'parameters.yml'), True) 
    
    print("Starting evaluate representationfor hybridModel".format(params['model']['trainable_parameters']))

    x, y, titles, transformation_names, imgs_deformed = [], [], [], [], []
    
    # rotation
    transform = torchvision.transforms.RandomAffine(degrees=[0,0])
    apply_transformation(max_value = 30, name = "rotation", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names, imgs_deformed=imgs_deformed,
                         transform = transform, device = device, num_data = 15)

    # distortion
    transform = torchvision.transforms.RandomPerspective(distortion_scale=0, p=1)
    apply_transformation(max_value = 0.2, name = "distortion", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,imgs_deformed=imgs_deformed,
                         transform = transform, device = device, num_data = 15)


    # shear
    transform = torchvision.transforms.RandomAffine(degrees = 0, shear= [0, 0])
    apply_transformation(max_value = 40, name = "shear", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,imgs_deformed=imgs_deformed,
                         transform = transform, device = device, num_data = 15)


    # Sharpness
    transform = torchvision.transforms.RandomAdjustSharpness(sharpness_factor=100, p=1)
    apply_transformation(max_value = 100, name = "sharpness", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,imgs_deformed=imgs_deformed,
                         transform = transform, device = device, num_data = 15)
    
    # Horizontal translation
    height = params['dataset']['height'] 
    max_translate = int(height* 0.1)
    transform = torchvision.transforms.RandomAffine(degrees = 0, translate=[0,0])
    apply_transformation(max_value = max_translate, name = "translation", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,imgs_deformed=imgs_deformed,
                         transform = transform, device = device, num_data = 15)

    # Mallat1
    # apply_transformation(max_value = 1, name = "Mallat1", hybridModel = hybridModel, img = img,
    #                      x=x,y=y,titles=titles,transformation_names=transformation_names,imgs_deformed=imgs_deformed,
    #                      transform = None, device = device, num_data = 15)

    distance  = get_baseline(img, (iter(test_loader)), hybridModel,  device)

    print("Done evaluating representationfor hybridModel: {}".format(model_path))
    
    values = {"model_path": model_path,"distance": distance,  "x":x, "y": y, "titles":titles, "params": params,
             "transformation_names":transformation_names,"image":imgs_deformed}
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
    
    figures = visualize_l2norm(model_values,len(model_values[0]["x"]))
    log_mlflow(params, model_values, figures, img)

def load_models_weights(model_path, device ):
    """
    TODO Shanel 
    TODO Laurent
    """
    hybridModel = mlflow.pytorch.load_model(model_path)
    hybridModel.to(device)
    hybridModel.eval()
    return hybridModel

def get_test_loader(params, use_cuda):
    """
    TODO Shanel 
    TODO Laurent
    """

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')
    train_loader, test_loader, params['general']['seed'], _ = datasetFactory(params,DATA_DIR,use_cuda) 
    return train_loader, test_loader


def compute_l2norm(hybridModel, deformation, img, deformation_list, transforms, device = None):
    """
    TODO Shanel 
    TODO Laurent
    """
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
                    transforms.distortion_scale = v.item()
                    img_deformed = transforms(img).to(device)
                elif deformation == 'sharpness':
                    transforms.sharpness_factor = v.item()
                    img_deformed = transforms(img).to(device)
                elif deformation == 'translation':
                    ret = transforms.get_params(transforms.degrees, transforms.translate, transforms.scale, transforms.shear, img.shape)
                    ret = list(ret)
                    ret[1] = (v.item(),0)
                    ret= tuple(ret)
                    img_deformed  = torchvision.transforms.functional.affine(img.to(device), *ret, interpolation=transforms.interpolation, fill=False)
                elif deformation == "Mallat1":
                    tau = lambda u : (v.item() *(0.5*u[0]+0.3*u[1]**2),v.item() *(0.3*u[1]) )      
                    img_deformed = diffeo(img.to(device),tau,device)

                
                representation = hybridModel.scatteringBase(img_deformed)
                if deformation == "Mallat1":
                    deformationSize = deformation_size(tau)
                    deformations.append(deformationSize)
                else:
                    deformations.append(v.item())
               # l2_norm.append(torch.dist(representation_0, representation ).item())
                l2_norm.append(torch.linalg.norm(representation_0 - representation).item()/
                              torch.linalg.norm(representation_0).item())
                
    l2_norm = np.array(l2_norm)
    deformations = np.array(deformations)
    return l2_norm, deformations, img_deformed

def append_to_list(x, y, l2_norm, deformations):
    """
    TODO Shanel 
    TODO Laurent
    """
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
                distances.append(torch.dist(representation_0, representation ).item()/
                                torch.linalg.norm(representation_0).item())
    return np.array(distances).mean()

    
def visualize_l2norm(model_values, num_transformations = 4):
    """
    TODO Shanel 
    TODO Laurent
    """
    plt.rcParams.update({'font.size': 20})
    colors = ['#ff0000','#0000ff', '#008000','#ffd700', '#800000', '#ff00ff' ]
    figures = []
    for idx in range(num_transformations):
        f = plt.figure(figsize=(10,10)) # create plots
        for c, model_value in enumerate(model_values):
            plt.scatter(x= model_value["x"][idx], y= model_value["y"][idx], 
                                    label= f'{model_value["params"]["scattering"]["init_params"]} + {model_value["params"]["scattering"]["learnable"]}',
                                    color =colors[c])
            plt.axhline(y= model_value['distance'], color=colors[c], linestyle='-')
            plt.xlabel("Deformation Size", fontsize=20)
            plt.title(model_value['titles'][idx],    fontsize=20)
            plt.ylabel('||S(x_tild) - S(x)||',   fontsize=20)
            plt.legend(fontsize = 20)
        figures.append(f)
    return figures

    

# Deforms the image given a function \tau.
def diffeo(img,tau,device):
    """
    TODO Shanel 
    TODO Laurent
    """
    img = img
    # Number of pixels. Suppose square image.
    dim = img.shape[-1]
    # Create a (dim x dim) matrix of 2d vectors. Each vector represents the normalized position in the grid. 
    # Normalized means (-1,-1) is top left and (1,1) is bottom right.
    grid = torch.tensor([[[x,y] for x in torch.linspace(-1,1,dim)] for y in torch.linspace(-1,1,dim)])
    # Apply u-tau(u) in Mallat's. 
    tau_mat = lambda grid : torch.tensor([[tau(grid[i,j,:]) for j in range(len(grid))] for i in range(len(grid))])
    grid_transf = (grid - tau_mat(grid)).unsqueeze(0).to(device)
    # Apply x(u-tau(u)) by interpolating the image at the index points given by grid_transf.
    img_transf = torch.nn.functional.grid_sample(img,grid_transf)
    return img_transf

# Calculate the deformation size : sup |J_{tau}(u)| over u.
def deformation_size(tau):
    """
    TODO Shanel 
    TODO Laurent
    """
    # Set a precision. This is arbitrary.
    precision = 128
    # Create a (flatten) grid of points between (-1,-1) and (1,1). This is the same grid as in the previous
    # function (but flatten), but it feels arbitrary also.
    points = [torch.tensor([x,y]) for x in torch.linspace(-1,1,precision) for y in torch.linspace(-1,1,precision)]
    # Evaluate the Jacobian of tau in each of those points. Returns a tensor of precision^2 x 2 x 2, i.e.
    # for each point in points the 2 x 2 jacobian. Is it necessary to compute on all points, or only on the
    # boundary would be sufficient?
    jac = torch.stack(list(map(lambda point : torch.stack(torch.autograd.functional.jacobian(tau,point)), points)))
    # Find the norm of those jacobians.
    norm_jac = torch.linalg.matrix_norm(jac,ord=2,dim=(1, 2))
    # Return the Jacobian with the biggest norm.
    return torch.max(norm_jac)

def log_mlflow(params, model_values, figures, img):
    """
    TODO Shanel 
    TODO Laurent
    """
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment('Exp Deformation')

    with mlflow.start_run():
        ToPILImage = torchvision.transforms.ToPILImage()
        img = ToPILImage(img.squeeze(0)).convert("L")
        #metrics = {'AVG- ' + str(key): val for key, val in metrics.items()}
        mlflow.log_params(rename_params('model', params['model']))   
        mlflow.log_params(rename_params('scattering', params['scattering']))
        mlflow.log_params(rename_params('dataset', params['dataset']))
        mlflow.log_params(rename_params('optim', params['optim']))
        mlflow.log_params(params['general'])
        mlflow.log_dict(params, "model/parameters.yml")
        for i,figure in enumerate(figures):
            mlflow.log_figure(figure, f'Deformation/{model_values[0]["transformation_names"][i]}/Deformation.pdf')        
            mlflow.log_image(img, f'Deformation/{model_values[0]["transformation_names"][i]}/Image_before.pdf')
            img_deformed = ToPILImage(model_values[0]["image"][i].squeeze(0)).convert("L")
            mlflow.log_image(img_deformed, f'Deformation/{model_values[0]["transformation_names"][i]}/Image_after.pdf')
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
