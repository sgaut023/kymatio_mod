"""Contains all the pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier
"""

import kymatio
import torch 
import torch.nn as nn

from kymatio import Scattering2D
import create_filters
from create_filters import *

def create_scatteringExclusive(J,N,M,params, device, use_cuda,initilization, seed=0,learnable=True):
    """Creates scattering parameters and replaces then with the specified initialization

    Creates the scattering network, adds it to the passed device, and returns its for modification. Next,
    based on input, we modify the filter initialization and use it to create new conv kernels. Then, we
    update the psi Kymatio object to match their api.

    arguments:
    params --
    device -- torch device, cuda or cpu 
    J -- scale of scattering (always 2 for now)
    N -- height of the input image
    M -- width of the input image
    learnable -- boolean saying if we want to learn params
    initilization -- the type of init: ['kymatio' or 'random']
    
    
   
    """
    J = params['scattering']['J']
    M, N= params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N']
    scattering = Scattering2D(J=J, shape=(M, N))
    L = scattering.L
    n_coefficients= 1 + L*J + L*L*J*(J-1)//2
    M_coefficient = params['preprocess']['dimension']['M']/(2**J)
    N_coefficient = params['preprocess']['dimension']['N']/(2**J)
    K = n_coefficients*3
    if use_cuda:
        scattering = scattering.cuda()
    phi, psi  = scattering.load_filters()
    params_filters = []
       
    # if we want to optimize the parameters used to create the filters
    #if params['model']['mode'] == 'scattering_dif' :      
    # We can initialize the parameters randomly or as the kymatio package does

    if learnable:
        requires_grad = True
    else:
        requires_grad = False

    if initilization == "kymatio":
        params_filters = create_filters_params(J, scattering.L,  requires_grad,  2) #kymatio init
    elif initilization == "Random":
        params_filters = create_filters_params_random( J* scattering.L ,  requires_grad,  2, seed) #random init
    else:
        raise Exception#refactor

    #old code
    if params['model']['mode'] == 'scattering':
        
    if params['model']['init_params'] =='Random':
        params_filters = create_filters_params_random( J* scattering.L ,  requires_grad,  2, seed) #random init
    else:
        # n_filters = J*scattering.L
        params_filters = create_filters_params(J, scattering.L,  requires_grad,  2) #kymatio init



    wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device )
    
    psi = update_psi(J, psi, wavelets, device) #update psi to reflect the new conv filters

    return scattering, psi, wavelets, params_filters

class sn_ScatteringBase(nn.Module):
    """A learnable scattering nn.module 
    """

    def __init__(self,N,J,learnable=True):
        super(ScatteringBase,self).__init__()


        model, scattering, psi, wavelets, params_filters = create_scattering(params, device, use_cuda, seed = params['model']['seed'])
        lr_scattering = params['model']['lr_scattering']  
        lr_orientation = params['model']['lr_orientation']  
        
        filters_plots_before = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(psi, J, 8, mode =mode) 
            filters_plots_before [mode]  = f  
        
        psi_skeleton = psi #build psi skeleton (kymatio data structure)

        #set optimizer
        parameters = [
            {'params': model.parameters()},
            {'params': params_filters[0], 'lr': lr_orientation},
            {'params': [params_filters[1], params_filters[2],
             params_filters[3]],'lr': lr_scattering}
        ]




class sn_CNNOnTop(nn.Module):
    pass
    