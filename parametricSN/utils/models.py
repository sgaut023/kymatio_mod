"""Contains all the pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier
"""

import kymatio
import torch 
import torch.nn as nn

from kymatio import Scattering2D
from .create_filters import *
from .wavelet_visualization import get_filters_visualization


class InvalidInitializationException(Exception):
    """Error thrown when an invalid initialization scheme is passed"""
    pass


class InvalidArchitectureError(Exception):
    """Error thrown when an invalid architecture name is passed"""
    pass

def create_scatteringExclusive(J,N,M,initilization,seed=0,requires_grad=True,CUDA=True):
    """Creates scattering parameters and replaces then with the specified initialization

    Creates the scattering network, adds it to the passed device, and returns its for modification. Next,
    based on input, we modify the filter initialization and use it to create new conv kernels. Then, we
    update the psi Kymatio object to match their api.

    arguments:
    CUDA -- True if were using gpu
    J -- scale of scattering (always 2 for now)
    N -- height of the input image
    M -- width of the input image
    learnable -- boolean saying if we want to learn params
    initilization -- the type of init: ['kymatio' or 'random']
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scattering = Scattering2D(J=J, shape=(M, N))

    L = scattering.L
    n_coefficients= 1 + L*J + L*L*J*(J-1)//2
    K = n_coefficients*3

    if CUDA:
        scattering = scattering.cuda()
    phi, psi  = scattering.load_filters()
    
    params_filters = []

    if initilization == "kymatio":
        params_filters = create_filters_params(J,L,requires_grad,2) #kymatio init
    elif initilization == "Random":
        params_filters = create_filters_params_random( J*L,requires_grad,2,seed) #random init
    else:
        raise InvalidInitializationException


    wavelets  = morlets((scattering.M_padded, scattering.N_padded,), params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device )
    
    psi = update_psi(J, psi, wavelets, device) #update psi to reflect the new conv filters

    return scattering, psi, wavelets, params_filters, n_coefficients

class sn_HybridModel(nn.Module):
    """An nn.Module incorporating scattering an a learnable network"""
    def __init__(self,scatteringBase,top):
        self.scatteringBase = scatteringBase
        self.top = top

    def parameters(self):
        """implement parameters method correctly (THIS MIGHT NOT BE NEEDED)"""
        temp = self.top.parameters()
        temp.extend(self.scatteringBase.parameters())
        return temp

    def forward(self,inp):
        return self.top(self.scatteringBase(inp))


class sn_ScatteringBase(nn.Module):
    """A learnable scattering nn.module 

    parameters:
    learnable -- should the filters be learnable parameters of this model
    CUDA -- True if we are using cuda
    J -- scale of scattering (always 2 for now)
    N -- height of the input image
    M -- width of the input image
    initilization -- the type of init: ['kymatio' or 'random']
    seed -- the random seed used to initialize the parameters

    """

    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        filter_viz = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(self.psi, self.J, 8, mode=mode) 
            filter_viz [mode]  = f  

        return filter_viz

    def __init__(self,J,N,M,initialization,seed,learnable=True,lr_orientation=0.1,lr_scattering=0.1,CUDA=True):
        """Creates scattering filters and adds them to the nn.parameters if learnable"""
        super(sn_ScatteringBase,self).__init__()
        self.J = J
        self.N = N
        self.M = M
        self.learnable = learnable
        self.CUDA = CUDA 
        self.initialization = initialization
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.M_coefficient = self.M/(2**self.J)
        self.N_coefficient = self.N/(2**self.J)

        self.scattering, self.psi, self.wavelets, self.params_filters, self.n_coefficients = create_scatteringExclusive(
            J,N,M,initilization=self.initialization,seed=seed,requires_grad=learnable,CUDA=self.CUDA
        )
        
        self.filters_plots_before = self.getFilterViz()

        self.scatteringTrain = False

    def train(self):
        super(sn_ScatteringBase,self).train()
        self.scatteringTrain = True

    def eval(self):
        super(sn_ScatteringBase,self).eval()
        if self.scatteringTrain:
            self.updateFilters()
        self.scatteringTrain = False

        
    def parameters(self):
        """ override parameters to include learning rates """
        if self.learnable:
            return [
                {'params': self.params_filters[0], 'lr': self.lr_orientation},
                {'params': [self.params_filters[1], self.params_filters[2],
                self.params_filters[3]],'lr': self.lr_scattering}
            ]
        else: 
            return []

    def updateFilters(self):
        """if were using learnable scattering, update the filters to reflect the new parameter values obtained from gradient descent"""
        if self.learnable:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.wavelets = morlets((self.scattering.M_padded, self.scattering.N_padded), self.params_filters[0], 
                                    self.params_filters[1], self.params_filters[2], self.params_filters[3], device=device)
            self.psi = update_psi(self.scattering.J, self.psi, self.wavelets, device) 
        else:
            pass


    def forward(self, ip):
        """ apply the scattering transform to the input image """
        if self.scatteringTrain:#update filters if training
            self.updateFilters()
        return construct_scattering(ip, self.scattering, self.psi)


def modelFactory(base,architecture,num_classes):
    """factory for the creation of different model architectures associated to a scattering base"""

    if architecture.lower() == 'cnn':
        return sn_CNN(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            standard=False
        )
    elif architecture.lower() == 'mlp':
        return sn_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            standard=False
        )
    elif architecture.lower() == 'linear_layer':
        return sn_LinearLayer(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            standard=False
        )
    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()


class sn_CNN(nn.Module):
    pass

class sn_MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, standard=False):
    super(sn_MLP,self).__init__()

    if standard:
        fc1 = nn.Linear(3*32*32, 512)
    else:
        fc1=  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients),  512)
    
    self.layers = nn.Sequential(
      fc1,
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, num_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    x = x.view(x.shape[0], -1)
    return self.layers(x)


class sn_LinearLayer(nn.Module):
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, standard=False):
        super(sn_LinearLayer,self).__init__()

        if standard:
            self.fc1 = nn.Linear(3*32*32, 256)
            self.fc2 = nn.Linear(256, num_classes)
        else:
            self.fc1=  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients), 1024)
            self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    