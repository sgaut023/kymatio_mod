"""Contains all the pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Exceptions:
    InvalidInitializationException --
    InvalidArchitectureError --

Functions: 
    create_scatteringExclusive -- creates scattering parameters
    model_factory -- creates different models based on input

Classes: 
    sn_Identity -- computes the identity function in forward pass
    sn_HybridModel -- combinations of a scattering and other nn.modules
    sn_ScatteringBase -- a scattering network
    sn_CNN -- CNN fitted for scattering input
    sn_LinearLayer -- Linear layer fitted for scattering input
    sn_MLP -- multilayer perceptron fitted for scattering input
"""

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

def modelFactory(base,architecture,num_classes, width =8, use_cuda=True):
    """factory for the creation of different model architectures associated to a scattering base"""

    if architecture.lower() == 'cnn':
        return sn_CNN(
        base.n_coefficients, width, num_classes=num_classes, standard=False
        )
    elif architecture.lower() == 'mlp':
        return sn_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            standard=False,use_cuda=use_cuda
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

def create_scatteringExclusive(J,N,M,initilization,seed=0,requires_grad=True,use_cuda=True):
    """Creates scattering parameters and replaces then with the specified initialization

    Creates the scattering network, adds it to the passed device, and returns its for modification. Next,
    based on input, we modify the filter initialization and use it to create new conv kernels. Then, we
    update the psi Kymatio object to match their api.

    arguments:
    use_cuda -- True if were using gpu
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

    if use_cuda:
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

class sn_Identity(nn.Module):
    """Identity nn.Module for identity"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    

class sn_HybridModel(nn.Module):
    """An nn.Module incorporating scattering an a learnable network"""
    def __init__(self,scatteringBase,top,use_cuda=True):
        super(sn_HybridModel,self).__init__()
        if use_cuda:
            self.scatteringBase = scatteringBase.cuda()
            self.top = top.cuda()
        else:
            self.scatteringBase = scatteringBase.cpu()
            self.top = top.cpu()

    def parameters(self):
        """implements parameters method allowing the use of scatteringBase's method """
        return list(self.top.parameters()) + list(self.scatteringBase.parameters())

    def forward(self,inp):
        return self.top(self.scatteringBase(inp))

    def train(self):
        self.scatteringBase.train()
        self.top.train()

    def eval(self):
        self.scatteringBase.eval()
        self.top.eval()


class sn_ScatteringBase(nn.Module):
    """A learnable scattering nn.module 

    parameters:
    learnable -- should the filters be learnable parameters of this model
    use_cuda -- True if we are using cuda
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

    def __init__(self,J,N,M,initialization,seed,learnable=True,lr_orientation=0.1,lr_scattering=0.1,use_cuda=True):
        """Creates scattering filters and adds them to the nn.parameters if learnable"""
        super(sn_ScatteringBase,self).__init__()
        self.J = J
        self.N = N
        self.M = M
        self.learnable = learnable
        self.use_cuda = use_cuda 
        self.initialization = initialization
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.M_coefficient = self.M/(2**self.J)
        self.N_coefficient = self.N/(2**self.J)

        self.scattering, self.psi, self.wavelets, self.params_filters, self.n_coefficients = create_scatteringExclusive(
            J,N,M,initilization=self.initialization,seed=seed,requires_grad=learnable,use_cuda=self.use_cuda
        )
        
        self.filters_plots_before = self.getFilterViz()

        self.scatteringTrain = False

    def train(self,mode=True):
        super().train(mode=mode)
        self.scatteringTrain = True

    def eval(self):
        super().eval()
        if self.scatteringTrain:
            self.updateFilters()
        self.scatteringTrain = False


    def parameters(self):
        """ override parameters to include learning rates """
        if self.learnable:
            yield {'params': self.params_filters[0], 'lr': self.lr_orientation}
            yield {'params': [self.params_filters[1], self.params_filters[2],
                self.params_filters[3]],'lr': self.lr_scattering}

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



class sn_MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, standard=False, use_cuda=True):
    super(sn_MLP,self).__init__()
    if use_cuda:
        self.cuda()

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
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, standard=False, use_cuda=True):
        super(sn_LinearLayer,self).__init__()
        if use_cuda:
            self.cuda()

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

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class sn_CNN(nn.Module):
    def __init__(self, in_channels ,  k=8, n=4, num_classes=10,standard=False):
        super(sn_CNN, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        in_channels = in_channels * 3
        if standard:

            self.init_conv = nn.Sequential(
                nn.Conv2d(3, self.ichannels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
                nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.standard = False

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            x = x.view(x.size(0), self.K, 8, 8)

        x = self.init_conv(x)

        if self.standard:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 