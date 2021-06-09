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

from numpy.core.numeric import False_
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

def modelFactory(base,architecture,num_classes, width=8, average=False, use_cuda=True):
    """factory for the creation of different model architectures associated to a scattering base"""

    if architecture.lower() == 'cnn':
        return sn_CNN(
            base.n_coefficients, k=width, num_classes=num_classes, standard=False
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
            standard=False, average=average
        )
    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()

def create_scatteringExclusive(J,N,M,second_order,device,initilization,seed=0,requires_grad=True,use_cuda=True):
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
    scattering = Scattering2D(J=J, shape=(M, N), frontend='torch')

    L = scattering.L
    if second_order:
        n_coefficients=  L*L*J*(J-1)//2 #+ 1 + L*J  
    else: 
        n_coefficients=  L*L*J*(J-1)//2 + 1 + L*J  
    
    K = n_coefficients*3

    if use_cuda:
        scattering = scattering.cuda()

    phi, psi  = scattering.load_filters()
    
    params_filters = []

    if initilization == "Kymatio":
        params_filters = create_filters_params(J,L,requires_grad,device) #kymatio init
    elif initilization == "Random":
        #num_filters = get_total_num_filters(J,L)
        num_filters= J*L
        params_filters = create_filters_params_random(num_filters,requires_grad,device) #random init
    else:
        raise InvalidInitializationException

    # params_filters = [p.to(device) for p in params_filters]
    shape = (scattering.M_padded, scattering.N_padded,)
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), 0).to(device)
    params_filters =  [ param.to(device) for param in params_filters]

    wavelets  = morlets(shape, params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device )
    
    psi = update_psi(J, psi, wavelets, initilization , device) #update psi to reflect the new conv filters

    return scattering, psi, wavelets, params_filters, n_coefficients, grid

class sn_Identity(nn.Module):
    """Identity nn.Module for identity"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_coefficients = 1

    def forward(self, x):
        return x
        
    def saveFilterGrads(self,scatteringActive):
        pass

    def saveFilterValues(self,scatteringActive):
        pass

    def plotFilterGrad(self):
        pass

    def plotFilterGrads(self):
        pass


    def plotFilterValue(self):
        pass

    def plotFilterValues(self):
        pass


    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        return 0
    

class sn_HybridModel(nn.Module):
    """An nn.Module incorporating scattering an a learnable network"""

    def __str__(self):
        return str(self.scatteringBase)

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
        temp = [{'params': list(self.top.parameters())}]
        temp.extend(list(self.scatteringBase.parameters()))
        return temp

    def forward(self,inp):
        return self.top(self.scatteringBase(inp))

    def train(self):
        self.scatteringBase.train()
        self.top.train()

    def eval(self):
        self.scatteringBase.eval()
        self.top.eval()

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        return self.scatteringBase.countLearnableParams() \
            + self.top.countLearnableParams()

    def showParams(self):
        """prints shape of all parameters and is_leaf"""
        for x in self.parameters():
            if type(x['params']) == list:
                for tens in x['params']:
                    print(tens.shape,tens.is_leaf)
            else:
                print(x['params'].shape,x['params'].is_leaf)

    


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

    def __str__(self):
        tempL = " L" if self.learnable else "NL"
        tempI = "K" if self.initialization == "Kymatio" else "R"
        return f"{tempI} {tempL}"

    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        filter_viz = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(self.psi, self.J, 8, mode=mode) 
            filter_viz[mode] = f  

        return filter_viz

    def __init__(self,J,N,M,second_order, initialization,seed,device,learnable=True,lr_orientation=0.1,lr_scattering=0.1,use_cuda=True):
        """Creates scattering filters and adds them to the nn.parameters if learnable"""
        super(sn_ScatteringBase,self).__init__()
        self.J = J
        self.N = N
        self.M = M
        self.second_order = second_order
        self.learnable = learnable
        self.use_cuda = use_cuda 
        self.device = device
        self.initialization = initialization
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.M_coefficient = self.M/(2**self.J)
        self.N_coefficient = self.N/(2**self.J)

        self.scattering, self.psi, self.wavelets, self.params_filters, self.n_coefficients, self.grid = create_scatteringExclusive(
            J,N,M,second_order, initilization=self.initialization,seed=seed,
            requires_grad=learnable,use_cuda=self.use_cuda,device=self.device
        )

        #self.filterTracker = {'orientation1':[],'orientation2':[],'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
        self.filterTracker = {'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
       
        self.filterGradTracker = {'angle': [],'1':[],'2':[],'3':[]}

        self.filters_plots_before = self.getFilterViz()
        self.scatteringTrain = False

    def saveFilterValues(self,scatteringActive):
        #print('orientation1',self.params_filters[0][:,0].detach().shape)
        #print('not ori',self.params_filters[1].detach().shape)
        # print(self.params_filters[3].detach())
        try:
            if scatteringActive:
                #orientations1 = self.params_filters[0][:,0].detach().clone()
                #orientations2 = self.params_filters[0][:,1].detach().clone()
                orientations1 = self.params_filters[0].detach().clone()
                #self.filterTracker['orientation1'].append(orientations1)
                #self.filterTracker['orientation2'].append(orientations2)
                self.filterTracker['1'].append(self.params_filters[1].detach().clone())
                self.filterTracker['2'].append(self.params_filters[2].detach().clone()) 
                self.filterTracker['3'].append(self.params_filters[3].detach().clone()) 
                scale = torch.mul(self.params_filters[1].detach().clone(), self.params_filters[2].detach().clone())
                self.filterTracker['scale'].append(scale) 
                #angle = torch.atan2(orientations1, orientations2)
                self.filterTracker['angle'].append(orientations1) 

            else:
                #self.filterGradTracker['orientation1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                #self.filterGradTracker['orientation2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass


    def saveFilterGrads(self,scatteringActive):
        #print('orientation1grad',self.params_filters[0].grad.shape)
        try:
            if scatteringActive:
                #self.filterGradTracker['orientation1'].append(self.params_filters[0].grad[:,0].clone()) 
                self.filterGradTracker['angle'].append(self.params_filters[0].grad.clone()) 
                #self.filterGradTracker['orientation2'].append(self.params_filters[0].grad[:,1].clone()) 
                self.filterGradTracker['1'].append(self.params_filters[1].grad.clone()) 
                self.filterGradTracker['2'].append(self.params_filters[2].grad.clone()) 
                self.filterGradTracker['3'].append(self.params_filters[3].grad.clone()) 
            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                #self.filterGradTracker['orientation1'].append(torch.zeros(self.params_filters[1].shape[0]))
                #self.filterGradTracker['orientation2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass


    def plotFilterGrad(self):
        """ plots the graph of the filter gradients """


        f = plt.figure (figsize=(7,7))
        temp = {
            'orientation1': [float(filters[0].cpu().numpy()) for filters in self.filterGradTracker['angle']],
            #'orientation2': [float(filters[0].cpu().numpy())  for filters in self.filterGradTracker['orientation2']],
            'xis': [float(filters[0].cpu().numpy())  for filters in self.filterGradTracker['1']],
            'sigmas': [float(filters[0].cpu().numpy())  for filters in self.filterGradTracker['2']],
            'slant': [float(filters[0].cpu().numpy())  for filters in self.filterGradTracker['3']]
        }



        plt.plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
        #plt.plot([x for x in range(len(temp['orientation2']))],temp['orientation2'],color='blue', label='theta2')
        plt.plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
        plt.plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
        plt.plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
        plt.legend()

        return f


    def plotFilterGrads(self):
        """ plots the graph of the filter gradients """
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterGradTracker['angle']],
                #'orientation2': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['orientation2']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['3']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
            #axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation2']))],temp['orientation2'],color='blue', label='orientation2')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')

            axarr[int(x/col),x%col].legend()

        return f

    def plotFilterValue(self):
        """ plots the graph of the filter 0 value  """

        f = plt.figure (figsize=(7,7))
        temp = {
            'orientation1': [float(filters[0].cpu().numpy()) for filters in self.filterTracker['angle']],
            #'orientation2': [float(filters[0].cpu().numpy())  for filters in self.filterTracker['orientation2']],
            'xis': [float(filters[0].cpu().numpy())  for filters in self.filterTracker['1']],
            'sigmas': [float(filters[0].cpu().numpy())  for filters in self.filterTracker['2']],
            'slant': [float(filters[0].cpu().numpy())  for filters in self.filterTracker['3']],
            'scale': [float(filters[0].cpu().numpy())  for filters in self.filterTracker['scale']],
            #'angle': [float(filters[0].cpu().numpy())  for filters in self.filterTracker['angle']]
        }

        plt.plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
        #plt.plot([x for x in range(len(temp['orientation2']))],temp['orientation2'],color='blue', label='theta2')
        plt.plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
        plt.plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
        plt.plot([x for x in range(len(temp['slant']))], temp['slant'], color='orange', label='slant')
        #plt.plot([x for x in range(len(temp['angle']))],temp['angle'],color='pink', label='theta')
        plt.plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
        
        plt.legend()

        return f
    
    def plotFilterValues(self):
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            #axarr[int(x/col),x%col].axis('off')
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterTracker['angle']],
                #'orientation2': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['orientation2']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['3']],
                'scale': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['scale']],
                'angle': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['angle']]
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
            #axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation2']))],temp['orientation2'],color='blue', label='orientation2')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
            #axarr[int(x/col),x%col].plot([x for x in range(len(temp['angle']))],temp['angle'],color='pink', label='theta')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
            axarr[int(x/col),x%col].legend()

        return f
        

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
            yield {'params': [self.params_filters[0]], 'lr': self.lr_orientation, 
                              'maxi_lr':self.lr_orientation , 'weight_decay': 0}
            yield {'params': [ self.params_filters[1],self.params_filters[2],
                               self.params_filters[3]],'lr': self.lr_scattering,
                               'maxi_lr':self.lr_scattering , 'weight_decay': 0}

    def updateFilters(self):
        """if were using learnable scattering, update the filters to reflect the new parameter values obtained from gradient descent"""
        if self.learnable:
            self.wavelets = morlets(self.grid, self.params_filters[0], self.params_filters[1], 
                                    self.params_filters[2], self.params_filters[3], device=self.device)
            self.psi = update_psi(self.scattering.J, self.psi, self.wavelets, self.initialization, self.device) 
        else:
            pass

    def forward(self, ip):
        """ apply the scattering transform to the input image """
        if self.scatteringTrain: #update filters if training
            self.updateFilters()
            
        x = construct_scattering(ip, self.scattering, self.psi)
        x = x[:,:, -self.n_coefficients:,:,:]
        x = x.reshape(x.size(0), self.n_coefficients*3, x.size(3), x.size(4))
        return x

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        if not self.learnable:
            return 0

        count = 0
        for t in self.parameters():
            if type(t["params"]) == list:
                for tens in t["params"]: 
                    count += tens.numel()
            else:
                count += t["params"].numel()

        print("Scattering learnable parameters: {}".format(count))
        return count



class sn_MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, standard=False, use_cuda=True):
        super(sn_MLP,self).__init__()
        selfnum_classes =num_classes
        if use_cuda:
            self.cuda()

        if standard:
            fc1 = nn.Linear(3*32*32, 512)
        else:
            fc1=  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients),  512)

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.n_coefficients*3,eps=1e-5,affine=True),
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

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        count = 0
        for t in self.parameters():
            count += t.numel()
        return count



class sn_LinearLayer(nn.Module):
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8, standard=False, average = False, use_cuda=True):
        super(sn_LinearLayer,self).__init__()
        self.n_coefficients = n_coefficients
        self.num_classes = num_classes
        self.average= average
        if use_cuda:
            self.cuda()

        if standard:
            self.fc1 = nn.Linear(3*32*32,num_classes)
            #self.fc2 = nn.Linear(256, num_classes)
        else:
            #self.fc1 = nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients), num_classes)
            self.fc1 = nn.Linear(int(3*n_coefficients), num_classes)
            # self.fc1 =  nn.Linear(int(3*M_coefficient*  N_coefficient*n_coefficients), 1024)
            # self.fc2 = nn.Linear(1024, num_classes)

        self.bn0 = nn.BatchNorm2d(self.n_coefficients*3,eps=1e-5,affine=True)


    def forward(self, x):
        # x = x[:,:, -self.n_coefficients:,:,:]
        x = self.bn0(x)
        if self.average:
            x = x.mean(dim=(2,3))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""

        count = 0
        for t in self.parameters():
            count += t.numel()
        return count


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
    def __init__(self, in_channels, k=8, n=4, num_classes=10, standard=False):
        super(sn_CNN, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_channels*3,eps=1e-5,affine=True)

        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.in_channels = in_channels
        self.num_classes =num_classes
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
            pass
            # x = x[:,:, -self.in_channels:,:,:]\
            # print(x.shape)
            # x = x.reshape(x.size(0), self.K, x.size(3), x.size(4))

        print("CNN shape:",x.shape)
        x = self.bn0(x)
        x = self.init_conv(x)

        if self.standard:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        count = 0
        for t in self.parameters():
            count += t.numel()
        return count
 
