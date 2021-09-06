"""Contains all the base pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Exceptions:
    InvalidInitializationException --
    InvalidArchitectureError --

Functions: 
    create_scatteringExclusive -- creates scattering parameters
    baseModelFactory -- creates different models based on input

Classes: 
    sn_Identity -- computes the identity function in forward pass
    sn_HybridModel -- combinations of a scattering and other nn.modules
    sn_ScatteringBase -- a scattering network
"""

import torch 

import torch.nn as nn

from numpy.core.numeric import False_
from kymatio import Scattering2D
from .create_filters import *
from .wavelet_visualization import get_filters_visualization, getOneFilter
from .sn_models_exceptions import InvalidInitializationException
from scipy.optimize import linear_sum_assignment




def create_scatteringExclusive(J,N,M,second_order,device,initialization,pixelwise,seed=0,requires_grad=True,use_cuda=True):
    """Creates scattering parameters and replaces then with the specified initialization

    Creates the scattering network, adds it to the passed device, and returns it for modification. Next,
    based on input, we modify the filter initialization and use it to create new conv kernels. Then, we
    update the psi Kymatio object to match the Kymatio API.

    arguments:
    use_cuda -- True if were using gpu
    J -- scale of scattering (always 2 for now)
    N -- height of the input image
    M -- width of the input image
    initilization -- the type of init: ['Tight-Frame' or 'Random']
    seed -- the seed used for creating randomly initialized filters
    requires_grad -- boolean idicating whether we want to learn params
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

    if initialization == "Tight-Frame":
        params_filters = create_filters_params(J,L,requires_grad,pixelwise,device) #kymatio init
    elif initialization == "Random":
        #num_filters = get_total_num_filters(J,L)
        num_filters= J*L
        params_filters = create_filters_params_random(num_filters,requires_grad,pixelwise,device) #random init
    else:
        raise InvalidInitializationException

    # params_filters = [p.to(device) for p in params_filters]
    shape = (scattering.M_padded, scattering.N_padded,)
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), 0).to(device)
    params_filters =  [ param.to(device) for param in params_filters]

    wavelets  = morlets(shape, params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device )

    if pixelwise:
        wavelets.requires_grad = True

    psi = update_psi(J, psi, wavelets, device) #update psi to reflect the new conv filters

    # filters = filters_pixelwise(psi)

    return scattering, psi, wavelets, params_filters, n_coefficients, grid#, filters


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
    
    def checkFilterDistance(self):
        return 0
    
    def checkParamDistance(self):
        pass

    def checkDistance(self,compared):
        pass
    
        
class sn_ScatteringBase(nn.Module):
    """A learnable scattering nn.module 

    parameters:
        learnable -- should the filters be learnable parameters of this model
        use_cuda -- True if we are using cuda
        J -- scale of scattering (always 2 for now)
        N -- height of the input image
        M -- width of the input image
        initilization -- the type of init: ['Tight-Frame' or 'Random']
        seed -- the random seed used to initialize the parameters
    """

    def __str__(self):
        tempL = " L" if self.learnable else "NL"
        tempI = "TF" if self.initialization == 'Tight-Frame' else "R"
        return f"{tempI} {tempL}"

    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        filter_viz = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(self.psi, self.J, 8, mode=mode) 
            filter_viz[mode] = f  

        return filter_viz

    def getOneFilter(self, count, scale, mode):
        return getOneFilter(self.psi, count, scale, mode)

    def __init__(self, J, N, M, second_order, initialization, seed, 
                 device, num_channels, pixelwise, learnable=True, lr_orientation=0.1, 
                 lr_scattering=0.1, monitor_filters=True, use_cuda=True):
        """Constructor for the leanable scattering nn.Module
        
        Creates scattering filters and adds them to the nn.parameters if learnable
        
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
            second_order -- 
            initilization -- the type of init: ['Tight-Frame' or 'Random']
            seed -- the random seed used to initialize the parameters
            device -- the device to place weights on
            learnable -- should the filters be learnable parameters of this model
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation                 
            monitor_filters -- boolean indicating whether to track filter distances from initialization
            use_cuda -- True if using GPU

        """
        super(sn_ScatteringBase,self).__init__()
        self.J = J
        self.N = N
        self.M = M
        self.second_order = second_order
        self.learnable = learnable
        self.use_cuda = use_cuda 
        self.device = device
        self.num_channels = num_channels
        self.pixelwise = pixelwise
        self.initialization = initialization
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.M_coefficient = int(self.M/(2**self.J))
        self.N_coefficient = int(self.N/(2**self.J))
        self.monitor_filters = monitor_filters


        self.scattering, self.psi, self.wavelets, self.params_filters, self.n_coefficients, self.grid = create_scatteringExclusive(
            J,N,M,second_order, initialization=self.initialization,pixelwise=self.pixelwise,seed=seed,
            requires_grad=learnable,use_cuda=self.use_cuda,device=self.device
        )

        self.filterTracker = {'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
        self.filterGradTracker = {'angle': [],'1':[],'2':[],'3':[]}

        self.filters_plots_before = self.getFilterViz()
        self.scatteringTrain = False

        if self.monitor_filters == True:
            _, self.compared_psi, self.compared_wavelets, self.compared_params, _, _ = create_scatteringExclusive(
                J,N,M,second_order, initialization='Tight-Frame', pixelwise=pixelwise,seed=seed,
                requires_grad=False,use_cuda=self.use_cuda,device=self.device
            )

            self.compared_params_grouped = torch.cat([x.unsqueeze(1) for x in self.compared_params[1:]],dim=1)
            self.compared_params_angle = self.compared_params[0] % (2 * np.pi)
            self.compared_wavelets = self.compared_wavelets.reshape(self.compared_wavelets.size(0),-1)
            self.compared_wavelets_complete = torch.cat([self.compared_wavelets.real,self.compared_wavelets.imag],dim=1)


    def checkParamDistance(self):
        """Method to checking the minimal distance between initialized filters and learned ones
        
        Euclidean distances are calculated between each filter for parameters other than orientations
        for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
        these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
        algorithm to compute the optimal match (I.E. the one that minizes total distance)        
        """

        def getAngleDistance(one,two):
            """returns the angle of arc between two points on the unit circle"""
            if one < 0 or (2 * np.pi) < one or two < 0 or (2 * np.pi) < two:
                raise Exception

            if one == two:
                return 0
            elif one < two:
                diff = min(
                    two - one,
                    one + (2 * np.pi) - two
                )
            elif two < one:
                diff = min(
                    one - two,
                    two + (2 * np.pi) - one
                )
            return diff
            

        with torch.no_grad():

            tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.params_filters[1:]],dim=1)
            tempParamsAngle = self.params_filters[0] % (2 * np.pi)

            groupDistances = torch.cdist(tempParamsGrouped,self.compared_params_grouped)
            angleDistances = torch.zeros(groupDistances.shape, device=self.device)
            avoidZero = torch.zeros(groupDistances.shape, device=self.device) + 0.0000000001

            for i in range(angleDistances.size(0)):
                for j in range(angleDistances.size(1)):
                    angleDistances[i,j] = getAngleDistance(tempParamsAngle[i],self.compared_params_angle[j])

            distances = groupDistances + angleDistances + avoidZero

            distNumpy = distances.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(distNumpy, maximize=False)

            return distNumpy[row_ind, col_ind].sum()




    def checkDistance(self,compared="wavelets"):
        """Method to checking the minimal distance between filter params of one matrix and another"""
        with torch.no_grad():
            if compared == 'wavelets':
                numCompared = self.wavelets.size(0)
                tempWavelets = self.wavelets.reshape(self.wavelets.size(0),-1)
                distances = torch.cdist(tempWavelets.real,self.compared_wavelets.real)
            elif compared == 'params':
                numCompared = self.params_filters[0].size(0)
                tempParams = torch.cat([x.unsqueeze(1) for x in self.params_filters],dim=1)
                distances = torch.cdist(tempParams,self.compared_params)
            elif compared == 'wavelets_complete':
                numCompared = self.wavelets.size(0)
                tempWavelets = self.wavelets.reshape(self.wavelets.size(0),-1)
                tempWavelets = torch.cat([tempWavelets.real, tempWavelets.imag],dim=1)
                distances = torch.cdist(tempWavelets,self.compared_wavelets_complete)

            sortedDistances = sorted([(i,j,distances[i,j].item()) for i in range(distances.size(0)) for j in range(distances.size(1))],key=lambda x: x[2],reverse=False)
            usedi = [False for x in range(numCompared)]
            usedj = [False for x in range(numCompared)]
            totalDist = []
            for i,j,dist in sortedDistances:
                if usedi[i] == False and usedj[j] == False:
                    totalDist.append(dist)
                    usedi[i] = True
                    usedj[j] = True
                elif len(totalDist) == numCompared:
                    break
                else:
                    pass

            return sum(totalDist)
    


    def saveFilterValues(self,scatteringActive):
        try:
            if scatteringActive:
                orientations1 = self.params_filters[0].detach().clone()
                self.filterTracker['1'].append(self.params_filters[1].detach().clone())
                self.filterTracker['2'].append(self.params_filters[2].detach().clone()) 
                self.filterTracker['3'].append(self.params_filters[3].detach().clone()) 
                scale = torch.mul(self.params_filters[1].detach().clone(), self.params_filters[2].detach().clone())
                self.filterTracker['scale'].append(scale) 
                self.filterTracker['angle'].append(orientations1) 

            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass


    def saveFilterGrads(self,scatteringActive):
        try:
            if scatteringActive:
                self.filterGradTracker['angle'].append(self.params_filters[0].grad.clone()) 
                self.filterGradTracker['1'].append(self.params_filters[1].grad.clone()) 
                self.filterGradTracker['2'].append(self.params_filters[2].grad.clone()) 
                self.filterGradTracker['3'].append(self.params_filters[3].grad.clone()) 
            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass



    def plotFilterGrads(self):
        """plots the graph of the filter gradients"""
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterGradTracker['angle']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['3']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')

            
            axarr[int(x/col),x%col].legend()

        return f

    
    def plotFilterValues(self):
        """plots the graph of the filter values"""
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            #axarr[int(x/col),x%col].axis('off')
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterTracker['angle']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['3']],
                'scale': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['scale']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
            axarr[int(x/col),x%col].legend()

        return f
        

    def plotParameterValues(self):
        size = (10, 10)
        f, axarr = plt.subplots(2, 2, figsize=size) # create plots
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        label = ['theta','xis','sigma','slant']

        for idx,param in enumerate(['angle',"1",'2','3']):#iterate over all the parameters
            for idx2,filter in enumerate(torch.stack(self.filterTracker[param]).T):
                filter = filter.cpu().numpy()
                # if param == 'angle':
                #     filter = filter%(2*np.pi)
                axarr[int(idx/2),idx%2].plot([x for x in range(len(filter))],filter)#, label=idx2)
            # axarr[int(idx/2),idx%2].legend()
            axarr[int(idx/2),idx%2].set_title(label[idx], fontsize=16)
            axarr[int(idx/2),idx%2].set_xlabel('Epoch', fontsize=12) # Or ITERATION to be more precise
            axarr[int(idx/2),idx%2].set_ylabel('Value', fontsize=12)
            

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
        if self.learnable and not self.pixelwise:
            yield {'params': [self.params_filters[0]], 'lr': self.lr_orientation, 
                              'maxi_lr':self.lr_orientation , 'weight_decay': 0}
            yield {'params': [ self.params_filters[1],self.params_filters[2],
                               self.params_filters[3]],'lr': self.lr_scattering,
                               'maxi_lr':self.lr_scattering , 'weight_decay': 0}
        
        elif self.learnable and self.pixelwise:
            yield {'params': self.wavelets, 'lr': self.lr_scattering, 'weight_decay': 0}

    def updateFilters(self):
        """if were using learnable scattering, update the filters to reflect 
        the new parameter values obtained from gradient descent"""
        if self.learnable and not self.pixelwise:
            self.wavelets = morlets(self.grid, self.params_filters[0], 
                                    self.params_filters[1], self.params_filters[2], 
                                    self.params_filters[3], device=self.device)
                                    
            self.psi = update_psi(self.scattering.J, self.psi, self.wavelets, 
                                  self.device)
        elif self.learnable and self.pixelwise:
            self.psi = update_psi(self.scattering.J, self.psi, self.wavelets, 
                                  self.device)
        else:
            pass

    def forward(self, ip):
        """ apply the scattering transform to the input image """
        if self.scatteringTrain: #update filters if training and not pixelwise
            self.updateFilters()
            
        x = construct_scattering(ip, self.scattering, self.psi)
        x = x[:,:, -self.n_coefficients:,:,:]
        x = x.reshape(x.size(0), self.n_coefficients*self.num_channels, x.size(3), x.size(4))

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
