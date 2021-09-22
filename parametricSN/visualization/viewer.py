import cv2
import matplotlib.pyplot as plt
import torch  #TODO REMOVE
import numpy as np  #TODO REMOVE

from parametricSN.models.create_filters import morlets, update_psi, create_filters_params
from .visualization_utils import get_filters_visualization, getOneFilter, getAllFilters, compareParams, compareParamsVisualization
#TODO move to utils
def toNumpy(x):
    return x.clone().cpu().numpy()

def getValue(x):
    return toNumpy(x.detach())

def getGrad(x):
    return toNumpy(x.grad)


class filterVisualizer(object):
    def __init__(self, scat):
        super(filterVisualizer, self).__init__()
        self.epoch = 0

        scat.pre_hook.remove()
        def updateFiltersVideo_hook(scattering, ip):
            """if were using learnable scattering, update the filters to reflect 
            the new parameter values obtained from gradient descent"""
            if (scattering.training or scattering.scatteringTrain):
                if scattering.learnable:
                    wavelets = morlets(scattering.grid, scattering.params_filters[0], 
                                        scattering.params_filters[1], scattering.params_filters[2], 
                                        scattering.params_filters[3])
                    phi, psi = scattering.load_filters()
                    scattering.psi = update_psi(scattering.J, psi, wavelets)
                    scattering.register_filters()

                if scattering.scatteringTrain:
                    self.saveFilterValues(True)

                # i think a more 'correct' way of doing this would be to place
                # this in the if statement above. Thoughts?
                self.writeVideoFrame()

                # scatteringTrain lags behind scattering.training
                # should we place this as a post forward hook?
                # we wouldnt need to define the rest of the code above. 
                scattering.scatteringTrain = scattering.training
        scat.pre_hook = scat.register_forward_pre_hook(updateFiltersVideo_hook)

        def print_hook(name):
            def updateFilterGrad(grad):
                self.filterGradTracker[name].append(toNumpy(grad))
            return updateFilterGrad
        scat.params_filters[0].register_hook(print_hook('angle'))
        scat.params_filters[1].register_hook(print_hook('1'))
        scat.params_filters[2].register_hook(print_hook('2'))
        scat.params_filters[3].register_hook(print_hook('3'))

        compared_params = create_filters_params(scat.J, scat.L, scat.learnable) #kymatio init

        #TODO turn into util function
        self.compared_params_grouped = torch.cat([x.unsqueeze(1) for x in compared_params[1:]], dim=1)
        self.compared_params_angle = compared_params[0] % (2 * np.pi)

        self.params_history = []

        self.scattering = scat

        #TODO turn into util function
        self.videoWriters = {}
        self.videoWriters['real'] = cv2.VideoWriter('videos/scatteringFilterProgressionReal{}epochs.avi'.format("--"),
                                          cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        self.videoWriters['imag'] = cv2.VideoWriter('videos/scatteringFilterProgressionImag{}epochs.avi'.format("--"),
                                          cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        self.videoWriters['fourier'] = cv2.VideoWriter('videos/scatteringFilterProgressionFourier{}epochs.avi'.format("--"),
                                             cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)

        # visualization code
        self.filterTracker = {'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
        self.filterGradTracker = {'angle': [],'1':[],'2':[],'3':[]}

        # take scattering as object
        self.filters_plots_before = self.getFilterViz()
                   

    def getOneFilter(self, count, scale, mode):
        phi, psi = self.scattering.load_filters()
        return getOneFilter(psi, count, scale, mode)

    def getAllFilters(self, totalCount, scale, mode):
        phi, psi = self.scattering.load_filters()
        return getAllFilters(psi, totalCount, scale, mode)

    def writeVideoFrame(self):
        """Writes frames to the appropriate video writer objects"""
        for vizType in self.videoWriters.keys():

            #TODO turn into util function
            temp = cv2.applyColorMap(np.array(self.getAllFilters(totalCount=16, scale=0, mode=vizType),dtype=np.uint8),cv2.COLORMAP_TURBO)
            temp = cv2.putText(temp, "Epoch {}".format(self.epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.videoWriters[vizType].write(temp)

    def releaseVideoWriters(self):
        for vizType in self.videoWriters.keys():
            self.videoWriters[vizType].release()

    def setEpoch(self, epoch):
        self.epoch = epoch

    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        phi, psi = self.scattering.load_filters()
        filter_viz = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(psi, self.scattering.J, 8, mode=mode) 
            filter_viz[mode] = f  
        return filter_viz

    def checkParamDistance(self):
        """Method to checking the minimal distance between initialized filters and learned ones
        
        Euclidean distances are calculated between each filter for parameters other than orientations
        for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
        these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
        algorithm to compute the optimal match (I.E. the one that minizes total distance)        

        return: 
            minimal distance
        """
        #TODO turn into util function
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.scattering.params_filters[1:]], dim=1).cpu()
        tempParamsAngle = (self.scattering.params_filters[0] % (2 * np.pi)).cpu()
        self.params_history.append({'params':tempParamsGrouped, 'angle':tempParamsAngle})
        return compareParams(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle
        )

    def compareParamsVisualization(self):
        """visualize the matched filters"""
        #TODO turn into util function
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.scattering.params_filters[1:]], dim=1).cpu()
        tempParamsAngle = (self.scattering.params_filters[0] % (2 * np.pi)).cpu()
        self.params_history.append({'params':tempParamsGrouped, 'angle':tempParamsAngle})
        return compareParamsVisualization(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle
        )

    def saveFilterValues(self, scatteringActive):
        #TODO turn into util function
        self.filterTracker['angle'].append(getValue(self.scattering.params_filters[0]))
        self.filterTracker['1'].append(getValue(self.scattering.params_filters[1]))
        self.filterTracker['2'].append(getValue(self.scattering.params_filters[2]))
        self.filterTracker['3'].append(getValue(self.scattering.params_filters[3]))
        self.filterTracker['scale'].append(np.multiply(self.filterTracker['1'][-1], self.filterTracker['2'][-1]))

    def saveFilterGrads(self,scatteringActive):
        print("shooooooooo")
        self.filterGradTracker['angle'].append(getGrad(self.scattering.params_filters[0]))
        self.filterGradTracker['1'].append(getGrad(self.scattering.params_filters[1]))
        self.filterGradTracker['2'].append(getGrad(self.scattering.params_filters[2]))
        self.filterGradTracker['3'].append(getGrad(self.scattering.params_filters[3]))

    def get_param_grad_per_epoch(self, x):
        return {
                    'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterGradTracker['angle']],
                    'xis': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['1']],
                    'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['2']],
                    'slant': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['3']],
                }

    def get_param_per_epoch(self, x):
        return {
                    'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterTracker['angle']],
                    'xis': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['1']],
                    'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['2']],
                    'slant': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['3']],
                    'scale': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['scale']],
                }

    def plotFilterGrads(self):
        paramsNum =self.params_filters[0].shape[0]
        if self.equivariant:
            col =  paramsNum
            row = 1
            size = (80, 10)
            f, axarr = plt.subplots(row, col, figsize=size) # create plots
            for x in range(paramsNum):
                temp= self.get_param_grad_per_epoch(x)
                axarr[x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
                axarr[x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[x%col].legend()
        
        else:
            col = 8
            row = int(paramsNum/col)
            size = (80, 10*row,)

            f, axarr = plt.subplots(row, col, figsize=size) # create plots

            for x in range(paramsNum):#iterate over all the filters
                temp= self.get_param_grad_per_epoch(x)
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[int(x/col),x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[int(x/col),x%col].legend()
        return f
    
    def plotFilterValues(self):
        """plots the graph of the filter values"""
        paramsNum = self.params_filters[0].shape[0]
        if self.equivariant:
            col = paramsNum
            row = 1
            size = (80, 10)
            f, axarr = plt.subplots(row, col, figsize=size) # create plots
            for x in range(paramsNum):
                temp = self.get_param_per_epoch(x)
                axarr[x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
                axarr[x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
                axarr[x%col].legend()

        else:
            col = 8
            row = int(self.filterNum/col)
            size = (80, 10*row,)
            f, axarr = plt.subplots(row, col, figsize=size) # create plots

            for x in range(paramsNum):
                temp = self.get_param_per_epoch(x)
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
        label = ['theta', 'xis', 'sigma', 'slant']
        for idx,param in enumerate(['angle', "1", '2', '3']):#iterate over all the parameters
            for idx2, filter in enumerate(np.stack(self.filterTracker[param]).T):
                axarr[int(idx/2), idx%2].plot([x for x in range(len(filter))], filter)
            axarr[int(idx/2), idx%2].set_title(label[idx], fontsize=16)
            axarr[int(idx/2), idx%2].set_xlabel('Epoch', fontsize=12) # Or ITERATION to be more precise
            axarr[int(idx/2), idx%2].set_ylabel('Value', fontsize=12)
        return f
