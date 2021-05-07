import torch
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from numpy.random import RandomState
from torch.utils.data import Subset

import gc
import time

class ImpossibleSampleNumException(Exception):
    """Error thrown when an impossible class balancedsample number is requested"""
    pass

class IncompatibleBatchSizeException(Exception):
    """Error thrown when an impossible class balancedsample number is requested"""
    pass

class IndicesNotSetupException(Exception):
    """Error thrown when an impossible class balancedsample number is requested"""
    pass

class SmallSampleController:
    """
    Class that holds one instance of the cifar 10 dataset, one test set generator
    and one train set generator. The test and train sets are managed by the controller
    who ensures they do not overlap. This class always provides class balanced samples
    """
    class Sampler:
        """
        Class responsible for sampling operations 
        """
        def __str__(self):
            return "Class Sampler"

        def __init__(self,numClasses,sampleNum,batchSize,multiplier):
            if sampleNum < numClasses:
                print(self,"Impossible train sample number passed.")
                raise ImpossibleSampleNumException

            if sampleNum % numClasses != 0:
                print(self,"Impossible train sample number passed.")
                raise ImpossibleSampleNumException

            self.numClasses = numClasses
            self.sampleNum = sampleNum
            self.batchSize = batchSize
            self.classBatchCount = int((self.sampleNum/self.numClasses)/self.batchSize)
            self.samplesPerClass = int(sampleNum/numClasses)
            self.multiplier = multiplier
            self.dataLoaders = None
            self.indexes = None


        def sample(self,dataset,offset,workers,RP):
            """Creates a list of dataloaders based on input"""
            
            if self.dataLoaders != None:
                del self.dataLoaders
                torch.cuda.empty_cache()
                gc.collect()

            if self.indexes != None:
                del self.indexes
                torch.cuda.empty_cache()
                gc.collect()
            
            self.indexes = []
            self.dataLoaders = []
            for x in range(self.multiplier):
                end = int(offset + self.samplesPerClass)
                indx = np.concatenate([np.where(np.array(dataset.targets) == class_)[0][RP[offset:end]]\
                     for class_ in range(0, self.numClasses)])

                subset = Subset(dataset, indx)
                self.indexes.append(indx)
                tempLoader = torch.utils.data.DataLoader(subset,
                                           batch_size=self.batchSize, shuffle=False,
                                           num_workers = workers, pin_memory = True)
                   
                self.dataLoaders.append(tempLoader)
                offset = end


        def load(self,device):
            if self.dataLoaders == None:
                print(self,"Please sample the dataset before trying to load it to a device")      
            else: 
                for ds in self.dataLoaders:
                    for b,t in ds:
                        b.to(device)
                        t.to(device)



            

            
    def __str__(self):
        return "[SmallSampleController] num classes:{}, batch size:{}".format(self.numClasses,batchSize)


    def __init__(self, numClasses, trainSampleNum, valSampleNum, batchSize, multiplier, trainDataset, valDataset, train=True):
        
        self.trainSampler = SmallSampleController.Sampler(
            numClasses=numClasses,sampleNum=trainSampleNum,
            batchSize=batchSize,multiplier=1
            )

        self.valSampler = SmallSampleController.Sampler(
            numClasses=numClasses,sampleNum=valSampleNum,
            batchSize=batchSize,multiplier=multiplier
            )

        self.trainDataset = trainDataset
        self.valDataset = valDataset
        self.numClasses = numClasses
        self.batchSize = batchSize
        self.train = train

        temp = [len(np.where(np.array(self.trainDataset.targets) == classe)[0]) for classe in range(0, self.numClasses)]
        self.maxIndex = min(temp)






    def sample(self,workers=5,valMultiplier=None,seed=None):
        """Samples a new random permutaiton of class balanced 
        training and validation samples from the dataset
        
        keyword arguments:
        workers -- the number of cpu cores to allocate
        valMultiplier -- the number of validation sets to sample
        seed -- the seed to sample with
        """

        if seed == None:
            seed = int(time.time())

        prng = RandomState(seed)
        RP = prng.permutation(np.arange(0, self.maxIndex))

        self.trainSampler.sample(
            dataset=self.trainDataset,offset=0,
            RP=RP,workers=workers
            )
        
        if valMultiplier != None:
            self.valSampler.multiplier = valMultiplier

        self.valSampler.sample(
            dataset=self.valDataset,offset=self.trainSampler.samplesPerClass,
            RP=RP,workers=workers
            )

        return seed
        
        


    def load(self,device):
        """loads each sampler's data to the device"""

        self.trainSampler.load(device=device)
        self.valSampler.load(device=device)

    def getDatasets(self):
        """returns dataloader list for each sampler"""

        return self.trainSampler.dataLoaders,self.valSampler.dataLoaders

    def generateNewSet(self,device,workers=5,valMultiplier=None,seed=None):
        """Generates a new random permutation of train and test data and returns it

        Samples new train and test loaders from their corresponding datasets using
        the seed, workers, and val multiplier passed. Then, we move these datasets to the 
        passed device. Finally, they are returned along with the seed used to create them

        arguments: 
        device -- torch device we want to more the data to
        workers -- cpu cores allocated to this task
        valMultiplier -- the number of validation sets to sample
        seed -- the seed to sample with
        """
        seed = self.sample(workers=workers,valMultiplier=valMultiplier,seed=seed)
        self.load(device)
        trainDL,valDL = self.getDatasets()
        if self.train == True:
            print("Generated new permutation of the CIFAR train dataset with seed:{}, train sample num: {}, test sample num: {}".format(
                    seed,self.trainSampler.sampleNum,self.valSampler.sampleNum))
        else:
            print("Generated new permutation of the CIFAR test dataset with seed:{}, train sample num: {}, test sample num: {}".format(
                    seed,self.trainSampler.sampleNum,self.valSampler.sampleNum))

        return trainDL,valDL,seed










