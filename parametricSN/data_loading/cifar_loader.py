"""Wrapper for subsampling the cifar-10 based on given input parameters

Author: Benjamin Therien

Exceptions: 
    ImpossibleSampleNumException --
    IncompatibleBatchSizeException -- 
    IncompatibleClassNumberException --
    IndicesNotSetupException --

Functions:
    cifar_getDataloaders -- samples from the cifar-10 dataset based on input
    cifar_augmentationFactory -- returns different augmentations for cifar-10

Classes: 
    SmallSampleController -- class used to sample a small portion from an existing dataset
"""

import torch
import gc
import time

import numpy as np

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from torchvision import datasets, transforms
from torch.utils.data import Subset
from numpy.random import RandomState



class ImpossibleSampleNumException(Exception):
    """Error thrown when an impossible class balancedsample number is requested"""
    pass

class IncompatibleBatchSizeException(Exception):
    """Error thrown when an impossible class balancedsample number is requested"""
    pass

class IncompatibleClassNumberException(Exception):
    """Error thrown when train and validation datasets dont have a compatible number of classes"""
    pass

class IndicesNotSetupException(Exception):
    """Error thrown when an impossible class balancedsample number is requested"""
    pass

def cifar_augmentationFactory(augmentation):
    """Factory for different augmentation choices"""

    if augmentation == 'autoaugment':
        # print("\n[get_dataset(params, use_cuda)] Augmenting data with AutoAugment augmentation")
        transform = [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        # print("\n[get_dataset(params, use_cuda)] Augmenting data with original-cifar augmentation")
        transform = [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        # print("\n[get_dataset(params, use_cuda)] No data augmentation")
        transform = []

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")
    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    # more precise cifar normalization thanks to:
    # https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/dataloader.py#L16

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])

def cifar_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, 
                         valBatchSize, multiplier, trainAugmentation,
                         seed=None, dataDir=".", num_workers=4, 
                         use_cuda=True,glico=False):
    """Samples a specified class balanced number of samples form the cifar dataset
    
    returns:
        train_loader, test_loader, seed, glico_dataset
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_train = cifar_augmentationFactory(trainAugmentation)
    transform_val = cifar_augmentationFactory("noaugment")

    dataset_train = datasets.CIFAR10(#load train dataset
        root=dataDir, train=True, 
        transform=transform_train, download=True
    )

    dataset_val = datasets.CIFAR10(#load test dataset
        root=dataDir, train=False, 
        transform=transform_val, download=True
    )

    ssc = SmallSampleController(
        trainSampleNum=trainSampleNum, valSampleNum=valSampleNum, 
        trainBatchSize=trainBatchSize, valBatchSize=valBatchSize,  
        trainDataset=dataset_train, valDataset=dataset_val 
    ) 

    return ssc

    train_loader_in_list, test_loader_in_list, seed = ssc.generateNewSet(#Sample from datasets
        device,workers=num_workers,
        valMultiplier=multiplier,
        seed=seed
    ) 

    if glico:
        glico_dataset = datasets.CIFAR10(#load train dataset again
            root=dataDir, train=True, download=True
        )
        glico_train = Subset(glico_dataset, ssc.trainSampler.indexes[0])
    else:
        glico_train = None

    return train_loader_in_list[0], test_loader_in_list[0], seed, glico_train

class SmallSampleController:
    """Interface for subsampling dataset of different classes.

    Class that holds one instance of a dataset, one test set generator
    and one train set generator. The test and train sets are managed by the controller
    who ensures they do not overlap. This class always provides class balanced samples
    """
    class Sampler:
        """
        Class responsible for sampling operations 
        """
        def __str__(self):
            return "Class Sampler"

        def __init__(self,numClasses,sampleNum,batchSize):
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
            self.dataLoaders = None
            self.indexes = None


        def getDataloader(self):
            """returns the first dataloder in the list"""
            return self.dataLoaders[0]



        def sample(self, dataset, offset, workers, RP, shuffle=True):
            """Creates a list of dataloaders based on input
            
            parameters:
                dataset -- The torch dataset object to sample from
                offset -- The offset position for used samples
                workers -- number of cores to use
                RP -- the index of random permutation (allows for seed based subsampling),
                shuffle -- boolean for shuffling the datasets
            """


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
            end = int(offset + self.samplesPerClass)
            indx = np.concatenate([np.where(np.array(dataset.targets) == class_)[0][RP[offset:end]]\
                    for class_ in range(0, self.numClasses)])

            subset = Subset(dataset, indx)
            self.indexes.append(indx)
            tempLoader = torch.utils.data.DataLoader(subset,
                                        batch_size=self.batchSize, shuffle=shuffle,
                                        num_workers = workers, pin_memory = True)
                
            self.dataLoaders.append(tempLoader)
            return len(indx)




        def load(self,device):
            if self.dataLoaders == None:
                print(self,"Please sample the dataset before trying to load it to a device")      
            else: 
                for ds in self.dataLoaders:
                    for b,t in ds:
                        b.to(device)
                        t.to(device)


    class DatasetContainer:
        """Designed to hold the information relevant to sampling from a pytorch dataset

        Parameters: 
        dataset -- the torch dataset object to be sampled from
        numClasses -- the number of classes in this dataset
        maxIndex -- the number of samples of the class with 
            the smallest number of samples in the dataset
        """

        def __init__(self,dataset):
            self.dataset = dataset
            self.numClasses = len(self.dataset.classes)
            self.maxIndex = min([len(np.where(np.array(self.dataset.targets) == classe)[0]) \
                for classe in range(0, self.numClasses)]) #finds the number of samples for the class with the least number of samples

        def __eq__(self,other):
            try:
                is_eq = self.dataset.data.shape == other.dataset.data.shape
            except:
                is_eq  = len(self.dataset) == len(other.dataset) 
            return is_eq

            
    def __str__(self):
        return "[SmallSampleController]"


    def __init__(self, trainSampleNum, valSampleNum, trainBatchSize, valBatchSize, trainDataset, valDataset):

        self.numClasses = len(trainDataset.classes)

        if self.numClasses != len(valDataset.classes):
            print("{} Incompatible number of classes for validation and train datasets".format(self))
            raise IncompatibleClassNumberException

        
        self.trainSampler = SmallSampleController.Sampler(
            numClasses=self.numClasses, sampleNum=trainSampleNum,
            batchSize=trainBatchSize
            )

        self.valSampler = SmallSampleController.Sampler(
            numClasses=self.numClasses, sampleNum=valSampleNum,
            batchSize=valBatchSize
            )

        self.trainDataset = SmallSampleController.DatasetContainer(trainDataset)
        self.valDataset = SmallSampleController.DatasetContainer(valDataset)




    def sample(self,workers=5,seed=None):
        """Samples a new random permutaiton of class balanced 
        training and validation samples from the dataset
        
        keyword arguments:
        workers -- the number of cpu cores to allocate
        seed -- the seed to sample with
        """


        if seed == None:
            seed = int(time.time()) #generate random seed

        prng = RandomState(seed)
        RP = prng.permutation(np.arange(0, self.trainDataset.maxIndex))

        trainSampleCount = self.trainSampler.sample(
            dataset=self.trainDataset.dataset,offset=0,
            RP=RP,workers=workers,shuffle=True
            )
        

        if self.valDataset == self.trainDataset: #offset to prevent train-test overlap

            valSampleCount = self.valSampler.samples(
                dataset=self.valDataset.dataset,
                offset=self.trainSampler.samplesPerClass,
                RP=RP,workers=workers,shuffle=False
            )
        else:

            prng = RandomState(seed)
            RP = prng.permutation(np.arange(0, self.valDataset.maxIndex))

            valSampleCount = self.valSampler.sample(
                dataset=self.valDataset.dataset,offset=0,
                RP=RP,workers=workers,shuffle=False
            )

        return trainSampleCount, valSampleCount, seed
        
        
    def load(self,device):
        """loads each sampler's data to the device"""

        self.trainSampler.load(device=device)
        self.valSampler.load(device=device)

    def getDataloaders(self):
        """returns dataloader list for each sampler"""

        return self.trainSampler.getDataloader(), self.valSampler.getDataloader()

    def generateNewSet(self, device, workers=5, seed=None, load=False):
        """Generates a new random permutation of train and test data and returns it

        Samples new train and test loaders from their corresponding datasets using
        the seed, workers, and val multiplier passed. Then, we move these datasets to the 
        passed device. Finally, they are returned along with the seed used to create them

        parameters: 
            device -- torch device we want to more the data to
            workers -- cpu cores allocated to this task
            seed -- the seed to sample with
            load -- boolean indicating whether to load dataset onto device
        """
        self.trainSampleCount, self.valSampleCount, seed = self.sample(workers=workers,seed=seed)

        if load:
            self.load(device)

        trainDL, valDL = self.getDataloaders()
        print("{} sumbsampled dataset with seed: {}, train sample num: {}, test sample num: {}".format(
                self, seed, self.trainSampleCount, self.valSampleCount))

        return trainDL, valDL, seed











