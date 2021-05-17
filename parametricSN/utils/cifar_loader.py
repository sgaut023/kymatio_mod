"""Wrapper for the cifar dataset with various options 

Author: Benjamin Therien

Exceptions: 
    ImpossibleSampleNumException --
    IncompatibleBatchSizeException -- 
    IncompatibleClassNumberException --
    IndicesNotSetupException --

Functions:
    cifar_getDataloaders -- 
    cifar_augmentationFactory -- 

Classes: 
    SmallSampleController -- class used to sample a small portion from an existing dataset
"""



import torch
import gc
import time

import numpy as np

from parametricSN.utils.auto_augment import AutoAugment, Cutout
from torchvision import datasets, transforms
from numpy.random import RandomState
from torch.utils.data import Subset


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
        print("\n[get_dataset(params, use_cuda)] Augmenting data with AutoAugment augmentation")
        trainTransform = [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        print("\n[get_dataset(params, use_cuda)] Augmenting data with original-cifar augmentation")
        trainTransform = [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        print("\n[get_dataset(params, use_cuda)] No data augmentation")
        trainTransform = []

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {AUGMENT} not implemented")
    else: 
        NotImplemented(f"augment parameter {AUGMENT} not implemented")
    pass

def cifar_getDataloaders(trainSampleNum, valSampleNum, 
                         trainBatchSize, valBatchSize, 
                         multiplier, trainDataset, 
                         valDataset,seed=None,
                         dataDir=".",use_cuda=True):
    """Factory for different augmentation choices"""

    if params['model']['data_root'] != None:
        DATA_DIR = Path(params['model']['data_root'])/params['model']['data_folder'] #scattering_datasets.get_dataset_dir('CIFAR')
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
        
    #CIFAR-10 normalization    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                  std=[0.247, 0.243, 0.261])

    #default normalization
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')

    
    transform_train = transforms.Compose(trainTransform + [transforms.ToTensor(), normalize]) 
    transform_val = transforms.Compose([transforms.ToTensor(), normalize]) #careful to keep this one same

    dataset_train = datasets.CIFAR10(#load train dataset
        root=DATA_DIR, train=True, 
        transform=transform_train, download=True
    )

    dataset_val = datasets.CIFAR10(#load test dataset
        root=DATA_DIR, train=False, 
        transform=transform_val, download=True
    )

    ss = SmallSampleController(
        trainSampleNum=TRAIN_SAMPLE_NUM, valSampleNum=VAL_SAMPLE_NUM, 
        trainBatchSize=TRAIN_BATCH_SIZE, valBatchSize=VAL_BATCH_SIZE, 
        multiplier=VALIDATION_SET_NUM, trainDataset=dataset_train, 
        valDataset=dataset_val 
    )  

    train_loader_in_list, test_loader_in_list, seed = ss.generateNewSet(
    device,workers=num_workers,valMultiplier=VALIDATION_SET_NUM,seed=SEED) #Sample from datasets

    params['model']['seed'] = seed
    train_loader, test_loader =  
    return train_loader_in_list[0], test_loader_in_list[0]

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


        def sample(self,dataset,offset,workers,RP,shuffle=True):
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
                                           batch_size=self.batchSize, shuffle=shuffle,
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
        return "[SmallSampleController] num classes:{}, train batch size:{}".format(self.numClasses,self.trainSampler.batchSize)


    def __init__(self, trainSampleNum, valSampleNum, trainBatchSize, valBatchSize, multiplier, trainDataset, valDataset):

        self.numClasses = len(trainDataset.classes)

        if self.numClasses != len(valDataset.classes):
            print("[SmallSampleController] Incompatble number of classes for validation and train datasets")
            raise IncompatibleClassNumberException

        
        self.trainSampler = SmallSampleController.Sampler(
            numClasses=self.numClasses,sampleNum=trainSampleNum,
            batchSize=trainBatchSize,multiplier=1
            )

        self.valSampler = SmallSampleController.Sampler(
            numClasses=self.numClasses,sampleNum=valSampleNum,
            batchSize=valBatchSize,multiplier=multiplier
            )

        self.trainDataset = SmallSampleController.DatasetContainer(trainDataset)
        self.valDataset = SmallSampleController.DatasetContainer(valDataset)




    def sample(self,workers=5,valMultiplier=None,seed=None):
        """Samples a new random permutaiton of class balanced 
        training and validation samples from the dataset
        
        keyword arguments:
        workers -- the number of cpu cores to allocate
        valMultiplier -- the number of validation sets to sample
        seed -- the seed to sample with
        """

        if valMultiplier != None:
            self.valSampler.multiplier = valMultiplier

        if seed == None:
            seed = int(time.time()) #generate random seed

        prng = RandomState(seed)
        RP = prng.permutation(np.arange(0, self.trainDataset.maxIndex))

        self.trainSampler.sample(
            dataset=self.trainDataset.dataset,offset=0,
            RP=RP,workers=workers,shuffle=True
            )
        

        if self.valDataset == self.trainDataset: #offset to prevent train-test overlap

            self.valSampler.sample(
                dataset=self.valDataset.dataset,
                offset=self.trainSampler.samplesPerClass,
                RP=RP,workers=workers,shuffle=False
            )
        else:

            prng = RandomState(seed)
            RP = prng.permutation(np.arange(0, self.valDataset.maxIndex))

            self.valSampler.sample(
                dataset=self.valDataset.dataset,offset=0,
                RP=RP,workers=workers,shuffle=False
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
        print("Generated new permutation of the dataset with seed:{}, train sample num: {}, test sample num: {}".format(
                seed,self.trainSampler.sampleNum,self.valSampler.sampleNum))

        return trainDL,valDL,seed











