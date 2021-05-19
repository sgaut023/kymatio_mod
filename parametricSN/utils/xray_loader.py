"""Wrapper for the cifar dataset with various options 

Author: Benjamin Therien, Shanel Gauthier

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
import os

import numpy as np

from parametricSN.utils.auto_augment import AutoAugment, Cutout
from torchvision import datasets, transforms
from torch.utils.data import Subset
from numpy.random import RandomState
from parametricSN.utils.cifar_loader import SmallSampleController


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

def xray_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation choices"""

    if augmentation == 'autoaugment':
        print("\n[get_dataset(params, use_cuda)] Augmenting data with AutoAugment augmentation")
        transform = [
            transforms.Resize((224,224)),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        print("\n[get_dataset(params, use_cuda)] Augmenting data with original-cifar augmentation")
        transform = [
            transforms.Resize((224,224)),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        print("\n[get_dataset(params, use_cuda)] No data augmentation")
        transform = [
            transforms.Resize((224,224)),
            transforms.CenterCrop((height, width)),
        ]

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")
    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])

def xray_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, 
                         valBatchSize, multiplier, trainAugmentation,
                         height , width , seed=None,   dataDir=".", num_workers=4, 
                         use_cuda=True):
    """Samples a specified class balanced number of samples form the cifar dataset"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_train = xray_augmentationFactory(trainAugmentation,  height, width)
    transform_val = xray_augmentationFactory("noaugment",  height, width)

    dataset_train= datasets.ImageFolder(root=os.path.join(dataDir,'train'), #use train dataset
                                            transform=transform_train)
    dataset_val = datasets.ImageFolder(root=os.path.join(dataDir,'test'), #use train dataset
                                            transform=transform_val)

    ssc = SmallSampleController(
        trainSampleNum=trainSampleNum, valSampleNum=valSampleNum, 
        trainBatchSize=trainBatchSize, valBatchSize=valBatchSize, 
        multiplier=multiplier, trainDataset=dataset_train, 
        valDataset=dataset_val 
    )  

    train_loader_in_list, test_loader_in_list, seed = ssc.generateNewSet(#Sample from datasets
        device,workers=num_workers,
        valMultiplier=multiplier,
        seed=seed
    ) 

    return train_loader_in_list[0], test_loader_in_list[0], seed

