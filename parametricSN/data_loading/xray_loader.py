"""Wrapper for the cifar dataset with various options 

Author: Benjamin Therien, Shanel Gauthier

Exceptions: 
    ImpossibleSampleNumException --
    IncompatibleBatchSizeException -- 
    IncompatibleClassNumberException --
    IndicesNotSetupException --

Functions:
    cifar_getDataloaders -- samples from the cifar-10 dataset based on input
    xray_augmentationFactory -- returns different augmentations for cifar-10

Classes: 
    SmallSampleController -- class used to sample a small portion from an existing dataset
"""

import torch
import os

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from parametricSN.data_loading.cifar_loader import SmallSampleController
from torchvision import datasets, transforms
from torch.utils.data import Subset



def xray_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation choices"""
    downsample = (260,260)

    if augmentation == 'autoaugment':
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        transform = [
            transforms.Resize(downsample),
            transforms.RandomCrop(size=(height, width)),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        transform = [
            transforms.Resize(downsample),
            transforms.CenterCrop((height, width)),
        ]

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")
    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #normalize = transforms.Normalize(mean=[0.5888, 0.5888, 0.5889],
                                     #std=[0.1882, 0.1882, 0.1882])

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])



def xray_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, 
                         valBatchSize, multiplier, trainAugmentation,
                         height , width , seed=None,   dataDir=".", num_workers=4, 
                         use_cuda=True, glico=False):
    """Samples a specified class balanced number of samples form the cifar dataset
    
    returns:
        ssc
    """

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
        trainDataset=dataset_train, valDataset=dataset_val
    )

    return ssc
