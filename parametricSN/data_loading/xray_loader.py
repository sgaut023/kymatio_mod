"""Subsamples COVIDX-CRX2

Author: Benjamin Therien, Shanel Gauthier

Functions:
    xray_augmentationFactory -- returns different augmentations for COVIDX-CRX2
    xray_getDataloaders -- returns different augmentations for COVIDX-CRX2


Classes: 
    SmallSampleController -- class used to sample a small portion from an existing dataset
"""

import os

from torchvision import datasets, transforms

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from parametricSN.data_loading.SmallSampleController import SmallSampleController


def xray_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation tranforms for the COVIDx CRX-2 dataset mnj"""
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
                        valBatchSize, trainAugmentation, height, 
                        width, dataDir="."):
    """Creates a SmallSampleController object from the COVIDx CRX-2 dataset
    
    returns:
        ssc
    """
    
    transform_train = xray_augmentationFactory(trainAugmentation, height, width)
    transform_val = xray_augmentationFactory("noaugment", height, width)

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
