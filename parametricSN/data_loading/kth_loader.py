"""Contains classes and functions for loading and sampling the KTH-TIPS2 dataset

Author: Benjamin Therien, Shanel Gauthier

Functions: 
    kth_augmentationFactory -- factory of KTH-TIPS2 augmentations 
    kth_getDataloaders -- returns dataloaders for KTH-TIPS2

class:
    KTHLoader -- loads KTH-TIPS2 from disk and creates dataloaders
"""

import torch
import time
import os

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from torchvision import datasets, transforms

def kth_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation choices for KTH-TIPS2"""

    if augmentation == 'autoaugment':
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]

    elif augmentation == 'original-cifar':
        transform = [
            transforms.Resize((200,200)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
        ]

    elif augmentation == 'noaugment':
        transform = [
            transforms.Resize((200,200)),
            transforms.CenterCrop((height, width))
        ]

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")

    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])



def kth_getDataloaders(trainBatchSize, valBatchSize, trainAugmentation,
                       height, width, sample, dataDir="."):
    """Samples a specified class balanced number of samples form the KTH-TIPS2 dataset
    
    returns:
        loader
    """
    transform_train = kth_augmentationFactory(trainAugmentation, height, width)
    transform_val = kth_augmentationFactory('noaugment', height, width)

    loader = KTHLoader(data_dir=dataDir, train_batch_size=trainBatchSize, 
                       val_batch_size=valBatchSize, transform_train=transform_train, 
                       transform_val=transform_val, sample=sample)

    return loader

class KTHLoader():
    """Class for loading the KTH texture dataset"""
    def __init__(self, data_dir, train_batch_size, val_batch_size, 
                 transform_train, transform_val, sample='a'):

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.transform_train =  transform_train
        self.transform_val = transform_val
        self.sample = sample

    def get_dataloaders(self, device, workers=5, seed=None, load=False):
        """ TODO Shanel add Comment

        returns:
            train_loader, test_loader, seed
        """
        datasets_val = []
        for s in ['a', 'b', 'c', 'd']:
            if self.sample == s:
                dataset = datasets.ImageFolder(#load train dataset
                    root=os.path.join(self.data_dir,f'sample_{s}'), 
                    transform=self.transform_train
                )
                dataset_train = dataset
            else:
                dataset = datasets.ImageFolder(#load train dataset
                    root=os.path.join(self.data_dir,f'sample_{s}'), 
                    transform=self.transform_val
                )

                datasets_val.append(dataset)

        dataset_val = torch.utils.data.ConcatDataset(datasets_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.train_batch_size, 
                                                   shuffle=True, num_workers=workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.val_batch_size, 
                                                  shuffle=True, num_workers=workers, 
                                                  pin_memory=True)

        self.trainSampleCount, self.valSampleCount = sum([len(x) for x in train_loader]), sum([len(x) for x in test_loader])

        if load:
            for batch,target in train_loader:
                batch.cuda()
                target.cuda()

            for batch,target in test_loader:
                batch.cuda()
                target.cuda()    

        if seed == None:
            seed = int(time.time()) #generate random seed
        else:
            seed = seed

        return train_loader, test_loader, seed