"""Contains classes and functions for loading and sampling from the kth texture dataset.

Author: Benjamin Therien, Shanel Gauthier

Functions: 
    kth_augmentationFactory -- factory of kth augmentations 
    kth_getDataloaders -- returns dataloaders for kth

class:
    KTHLoder -- loads and tracks parameters from the kth dataset
"""

import torch
import time
import os

from parametricSN.utils.auto_augment import AutoAugment, Cutout
from torchvision import datasets, transforms

def kth_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation choices"""

    if augmentation == 'autoaugment':
        print("\n[get_dataset(params, use_cuda)] Augmenting data with AutoAugment augmentation")
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        print("\n[get_dataset(params, use_cuda)] Augmenting data with original-cifar augmentation")
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        print("\n[get_dataset(params, use_cuda)] No data augmentation")
        transform = [
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
                       height, width, sample, seed=None, dataDir=".", 
                       num_workers=4, use_cuda=True):
    """Samples a specified class balanced number of samples form the kth dataset"""
    transform_train = kth_augmentationFactory(trainAugmentation, height, width)
    transform_val = kth_augmentationFactory('noaugment', height, width)

    loader = KTHLoader(data_dir=dataDir, train_batch_size=trainBatchSize, 
                       val_batch_size=valBatchSize, transform_train=transform_train, 
                       transform_val=transform_val, 
                       num_workers=num_workers, seed=seed, 
                       sample=sample)

    train_loader, test_loader = loader.get_dataloaders()

    if use_cuda:
        for batch,target in train_loader:
            batch.cuda()
            target.cuda()

        for batch,target in test_loader:
            batch.cuda()
            target.cuda()
    
    return train_loader, test_loader, loader.seed

class KTHLoader():
    """Class for loading the KTH texture dataset"""
    def __init__(self, data_dir, train_batch_size, 
                 val_batch_size, transform_train, transform_val, 
                 num_workers, seed=None, sample='a'):

        self.data_dir = data_dir
        if seed == None:
            self.seed = int(time.time()) #generate random seed
        else:
            self.seed = seed

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.transform_train =  transform_train
        self.transform_val = transform_val
        self.num_workers =num_workers
        self.sample = sample

    def get_dataloaders(self):
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
                                                   shuffle=True, num_workers=self.num_workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.val_batch_size, 
                                                  shuffle=True, num_workers=self.num_workers, 
                                                  pin_memory=True)

        return train_loader, test_loader