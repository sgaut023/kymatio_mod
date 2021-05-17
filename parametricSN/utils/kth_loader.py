import torch 
from torchvision import datasets
from pathlib import Path 
import time
import os

class KTHLoader():
    def __init__(self, data_dir, train_batch_size, val_batch_size, transform_train, transform_val, 
                num_workers, seed = None, sample = 'a'):
        
        self.data_dir =data_dir
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
                dataset = datasets.ImageFolder(root=os.path.join(self.data_dir,f'sample_{s}'), #use train dataset
                                            transform=self.transform_train)
                dataset_train = dataset
            else:
                dataset = datasets.ImageFolder(root=os.path.join(self.data_dir,f'sample_{s}'), #use train dataset
                                            transform=self.transform_val)
                datasets_val.append(dataset)
        
        dataset_val = torch.utils.data.ConcatDataset(datasets_val)
                
        train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=self.train_batch_size, shuffle=True,
                                           num_workers = self.num_workers, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(dataset_val,
                                           batch_size=self.val_batch_size, shuffle=True,
                                           num_workers = self.num_workers, pin_memory = True)
        return train_loader, test_loader
