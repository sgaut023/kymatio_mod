from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from sklearn.decomposition import PCA
from PIL import Image
import os
import torch
import time

class ChestXrayDataset(Dataset):
  def __init__(self, dataframe: pd.DataFrame, root_dir, 
                label_coln: str, path_coln: str, transforms):
        self.dataframe = dataframe
        self.label_coln = label_coln
        self.path_coln = path_coln
        self.transforms = transforms
        self.root_dir = root_dir
        self.classes = ['positive', 'negative']

  def __len__(self):
        return self.dataframe.shape[0]
  def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img = Image.open(os.path.join(self.root_dir, row[self.path_coln]))

        if self.transforms:
          img = self.transforms(img)
        else:
          img = torchvision.transforms.functional.to_tensor(img)
     
        return (img, row[self.label_coln])

    
