"""Script to download and generate xray dataset 
Dataset folder organization
├── train        
│   ├── positive     
│   ├── negative    
├── test        
│   ├── positive    
│   ├── negative     

Author: Shanel Gauthier

Functions:
    extract_zip           -- Extract files from zip
    create_train_folder   -- Create train set in the target folder
    create_test_folder    -- Create test set in the target folder

"""
from typing import Optional
import pandas as pd
import argparse
import shutil
import os
from tqdm import tqdm
import sys
import shutil
from zipfile import ZipFile


def extract_zip(dataset_path, target_path):
    """Extract files from zip
    Parameters:
        dataset_path -- url to dataset
        file_name -- file name of the dataset
    """
    dataset_path = os.path.join(dataset_path,'covidx-cxr2.zip')
    print(f'Extracting zip file: {dataset_path}')
    with ZipFile(file=dataset_path) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=os.path.join(target_path, 'xray'))
    os.remove(dataset_path)

def create_train_folder(df_train, target_path):
    """Create train set in the target folder
    Parameters:
        df_train    -- dataframe that contains all the train set details 
                       ('patient_id', 'filename', 'class', 'data_source')
        target_path -- path to the new dataset folder
    """
    folder_path = os.path.join(target_path, 'xray_preprocess/train')
    print(f'Create train set at: {folder_path}')
    for _, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        if row['class']=='negative':
            destination_path = os.path.join(folder_path, 'negative')
        elif row['class']=='positive':
            destination_path = os.path.join(folder_path, 'positive')
        if not os.path.exists(destination_path):
            os.makedirs(destination_path) 
        img = os.path.join(target_path, 'xray', 'train', row['filename'])
        shutil.copy(img, destination_path )

def create_test_folder(df_test, target_path):
    """Create test set in the target folder
    Parameters:
        df_test    -- dataframe that contains all the test set details 
                       ('patient_id', 'filename', 'class', 'data_source')
        target_path -- path to the new dataset folder
    """
    folder_path = os.path.join(target_path, 'xray_preprocess/test')
    print(f'Create test set at: {folder_path}')
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        if row['class']=='negative':
            destination_path = os.path.join(folder_path, 'negative')
        elif row['class']=='positive':
            destination_path = os.path.join(folder_path, 'positive')
        if not os.path.exists(destination_path):
            os.makedirs(destination_path) 
        img = os.path.join(target_path, 'xray', 'test', row['filename'])
        shutil.copy(img, destination_path )

def create_train_test_df(target_path):
    """Create train et test dataframe based on text file
        target_path -- path to the new dataset folder
        Parameters:
            target_path -- path to the new dataset folder
        Returns:
            df_test     -- dataframe that contains all the test set details 
                       ('patient_id', 'filename', 'class', 'data_source')
            df_train    -- dataframe that contains all the train set details 
                       ('patient_id', 'filename', 'class', 'data_source')

    """
    df_train = pd.read_csv(os.path.join(target_path, 'xray', 'train.txt'), delimiter=' ',
                                        header = 0 )
    df_test = pd.read_csv(os.path.join(target_path, 'xray', 'test.txt'), delimiter=' ', header = 0)
    df_train.columns=['patient_id', 'filename', 'class', 'data_source']
    df_test.columns=['patient_id', 'filename', 'class', 'data_source']

    return df_train, df_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paths to folders')
    parser.add_argument("--target_path", "-tp", type=str, help='Path to target folder', required=True)
    parser.add_argument("--dataset_path", "-dp", type=str, help='Path to dataset zip', required=True)
    args = parser.parse_args()  
    extract_zip(args.dataset_path, args.target_path)
    df_train, df_test = create_train_test_df(args.target_path)
    create_train_folder(df_train, args.target_path)
    create_test_folder(df_test, args.target_path)
    
    # remove extracted folder (not necessary)
    mydir = os.path.join(args.target_path,'xray')
    try:
        shutil.rmtree( mydir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
