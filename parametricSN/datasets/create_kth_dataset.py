"""Script to download and generate KTH dataset 
The generated dataset contains 4 folders per sample (a,b,c and d)

Dataset folder organization
├── a    <- Folder contains 11 folders (one per material)   
├── b    <- Folder contains 11 folders (one per material) 
├── c    <- Folder contains 11 folders (one per material) 
├── d    <- Folder contains 11 folders (one per material)         

Author: Shanel Gauthier

Functions:
    download_drom_url -- Download Dataset from URL
    extract_tar       -- Extract files from tar
    create_dataset    -- Create dataset in the target folder

"""
from glob import glob
import shutil
import os
from tqdm import tqdm
import requests
import tarfile
import sys
import shutil

def download_drom_url(link, file_name):
    """Download Dataset from URL
    FROM: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Parameters:
        link -- url to dataset
        file_name -- file name of the dataset
    """
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

def extract_tar(file_name, target_path):
    """Extract files from tar
    Parameters:
        file_name   -- file name of the dataset
        target_path -- path to the target dataset folder
    """
    print("Extracting %s" % file_name)
    with tarfile.open(name=file_name) as tar:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(path= target_path, member=member)
    os.remove(file_name)

def create_dataset(target_path):
    """Create KTH dataset in the target folder
    Parameters:
        target_path -- path to the new dataset folder
    """
    folders = glob(f'{target_path}/KTH-TIPS2-b/*/*')
    print("Creating new dataset folder")
    for folder in tqdm(folders):
        new_folder = os.path.join(target_path, "KTH")
        sample = folder.split('/')[-1][-1]
        label = folder.split('/')[-2]
        destination_path = os.path.join(new_folder, f'{sample}/{label}')
        print(destination_path)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)   
        pattern = f'{folder}*/*' 
        for img in glob(pattern):
            shutil.copy(img, destination_path)

if __name__ == '__main__':
    try:
        target_path  =sys.argv[1]
    except IndexError:
        print("ERROR: Must provide a target path to save the KTH dataset")
        exit()
    link = 'https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar'
    file_name ='kth-tips2-b_col_200x200.tar'
    #download_drom_url(link, file_name)
    #extract_tar(file_name, target_path)
    create_dataset(target_path)
    
    # remove extracted folder (not necessary)
    mydir = os.path.join(target_path,'KTH-TIPS2-b')
    try:
        shutil.rmtree( mydir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
