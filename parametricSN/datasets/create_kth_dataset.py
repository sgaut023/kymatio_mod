"""Script to download and generate KTH dataset 
The generated dataset contains 4 folders per sample (a,b,c and d)

     

Author: Shanel Gauthier

Functions:
    download_from_url -- Download Dataset from URL
    extract_tar       -- Extract files from tar
    create_dataset    -- Create dataset in the target folder

"""
from glob import glob
import shutil
from pathlib import Path
import os
from tqdm import tqdm
import requests
import tarfile
import sys
import shutil



# if __name__ == '__main__':
#     target_path = Path(os.path.realpath(__file__)).parent.parent.parent/'data'
#     target_path.mkdir(parents=True, exist_ok= True)

#     link = 'https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar'
#     file_name ='kth-tips2-b_col_200x200.tar'
#     download_from_url(link, file_name)
#     extract_tar(file_name, target_path)
#     create_dataset(target_path)
    
#     # remove extracted folder (not necessary)
#     mydir = os.path.join(target_path,'KTH-TIPS2-b')
#     try:
#         shutil.rmtree( mydir)
#     except OSError as e:
#         print ("Error: %s - %s." % (e.filename, e.strerror))

def download():

