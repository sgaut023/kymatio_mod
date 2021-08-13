#!/bin/bash
echo
echo "------------------Create Xray Dataset-----------------------"
echo
echo "Please make sure that you already downloaded your kaggle.json file by following these instructions: https://github.com/Kaggle/kaggle-api#api-credentials "
echo "Please make sure that you put it in the location ~/.kaggle/kaggle.json"
echo
echo "The current folder is: $PWD"
kaggle datasets download -d andyczhao/covidx-cxr2 --force
echo
python parametricSN/datasets/create_xray_dataset.py -dp $PWD 
echo
echo "Done creating xray dataset"
