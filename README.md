Parametric Scattering Networks
==============================

This repository contains our implementation of learnable scattering networks: https://arxiv.org/abs/2107.09539


Get Setup
------------

Prerequisites
- Anaconda/Miniconda

Start by cloning the repository

To create the `parametricSN` conda environment, enter the following in the command prompt: 
```
conda env create -f parametricSN/environment.yml
```
To active the `parametricSN` conda environment, enter the following: 
```
conda activate parametricSN
```

Experiments
------------


Results
------------


Project Organization
------------

    ├── conf                    <- Configuration folder
    ├── experiments        
    │   ├── cifar_experiments   <- All scripts to reproduce cifar experiments.
    │   ├── kth_experiments     <- All scripts to reproduce KTH-TPIS2 experiments.
    │   └── xray_experiments    <- All scripts to reproduce Covidx CRX-2 experiments.
    ├── kymatio                 <- Folder copied from: https://github.com/kymatio/kymatio.
    ├── parametricSN 
    │   ├── data_loading        <- Wrapper for subsampling the cifar-10, KTH-TIPS2 and Covidx CRX-2 based on given input.
    │   ├── models              <- Contains all the  pytorch NN.modules for this project.
    │   └── notebooks           <- Jupyter notebooks.
    │   └── training            <- Contains train et test functions.
    │   └── utils               <- Helpers Functions.
    │   └── main.py             <- Source code for use in this project.
    │   └── environment.yml     <- The conda environment file for reproducing the analysis environment
    




