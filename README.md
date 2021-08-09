Parametric Scattering Networks
==============================

This repository contains our implementation of learnable scattering networks: https://arxiv.org/abs/2107.09539
![Screen Shot 2021-08-09 at 9 39 37 AM](https://user-images.githubusercontent.com/23482039/128716737-95fe42fa-32b7-4234-bc63-7d500a092636.png)

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
![Screen Shot 2021-08-09 at 9 49 14 AM](https://user-images.githubusercontent.com/23482039/128716927-e73247a1-5423-4408-bea5-06fecfbd8396.png)


Results
------------
We consider an architecture inspired by [Edouard Oyallon et al.](https://arxiv.org/abs/1809.06367), where scattering is combined with a Wide Residual Network (WRN), and another simpler one, denoted LL, where scattering is followed by a linear model. We compare learned parametric scattering networks (LS) to fixed ones (S), for both random (Rand) and tight frame (TF) initializations. We also compare our approach to a fully learned WRN. Our evaluations are based on three image datasets: CIFAR-10, COVIDx CRX-2 and KTH-TIPS2. For CIFAR-10, the training set is augmented with pre-specified autoaugment. The Table below reports the results with J=2.
| Init. | Arch.           | 100 samples             | 500 samples             | 1000 samples            | All                                           |
|-------|-----------------|-------------------------|-------------------------|-------------------------|-----------------------------------------------|
| TF    | LS+LL           | 37.84±0.57              | 52.68±0.31 | $\mathbf{57.43}\pm0.17$ | $\mathbf{69.57}\pm0.1$                        |
| TF    | S +LL           | 36.01±0.55              | $48.12±0.25$          | $53.25\pm 0.24$         | $65.58\pm0.04$                                |
| Rand  | LS+LL           | 34.81±0.6$              | $49.6±0.39$           | $55.72\pm0.39$          | $69.39\pm0.41$                                |
| Rand  | S +LL           | 29.77±0.47              | $41.85±.41$          | $46.3\pm0.37$           | $57.72\pm0.1$                                 |
| TF    | LS+WRN          | 43.60±0.87$             | 63.13}\pm0.29$ | $70.14\pm0.26$          | $93.61\pm0.12$                                |
| TF    | S +WRN          | 43.16±0.78              | $61.66\pm0.32$          | $68.16\pm0.27$          | $92.27\pm0.05$                                |
| Rand  | LS+WRN          | 41.42±0.65              | $59.84\pm0.40$          | $67.4\pm0.28$           | $93.36\pm0.19$                                |
| Rand  | S +WRN          | 32.08±0.46              | $46.84\pm0.21$          | $52.76\pm0.33$          | $85.35\pm1.06$                                |
|       | WRN             | 38.78±0.72$             | $62.97\pm 0.41$         | $\mathbf{71.37}\pm0.31$ | [95.7](https://arxiv.org/abs/1809.06367)

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
    │   └── environment.yml     <- The conda environment file for reproducing the analysis environment.
    




