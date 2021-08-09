Parametric Scattering Networks
==============================

This repository contains our implementation of learnable scattering networks: https://arxiv.org/abs/2107.09539
![Screen Shot 2021-08-09 at 9 39 37 AM](https://user-images.githubusercontent.com/23482039/128716737-95fe42fa-32b7-4234-bc63-7d500a092636.png)

You can use the following notebook to explore the parameters used to create the filters.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sgaut023/kymatio_mod/blob/master/parametricSN/notebooks/FilterParamsEffect.ipynb)

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
Datasets
------------
Our empirical evaluations are based on three image datasets, illustrated in the Figure below. We subsample each dataset at various sample sizes in order to showcase the performance of scattering-based architectures in the small data regime. CIFAR-10 and [KTH-TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/credits.html) are natural image and texture recognition datasets (correspondingly). They are often used as general-purpose benchmarks in similar image analysis settings. [COVIDx CRX-2](https://www.kaggle.com/andyczhao/covidx-cxr2) is a dataset of X-ray scans for COVID-19 diagnosis; its use here demonstrates the viability of our parametric scattering approach in practice, e.g., in medical imaging applications.
![Screen Shot 2021-08-09 at 9 49 14 AM](https://user-images.githubusercontent.com/23482039/128716927-e73247a1-5423-4408-bea5-06fecfbd8396.png)

#### 1. KTH-TIPS2
To download the [KTH-TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/credits.html) dataset, run this command where target_path is the path to the target folder.
```
python parametricSN/datasets/create_kth_dataset.py target_path
```

#### 2. COVIDx CRX-2
To download the [COVIDx CRX-2](https://www.kaggle.com/andyczhao/covidx-cxr2) dataset, you need to download your kaggle.json file by following these [instructions](https://github.com/Kaggle/kaggle-api#api-credentials) and place it in the location ~/.kaggle/kaggle.json. Then, run this command where target_path is the path to the target folder.
```
bash parametricSN/datasets/create_xray_dataset.sh target_path
```
Experiments
------------
To run an experiment with the CIFAR-10 dataset, run the command below:
```
python parametricSN/main.py run-train -pf parameters.yml
```
To run an experiment with the [KTH-TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/credits.html) dataset, run the command below:
```
python parametricSN/main.py run-train -pf parameters_texture.yml
```
To run an experiment with the [COVIDx CRX-2](https://www.kaggle.com/andyczhao/covidx-cxr2) dataset, run the command below:
```
python parametricSN/main.py run-train -pf parameters_xray.yml
```


Results
------------
We consider an architecture inspired by [Edouard Oyallon et al.](https://arxiv.org/abs/1809.06367), where scattering is combined with a Wide Residual Network (WRN), and another simpler one, denoted LL, where scattering is followed by a linear model. We compare learned parametric scattering networks (LS) to fixed ones (S), for both random (Rand) and tight frame (TF) initializations. We also compare our approach to a fully learned WRN. Our evaluations are based on three image datasets: CIFAR-10, COVIDx CRX-2 and KTH-TIPS2. For CIFAR-10, the training set is augmented with pre-specified autoaugment. The Table below reports the results with J=2. Learnable scattering with tight frame configuration improves performance for all architectures, showing benefits for small sample sizes. Randomly-initialized scattering can reach similar performance to tight frame after optimization.
| Init. | Arch.           | 100 samples             | 500 samples             | 1000 samples            | All                                           |
|-------|-----------------|-------------------------|-------------------------|-------------------------|-----------------------------------------------|
| TF    | LS+LL           | 37.84±0.57              | 52.68±0.31              | 57.43±0.17              | 69.57±0.1                                     |
| TF    | S +LL           | 36.01±0.55              | 48.12±0.25              | 53.25± 0.24             | 65.58±0.04                                    |
| Rand  | LS+LL           | 34.81±0.6               | 49.6±0.39               | 55.72±0.39              | 69.39±0.41                                    |
| Rand  | S +LL           | 29.77±0.47              | 41.85±.41               | 46.3±0.37               | 57.72±0.1                                     |
| TF    | LS+WRN          | 43.60±0.87              | 63.13±0.29             | 70.14±0.26              | 93.61±0.12                                    |
| TF    | S +WRN          | 43.16±0.78              | 61.66±0.32              | 68.16±0.27              | 92.27±0.05                                    |
| Rand  | LS+WRN          | 41.42±0.65              | 59.84±0.40              | 67.4±0.28               | 93.36±0.19                                    |
| Rand  | S +WRN          | 32.08±0.46              | 46.84±0.21              | 52.76±0.33              | 85.35±1.06                                    |
|       | WRN             | 38.78±0.72              | 62.97± 0.41             | 71.37±0.31              | [95.7](https://arxiv.org/abs/1809.06367)

Table below reports our evaluation on [COVIDx CRX-2](https://www.kaggle.com/andyczhao/covidx-cxr2)  and [KTH-TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/credits.html) using J=3. For COVIDx CRX-2, we use the same evaluation protocol as CIFAR-10.  We observe that the WRN alone performs worse than the other architectures, demonstrating the effectiveness of the scattering prior in the small data regime. For KTH-TIPS2, following the [standard protocol](https://openaccess.thecvf.com/content_ICCV_2017/papers/Song_Locally-Transferred_Fisher_Vectors_ICCV_2017_paper.pdf), we train the model on one of the four physical samples of 11 different materials (1188 images), while the rest are used for testing. Out of all the WRN hybrid models, the random learnable model achieves the highest accuracy and is the only one to improve over its linear counterpart. 

| Init. | Arch.           | CIFAR-100 samples       | CIFAR-500 samples       | CIFAR-1000 samples      | KTH                     |
|-------|-----------------|-------------------------|-------------------------|-------------------------|-------------------------|
| TF    | LS+LL           | 74.80±1.65              | 83.10±0.84              | 84.58±0.79              | 66.83±0.94              |
| TF    | S +LL           | 75.48±1.77              | 83.58±0.91              | 86.18±0.49              | 63.91±0.57              |
| Rand  | LS+LL           | 73.15±1.35              | 82.33±1.1               | 84.73±0.59              | 65.98±0.73              |
| Rand  | S +LL           | 74.25±0.86              | 82.53±0.76              | 85.43±0.51              | 60.42±0.34              |
| TF    | LS+WRN          | 78.1±1.62               | 86.15±0.63              | 89.65±0.42              | 66.46±1.09              |
| TF    | S +WRN          | 76.23±2                 | 86.5±0.66               | 89.13±0.36              | 63.77±0.59              |
| Rand  | LS+WRN          | 74.86±1.22              | 84.15±0.79              | 87.63±0.55              | 67.35±0.51              |
| Rand  | S +WRN          | 75.4±1.03               | 83.75±0.58              | 87.48±0.61              | 65.05±0.38              |
|       | WRN             | 69.15±1.13              | 80.04±2.41              | 87.81±1.37              | $51.24±1.37             |



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
    │   ├── datasets            <- Script to download KTH-TPIS2 and COVIDx CRX-2 datasets.
    │   └── notebooks           <- Jupyter notebooks.
    │   └── training            <- Contains train and test functions.
    │   └── utils               <- Helpers Functions.
    │   └── main.py             <- Source code.
    │   └── environment.yml     <- The conda environment file for reproducing the analysis environment.
    




