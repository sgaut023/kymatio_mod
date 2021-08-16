"""Cifar-10 500 sample experiment script

This files runs one model in the following settings: (Learnable,"Random"),(Not Leanable,"Random"),(Learnable,"Tight-Frame"),(Not Leanable,"Tight-Frame")

Experiment: learnable vs non-learnable scattering for cifar-10 500 samples 

example command:

    python parametricSN/refactor_cifar_small_sample.py run-train -oname sgd -olr 0.1 -slrs 0.1 -slro 0.1 -gseed 1620406577 -sl True -me 10

"""

import os
import time
import sys
sys.path.append(str(os.getcwd()))

import numpy as np

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

PROCESS_BATCH_SIZE = 4

mlflow_exp_name = "\"new Cifar-10 500 Samples + batch norm affine\""

PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.13
DF = 25
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 1
EPOCHS = 1000
INIT = "Kymatio"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 500
TRAIN_BATCH_SIZE = 500
AUGMENT = "autoaugment"



if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [491659600,207715039,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
        for aa in [(1,"Random"),(0,"Random"),(1,"Tight-Frame"),(0,"Tight-Frame")]:
            LEARNABLE, INIT = aa

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -en {} -dtbs {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,DATA_ARG)

            commands.append(command)
    

    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )






