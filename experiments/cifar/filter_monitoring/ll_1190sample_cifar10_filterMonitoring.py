import os
import sys
import torch
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands, logComparison 

mlflow_exp_name = os.path.basename(__file__)
PROCESS_BATCH_SIZE = 2


RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.2
DF = 25
LEARNABLE = 1
EPOCHS = 1
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 1190
TRAIN_BATCH_SIZE = 1200
AUGMENT = "autoaugment"
SCATT_PARAM_DISTANCE = 1
J = 2
L=16

if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    for EPOCHS in [500]:
        commands = []

        for SEED in [207715039]:
            

            for aa in [(1,"Tight-Frame")]:
                LEARNABLE, INIT = aa

                command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -en {} -dtbs {} -spd {} -sj {} {}".format(
                    PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,SCATT_PARAM_DISTANCE,J,DATA_ARG)

                commands.append(command)
        
        experiments_mpCommands(
            processBatchSize=PROCESS_BATCH_SIZE,
            commands=commands
        )

        logComparison(mlflow_exp_name)
    exit(0)
