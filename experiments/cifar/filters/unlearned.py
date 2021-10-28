import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)

PROCESS_BATCH_SIZE = 1

RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.06
LRS = 0.2
LRO = 0.2
LRMAX = 0.06
DF = 25
LEARNABLE = 1
EPOCHS = 500
INIT = "Kymatio"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 50000
TRAIN_BATCH_SIZE = 128
AUGMENT = "original-cifar"
L=16
# 14, 3
# 12, 4
# 10, 5
# 8, 6, 7
J=1
if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [207715039]:
        for L in [16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2]:
            for aa in [(0,"Tight-Frame")]:
                LEARNABLE, INIT = aa

                command = "{} {} run-train -oname {} -olr {} -gseed {}  -sj {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -en {} -dtbs {} -sll {} {}".format(
                    PYTHON,RUN_FILE,OPTIM,LR,SEED,J,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,L,DATA_ARG)

                commands.append(command)
    

    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )






