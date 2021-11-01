import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)

PROCESS_BATCH_SIZE = 1

RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.06
LRS = 0.1
LRO = 0.1
LRMAX = 0.06
DF = 25
LEARNABLE = 1
EPOCHS = 1000
INIT = "Kymatio"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 500
TRAIN_BATCH_SIZE = 128
AUGMENT = 'original-cifar'
SCATT_LRMAX = 0.000
SCATT_DF = 25
SCATT_THREE_PHASE = 1

PIXELWISE = 'pixelwise'

if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [491659600,207715039,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
        for aa in [(1,"Tight-Frame"),(1,"Random")]:#,
            LEARNABLE, INIT = aa

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -en {} -dtbs {} -smaxlr {} -sdivf {} -stp {} -spw {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE, SCATT_LRMAX, SCATT_DF, SCATT_THREE_PHASE, PIXELWISE, DATA_ARG)            
            commands.append(command)
    

    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )






