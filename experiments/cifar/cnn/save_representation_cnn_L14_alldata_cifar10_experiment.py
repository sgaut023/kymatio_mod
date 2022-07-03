import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)

PROCESS_BATCH_SIZE = 2

RUN_FILE = "parametricSN/save_rep.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.1
DF = 25
THREE_PHASE = 1
LEARNABLE = 1
EPOCHS = 500
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 50000
TRAIN_BATCH_SIZE = 1024
AUGMENT = "autoaugment"
MODEL = "cnn"
PHASE_ENDS = " ".join(["100","200"])
SCATT_ARCH = 'scattering'
MODEL_LOSS = 'cross-entropy'
SCATT_LRMAX = 0.2
SCATT_DF = 25
SCATT_THREE_PHASE = 1
L=14


if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [491659600]:#,207715039,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
        for aa in [(1,"Tight-Frame")]:#,(0,"Tight-Frame"),(1,"Random"),(0,"Random")]:
            LEARNABLE, INIT = aa

            args1 = "-oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {}".format(
                OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM
            )

            args2 = "-os {} -daug {} -en {} -dtbs {} -mname {} -ope {}".format(
                SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,MODEL,PHASE_ENDS
            )

            args3 = "-smaxlr {} -sdivf {} -stp {} -mloss {} -sa {} -slrs {} -slro {} -sll {}".format(
                SCATT_LRMAX,SCATT_DF,SCATT_THREE_PHASE,MODEL_LOSS,SCATT_ARCH,LRS,LRO,L
            )

            command = "{} {} run-train {} {} {} {}".format(
                PYTHON,RUN_FILE,args1,args2,args3,DATA_ARG)

            commands.append(command)
    

    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )




