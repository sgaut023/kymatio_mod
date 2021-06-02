"""CNN Cifar-10 1000 sample experiment script

This files runs one model in the following settings: (Learnable,"Random"),(Not Leanable,"Random"),(Learnable,"Kymatio"),(Not Leanable,"Kymatio")

Experiment: learnable vs non-learnable scattering for cifar-10 1000 samples 

example command:

    python parametricSN/refactor_cifar_small_sample.py run-train -oname sgd -olr 0.1 -slrs 0.1 -slro 0.1 -gseed 1620406577 -sl True -me 10

"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 1

mlflow_exp_name = "\"CNN cosine loss Cifar-10 1000 samples batch norm affine\""

PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.06
DF = 25
THREE_PHASE = 1
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 1
EPOCHS = 1000
INIT = "Kymatio"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 1000
TRAIN_BATCH_SIZE = 128
AUGMENT = "autoaugment"
ALTERNATING = 1
MODEL = "cnn"
PHASE_ENDS = " ".join(["200","300","600","700","900","1000"])
PHASE_ENDS = " ".join(["1","300","600","700","900","1000"])
# PHASE_ENDS = " ".join(["5","10"])

MODEL_LOSS = 'cross-entropy'
SCATT_LRMAX = 0.6
SCATT_DF = 25
SCATT_THREE_PHASE = 1


def runCommand(cmd):
    print("[Running] {}".format(cmd))
    os.system(cmd)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=int)
    parser.add_argument("--data-folder", "-df", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = cli()

    if args.data_root != None and args.data_folder != None:
        DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    else:
        DATA_ARG = ""

    commands = []

    # for x in range(RUNS_PER_SEED):
    for SEED in [737523103,207715039,491659600,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:

        # SEED = int(time.time() * np.random.rand(1))
        for aa in [(1,"Kymatio"),(0,"Kymatio"),(1,"Kymatio"),(0,"Kymatio")]:#(1,"Random"),(0,"Random")]:
            LEARNABLE, INIT = aa

            args1 = "-oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {}".format(
                OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM
            )

            args2 = "-os {} -daug {} -oalt {} -en {} -dtbs {} -mname {} -ope {}".format(
                SCHEDULER,AUGMENT,ALTERNATING,mlflow_exp_name,TRAIN_BATCH_SIZE,MODEL,PHASE_ENDS
            )

            args3 = "-smaxlr {} -sdivf {} -stp {} -mloss {}".format(
                SCATT_LRMAX,SCATT_DF,SCATT_THREE_PHASE,MODEL_LOSS
            )

            command = "{} {} run-train {} {} {} {}".format(
                PYTHON,RUN_FILE,args1,args2,args3,DATA_ARG)

            commands.append(command)
    

    for cmd in commands:
        print(cmd)

    processes = [Process(target=runCommand,args=(commands[i],)) for i,cmd in enumerate(commands)]
    processBatches = [processes[i*PROCESS_BATCH_SIZE:(i+1)*PROCESS_BATCH_SIZE] for i in range(math.ceil(len(processes)/PROCESS_BATCH_SIZE))]

    for i,batch in enumerate(processBatches):
        print("Running process batch {}".format(i))
        startTime = time.time()

        for process in batch:
            process.start()
            time.sleep(5)

        for process in batch:
            process.join()

        print("\n\nRunning Took {} seconds".format(time.time() - startTime))
        time.sleep(1)






