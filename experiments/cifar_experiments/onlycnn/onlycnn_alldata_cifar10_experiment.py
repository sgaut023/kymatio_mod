""" Only CNN Cifar-10 1000 sample experiment script
"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 1

mlflow_exp_name = "\"All Data Cifar-10 CNN No SCATT\""

PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.1
DF = 25
THREE_PHASE = 1
SEED = int(time.time() * np.random.rand(1))
EPOCHS = 50
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TEST_BATCH_SIZE = 256
TRAIN_SAMPLE_NUM = 50000
TRAIN_BATCH_SIZE = 256
AUGMENT = "autoaugment"
MODEL = "cnn"
PHASE_ENDS = " ".join(["100","200"])
MODEL_WIDTH = 8
SCATT_ARCH = 'identity'
MODEL_LOSS = 'cross-entropy'
SCATT_LRMAX = 0.2
SCATT_DF = 25
SCATT_THREE_PHASE = 1


def runCommand(cmd):
    print("[Running] {}".format(cmd))
    os.system(cmd)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str)
    parser.add_argument("--data-folder", "-df", type=str)
    parser.add_argument("--python", "-p", type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = cli()

    if args.data_root != None and args.data_folder != None:
        DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    else:
        DATA_ARG = ""

    if args.python != None:
        PYTHON = args.python

    commands = []

    for SEED in [491659600,207715039,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
            args1 = "-oname {} -olr {} -gseed {} -me {} -omaxlr {} -odivf {} -dtsn {}".format(
                OPTIM,LR,SEED,EPOCHS,LRMAX,DF,TRAIN_SAMPLE_NUM
            )

            args2 = "-os {} -daug {} -en {} -dtbs {} -mname {} -ope {}".format(
                SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,MODEL,PHASE_ENDS
            )

            args3 = "-smaxlr {} -sdivf {} -stp {} -mloss {} -sa {} -mw {} -dtstbs {}".format(
                SCATT_LRMAX,SCATT_DF,SCATT_THREE_PHASE,MODEL_LOSS,SCATT_ARCH,MODEL_WIDTH,TEST_BATCH_SIZE
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





