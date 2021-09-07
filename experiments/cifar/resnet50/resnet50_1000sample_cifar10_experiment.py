"""CNN No-SCAT 1000 samples Cifar-10
"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 5

mlflow_exp_name = "\"OnlyCNN 1000 samples Cifar-10\""

PYTHON = '/home/benjamin/venv/torch11/bin/python'
PARAMS_FILE = "parameters.yml"
RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.1
DF = 25
SEED = int(time.time() * np.random.rand(1))
EPOCHS = 1000
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 1000
TEST_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 64
AUGMENT = "autoaugment"
SECOND_ORDER = 0
MODEL = 'resnet50'

MODEL_WIDTH = 8
SCATT_ARCH = 'identity'

ACCUM_STEP_MULTIPLE = 128
MODEL_LOSS = 'cross-entropy'

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

    for SEED in [207715039, 491659600,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
        args1 = "-daug {} -en {} -pf {} -sso {} -mname {} {}".format(
        AUGMENT,mlflow_exp_name,PARAMS_FILE,SECOND_ORDER,MODEL,DATA_ARG)

        args2 = "-oname {} -olr {} -gseed {} -me {} -omaxlr {} -odivf {} -dtsn {} -dtbs {} -os {}".format(
            OPTIM,LR,SEED,EPOCHS,LRMAX,DF,TRAIN_SAMPLE_NUM,TRAIN_BATCH_SIZE,SCHEDULER)

        args3 = "-slrs {} -slro {} -mw {} -mloss {} -sa {} -dtstbs {}".format(
            LRS,LRO,MODEL_WIDTH,MODEL_LOSS,SCATT_ARCH,TEST_BATCH_SIZE)
        
        command = "{} {} run-train {} {} {}".format(
            PYTHON,RUN_FILE,args1,args2,args3)

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
            time.sleep(10)

        for process in batch:
            process.join()

        print("\n\nRunning Took {} seconds".format(time.time() - startTime))
        time.sleep(1)






