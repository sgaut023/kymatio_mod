""" SN+LL 500 Samples Xray
"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 3

mlflow_exp_name = "\"SN+LL 500 Samples Xray\""
PARAMS_FILE = "parameters_xray.yml"
PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.06
DF = 25
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 1
EPOCHS = 400
INIT = "Kymatio"
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 500
TRAIN_BATCH_SIZE = 128
AUGMENT = "original-cifar"
ALTERNATING = 0
SECOND_ORDER = 0
MODEL="linear_layer"

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

    commandsL = []
    commandsNL = []

    for SEED in [43861066,839243457,50463677,1126614176,977417033,1441538770,1410873071]:#287946934,152241024,1049360704,

        # SEED = int(time.time() * np.random.rand(1))
        for aa in [(1,"Random"),(0,"Random"),(1,"Kymatio"),(0,"Kymatio")]:
            LEARNABLE, INIT = aa

            args1 = "-daug {} -oalt {} -en {} -pf {} -sso {} -mname {} {}".format(
                AUGMENT,ALTERNATING,mlflow_exp_name,PARAMS_FILE,SECOND_ORDER,MODEL,DATA_ARG)

            args2 = "-oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -dtbs {} -os {}".format(
                OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,TRAIN_BATCH_SIZE,SCHEDULER)

            args3 = "-slrs {} -slro {}".format(
                LRS,LRO)
            
            command = "{} {} run-train {} {} {}".format(
                PYTHON,RUN_FILE,args1,args2,args3)

            if LEARNABLE == 1:
                commandsL.append(command)
            else:
                commandsNL.append(command)
    
    commands = commandsL + commandsNL
    
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






