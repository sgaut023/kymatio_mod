""" SN+CNN 100 Samples Xray
"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 3

mlflow_exp_name = "\"SN+CNN 1000 Samples Xray\""
PARAMS_FILE = "parameters_xray.yml"
PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.01
LRS = 0.01
LRO = 0.01
LRMAX = 0.01
DF = 25
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 1
EPOCHS = 200
INIT = "Kymatio"
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 1000
TRAIN_BATCH_SIZE = 128
AUGMENT = "original-cifar"
ALTERNATING = 0
SECOND_ORDER = 0
MODEL = 'cnn'

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

    # for x in range(RUNS_PER_SEED):
    for SEED in [706825958,750490779,884698041,1614090550]:#637789757,433842091,313350229,22942091,1452034008,1065155395
        if SEED == 750490779:
            rns = [(0,"Random")]
        elif SEED == 706825958:
            rns = [(0,"Random")]
        elif SEED == 884698041:
            rns = [(0,"Kymatio")]
        elif SEED == 1614090550:
            rns = [(0,"Random")]
        else: 
            rns = [(1,"Kymatio"),(0,"Kymatio"),(1,"Random"),(0,"Random")]


        # SEED = int(time.time() * np.random.rand(1))
        for aa in rns:
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






