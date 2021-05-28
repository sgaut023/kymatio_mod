""" 1000 sample xray experiment script

This files runs one model in the following settings: (Learnable,"Random"),(Not Leanable,"Random"),(Learnable,"Kymatio"),(Not Leanable,"Kymatio")

Experiment: learnable vs non-learnable & Kymatio vs Random for xray 1000 samples 

example command:

    python parametricSN/cifar_small_sample.py run-train -oname sgd -olr 0.1 -gseed 1371927268 -sl 1 -me 500 -omaxlr 0.06 -odivf 25 -sip Random -dtsn 100 -os OneCycleLR -daug original-cifar -oalt 0 -en "Xray 100 Samples" -pf parameters_xray.yml 

"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 4

mlflow_exp_name = "\"Xray 1000 Samples batch norm affine\""
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
TRAIN_SAMPLE_NUM = 1000
TRAIN_BATCH_SIZE = 128
AUGMENT = "original-cifar"
ALTERNATING = 0


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


    for x in range(RUNS_PER_SEED):

        SEED = int(time.time() * np.random.rand(1))
        for aa in [(1,"Random"),(0,"Random"),(1,"Kymatio"),(0,"Kymatio")]:
            LEARNABLE, INIT = aa

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -oalt {} -en {} -pf {} -dtbs {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,ALTERNATING,mlflow_exp_name,PARAMS_FILE,TRAIN_BATCH_SIZE,DATA_ARG)

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






