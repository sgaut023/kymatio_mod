"""Cifar-10 100 sample experiment script

This files runs one model in the following settings: (Learnable,"Random"),(Not Leanable,"Random"),(Learnable,"Kymatio"),(Not Leanable,"Kymatio")

Experiment: learnable vs non-learnable scattering for cifar-10 100 samples with kymatio initialization

example command:

    python parametricSN/refactor_cifar_small_sample.py run-train -oname sgd -olr 0.1 -slrs 0.1 -slro 0.1 -gseed 1620406577 -sl True -me 10

"""

import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 4

mlflow_exp_name = "\"new Cifar-10 100 Samples batch norm affine\""

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
EPOCHS = 5000
INIT = "Kymatio"
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 100
TRAIN_BATCH_SIZE = 128
AUGMENT = "autoaugment"
ALTERNATING = 0


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

    # for x in range(RUNS_PER_SEED):
    for SEED in [235992187,306511717,320845575,444050326,497316683,788057338,831046333,863982375,874034868,916884331]:

        # SEED = int(time.time() * np.random.rand(1))
        for aa in [(1,"Random"),(0,"Random"),(1,"Kymatio"),(0,"Kymatio")]:
            LEARNABLE, INIT = aa

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -dtbs {} -os {} -daug {} -oalt {} -en {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,TRAIN_BATCH_SIZE,SCHEDULER,AUGMENT,ALTERNATING,mlflow_exp_name,DATA_ARG)

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





