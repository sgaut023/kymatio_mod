
import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 1

mlflow_exp_name = "\"KTH second order Samples Kymatio Initialization\""

PYTHON = '/home/gauthiers/.conda/envs/ultra/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
PARAMS_FILE = "parameters_texture.yml"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
DF = 25
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 0
INIT = "Kymatio"
EPOCHS = 500
RUNS_PER_SEED = 1
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "StepLR"
AUGMENT = "original-cifar"
ALTERNATING = 1



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
    for sample in ['d', 'c', 'b', 'a']:
        SEED = int(time.time() * np.random.rand(1))
        for x in range(TOTALRUNS):

            LEARNABLE = 0 if LEARNABLE == 1 else 1
            # if x % 1 == 0  and x != 0:
            #     SEED = int(time.time() * np.random.rand(1))
            # if LEARNABLE  ==1:
            #     INIT = "Random"
            # else:
            #     INIT = "Kymatio"

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -odivf {} -sip {}  -os {} -daug {} -oalt {} -en {} -pf {} -dsam {} {}".format(
            PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,DF,INIT,SCHEDULER,AUGMENT,ALTERNATING,mlflow_exp_name,PARAMS_FILE, sample, DATA_ARG)

            commands.append(command)
    
    INIT = "Random"
    for sample in ['d', 'c', 'b', 'a']:
        SEED = int(time.time() * np.random.rand(1))
        for x in range(TOTALRUNS):

            LEARNABLE = 0 if LEARNABLE == 1 else 1
            # if x % 2 == 0  and x != 0:
            #     SEED = int(time.time() * np.random.rand(1))
            # if LEARNABLE  ==1:
            #     INIT = "Random"
            # else:
            #     INIT = "Kymatio"

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -odivf {} -sip {}  -os {} -daug {} -oalt {} -en {} -pf {} -dsam {} {}".format(
            PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,DF,INIT,SCHEDULER,AUGMENT,ALTERNATING,mlflow_exp_name,PARAMS_FILE, sample, DATA_ARG)

            commands.append(command)

    for cmd in commands:
        print(cmd)

    processes = [Process(target=runCommand,args=(commands[i],)) for i,cmd in enumerate(commands)]
    processBatches = [processes[i*PROCESS_BATCH_SIZE:(i+1)*PROCESS_BATCH_SIZE] for i in range(math.ceil(len(processes)/PROCESS_BATCH_SIZE))]

    for i,batch in enumerate(processBatches):
        print("Running process batch {}".format(i))
        startTime = time.time()

        for process in batch:
            print("From Main: {}".format(process._args))
            process.start()
            time.sleep(5)

        for process in batch:
            process.join()

        print("\n\nRunning Took {} seconds".format(time.time() - startTime))
        time.sleep(1)

