
import os
import math
import time
import argparse

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 2

mlflow_exp_name = "\" Scattering + CNN FOR KTH \""

PYTHON = '/home/gauthiers/.conda/envs/ultra/bin/python'
RUN_FILE = "parametricSN/main.py"

PARAMS_FILE = "parameters_texture.yml"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
DF = 25
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = 0
INIT = "Tight-Frame"
EPOCHS = 50
RUNS_PER_SEED = 1
TOTALRUNS = 4
SCHEDULER = "OneCycleLR"
AUGMENT = "original-cifar"
MODEL = 'cnn'
MODEL_LOSS = 'cross-entropy'
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
    for SEED in [1390666426,432857963,1378328753,1118756524]:
        for sample in ['d', 'c', 'b', 'a']:
            for x in range(TOTALRUNS):

                LEARNABLE = 0 if LEARNABLE == 1 else 1
                if x % 2 == 0  and x != 0:
                    INIT = "Random" if INIT == "Tight-Frame" else "Tight-Frame"

                command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -odivf {} -sip {}  -os {} -daug {} -en {} -mname {} -pf {} -dsam {} -mloss {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,DF,INIT,SCHEDULER,AUGMENT,mlflow_exp_name, MODEL, PARAMS_FILE, sample, MODEL_LOSS, DATA_ARG)

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

