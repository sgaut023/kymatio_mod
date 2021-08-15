import os
import math
import time
import argparse
import numpy as np
from multiprocessing import Process
PROCESS_BATCH_SIZE = 4
mlflow_exp_name = "\"KTH Only Resnet\""
PYTHON = '/home/gauthiers/.conda/envs/ultra/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
PARAMS_FILE = "parameters_texture.yml"
OPTIM = "sgd"
LR = 0.0001
LRS = 0.1
LRO = 0.1
DF = 25
SEED = int(time.time() * np.random.rand(1))
EPOCHS = 100
RUNS_PER_SEED = 4
TOTALRUNS = 1
SCHEDULER = "OneCycleLR"
AUGMENT = "original-cifar"
ACCUM_STEP_MULTIPLE = 128
TEST_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 16
SECOND_ORDER = 0
MODEL = 'resnet50'
MODEL_WIDTH = 8
SCATT_ARCH = 'identity'
MODEL_LOSS = 'cross-entropy-accum'
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
    for SEED in [1390666426,432857963,1378328753,1118756524]:
        for sample in ['a', 'b', 'c', 'd']:
            args1 = "-oname {} -olr {} -gseed {} -me {} -odivf {}  -os {} -daug {} -en {} -pf {} -dsam {} {}".format(
                OPTIM,LR,SEED,EPOCHS,DF,SCHEDULER,AUGMENT,mlflow_exp_name,PARAMS_FILE, sample, DATA_ARG
            )
            args2 = "-mw {} -mloss {} -sa {} -dtstbs {} -dtbs {} -mname {} -dasm {}".format(
            MODEL_WIDTH,MODEL_LOSS,SCATT_ARCH,TEST_BATCH_SIZE,TRAIN_BATCH_SIZE,MODEL,ACCUM_STEP_MULTIPLE)
            command = "{} {} run-train {} {}".format(
            PYTHON,RUN_FILE,args1,args2)
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