"""Cifar-10 100 samples kymatio initialization experiment script

Experiment: learnable vs non-learnable scattering for cifar-10 100 samples with kymatio initialization

example command:

    python parametricSN/refactor_cifar_small_sample.py run-train -oname sgd -olr 0.1 -slrs 0.1 -slro 0.1 -gseed 1620406577 -sl True -me 10

"""

import os
import math
import time

import numpy as np

from multiprocessing import Process

PROCESS_BATCH_SIZE = 1

mlflow_exp_name = "\"Cifar-10 100 Samples Kymatio Initialization\""

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
EPOCHS = 10000
INIT = "Kymatio"
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 100
AUGMENT = "autoaugment"
ALTERNATING = 0

def runCommand(cmd):
    print("[Running] {}".format(command))
    os.system(command)

if __name__ == '__main__':

    commands = []

    for x in range(TOTALRUNS):

        LEARNABLE = 1 if LEARNABLE == 1 else 0
        
        if x % 2 == 0  and x != 0:
            SEED = int(time.time() * np.random.rand(1))

        command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -oalt {} -en {}".format(
        PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,ALTERNATING,mlflow_exp_name)

        commands.append(command)

    processes = [Process(target=runCommand,args=(cmd,)) for cmd in commands]
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






