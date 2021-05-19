"""Cifar-10 100 samples kymatio initialization experiment script

Experiment: learnable vs non-learnable scattering for cifar-10 100 samples with kymatio initialization

example command:

    python parametricSN/refactor_cifar_small_sample.py run-train -oname sgd -olr 0.1 -slrs 0.1 -slro 0.1 -gseed 1620406577 -sl True -me 10

"""
import numpy as np
import os
import math
import time
import argparse

from multiprocessing import Process

PROCESS_BATCH_SIZE = 2

mlflow_exp_name = "\"KTH Samples Kymatio Initialization\""

PYTHON = '/home/gauthiers/.conda/envs/ultra/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
PARAMS_FILE = "parameters_texture_mila.yml"
OPTIM = "sgd"
LR = 0.01
LRS = 0.01
LRO = 0.01
DF = 25
SEED = int(time.time() * np.random.rand(1))
LEARNABLE = True
EPOCHS = 10
RUNS_PER_SEED = 1
TOTALRUNS = 2 * RUNS_PER_SEED

def runCommand(cmd):
    print("[Running] {}".format(command))
    os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser .add_argument('--data-root', "-dr", type=str, help="Data Root")
    args = parser.parse_args()

    commands = []
    for sample in ["a", "b", "c", "d"]:
        for x in range(TOTALRUNS):

            LEARNABLE = not LEARNABLE
            if LEARNABLE :
                INIT = "Random"
            else:
                INIT = "Kymatio"
            # if x % 2 == 0  and x != 0d:
            #     SEED = int(time.time() * np.random.rand(1))

            command = "{} {} run-train -oname {} -olr {} -slrs {} -slro {} -gseed {} -sl {} -me {} -odivf {} -sip {} -en {} -dsam {} -pf {} -ddr {}".format(
            PYTHON,RUN_FILE,OPTIM,LR,LRS,LRO,SEED,LEARNABLE,EPOCHS,DF,INIT, mlflow_exp_name, sample, PARAMS_FILE, args.data_root )

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
    
    