"""Script to make multiple train runs in sequence 


example command:

python parametricSN/refactor_cifar_small_sample.py run-train -o sgd -lr 0.1 -lrs 0.1 -lro 0.1 -s 1620406577 -m scattering -e 3000
"""


import os
import time

PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.06
DF = 25
SEED = int(time.time())
MODE = "scattering_dif"
EPOCHS = 10000
INIT = "Kymatio"
TOTALRUNS = 2 * (2)



for x in range(TOTALRUNS):
    if x % 2 == 0 and x != 0:
        SEED = int(time.time())

    if MODE == "scattering":
        MODE = "scattering_dif"
    else:
        MODE = "scattering"    

    command = "{} {} run-train -o {} -lr {} -lrs {} -lro {} -s {} -m {} -e {} -lrmax {} -df {} -ip {}".format(
    PYTHON,RUN_FILE,OPTIM,LR,LRS,LRO,SEED,MODE,EPOCHS,LRMAX,DF,INIT)
    print("[Running] {}".format(command))
    os.system(command)

OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.06
DF = 25
SEED = int(time.time())
MODE = "scattering_dif"
EPOCHS = 10000
INIT = "Random"
TOTALRUNS = 2 * (2)

for x in range(TOTALRUNS):
    if x % 2 == 0 and x != 0:
        SEED = int(time.time())

    if MODE == "scattering":
        MODE = "scattering_dif"
    else:
        MODE = "scattering"    

    command = "{} {} run-train -o {} -lr {} -lrs {} -lro {} -s {} -m {} -e {} -lrmax {} -df {} -ip {}".format(
    PYTHON,RUN_FILE,OPTIM,LR,LRS,LRO,SEED,MODE,EPOCHS,LRMAX,DF,INIT)
    print("[Running] {}".format(command))
    os.system(command)





exit(0)

for x in range(TOTALRUNS):
    if x % 4 == 0 and x != 0:
        if MODE == "scattering":
            MODE = "scattering_dif"
        else:
            MODE = "scattering"

    if x % 2 == 0:
        SEED = int(time.time())
        OPTIM = "adam"
        LR = 0.001
        LRS = 0.001
        LRO = 0.001
        command = "{} {} run-train -o {} -lr {} -lrs {} -lro {} -s {} -m {} -e {}".format(
    PYTHON,RUN_FILE,OPTIM,LR,LRS,LRO,SEED,MODE,EPOCHS)
    else:
        OPTIM = "sgd"
        LR = 0.1
        LRS = 0.1
        LRO = 0.1
        command = "{} {} run-train -o {} -lr {} -lrs {} -lro {} -s {} -m {} -e {}".format(
    PYTHON,RUN_FILE,OPTIM,LR,LRS,LRO,SEED,MODE,EPOCHS)
    print("[Running] {}".format(command))
    os.system(command)
