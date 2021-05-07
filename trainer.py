"""Script to make multiple train runs in sequence 


example command:

python parametricSN/cifar_small_sample.py run-train -o sgd -lr 0.1 -lrs 0.1 -lro 0.1 -s 1620406577 -m scattering -e 3000
"""


import os
import time

PYTHON = '/home/benjamin/venv/torch11/bin/python'
RUN_FILE = "parametricSN/cifar_small_sample.py"
OPTIM = "adam"
LR = 0.1
LRS = 0.1
LRO = 0.1
SEED = int(time.time())
MODE = "scattering_dif"
EPOCHS = 3000

TOTALRUNS = 4 * (4)


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
