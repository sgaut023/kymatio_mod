"""Script to run scripts"""

import os

PYTHON = '/home/benjamin/venv/torch11/bin/python'

os.system("{} cifar10-1000sample-experiment.py".format(PYTHON))
os.system("{} cifar10-500sample-experiment.py".format(PYTHON))
os.system("{} cifar10-100sample-experiment.py".format(PYTHON))
