"""Script to run scripts"""

import os

PYTHON = '/home/benjamin/venv/torch11/bin/python'

#cifar
os.system("{} cifar_experiments/100sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/500sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/1000sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/cnn1000sample_cifar10_experiment.py".format(PYTHON))
exit(0)

#cifar cnn

os.system("{} cifar_experiments/cnn500sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/cnn100sample_cifar10_experiment.py".format(PYTHON))

exit(0)

#cifar
os.system("{} cifar_experiments/100sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/500sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/1000sample_cifar10_experiment.py".format(PYTHON))

#xray
os.system("{} xray_experiments/100sample_xray_experiment.py".format(PYTHON))
os.system("{} xray_experiments/500sample_xray_experiment.py".format(PYTHON))
os.system("{} xray_experiments/1000sample_xray_experiment.py".format(PYTHON))
