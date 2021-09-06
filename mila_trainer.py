"""Script to run scripts"""

import os
import argparse

PYTHON = 'python'
DATA_ARG = ""

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str)
    parser.add_argument("--data-folder", "-df", type=str)
    parser.add_argument("--python", "-p", type=str)

    return parser.parse_args()

args = cli()

if args.data_root != None and args.data_folder != None:
    DATA_ARG = "-dr {} -df {}".format(args.data_root,args.data_folder)
    
if args.python != None:
    DATA_ARG = DATA_ARG + " -p {}".format(args.python)




#cifar
#os.system("{} cifar_experiments/cnn_100sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} cifar_experiments/cnn_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} cifar_experiments/cnn_1000sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} experiments/cifar_experiments/cnn_alldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))

#os.system("{} cifar_experiments/ll_100sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} cifar_experiments/ll_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} cifar_experiments/ll_1000sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} cifar_experiments/ll_alldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))

os.system("{} experiments/cifar_experiments/resnet50/resnet50_100sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/cifar_experiments/resnet50/resnet50_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/cifar_experiments/resnet50/resnet50_1000sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/cifar_experiments/resnet50/resnet50_alldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))

# os.system("{} cifar_experiments/llalldata_cifar10_experiment.py".format(PYTHON))
#os.system("{} experiments/cifar_experiments/onlycnn_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/cifar_experiments/onlycnn_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/cifar_experiments/onlycnn_1000sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/cifar_experiments/onlycnn_alldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))


exit(0)

#cifar cnn
os.system("{} cifar_experiments/cnn500sample_cifar10_experiment.py".format(PYTHON))

os.system("{} cifar_experiments/cnn100sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/occnn1000sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/cnn1000sample_cifar10_experiment.py".format(PYTHON))


os.system("{} cifar_experiments/100sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/500sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/1000sample_cifar10_experiment.py".format(PYTHON))


exit(0)

#cifar
os.system("{} cifar_experiments/100sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/500sample_cifar10_experiment.py".format(PYTHON))
os.system("{} cifar_experiments/1000sample_cifar10_experiment.py".format(PYTHON))

#xray
os.system("{} xray_experiments/100sample_xray_experiment.py".format(PYTHON))
os.system("{} xray_experiments/500sample_xray_experiment.py".format(PYTHON))
os.system("{} xray_experiments/1000sample_xray_experiment.py".format(PYTHON))
