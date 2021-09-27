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

#CIFAR RESNET-50
os.system("{} experiments/cifar/resnet50/resnet50_100sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/cifar/resnet50/resnet50_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/cifar/resnet50/resnet50_1000sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/cifar/resnet50/resnet50_alldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))


#KTH
# os.system("{} experiments/kth/cnn_pixelwise_kth_sample-experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/kth/ll_pixelwise_kth_sample-experiment.py {}".format(PYTHON,DATA_ARG))

#Xray
# os.system("{} experiments/cifar/cnn/cnn_100sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/cifar/cnn/cnn_500sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/cifar/cnn/cnn_1000sample_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/cifar/cnn/cnn_alldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))

# os.system("{} experiments/xray/ll/ll_100sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/xray/ll/ll_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/xray/ll/ll_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))

# os.system("{} experiments/xray/cnn/cnn_100sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/xray/cnn/cnn_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} experiments/xray/cnn/cnn_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))

exit(0)
