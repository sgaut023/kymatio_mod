"""Script to run scripts"""

import os
import argparse

PYTHON = '/home/gauthiers/.conda/envs/ultra/bin/python'
DATA_ARG = ""

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str)
    parser.add_argument("--data-folder", "-df", type=str)
    parser.add_argument("--python", "-p", type=str)

    return parser.parse_args()

args = cli()

if args.data_root != None and args.data_folder != None:
    DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    
if args.python != None:
    DATA_ARG = DATA_ARG + " -p {}".format(args.python)


os.system("{} experiments/xray_experiments/onlycnn/onlycnn_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} experiments/xray_experiments/onlycnn/onlycnn_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} experiments/xray_experiments/onlycnn_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} experiments/xray_experiments/onlycnn_1000sample_xray_experiment.py{}".format(PYTHON,DATA_ARG))
#os.system("{} xray_experiments/resnet50_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} xray_experiments/cnn_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} experiments/xray_experiments/ll_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} experiments/xray_experiments/ll_allsample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} experiments/xray_experiments/ll_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} xray_experiments/resnet50_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} xray_experiments/resnet50_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
#os.system("{} xray_experiments/ll_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} xray_experiments/cnn_100sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))







#cifarxray_experiments/
# os.system("{} cifar_experiments/onlycnnalldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))
# os.system("{} cifar_experiments/cnnalldata_cifar10_experiment.py {}".format(PYTHON,DATA_ARG))

# os.system("{} cifar_experiments/llalldata_cifar10_experiment.py".format(PYTHON))


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
