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
    DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    
if args.python != None:
    DATA_ARG = DATA_ARG + " -p {}".format(args.python)


#cifar
os.system("{} xray_experiments/onlycnn_100sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} xray_experiments/onlycnn_500sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))
os.system("{} xray_experiments/onlycnn_1000sample_xray_experiment.py {}".format(PYTHON,DATA_ARG))

exit(0)
