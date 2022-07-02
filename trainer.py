"""Script to run scripts"""

import os
import sys
import argparse

# PYTHON = '/home/gauthiers/.conda/envs/ultra/bin/python'
# DATA_ARG = ""

# def cli():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data-root", "-dr", type=str)
#     parser.add_argument("--data-folder", "-df", type=str)
#     parser.add_argument("--python", "-p", type=str)

#     return parser.parse_args()

# args = cli()

# if args.data_root != None and args.data_folder != None:
#     DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    
# if args.python != None:
#     DATA_ARG = DATA_ARG + " -p {}".format(args.python)



# def runAllInPath(path):
#     for fn in os.listdir(path):
#         os.system("{} {} &".format(sys.executable,os.path.join(path,fn)))

# ds = 'cifar'
# path = "/home/therien/Documents/github/kymatio_mod/experiments/{}/learning_curve_comparison".format(ds)

# runAllInPath(path)
# exit(0)
# 22.320272M



runs = [
    'experiments/cifar/filter_monitoring/ll_1190sample_cifar10_filterMonitoring_qualitative.py',
    'experiments/kth/filter_monitoring/ll_kth_filter_monitoring_qualitative.py',
    'experiments/xray/filter_monitoring/ll_1188sample_xray_filter_monitoring_qualitative.py'
]


for c in runs:
    os.system("{} {}".format(sys.executable, c))