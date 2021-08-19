"""Script to run scripts"""

import os
import sys
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

for ds in ['xray','kth','cifar']:
    for fn in os.listdir("/home/benjamin/Documents/github/kymatio_mod/experiments/{}/filter_monitoring".format(ds)):
        os.system("{} experiments/{}/filter_monitoring/{}".format(sys.executable,ds,fn))


exit(0)
