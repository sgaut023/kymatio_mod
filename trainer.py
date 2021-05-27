"""Script to run scripts"""

import os

PYTHON = '/home/benjamin/venv/torch11/bin/python'
os.system("{} xray_experiments/100sample_xray_experiment.py".format(PYTHON))
os.system("{} xray_experiments/500sample_xray_experiment.py".format(PYTHON))
os.system("{} xray_experiments/1000sample_xray_experiment.py".format(PYTHON))
