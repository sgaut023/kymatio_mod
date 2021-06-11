#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=6                     # Ask for 2 CPUs
#SBATCH --gres=gpu:v100:32gb:1                          # Ask for 1 GPU
#SBATCH --mem=32G                             # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                        # The job will run for 3 hours
#SBATCH -o /home/mila/b/benjamin.therien/logging/slurm-%j.out  # Write the log on tmp1

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate parametricSN

# 3. Copy your dataset on the compute node
cp -r /home/mila/b/benjamin.therien/github/data/xray/chest_xrays_preprocess $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python mila_trainer.py -dr $SLURM_TMPDIR -df chest_xrays_preprocess -p python

# 5. Copy whatever you want to save on $SCRATCH
#cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
