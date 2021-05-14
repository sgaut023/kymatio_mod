#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=3G                             # Ask for 10 GB of RAM
#SBATCH --time=1:00:00                        # The job will run for 3 hours
#SBATCH -o /home/mila/g/gauthies/logging/slurm-%j.out  # Write the log on tmp1

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate parametricSN

# 3. Copy your dataset on the compute node
cp -r /home/mila/g/gauthies/datasets/KTH $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python parametricSN/cifar_small_sample.py run-train -dr $SLURM_TMPDIR  -dfo KTH -pf parameters_texture.yml

# 5. Copy whatever you want to save on $SCRATCH
#cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
