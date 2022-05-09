#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-20:00          # Runtime in D-HH:MM
#SBATCH -p ccgpu            # Partition to submit to
#SBATCH --mem-per-cpu 10000 # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Not loading the conda environent, because as per Kyle Heuton, shell jobs should inherit
# the environment from the process that launched it.  Hence, `conda activate` shouldn't be necessary

# WARNING: When I tried to run this slurm file on the Tufts cluster, the program seemed to stop running after a while and not
# write the results anywhere accessible.  Not sure why.  

python src/categorical_from_binary/experiments/glass_identification/main/run_many_inference_strategies.py
