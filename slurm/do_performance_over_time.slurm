#!/usr/bin/env bash
#SBATCH -n 4                # Number of cores
#SBATCH -t 4-23:00          # Runtime in D-HH:MM
#SBATCH -p ccgpu            # Partition to submit to
#SBATCH --mem-per-cpu 40000 # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Not loading the conda environent, because as per Kyle Heuton, shell jobs should inherit
# the environment from the process that launched it.  Hence, `conda activate` shouldn't be necessary

echo "Now running script with args:"
echo $args

python src/categorical_from_binary/performance_over_time/main.py  $args  

# Usage:
#  args='--path_to_configs configs/performance_over_time/demo_sims.yaml --only_run_this_inference 1' bash slurm/do_performance_over_time.slurm
