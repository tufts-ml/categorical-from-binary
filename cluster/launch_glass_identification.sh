#!/bin/bash
#
# Usage
# -----
# $ bash launch_cyber_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'


# WARNING: When I tried to run this slurm file on the Tufts cluster, the program seemed to stop running after a while and not
# write the results anywhere accessible.  Not sure why.  

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < slurm/do_glass_identification.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash slurm/do_glass_identification.slurm
fi