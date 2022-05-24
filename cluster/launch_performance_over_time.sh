#!/bin/bash
#
# Usage
# -----
# $ bash launch_performance_over_time.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

# The script below runs ALL inference methods for some given configs. 
#
# If I just want to run a SINGLE inference method on the cluster for some given configs,
# I can do something like this:
#
# args='--path_to_configs configs/performance_over_time/demo_sims.yaml --only_run_this_inference 3'
# export args=$args
# sbatch < slurm/do_performance_over_time.slurm


for only_run_this_inference in 1 2 3 4 5
do
    args='--path_to_configs configs/performance_over_time/demo_sims.yaml
    --only_run_this_inference '
    args+=$only_run_this_inference  

    export args=$args

    ## Use this line to see where you are in the loop
    echo "args=$args"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < slurm/do_performance_over_time.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash slurm/do_performance_over_time.slurm
    fi

done