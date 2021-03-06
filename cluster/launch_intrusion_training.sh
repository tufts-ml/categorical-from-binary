#!/bin/bash
#
# Usage
# -----
# $ bash cluster/launch_train_intrusion_over_time.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

for cyber_user_idx_relative_to_subset_override in {0..50}
do
    for only_run_this_inference in 1 5
    do 
        args='--path_to_configs configs/intrusion_detection/base.yaml --only_run_this_inference '
        args+=$only_run_this_inference  
        args+=' --cyber_user_idx_relative_to_subset_override '
        args+=$cyber_user_idx_relative_to_subset_override

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
done 