#!/bin/bash
#
# Usage
# -----
# $ bash launch_cyber_training.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export max_n_iterations=100 
export number_of_users_in_subset=50
export start_idx_for_users_in_subset=300 

for user_idx_relative_to_subset in {0..50}
do
    export user_idx_relative_to_subset=$user_idx_relative_to_subset

    ## Use this line to see where you are in the loop
    echo "user_idx_relative_to_subset=$user_idx_relative_to_subset number_of_users_in_subset=$number_of_users_in_subset start_idx_for_users_in_subset=$start_idx_for_users_in_subset max_n_iterations=$max_n_iterations "

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < slurm/do_cyber_training.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash slurm/do_cyber_training.slurm
    fi

done