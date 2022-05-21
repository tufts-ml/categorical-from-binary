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


export path_to_configs='configs/intrusion_detection/base.yaml'
export scoring_results_dir='/cluster/tufts/hugheslab/mwojno01/data/results/intrusion/'
export cavi_time_units=1
export advi_time_units=10

for cyber_user_idx_relative_to_subset_override in {0..50}
do
    export cyber_user_idx_relative_to_subset_override=$cyber_user_idx_relative_to_subset_override

    ## Use this line to see where you are in the loop
    echo "path_to_configs=$path_to_configs cyber_user_idx_relative_to_subset_override=$cyber_user_idx_relative_to_subset_override scoring_results_dir=$scoring_results_dir cavi_time_units=$cavi_time_units advi_time_units=$advi_time_units "
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < slurm/do_intrusion_scoring.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash slurm/do_intrusion_scoring.slurm
    fi

done 