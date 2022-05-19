#!/bin/bash
#
# Usage
# -----
# $ bash launch_MLE_comparison.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi



for seed in 0 1 2 3 4 5 6 7 8 9
do
    args='--list_of_n_categories 30 
    --multipliers_on_n_categories_to_create_n_covariates 2
    --multipliers_on_n_parameters_to_create_n_samples 1 100
    --list_of_scales_for_predictive_categories 0.01 .5 1 2 5 10 20 50 100
    --ib_model_as_string probit
    --data_generating_link_as_string multi_logit 
    --convergence_criterion_drop_in_mean_elbo 0.1
    --results_dir /cluster/tufts/hugheslab/mwojno01/data/results/sims/probit_fixed/
    --seed '
    args+=$seed  

    export args=$args

    ## Use this line to see where you are in the loop
    echo "args=$args"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < slurm/do_MLE_comparison.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash slurm/do_MLE_comparison.slurm
    fi

done