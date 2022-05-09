#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-23:00          # Runtime in D-HH:MM
#SBATCH -p ccgpu            # Partition to submit to
#SBATCH --mem-per-cpu 30000 # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Not loading the conda environent, because as per Kyle Heuton, shell jobs should inherit
# the environment from the process that launched it.  Hence, `conda activate` shouldn't be necessary

echo "Now running script with args:"
echo $args

python src/categorical_from_binary/experiments/sims/main/run_CAVI_vs_MLE_simulations.py  $args  

### Examples of how to call
# 1) Test run:
#  args='--seed 0 --test_run true' bash slurm/do_sims.slurm

# 2) More extensive test run 
# args='--seed 0 
#                 --list_of_n_categories 3 10 
#                 --multipliers_on_n_categories_to_create_n_covariates 1
#                 --multipliers_on_n_parameters_to_create_n_samples 10 
#                 --list_of_scales_for_predictive_categories 0.1
#                 --ib_model_as_string probit
#                 --data_generating_link_as_string multi_logit 
#                 --convergence_criterion_drop_in_mean_elbo 0.1 
#                 --results_dir ./tmp '  bash slurm/do_sims.slurm

# 3) A real run (the default results_dir works on the cluster)
# args='--seed 0 
#                 --list_of_n_categories 3 10 
#                 --multipliers_on_n_categories_to_create_n_covariates 1 2
#                 --multipliers_on_n_parameters_to_create_n_samples 10 100
#                 --list_of_scales_for_predictive_categories 0.1 2.0
#                 --ib_model_as_string probit
#                 --data_generating_link_as_string multi_logit 
#                 --convergence_criterion_drop_in_mean_elbo 0.1 '  bash slurm/do_sims.slurm



