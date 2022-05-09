from categorical_from_binary.experiments.compare_to_mle.main.run_comparison_to_mle import (
    run_CAVI_vs_MLE_simulations,
)
from categorical_from_binary.experiments.compare_to_mle.results_collector import (
    make_df_from_directory_taking_one_file_for_each_seed,
)


seed = 0
list_of_n_categories = [3]
multipliers_on_n_categories_to_create_n_covariates = [2]
list_of_scales_for_predictive_categories = [10]
n_datasets_per_data_dimension = 1
multipliers_on_n_parameters_to_create_n_samples = [1, 10]
ib_model_as_string = "probit"
data_generating_link_as_string = "multi_logit"
convergence_criterion_drop_in_mean_elbo = 0.1
results_dir = "./tmp"
test_run = True

run_CAVI_vs_MLE_simulations(
    seed,
    list_of_n_categories,
    multipliers_on_n_categories_to_create_n_covariates,
    multipliers_on_n_parameters_to_create_n_samples,
    list_of_scales_for_predictive_categories,
    ib_model_as_string,
    data_generating_link_as_string,
    convergence_criterion_drop_in_mean_elbo,
    results_dir,
    test_run,
)

df = make_df_from_directory_taking_one_file_for_each_seed(
    results_dir, prefix_for_filenames_of_interest="sim_results_CAVI_vs_MLE_"
)
print(f"{df}")

# TODO maybe, show example plots in the demp.
