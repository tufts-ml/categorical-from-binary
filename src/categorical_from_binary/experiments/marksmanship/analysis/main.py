import numpy as np


np.set_printoptions(precision=3, suppress=True)

from matplotlib import pyplot as plt

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.experiments.marksmanship.analysis.covariate_modulation import (
    construct_dataframe_showing_covariate_modulation_for_binary_features,
    print_report_on_binary_features,
    print_report_on_continuous_features,
)
from categorical_from_binary.experiments.marksmanship.preprocess.join import (
    make_marksmanship_data_frame,
)
from categorical_from_binary.experiments.marksmanship.preprocess.util import (
    exclude_based_on_missingness,
    exclude_based_on_zero_std,
)
from categorical_from_binary.experiments.marksmanship.preprocess.xy import (
    add_previous_label_to_covariates,
    add_second_previous_label_to_covariates,
    make_covariates_and_covariate_dict_from_df,
    make_labels_and_label_dict_from_df,
)
from categorical_from_binary.experiments.marksmanship.show.covariate_modulation import (
    plot_response_probabilities_by_covariate_value_using_CAVI_model,
    plot_response_probabilities_by_covariate_value_using_SKLEARN_model,
)
from categorical_from_binary.experiments.marksmanship.show.heatmap import (
    plot_heatmap_for_marksmanship,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_gamma_prior,
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.shrinkage_groups import (
    ShrinkageGroupingStrategy,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.util import beta_stds_from_beta_cov
from categorical_from_binary.selection.core import (
    compute_feature_inclusion_data_frame_using_scaled_neighborhood_method,
)
from categorical_from_binary.selection.hyperparameters import (
    hyperparameters_from_lambda_and_desired_marginal_beta_variance,
)
from categorical_from_binary.sklearn import fit_sklearn_multiclass_logistic_regression


###
# Configs
###

### For dataset construction

# Using autoregressive labels as covariates
index_of_previous_label_to_add_to_covariates = 2
remove_intercept_when_adding_autoregressive_labels = False  # True
use_one_label_as_autoregressive_baseline_label = True  # False
autoregressive_baseline_label = "miss threat"  # None

# Incorporating baseline covariates
threshold_on_proportion_missing_in_columns_for_baseline_data = 0.50
threshold_on_proportion_missing_in_columns_for_merged_data = 0.50
use_baseline_covariates = False


###
# Preprocessing
###

df_all_covariates = make_marksmanship_data_frame(
    use_baseline_covariates,
    threshold_on_proportion_missing_in_columns_for_baseline_data,
)

# handle exclusions
df_all_covariates_without_missing_data = exclude_based_on_missingness(
    df_all_covariates,
    threshold_on_proportion_missing_in_columns_for_merged_data,
)
df = exclude_based_on_zero_std(df_all_covariates_without_missing_data)

(
    labels_before_autoregressive_adjustment,
    label_dict,
) = make_labels_and_label_dict_from_df(df)
(
    covariates_before_autoregressive_adjustment,
    covariate_dict,
) = make_covariates_and_covariate_dict_from_df(df)


if index_of_previous_label_to_add_to_covariates == 0:
    covariates = covariates_before_autoregressive_adjustment
    labels = labels_before_autoregressive_adjustment
elif index_of_previous_label_to_add_to_covariates == 1:
    covariates, covariate_dict, labels = add_previous_label_to_covariates(
        covariates_before_autoregressive_adjustment,
        covariate_dict,
        labels_before_autoregressive_adjustment,
        label_dict,
        remove_intercept_when_adding_autoregressive_labels,
        use_one_label_as_autoregressive_baseline_label,
        autoregressive_baseline_label,
        where_to_put_autoregressive_labels_after="end",
    )
# Providing the option to add 2 steps back and only 2 steps back is a weird
# hack to account for the fact that the soliders were told to shoot at a target twice.
elif index_of_previous_label_to_add_to_covariates == 2:
    covariates, covariate_dict, labels = add_second_previous_label_to_covariates(
        covariates_before_autoregressive_adjustment,
        covariate_dict,
        labels_before_autoregressive_adjustment,
        label_dict,
        remove_intercept_when_adding_autoregressive_labels,
        use_one_label_as_autoregressive_baseline_label,
        autoregressive_baseline_label,
        where_to_put_autoregressive_labels_after="end",
    )
else:
    raise ValueError(
        f"I don't understand  {index_of_previous_label_to_add_to_covariates }, the value provided "
        f"for index_of_previous_label_to_add_to_covariates."
    )

###
# Variational inference
###

# configs
max_n_iterations = 10
use_normal_not_normal_gamma_prior = False
lambda__if_using_normal_gamma_prior = 0.001
variance_if_using_normal_gamma_prior = 1.0
shrinkage_grouping_strategy = (
    ShrinkageGroupingStrategy.FREE_INTERCEPTS_BUT_GROUPS_FOR_OTHER_COVARIATES
)

if use_normal_not_normal_gamma_prior:
    results = compute_multiclass_probit_vi_with_normal_prior(
        labels,
        covariates,
        variational_params_init=None,
        max_n_iterations=max_n_iterations,
    )
else:
    hyperparameters = hyperparameters_from_lambda_and_desired_marginal_beta_variance(
        variance_if_using_normal_gamma_prior, lambda__if_using_normal_gamma_prior
    )

    results = compute_multiclass_probit_vi_with_normal_gamma_prior(
        labels,
        covariates,
        variational_params_init=None,
        max_n_iterations=max_n_iterations,
        hyperparameters=hyperparameters,
        shrinkage_grouping_strategy=shrinkage_grouping_strategy,
    )
variational_params = results.variational_params
beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov


###
# Info about model
###

### print info about model
feature_names = list(covariate_dict.keys())
beta_stds = beta_stds_from_beta_cov(beta_cov)

print(f"Feature names: \n {feature_names}")
print(f"Beta means (one row for each category):\n {beta_mean.T}")
print(f"Beta standard devs: \n {beta_stds}")

### plot model
neighborhood_probability_threshold_for_exclusion = 0.90
neighborhood_radius_in_units_of_std_devs = 3
feature_inclusion_df = (
    compute_feature_inclusion_data_frame_using_scaled_neighborhood_method(
        beta_mean,
        beta_stds,
        neighborhood_probability_threshold_for_exclusion,
        neighborhood_radius_in_units_of_std_devs,
        covariate_dict,
        label_dict,
    )
)

plot_heatmap_for_marksmanship(
    beta_mean,
    feature_inclusion_df,
    white_out_excluded_covariates=True,
    give_each_intercept_term_its_own_heatmap_color=True,
    x_ticks_fontsize=7,
)

### some printouts
print_report_on_binary_features(
    covariate_dict, beta_mean, category_probability_link=Link.CBM_PROBIT
)
print_report_on_continuous_features(
    covariate_dict, beta_mean, category_probability_link=Link.CBM_PROBIT
)

binary_feature_names_to_use = [
    "Platoon C, not A",
    "Session 2, not 1",
    "Session 3, not 1",
]

df = construct_dataframe_showing_covariate_modulation_for_binary_features(
    covariate_dict,
    label_dict,
    beta_mean,
    category_probability_link=Link.CBM_PROBIT,
    binary_feature_names_to_use=binary_feature_names_to_use,
)
print(df.to_latex(float_format="%.3f"))

input("Press any key to continue the program")

binary_feature_names_to_use = [
    "Prev. shoot at non-threat, not miss threat  2 trials back",
    "Prev. hit threat at periphery, not miss threat  2 trials back",
    "Prev. hit threat at center, not miss threat  2 trials back",
]
df = construct_dataframe_showing_covariate_modulation_for_binary_features(
    covariate_dict,
    label_dict,
    beta_mean,
    category_probability_link=Link.CBM_PROBIT,
    binary_feature_names_to_use=binary_feature_names_to_use,
)
print(df.to_latex(float_format="%.3f"))

###
# Plots of covariate modulation
###

### Using VB
category_probability_link = Link.CBM_PROBIT

# TODO: I'm writing out here by hand which variables were selected. I should
# have the code spit this out for me automatically.
feature_names = ["Log aim time", "Absolute rotation"]
for feature_name in feature_names:
    ax = plot_response_probabilities_by_covariate_value_using_CAVI_model(
        covariate_dict, label_dict, beta_mean, feature_name, category_probability_link
    )
    plt.show()

### Using SKLEARN

sklearn_models = [
    fit_sklearn_multiclass_logistic_regression(covariates, labels, penalty="l2"),
    fit_sklearn_multiclass_logistic_regression(
        covariates, labels, penalty="l1", solver="saga"
    ),
]
for sklearn_model in sklearn_models:
    for feature_name in feature_names:
        ax = plot_response_probabilities_by_covariate_value_using_SKLEARN_model(
            covariate_dict,
            label_dict,
            sklearn_model,
            feature_name,
        )
        plt.show()
