"""
We explore IB+CBM vs IB+CBC (with a probit link function)
when there are covariates.  

In particular, we:
1) Investigate the relationship between prediction error and various normalizing constants.
2) Investigate whether a per-sample combination rule for choosing between IB+CBM and IB+CBC
reduces approximation error and error to ground truth. 
"""
from collections import defaultdict

import numpy as np


np.set_printoptions(precision=3, suppress=True)

import pandas as pd
from matplotlib import pyplot as plt

from categorical_from_binary.autodiff.jax_helpers import (
    compute_CBC_Probit_predictions,
    compute_CBM_Probit_predictions,
    compute_IB_Probit_predictions,
    optimize_beta_for_CBC_model,
    optimize_beta_for_CBM_model,
    optimize_beta_for_IB_model,
)
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.approximation_error import (
    compute_approximation_error_in_category_probs_using_l1_distance,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.combinations import (
    CombinationRule,
    construct_combined_category_probabilities,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.normalizing_constants import (
    compute_log_Cs_CBC_probit,
    compute_log_Cs_CBM_probit,
    compute_log_sum_of_psis,
    compute_sum_of_IB_probs,
)
from categorical_from_binary.plot_helpers import (
    create_hot_pairplot,
    make_linear_regression_plot,
)


###
# Configs
###

n_sims = 10

# configs related to data generation
scale_for_predictive_categories = 2.0
n_categories = 3
n_features = 6
n_samples = 1800
beta_0 = None
include_intercept = True
link = Link.MULTI_LOGIT
dict_of_error_before_and_after_averaging = defaultdict(list)
results_by_dataset = []

# configs related to combining IB+CBC and IB+CBM
combination_rule = (
    CombinationRule.WEIGH_IB_PLUS_CBC_VIA_PROBABILITY_THAT_IB_IS_IN_CBC_REGION_BUT_OVERRIDE_TO_ONLY_USE_IB_PLUS_CBM_WHEN_IB_ASSIGNS_LESS_THAN_HALF_OF_ITS_MASS_TO_THE_CBC_SET
)

for seed in range(n_sims):
    print(f"---Now running simulation {seed+1}/{n_sims}---")

    ###
    # Construct dataset
    ###
    beta_category_strategy = ControlCategoryPredictability(
        scale_for_predictive_categories=scale_for_predictive_categories
    )
    dataset = generate_multiclass_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        beta_0=beta_0,
        link=link,
        seed=seed,
        include_intercept=include_intercept,
        beta_category_strategy=beta_category_strategy,
    )

    ###
    # Compute MLE's
    ###

    beta_init = np.zeros((n_features + 1, n_categories))

    print("Now doing MLE with IB model")
    beta_star_IB_MLE = optimize_beta_for_IB_model(
        beta_init, dataset.features, dataset.labels
    )

    print("Now doing MLE with CBC model")
    beta_star_CBC_MLE = optimize_beta_for_CBC_model(
        beta_init, dataset.features, dataset.labels
    )

    print("Now doing MLE with CBM model")
    beta_star_CBM_MLE = optimize_beta_for_CBM_model(
        beta_init, dataset.features, dataset.labels
    )

    ###
    # Compute predictions
    ###

    CBC_predictions_with_IB_MLE = compute_CBC_Probit_predictions(
        beta_star_IB_MLE, dataset.features, return_numpy=True
    )
    CBC_predictions_with_CBC_MLE = compute_CBC_Probit_predictions(
        beta_star_CBC_MLE, dataset.features, return_numpy=True
    )

    CBM_predictions_with_IB_MLE = compute_CBM_Probit_predictions(
        beta_star_IB_MLE, dataset.features, return_numpy=True
    )
    CBM_predictions_with_CBM_MLE = compute_CBM_Probit_predictions(
        beta_star_CBM_MLE, dataset.features, return_numpy=True
    )

    IB_predictions_with_IB_MLE = compute_IB_Probit_predictions(
        beta_star_IB_MLE, dataset.features
    )

    true_data_generating_probabilities = construct_category_probs(
        dataset.features, dataset.beta, link
    )

    ###
    # Compute error in predictions
    ###

    # Error is computed relative to the predictions one would have obtained if one had used
    # a proper MLE for the CBC vs CBM model, instead of swapping in the IB MLE.
    error_in_IB_plus_CBC = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBC_predictions_with_CBC_MLE, CBC_predictions_with_IB_MLE
        )
    )
    error_in_IB_plus_CBM = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBM_predictions_with_CBM_MLE, CBM_predictions_with_IB_MLE
        )
    )

    ###
    # Compute normalizing constants (which MIGHT predict error)
    ###

    # Compute the normalizing constants of the marginal likelihood (i.e. the category probabilities)

    # Quantities related to the MLE constraints.  the sum of IB probs is the the normalizing constant for the
    # CBM's MARGINAL data likelihood.  The sum of psis is the normalizing constant for the alternate form
    # of the CBC's MARGINAL data likelihood (otherwise, the normalizing constant for the CBC marginal and
    # complete data likelihoods are identical.)
    sum_of_IB_probs = compute_sum_of_IB_probs(dataset.features, beta_star_IB_MLE)
    log_sum_of_psis = compute_log_sum_of_psis(dataset.features, beta_star_IB_MLE)

    # Compute the normalizing constants of the complete data likelihoods.
    # These might be related to the error because they are the term we drop when
    # going from the true model (CBM or CBC) to the surrogate model (IB)

    log_Cs_CBC = compute_log_Cs_CBC_probit(dataset.features, beta_star_IB_MLE)
    log_Cs_CBM = compute_log_Cs_CBM_probit(
        dataset.features, beta_star_IB_MLE, dataset.labels
    )

    ###
    # Investigate relationships between error and normalizing constants
    ###

    df_normalizing_constants_and_approximation_errors = pd.DataFrame(
        {
            "error_in_IB_plus_CBM": error_in_IB_plus_CBM,
            "error_in_IB_plus_CBC": error_in_IB_plus_CBC,
            "sum_of_IB_probs": sum_of_IB_probs,
            "log_sum_of_psis": log_sum_of_psis,
            "log_Cs_CBC ": log_Cs_CBC,
            "log_Cs_CBM ": log_Cs_CBM,
        }
    )

    # Show correlations
    pd.set_option("display.max_columns", None)
    correlations = df_normalizing_constants_and_approximation_errors.corr()
    print(correlations)
    results_by_dataset.append(df_normalizing_constants_and_approximation_errors)

    ###
    # Try to combine predictions
    ###

    combined_predictions_with_IB_MLE = construct_combined_category_probabilities(
        CBC_predictions_with_IB_MLE,
        CBM_predictions_with_IB_MLE,
        log_Cs_CBC,
        sum_of_IB_probs,
        combination_rule,
    )

    # compute error
    error_in_combined_approach_to_CBC = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBC_predictions_with_CBC_MLE, combined_predictions_with_IB_MLE
        )
    )
    error_in_combined_approach_to_CBM = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBM_predictions_with_CBM_MLE, combined_predictions_with_IB_MLE
        )
    )
    error_in_IB_plus_CBC_to_true = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBC_predictions_with_IB_MLE, true_data_generating_probabilities
        )
    )
    error_in_IB_plus_CBM_to_true = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBM_predictions_with_IB_MLE, true_data_generating_probabilities
        )
    )
    error_in_combined_approach_to_true = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            combined_predictions_with_IB_MLE, true_data_generating_probabilities
        )
    )
    error_in_CBC_to_true = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBC_predictions_with_CBC_MLE, true_data_generating_probabilities
        )
    )
    error_in_CBM_to_true = (
        compute_approximation_error_in_category_probs_using_l1_distance(
            CBM_predictions_with_CBM_MLE, true_data_generating_probabilities
        )
    )

    # did the error go down?
    mean_error_in_IB_plus_CBC = np.mean(error_in_IB_plus_CBC)
    mean_error_in_IB_plus_CBM = np.mean(error_in_IB_plus_CBM)
    mean_error_in_combined_approach_to_CBC = np.mean(error_in_combined_approach_to_CBC)
    mean_error_in_combined_approach_to_CBM = np.mean(error_in_combined_approach_to_CBM)
    mean_error_in_combined_approach_to_true = np.mean(
        error_in_combined_approach_to_true
    )
    mean_error_in_IB_plus_CBC_to_true = np.mean(error_in_IB_plus_CBC_to_true)
    mean_error_in_IB_plus_CBM_to_true = np.mean(error_in_IB_plus_CBM_to_true)
    mean_error_in_CBC_to_true = np.mean(error_in_CBC_to_true)
    mean_error_in_CBM_to_true = np.mean(error_in_CBM_to_true)

    dict_of_error_before_and_after_averaging["mean_error_in_IB_plus_CBC"].append(
        mean_error_in_IB_plus_CBC
    )
    dict_of_error_before_and_after_averaging[
        "mean_error_in_combined_approach_to_CBC"
    ].append(mean_error_in_combined_approach_to_CBC)
    dict_of_error_before_and_after_averaging["mean_error_in_IB_plus_CBM"].append(
        mean_error_in_IB_plus_CBM
    )
    dict_of_error_before_and_after_averaging[
        "mean_error_in_combined_approach_to_CBM"
    ].append(mean_error_in_combined_approach_to_CBM)
    dict_of_error_before_and_after_averaging[
        "mean_error_in_combined_approach_to_true"
    ].append(mean_error_in_combined_approach_to_true)
    dict_of_error_before_and_after_averaging[
        "mean_error_in_IB_plus_CBC_to_true"
    ].append(mean_error_in_IB_plus_CBC_to_true)
    dict_of_error_before_and_after_averaging[
        "mean_error_in_IB_plus_CBM_to_true"
    ].append(mean_error_in_IB_plus_CBM_to_true)
    dict_of_error_before_and_after_averaging["mean_error_in_CBC_to_true"].append(
        mean_error_in_CBC_to_true
    )
    dict_of_error_before_and_after_averaging["mean_error_in_CBM_to_true"].append(
        mean_error_in_CBM_to_true
    )


error_before_and_after_averaging = pd.DataFrame(
    dict_of_error_before_and_after_averaging
)
error_before_and_after_averaging

###
# Plots
###


### First check if combining the predictions helped.

# Show regression of <error from combined method> on <error from single method>

ax = make_linear_regression_plot(
    error_before_and_after_averaging,
    x_col_name="mean_error_in_IB_plus_CBC_to_true",
    y_col_name="mean_error_in_combined_approach_to_true",
    reg_line_label="combined model",
    x_axis_label="mean error in IB plus CBC to true ",
    y_axis_label="mean error in combined method to true",
    title="Mean approximation error to true by dataset",
    add_y_equals_x_line=True,
)
plt.show()

ax = make_linear_regression_plot(
    error_before_and_after_averaging,
    x_col_name="mean_error_in_IB_plus_CBM_to_true",
    y_col_name="mean_error_in_combined_approach_to_true",
    reg_line_label="combined model",
    x_axis_label="mean error in IB plus CBM to true ",
    y_axis_label="mean error in combined method to true",
    title="Mean approximation error to true by dataset",
    add_y_equals_x_line=True,
)
plt.show()

ax = make_linear_regression_plot(
    error_before_and_after_averaging,
    x_col_name="mean_error_in_IB_plus_CBC",
    y_col_name="mean_error_in_combined_approach_to_CBC",
    reg_line_label="combined model",
    x_axis_label="mean error in IB plus CBC",
    y_axis_label="mean error in combined method",
    title="Mean approximation error to CBC by dataset",
    add_y_equals_x_line=True,
)
plt.show()


ax = make_linear_regression_plot(
    error_before_and_after_averaging,
    x_col_name="mean_error_in_IB_plus_CBM",
    y_col_name="mean_error_in_combined_approach_to_CBM",
    reg_line_label="combined model",
    x_axis_label="mean error in IB plus CBM",
    y_axis_label="mean error in combined method",
    title="Mean approximation error to CBM by dataset",
    add_y_equals_x_line=True,
)
plt.show()


### Plot one correlation matrix
df_normalizing_constants_and_approximation_errors = results_by_dataset[0]
g = create_hot_pairplot(df_normalizing_constants_and_approximation_errors)
plt.show()
