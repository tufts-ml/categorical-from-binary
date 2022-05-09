import pandas as pd

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.deprecated.advi_via_bambi.helpers import (
    create_bambi_softmax_model,
    force_bambi_model_to_have_standard_normal_priors,
    get_beta_mean_from_variational_approximation,
    get_fitted_advi_model_and_intermediate_results,
)
from categorical_from_binary.deprecated.advi_via_bambi.performance_over_time import (
    construct_holdout_performance_over_time_for_legacy_ADVI_via_bambi,
)
from categorical_from_binary.ib_cavi.cbm_vs_cbc.bma import (
    compute_weight_on_CBC_from_bayesian_model_averaging,
    construct_category_probabilities_from_bayesian_model_averaging,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_ib_cavi_with_normal_prior,
)
from categorical_from_binary.metrics import (
    append_metrics_dict_for_one_dataset_to_results_dict,
    compute_metrics,
)
from categorical_from_binary.timing import time_me


###
# Initialize outputs
###
metrics_dict = {}


###
# Construct dataset
###
n_categories = 3  # 30
n_features = 6  # 60
n_samples = 1800  # 18000
scale_for_predictive_categories = 2.0  # 20.0
include_intercept = True
link_for_data_generation = Link.MULTI_LOGIT
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=scale_for_predictive_categories
)
seed = 1
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link_for_data_generation,
    seed=seed,
    include_intercept=include_intercept,
    beta_category_strategy=beta_category_strategy,
)


# Prep training / test split
n_train_samples = int(0.8 * n_samples)
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]


###
# ADVI: Inference, category probs, metrics
###
n_advi_iterations = 1000
n_mc_samples_for_advi_gradient_estimation = 100
init_advi = "advi"
n_init_advi = 100
random_seed_advi = 1
method_as_string = "advi"
model = create_bambi_softmax_model(covariates_train, labels_train)
model = force_bambi_model_to_have_standard_normal_priors(model)

(model_fitted_by_ADVI, intermediate_results), time_for_ADVI = time_me(
    get_fitted_advi_model_and_intermediate_results
)(
    model,
    method_as_string=method_as_string,
    n_its=n_advi_iterations,
    init=init_advi,
    n_init=n_init_advi,
    n_mc_samples_for_gradient_estimation=n_mc_samples_for_advi_gradient_estimation,
    random_seed=random_seed_advi,
)
beta_mean_ADVI = get_beta_mean_from_variational_approximation(
    model_fitted_by_ADVI,
    labels_train,
)
probs_test_ADVI = construct_category_probs(
    covariates_test,
    beta_mean_ADVI,
    Link.MULTI_LOGIT_NON_IDENTIFIED,
)
metrics_dict["multi_logit_ADVI"] = compute_metrics(
    probs_test_ADVI, labels_test, min_allowable_prob=None
)

holdout_performance_over_time_for_ADVI = (
    construct_holdout_performance_over_time_for_legacy_ADVI_via_bambi(
        intermediate_results,
        model_fitted_by_ADVI,
        time_for_ADVI,
        labels_train,
        covariates_test,
        labels_test,
        Link.MULTI_LOGIT_NON_IDENTIFIED,
    )
)


###
# Data generating process
###
probs_test_true = construct_category_probs(
    covariates_test,
    dataset.beta,
    link_for_data_generation,
)
metrics_dict["dgp"] = compute_metrics(
    probs_test_true, labels_test, min_allowable_prob=None
)


###
# Inference: CAVI
###

ib_model = IB_Model.LOGIT
# convergence_criterion_drop_in_mean_elbo = 10.0
max_n_iterations = 1000
results_CAVI, time_for_CAVI = time_me(compute_ib_cavi_with_normal_prior)(
    ib_model,
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    # convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
    max_n_iterations=max_n_iterations,
)
variational_beta = results_CAVI.variational_params.beta
n_monte_carlo_samples = 10
CBC_weight = compute_weight_on_CBC_from_bayesian_model_averaging(
    covariates_train,
    labels_train,
    variational_beta,
    n_monte_carlo_samples,
    ib_model,
)
probs_test_IB_CAVI_with_BMA = (
    construct_category_probabilities_from_bayesian_model_averaging(
        covariates_test,
        variational_beta.mean,
        CBC_weight,
        ib_model,
    )
)
metrics_dict["IB_CAVI_plus_BMA"] = compute_metrics(
    probs_test_IB_CAVI_with_BMA, labels_test, min_allowable_prob=None
)


# we set the betas for the 0th cat prob to zero (i.e. we have identified the beta externally), so we can now apply the non-identified link.
# we do this instead of callign the identified multi logit because it seems pymc3 identifies by setting the FIRST Beta column to be 0,
# whereas our link IIRC identifies by setting the LAST beta column to 0.


###
# Form results dict from performnace metrics and time info
###
results_dict = append_metrics_dict_for_one_dataset_to_results_dict(metrics_dict)

# append time info to results dict
results_dict["time_for_ADVI"] = time_for_ADVI
results_dict["time_for_CAVI"] = time_for_CAVI

results_df = pd.DataFrame(results_dict)

pd.set_option("display.max_columns", None)
print(f"Some metrics on the inference: \n {results_df}")
input("Press any key to continue.")


###
# Compare holdout performance over tiem
###
pd.set_option("display.max_rows", None)

print(
    f"CAVI holdout performance over time: \n {results_CAVI.holdout_performance_over_time}"
)
print(
    f"ADVI holdout performance over time: \n {holdout_performance_over_time_for_ADVI}"
)
