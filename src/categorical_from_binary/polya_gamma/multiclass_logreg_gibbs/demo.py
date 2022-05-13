"""
This demos a Gibbs sampler for sampling from a multinomial logit model
with polya gamma augmentation. 
"""

import numpy as np


np.set_printoptions(precision=3, suppress=True)

import pandas as pd


pd.set_option("display.max_columns", None)

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    ControlCategoryPredictability,
    Link,
    construct_category_probs,
    generate_multiclass_regression_dataset,
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
from categorical_from_binary.performance_over_time.for_mcmc import (
    construct_performance_over_time_for_MCMC,
)
from categorical_from_binary.polya_gamma.multiclass_logreg_gibbs.inference import (
    sample_from_posterior_of_multiclass_logistic_regression_with_pga,
)
from categorical_from_binary.timing import time_me


###
# Construct dataset
###
n_categories = 20
n_features = 40
n_samples = 8000  # 800
sigma_high = 2.0
include_intercept = True
link_for_data_generation = Link.SOFTMAX
beta_category_strategy = ControlCategoryPredictability(
    scale_for_predictive_categories=sigma_high
)
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link_for_data_generation,
    seed=None,
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
# Inference
###


# CBC Gibbs sampling on MNL with PGA
num_MCMC_samples = 100
beta_samples, time_for_gibbs = time_me(
    sample_from_posterior_of_multiclass_logistic_regression_with_pga
)(covariates_train, labels_train, num_MCMC_samples, prior_info=None, beta_init=None)


# IB-Logit
ib_model = IB_Model.LOGIT
convergence_criterion_drop_in_mean_elbo = 0.1
results_CAVI, time_for_CAVI = time_me(compute_ib_cavi_with_normal_prior)(
    ib_model,
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
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


###
# Category Probs
###
num_burn_in = int(0.2 * num_MCMC_samples)
beta_posterior_mean_from_gibbs = np.mean(beta_samples[num_burn_in:], 0)
probs_test_multi_logit_pga_gibbs = construct_category_probs(
    covariates_test,
    beta_posterior_mean_from_gibbs,
    Link.SOFTMAX,
)
probs_test_true = construct_category_probs(
    covariates_test,
    dataset.beta,
    link_for_data_generation,
)
probs_test_IB_CAVI_with_BMA = (
    construct_category_probabilities_from_bayesian_model_averaging(
        covariates_test,
        variational_beta.mean,
        CBC_weight,
        ib_model,
    )
)

metrics_dict = {}
metrics_dict["dgp"] = compute_metrics(
    probs_test_true, labels_test, min_allowable_prob=None
)
metrics_dict["multi_logit_pga_gibbs"] = compute_metrics(
    probs_test_multi_logit_pga_gibbs, labels_test, min_allowable_prob=None
)
metrics_dict["IB_CAVI_plus_BMA"] = compute_metrics(
    probs_test_IB_CAVI_with_BMA, labels_test, min_allowable_prob=None
)
results_dict = append_metrics_dict_for_one_dataset_to_results_dict(metrics_dict)

# append time info to results dict
results_dict["time_for_gibbs"] = time_for_gibbs
results_dict["time_for_CAVI"] = time_for_CAVI
results_df = pd.DataFrame(results_dict)

print(f"Some metrics on the inference: \n {results_df}")
input("Press any key to continue.")

###
# Likelihood over time
###

# for gibbs, we're currently just plotting all of the samples.
# we could perhaps cut out the first x% samples if we wanted to.
num_burn_in = 0
stride_for_evaluating_holdout_performance = num_MCMC_samples / 5
holdout_performance_over_time_gibbs = construct_performance_over_time_for_MCMC(
    beta_samples,
    time_for_gibbs,
    covariates_train,
    labels_train,
    covariates_test,
    labels_test,
    Link.SOFTMAX,
    stride=stride_for_evaluating_holdout_performance,
    n_warmup_samples=num_burn_in,
)
print(
    f"\n\nGibbs holdout performance over time: \n {holdout_performance_over_time_gibbs }"
)


# for comparison
pd.set_option("display.max_columns", None)
print(
    f"\nIB-CAVI performance over time: \n {results_CAVI.holdout_performance_over_time}"
)
# can probably just pick the best model and use that as an "over time" proxy so long as the BMA
# weights select the best model at the end.

"""
Review:

I see some smallish benefits of CAVI at:

n_categories = 20
n_features = 40
n_samples = 800
sigma_high = 2.0
ib_model = IB_Model.LOGIT

But Gibbs is pretty dang good.  When the number of samples N is high (say, increase N to 800),
Gibbs actually seems to be faster?!!  It might be an implementational thing (e.g. the polya 
gamma sampler is implemented in C and is lightning fast).   And/or perhaps we need
SVI or memoized VI to see an actual benefit of VI.  Otherwise, we just have to resort to saying that
sometimes VI is the desired approach (e.g. composing PGM's with NN's).
"""
