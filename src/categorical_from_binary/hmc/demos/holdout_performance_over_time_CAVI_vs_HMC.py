import numpy as np
import pandas as pd

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.hmc.core import (
    create_categorical_model,
    run_nuts_on_categorical_data,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.performance_over_time.for_mcmc import (
    construct_performance_over_time_for_MCMC,
)
from categorical_from_binary.timing import time_me


###
# Construct dataset
###
n_categories = 3
n_sparse_categories = 0
n_features = 1
n_samples = 1000
n_train_samples = 900
include_intercept = True
link = Link.MULTI_LOGIT
seed = 0
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    n_categories_where_all_beta_coefficients_are_sparse=n_sparse_categories,
    beta_0=None,
    link=link,
    seed=seed,
    include_intercept=include_intercept,
)


# Prep training / test split
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

####
# Variational Inference
####
results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=20,
    # convergence_criterion_drop_in_mean_elbo=0.01,
)
performance_over_time_CAVI = results.performance_over_time


####
# Hamiltonian Monte Carlo
####

num_warmup, num_mcmc_samples = 300, 1000
stride_for_evaluating_holdout_performance = num_mcmc_samples / 20

Nseen_list = [n_train_samples]
link = Link.CBC_PROBIT
beta_samples_HMC_dict, time_for_HMC = time_me(run_nuts_on_categorical_data)(
    num_warmup,
    num_mcmc_samples,
    Nseen_list,
    create_categorical_model,
    link,
    labels_train,
    covariates_train,
    random_seed=0,
)


beta_samples_HMC = np.array(beta_samples_HMC_dict[n_train_samples])
holdout_performance_over_time_HMC = construct_performance_over_time_for_MCMC(
    beta_samples_HMC,
    time_for_HMC,
    covariates_train,
    labels_train,
    covariates_test,
    labels_test,
    Link.CBC_PROBIT,
    stride=stride_for_evaluating_holdout_performance,
    n_warmup_samples=num_warmup,
    one_beta_sample_has_transposed_orientation=True,
)

###
# Show results
###

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)
print(f"\n\nHMC holdout performance over time: \n {holdout_performance_over_time_HMC }")
print(f"\n\nCAVI performance over time: \n {performance_over_time_CAVI }")
