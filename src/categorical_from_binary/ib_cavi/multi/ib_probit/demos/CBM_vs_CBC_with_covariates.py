"""
Here we compare two estimators (CBC probit and CBM Probit) for the category probabilities 
given beta-hat when 
(a) we use the variational posterior expectation for beta-hat
(b) covariates are present
(c) the data generation process uses various links.
"""

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.data_generation.splitter import SplitDataset
from categorical_from_binary.evaluate.multiclass import (
    take_measurements_comparing_CBM_and_CBC_estimators,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


# TODO: Convergence criterion should be based on mean ELBO not ELBO!
# For example, a drop of 1.0 in the ELBO would be HUGE for a small dataset but MINISCULE
# for a large dataset.


link_for_generating_data = Link.MULTI_LOGIT
# Possible Link functions to use:  Link.MULTI_LOGIT, Link.CBC_PROBIT, Link.CBM_Probit

###
# Construct dataset
###
seed = 2
n_categories = 3
n_features = 3
n_samples = 5000
n_train_samples = 4000
include_intercept = True
link_for_generating_data = link_for_generating_data
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    beta_0=None,
    link=link_for_generating_data,
    seed=seed,
    include_intercept=include_intercept,
)


# Prep training data
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
# `labels_train` gives one-hot encoded representation of category

####
# Variational Inference
####
convergence_criterion_drop_in_mean_elbo = 0.001
results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    variational_params_init=None,
    convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
)
variational_params = results.variational_params
beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov


###
# Evaluate the model quality
###

covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

split_dataset = SplitDataset(
    covariates_train, labels_train, covariates_test, labels_test
)
results = take_measurements_comparing_CBM_and_CBC_estimators(
    split_dataset, beta_mean, link_for_generating_data, beta_ground_truth=dataset.beta
)
