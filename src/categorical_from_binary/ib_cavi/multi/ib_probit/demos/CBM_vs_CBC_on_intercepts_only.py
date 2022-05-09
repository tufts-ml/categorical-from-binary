"""
Here we compare two estimators (CBC probit and CBM Probit) for the category probabilities 
given beta-hat when 
(a) we use the variational posterior expectation for beta-hat
(b) the dataset is intercept-only (i.e., we're just trying to match empirical frequencies)

RESULTS:
* The CBC estimator has trouble predicting category probabilities
even for very simple data (no covariates) and even after lots of it.   
* The CBM estimator gives great performance 
"""

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
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

###
# Construct dataset
###
n_categories = 3
n_features = 0
n_samples = 5000
n_train_samples = 4000
seed = 1234
include_intercept = True
link_for_generating_data = Link.CBC_PROBIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
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
# Evaluate the goodness of fit: Does the model match the data frequencies in the training set?
###

# We might hope that the model could at least match the empirical probabilities.

np.set_printoptions(precision=3, suppress=True)
print("Results for a simple categorical dataset -- no covariates")
empirical_probs = np.sum(labels_train, 0) / n_train_samples
features_intercept = np.array([1])[:, np.newaxis]
category_probs_CBC_with_posterior_expectation = construct_category_probs(
    features_intercept, beta_mean, Link.CBC_PROBIT
)
category_probs_CBM_with_posterior_expectation = construct_category_probs(
    features_intercept, beta_mean, Link.CBM_PROBIT
)
print(
    f"The empirical probabilities in {n_train_samples} training samples were: {empirical_probs}"
)
print(
    f"The estimated category probabilities, plugging the posterior expectation into the CBM formula, are {np.array(category_probs_CBM_with_posterior_expectation[0])}"
)
print(
    f"The estimated category probabilities, plugging the posterior expectation into the CBC formula, are {np.array(category_probs_CBC_with_posterior_expectation[0])}"
)


"""
n_categories = 3
n_features = 0
n_samples = 5000
n_train_samples = 4000

The empirical probabilities in 4000 training samples were: [0.021 0.774 0.205]
The estimated category probabilities, plugging the posterior expectation into the CBM formula, are [0.022 0.774 0.205]
The estimated category probabilities, plugging the posterior expectation into the CBC formula, are [0.006 0.924 0.07 ]
"""


###
# Evaluate the goodness of fit: Does the model assign high probability to the actual observations
###

covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]

split_dataset = SplitDataset(
    covariates_train, labels_train, covariates_test, labels_test
)
take_measurements_comparing_CBM_and_CBC_estimators(
    split_dataset, beta_mean, link_for_generating_data, beta_ground_truth=dataset.beta
)


"""

Results on sample run (seed 1234):

Now running evaluations on train data
The Metric.MEAN_LOG_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBC_PROBIT is -0.586
The Metric.MEAN_LOG_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBM_PROBIT is -0.456
The Metric.MEAN_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBC_PROBIT is 0.826
The Metric.MEAN_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBM_PROBIT is 0.74



Now running evaluations on test data
The Metric.MEAN_LOG_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBC_PROBIT is -0.618
The Metric.MEAN_LOG_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBM_PROBIT is -0.476
The Metric.MEAN_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBC_PROBIT is 0.82
The Metric.MEAN_LIKELIHOOD from plugging the variational posterior mean into the category probabilities of Link.CBM_PROBIT is 0.735
"""
