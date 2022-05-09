import numpy as np
import pandas as pd

from categorical_from_binary.data_generation.bayes_binary_reg import (
    construct_binary_logistic_probs,
    generate_logistic_regression_dataset,
)
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.ib_cavi.binary_logit_with_pga_and_albert_chib_aug.gibbs import (
    prior_info_from_prior_params,
    sample_from_posterior,
)


###
# Demo -- generate data, do variational inference
###
dataset = generate_logistic_regression_dataset(n_samples=5000, n_features=5, seed=10)
covariates = prepend_features_with_column_of_all_ones_for_intercept(dataset.features)
labels = dataset.labels


###
# Inference -- Gibbs sampling
###

# prior
N, M = np.shape(covariates)
mu_0 = np.zeros(M)
Sigma_0 = np.eye(M)
prior_info = prior_info_from_prior_params(mu_0, Sigma_0)

# initialization
z_init = np.zeros(N)
beta_init = np.zeros(M)
num_MCMC_samples = 200

# sample
beta_mcmc_samples = sample_from_posterior(
    covariates, labels, prior_info, num_MCMC_samples, z_init, beta_init
)

# compute estimated expected posterior mean
n_burn_in = 50
beta_hat = np.mean(beta_mcmc_samples[:, n_burn_in:], 1)

###
# Evaluate performance
###

# We print a table comparing p-hat to the truth
probs_train_mcmc = construct_binary_logistic_probs(beta_hat, covariates)
probs_train_true = construct_binary_logistic_probs(dataset.beta, covariates)
performance = np.vstack((labels, probs_train_mcmc, probs_train_true)).T
df = pd.DataFrame(performance, columns=["labels", "probs_mcmc", "probs_true"])
print(df)
