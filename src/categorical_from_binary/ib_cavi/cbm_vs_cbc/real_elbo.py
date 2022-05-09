"""
Unlike the "elbo" modules, this approximates the TRUE elbo (via sampling from the energy)
for the CBC and CBM models, rather than going through IB 
"""

import numpy as np
import scipy.stats

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_category_probs,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.elbo import (
    compute_variational_entropy_of_beta,
)


def estimate_energy_with_one_sample(
    features, labels, link, variational_beta, prior_beta_mean=None, prior_beta_cov=None
) -> float:
    beta_mean, beta_cov = variational_beta.mean, variational_beta.cov
    M, K = np.shape(beta_mean)

    if prior_beta_mean is None:
        prior_beta_mean = np.zeros((M, K))
    if prior_beta_cov is None:
        prior_beta_cov = np.eye(M)

    choices = np.argmax(labels, 1)
    M, K = np.shape(beta_mean)
    beta_sample = np.zeros((M, K))
    for k in range(K):
        if np.ndim(beta_cov) == 3:
            beta_cov_for_this_category = beta_cov[:, :, k]
        elif np.ndim(beta_cov) == 2:
            beta_cov_for_this_category = beta_cov
        else:
            raise ValueError("Not sure how to get beta covariance for one category.")

        beta_sample[:, k] = scipy.stats.multivariate_normal(
            beta_mean[:, k], beta_cov_for_this_category
        ).rvs()

    # compute log_prior
    log_prior = 0
    for k in range(K):
        log_prior += scipy.stats.multivariate_normal(
            prior_beta_mean[:, k], prior_beta_cov
        ).logpdf(beta_sample[:, k])

    probs = construct_category_probs(features, beta_sample, link)
    log_choice_probs = np.log(np.array([probs[i, k] for (i, k) in enumerate(choices)]))
    return log_prior + np.sum(log_choice_probs)


def approximate_true_elbo_with_samples(
    features,
    labels,
    link,
    variational_beta,
    n_monte_carlo_samples,
    prior_beta_mean=None,
    prior_beta_cov=None,
) -> float:
    energy_hat = np.mean(
        [
            estimate_energy_with_one_sample(
                features,
                labels,
                link,
                variational_beta,
                prior_beta_mean,
                prior_beta_cov,
            )
            for s in range(n_monte_carlo_samples)
        ]
    )
    entropy_exact = compute_variational_entropy_of_beta(
        variational_beta.mean, variational_beta.cov
    )
    return energy_hat + entropy_exact
