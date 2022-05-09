from typing import List, NamedTuple

import numpy as np
import scipy

from categorical_from_binary.data_generation.hierarchical_logreg import (
    HierarchicalLogisticRegressionDataset,
)
from categorical_from_binary.laplace_vi.bayes_logreg.inference import (
    get_beta_covariance,
    optimize_beta_for_bayesian_logreg_using_laplace_vi,
)


class GroupParams(NamedTuple):
    """
    The mean and covariance of the beta for a given group

    Called mu_m and Sigma_m in the paper

    See pp.1019 of Wang and Blei (2013), JMLR.
    """

    mean: np.ndarray
    cov: np.ndarray


class GlobalParams(NamedTuple):
    """
    The mean and covariance of the distributon of betas across groups.

    Called mu_0 and Sigma_0 in the paper

    See pp.1019 of Wang and Blei (2013), JMLR.
    """

    mean: np.ndarray
    cov: np.ndarray


class VariationalParams(NamedTuple):
    global_: GlobalParams
    local: List[GroupParams]


class GlobalHyperparams(NamedTuple):
    """
    The hyperparams governing the prior on the mean and covariance of the
    distribution of betas across groups

    See pp.1019 of Wang and Blei (2013), JMLR.
    """

    # TODO: Use better names...
    nu: float  # first param (dof) for the Wishart prior on global precision, inv(cov_0)
    phi_0: np.ndarray  # second param for the Wishart prior on global precision inv(cov_0)
    phi_1: np.ndarray  # cov for Normal prior on global mean (mu_0)


def update_global_parameters(
    beta_means: np.ndarray,
    global_params: GlobalParams,
    global_hyperparams: GlobalHyperparams,
    n_iterations: int,
) -> GlobalParams:
    """
    Arguments:
        beta_means:  has shape (n_groups, n_features + 1)

    """
    n_groups, beta_dim = np.shape(beta_means)
    global_mean_precision_hyperparam = scipy.linalg.inv(
        global_hyperparams.phi_1
    )  # precision for Normal prior on global mean (mu_0)

    # TODO: run this until tolerance is reached, instead of hard-coding a number of iterations
    for i in range(n_iterations):
        mu_0 = scipy.linalg.inv(
            (global_params.cov @ global_mean_precision_hyperparam / n_groups)
            + np.eye(beta_dim)
        ) @ np.mean(beta_means, 0)
        sum_of_outer_products = np.zeros((beta_dim, beta_dim))
        for group in range(n_groups):
            sum_of_outer_products += np.outer(
                beta_means[group] - mu_0, beta_means[group] - mu_0
            )

        Sigma_0 = (
            scipy.linalg.inv(global_hyperparams.phi_0) + sum_of_outer_products
        ) / (n_groups + global_hyperparams.nu + beta_dim - 1)
    return GlobalParams(mean=mu_0, cov=Sigma_0)


def get_independent_betas_for_each_group(
    dataset_grouped: HierarchicalLogisticRegressionDataset,
) -> List[np.ndarray]:
    """
    The fitted betas if we fit each of the groups independently

    Used for comparison,  we would hope that sharing of information across groups
    leads to better estimates, at least when the sample size per group is small and when
    the group-specific betas are sufficiently tightly clustered around global betas
    """
    n_groups = len(dataset_grouped.logistic_regression_datasets)
    beta_dim = len(dataset_grouped.beta_global)

    beta_means_init = np.zeros(beta_dim)
    mean_global_init = np.zeros(beta_dim)
    cov_global_init = np.eye(beta_dim)

    beta_means_independent = [None] * n_groups
    for group in range(n_groups):
        # print(f"Now independently computing beta for group {group}/{n_groups}")
        dataset = dataset_grouped.logistic_regression_datasets[group]
        beta_means_independent[
            group
        ] = optimize_beta_for_bayesian_logreg_using_laplace_vi(
            beta_means_init,
            dataset,
            mean_global_init,
            cov_global_init,
            display_output=False,
        )
    return beta_means_independent


def compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi(
    dataset_hierarchical: HierarchicalLogisticRegressionDataset,
    beta_means_for_groups_init: np.array,
    beta_mean_global_init: np.array,
    beta_cov_global_init: np.array,
    global_hyperparams: GlobalHyperparams,
    n_iterations_for_vi: int = 10,
    n_iterations_for_global_updates: int = 10,
    display_output_vi: bool = True,
    display_output_optimization: bool = False,
) -> VariationalParams:
    """
    Definitions:
        beta_dim: the dimensionality of the beta vector
            It is the covariate dimensionality + 1 (for the intercept term)

    Arguments:
        beta_means_for_groups_init:  has shape (n_groups, beta_dim),
        beta_mean_global_init:  has shape (beta_dim, )
        beta_cov_global_init: has shape (beta_dim, )
    """

    # TODO [***]: Run this until a tolerance is reached , instead of hard-coding a number of iterations
    # Probably the best way is to monitor changes in the ELBO.  I might instead monitor changes in the
    # values of the parameters themselves.

    # TODO: Could give defaults for all the initializations.  See what I'm using.
    # Lots of zero vectors and identity matrices.

    beta_means_for_groups = beta_means_for_groups_init
    global_params = GlobalParams(beta_mean_global_init, beta_cov_global_init)

    n_groups = len(beta_means_for_groups)

    for i in range(n_iterations_for_vi):
        if display_output_vi:
            print(
                f"......Now running variational inference iteration {i}/{n_iterations_for_vi}"
            )

        for group in range(n_groups):
            if display_output_vi:
                print(
                    f"Now updating variational distribution on beta for group {group}/{n_groups}"
                )
            dataset = dataset_hierarchical.logistic_regression_datasets[group]
            beta_mean_for_this_group = (
                optimize_beta_for_bayesian_logreg_using_laplace_vi(
                    beta_means_for_groups[group],
                    dataset,
                    global_params.mean,
                    global_params.cov,
                    display_output=display_output_optimization,
                )
            )
            beta_means_for_groups[group] = beta_mean_for_this_group

        global_params = update_global_parameters(
            beta_means_for_groups,
            global_params,
            global_hyperparams,
            n_iterations_for_global_updates,
        )

    # Once we've established the global parameters, run a loop that computes the
    # the variational distribution on all the group-specific betas, by
    # computing the missing covariance term for each group-specific beta
    local_params = []
    for group in range(n_groups):
        beta_mean_for_this_group = beta_means_for_groups[group]
        beta_cov_for_this_group = get_beta_covariance(
            beta_mean=beta_mean_for_this_group,
            prior_mean=global_params.mean,
            prior_cov=global_params.cov,
            dataset=dataset_hierarchical.logistic_regression_datasets[group],
        )
        local_params.append(
            GroupParams(beta_mean_for_this_group, beta_cov_for_this_group)
        )

    return VariationalParams(global_params, local_params)
