"""
Sanity checks for the variational approximation to the posterior of the 
hierarhcical Bayesian logistic regression.
"""

from typing import List, NamedTuple

import numpy as np

from categorical_from_binary.data_generation.hierarchical_logreg import (
    HierarchicalLogisticRegressionDataset,
)
from categorical_from_binary.laplace_vi.hierarchical_logreg.inference import (
    VariationalParams,
    get_independent_betas_for_each_group,
)


class GroupBetaEstimationData(NamedTuple):
    posterior_mean: List[np.array]
    true: List[np.array]


class GroupBetaEstimationErrors(NamedTuple):
    independent: List[float]
    hierarchical: List[float]


def sanity_check_global_betas(
    variational_params: VariationalParams,
    dataset_hierarchical: HierarchicalLogisticRegressionDataset,
) -> None:
    """
    How close did we get to the true global betas?
    Should get closer as (overall) sample size increases.
    """
    print("\n\n...Now sanity checking global betas....")
    global_beta_variational_mean = variational_params.global_.mean
    global_beta_true = dataset_hierarchical.beta_global
    print(
        f"\nglobal beta variational mean:{global_beta_variational_mean} \n  global beta true:{global_beta_true}"
    )


def sanity_check_group_betas(
    variational_params: VariationalParams,
    dataset_hierarchical: HierarchicalLogisticRegressionDataset,
) -> GroupBetaEstimationData:
    """
    How close did we get to the true group betas?
    Should get closer as sample size (within the groups) increases.
    """
    print("\n\n...Now sanity checking group betas....")
    group_beta_means_hierarchical = [
        group_params.mean for group_params in variational_params.local
    ]
    n_groups = len(group_beta_means_hierarchical)

    beta_posterior_mean_by_group, true_beta_by_group = [], []
    for group in range(n_groups):
        # TODO: represent beta consistently as a single vector across data generation and inference
        dataset = dataset_hierarchical.logistic_regression_datasets[group]
        beta_true = dataset.beta
        beta_posterior_mean = group_beta_means_hierarchical[group]
        print(
            f"\ngroup:{group} \n beta posterior mean:{ beta_posterior_mean } \n beta true value:   {beta_true}"
        )

        beta_posterior_mean_by_group.append(beta_posterior_mean)
        true_beta_by_group.append(beta_true)

    return GroupBetaEstimationData(beta_posterior_mean_by_group, true_beta_by_group)


def sanity_check_hierarchical_model(
    variational_params: VariationalParams,
    dataset_hierarchical: HierarchicalLogisticRegressionDataset,
) -> GroupBetaEstimationErrors:
    """
    Any benefit to the hierarchical model?  Compare to the independent model.
    We'd expect the hierarchical model to be helpful when n_per_group is small relative to the total n,
    and when the difference between betas across groups isn't overly large.
    """
    print("\n\n...Now sanity checking hierarchical model....")
    group_beta_means_independent = get_independent_betas_for_each_group(
        dataset_hierarchical
    )
    group_beta_means_hierarchical = [
        group_params.mean for group_params in variational_params.local
    ]
    n_groups = len(group_beta_means_hierarchical)

    errors_independent, errors_hierarchical = [], []
    for group in range(n_groups):
        # TODO: represent beta consistently as a single vector across data generation and inference
        dataset = dataset_hierarchical.logistic_regression_datasets[group]
        beta_true = dataset.beta
        print(
            f"\n\ngroup:{group} \n beta_independent:{group_beta_means_independent[group]} \n beta_hierarchical:{group_beta_means_hierarchical[group]} \n  beta_true:{beta_true} "
        )
        error_independent = np.sum(
            abs((group_beta_means_independent[group] - beta_true) / beta_true)
        )
        error_hierarchical = np.sum(
            abs((group_beta_means_hierarchical[group] - beta_true) / beta_true)
        )
        print(
            f"Error independent: {error_independent}, error hierarchical: {error_hierarchical}"
        )

        errors_independent.append(error_independent)
        errors_hierarchical.append(error_hierarchical)
    return GroupBetaEstimationErrors(errors_independent, errors_hierarchical)
