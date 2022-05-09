"""
Generate some logistic regression data
"""
from typing import List, NamedTuple

import numpy as np

from categorical_from_binary.data_generation.bayes_binary_reg import (
    BinaryRegressionDataset,
    Link,
    generate_features_via_independent_normals,
    generate_labels_via_features_and_binary_regression_weights,
)


class HierarchicalLogisticRegressionDataset(NamedTuple):
    logistic_regression_datasets: List[BinaryRegressionDataset]
    beta_global: np.ndarray  # shape: (n_features+1, ) ; 0th entry is the intercept term.


def generate_hierarchical_logistic_regression_dataset(
    n_samples: int, n_features: int, n_groups: int, seed: int = 1
) -> HierarchicalLogisticRegressionDataset:

    # generate global regression coefficients
    np.random.seed(seed)
    beta_0_global = np.random.normal(loc=0, scale=0.25)
    beta_for_features_global = np.random.normal(loc=0, scale=1, size=n_features)
    beta_global = np.concatenate(([beta_0_global], beta_for_features_global))

    log_reg_datasets = []

    for group in range(n_groups):
        # generate perturbations for each group
        # TODO: Have the betas be nonindependent.
        beta_0_noise = np.random.normal(loc=0, scale=0.1)
        beta_for_features_noise = np.random.normal(loc=0, scale=0.33, size=n_features)
        beta_noise = np.concatenate(([beta_0_noise], beta_for_features_noise))

        beta_for_this_group = beta_global + beta_noise

        features = generate_features_via_independent_normals(
            n_samples, n_features, mean=0.0, scale=1.0
        )
        labels = generate_labels_via_features_and_binary_regression_weights(
            features, beta_for_this_group, Link.LOGISTIC
        )

        log_reg_dataset_for_this_group = BinaryRegressionDataset(
            features, labels, beta_for_this_group, Link.LOGISTIC
        )
        log_reg_datasets.append(log_reg_dataset_for_this_group)

    return HierarchicalLogisticRegressionDataset(log_reg_datasets, beta_global)
