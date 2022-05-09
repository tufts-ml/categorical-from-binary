import numpy as np

from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    HierarchicalMulticlassRegressionDataset,
    generate_hierarchical_multiclass_regression_dataset,
)


def _compute_mean_variance_in_betas_across_groups(
    hierarchical_dataset: HierarchicalMulticlassRegressionDataset,
):
    """
    Computes the variance in the beta across the groups,
    and then takes the mean across all elements of the (n_designed_features x n_categories)
    beta matrix
    """
    group_betas = np.array([dataset.beta for dataset in hierarchical_dataset.datasets])
    mean_variance = np.mean(np.var(group_betas, 0))
    return mean_variance


def test_generate_hierarchical_multiclass_regression_dataset():
    """
    We test that the function runs without error, and also that
    the beta's for each group are more dissimilar as s2 increases.
    """
    n_samples = 50
    n_features_exogenous = 5
    n_categories = 3
    n_groups = 10

    for is_autoregressive in [True, False]:
        hd = generate_hierarchical_multiclass_regression_dataset(
            n_samples,
            n_features_exogenous,
            n_categories,
            n_groups,
            s2_beta=0.1,
            is_autoregressive=is_autoregressive,
        )
        mean_variance_when_expecting_small_variance_in_beta_across_groups = (
            _compute_mean_variance_in_betas_across_groups(hd)
        )
        hd = generate_hierarchical_multiclass_regression_dataset(
            n_samples,
            n_features_exogenous,
            n_categories,
            n_groups,
            s2_beta=2.0,
            is_autoregressive=is_autoregressive,
        )
        mean_variance_when_expecting_large_variance_in_beta_across_groups = (
            _compute_mean_variance_in_betas_across_groups(hd)
        )
        assert (
            mean_variance_when_expecting_small_variance_in_beta_across_groups
            < mean_variance_when_expecting_large_variance_in_beta_across_groups
        )
