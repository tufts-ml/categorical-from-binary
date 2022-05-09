import numpy as np
import pytest
from scipy.sparse import csc_matrix

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
    compute_variational_expectation_of_z_with_dense_inputs,
    compute_variational_expectation_of_z_with_sparse_inputs,
)


def test_matching_results_for__compute_variational_expectation_of_z_with_dense_inputs__and_compute_variational_expectation_of_z_with_sparse_inputs():
    # Small dataset: M=2, K=3, N=4
    labels = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    design_matrix = np.array([[-0.3, 0.3], [-0.2, 0.2], [-10, -10], [10, 10]])
    beta_mean = np.array([[1, 0, 0], [1, 0, 0]])
    ez_true = compute_variational_expectation_of_z_with_dense_inputs(
        labels, design_matrix, beta_mean
    )
    ez_sparse = compute_variational_expectation_of_z_with_sparse_inputs(
        labels, design_matrix, beta_mean
    )
    ez_sparse = ez_sparse.toarray()
    assert np.isclose(ez_true, ez_sparse).all()

    beta_mean = np.array([[0, 0, 0], [0, 0, 0]])
    ez_true = compute_variational_expectation_of_z_with_dense_inputs(
        labels, design_matrix, beta_mean
    )
    ez_sparse = compute_variational_expectation_of_z_with_sparse_inputs(
        labels, design_matrix, beta_mean
    )
    ez_sparse = ez_sparse.toarray()
    assert np.isclose(ez_true, ez_sparse).all()


@pytest.fixture
def dataset():
    n_categories = 3
    n_features = 3
    n_samples = 1000
    include_intercept = True
    link = Link.MULTI_LOGIT  # Link.MULTI_LOGIT  # Link.CBC_PROBIT
    return generate_multiclass_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        beta_0=None,
        link=link,
        seed=None,
        include_intercept=include_intercept,
    )


def test__compute_multiclass_probit_vi_with_normal_prior__gives_same_variational_beta_mean_under_sparse_and_dense_computation(
    dataset,
):

    # Prep training / test split
    n_samples = len(dataset.labels)
    n_train_samples = int(0.8 * n_samples)
    covariates_train = dataset.features[:n_train_samples]
    labels_train = dataset.labels[:n_train_samples]
    covariates_test = dataset.features[n_train_samples:]
    labels_test = dataset.labels[n_train_samples:]

    max_n_iterations = 5

    results_dense = compute_multiclass_probit_vi_with_normal_prior(
        labels_train,
        covariates_train,
        labels_test=labels_test,
        covariates_test=covariates_test,
        max_n_iterations=max_n_iterations,
    )

    results_sparse = compute_multiclass_probit_vi_with_normal_prior(
        csc_matrix(labels_train),
        csc_matrix(covariates_train),
        labels_test=csc_matrix(labels_test),
        covariates_test=csc_matrix(covariates_test),
        max_n_iterations=max_n_iterations,
    )

    beta_mean_dense = results_dense.variational_params.beta.mean
    beta_mean_sparse = results_sparse.variational_params.beta.mean

    assert np.isclose(beta_mean_sparse.toarray(), beta_mean_dense).all()
