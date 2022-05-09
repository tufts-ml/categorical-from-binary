import numpy as np
import pytest
import scipy
from scipy.stats import multivariate_normal as mvn

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_probit_regression_dataset,
)
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.ib_cavi.binary_probit.elbo import (
    compute_monte_carlo_approximate_entropy_for_z,
    compute_monte_carlo_approximate_likelihood_energy,
    compute_variational_entropy_of_beta,
    compute_variational_entropy_of_z,
    compute_variational_expectation_of_complete_data_likelihood,
    compute_variational_expectation_of_prior,
    sample_z_from_natural_parameters,
)
from categorical_from_binary.ib_cavi.binary_probit.inference.main import (
    compute_variational_expectation_of_z,
)
from categorical_from_binary.ib_cavi.binary_probit.inference.structs import (
    VariationalBeta,
    VariationalParams,
    VariationalZs,
)


def test_compute_variational_expectation_of_prior_function():
    # TODO: randomly sample these
    RHOS = [-0.99, -0.5, 0.0, 0.5, 0.99]
    MUS = [-2, -1, 0, 1, 2]
    VAR_SCALES = [0.1, 1.0, 10.0]

    for rho in RHOS:
        for mu1, mu2 in zip(MUS, MUS):
            for var_scale in VAR_SCALES:
                beta_mean = np.array([mu1, mu2])
                beta_cov = var_scale * np.array([[1.0, rho], [rho, 1.0]])

                beta_cross_entropy = -compute_variational_expectation_of_prior(
                    beta_mean, beta_cov
                )
                beta_entropy = scipy.stats.multivariate_normal(
                    beta_mean, beta_cov
                ).entropy()
                beta_kl = beta_cross_entropy - beta_entropy

                assert beta_kl >= 0


# small fake dataset for testing
@pytest.fixture
def dataset():
    return generate_probit_regression_dataset(n_samples=10, n_features=1, seed=10)


@pytest.fixture
def covariates(dataset):
    return prepend_features_with_column_of_all_ones_for_intercept(dataset.features)


@pytest.fixture
def labels(dataset):
    return dataset.labels


@pytest.fixture
def variational_params(covariates):
    # crappy MF posterior approximation; didn't actually run VI
    n_covariates = np.shape(covariates)[1]
    beta_mean = np.zeros(
        n_covariates,
    )
    beta_cov = np.eye(
        n_covariates,
    )
    variational_beta = VariationalBeta(beta_mean, beta_cov)
    z_natural_params = covariates @ beta_mean
    variational_zs = VariationalZs(z_natural_params)
    return VariationalParams(variational_beta, variational_zs)


@pytest.fixture
def n_samples():
    return 10000


@pytest.fixture
def beta_samples(variational_params, n_samples):
    return mvn(variational_params.beta.mean, variational_params.beta.cov).rvs(
        size=n_samples
    )


@pytest.fixture
def z_samples(variational_params, n_samples, labels):
    vp = variational_params
    return sample_z_from_natural_parameters(labels, vp.zs.parent_mean, n_samples)


def test_compute_variational_entropy_of_beta(variational_params, beta_samples):
    """The `regression weights entropy` term of the ELBO"""
    beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov
    entropy_beta_empirical = -np.mean(mvn(beta_mean, beta_cov).logpdf(beta_samples))
    entropy_beta_computed = compute_variational_entropy_of_beta(beta_mean, beta_cov)
    assert np.isclose(entropy_beta_empirical, entropy_beta_computed, atol=0.1)


def test_compute_variational_entropy_of_z(
    variational_params, labels, covariates, z_samples
):
    """The `latent variable entropy` term of the ELBO"""
    beta_mean = variational_params.beta.mean
    z_natural_params = variational_params.zs.parent_mean
    entropy_z_empirical = compute_monte_carlo_approximate_entropy_for_z(
        z_samples, z_natural_params, labels
    )
    entropy_z_computed = compute_variational_entropy_of_z(labels, covariates, beta_mean)
    assert np.isclose(entropy_z_empirical, entropy_z_computed, atol=0.1)


def test_compute_variational_expectation_of_prior(variational_params, beta_samples):
    """The `prior energy` term of the ELBO"""
    beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov
    prior_energy_beta_empirical = np.mean(mvn(beta_mean, beta_cov).logpdf(beta_samples))
    prior_energy_beta_computed = compute_variational_expectation_of_prior(
        beta_mean, beta_cov
    )
    assert np.isclose(prior_energy_beta_empirical, prior_energy_beta_computed, atol=0.1)


def test_compute_variational_expectation_of_complete_data_likelihood(
    variational_params, labels, covariates, beta_samples, z_samples
):
    """The `likelihood energy` term of the ELBO"""
    beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov
    z_mean = compute_variational_expectation_of_z(labels, covariates, beta_mean)
    likelihood_energy_computed = (
        compute_variational_expectation_of_complete_data_likelihood(
            z_mean, beta_mean, beta_cov, covariates
        )
    )
    likelihood_energy_empirical = compute_monte_carlo_approximate_likelihood_energy(
        z_samples,
        beta_samples,
        covariates,
    )
    assert np.isclose(likelihood_energy_empirical, likelihood_energy_computed, atol=0.1)
