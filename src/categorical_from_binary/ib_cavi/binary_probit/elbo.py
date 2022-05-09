import numpy as np
import scipy

from categorical_from_binary.ib_cavi.trunc_norm import (
    compute_density_normal_minus,
    compute_density_normal_plus,
    compute_entropy_normal_minus,
    compute_entropy_normal_plus,
    sample_normal_minus,
    sample_normal_plus,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


###
# ELBO -- as used in inference
###


def compute_variational_entropy_of_individual_entries_of_z(
    labels: NumpyArray1D,
    z_natural_parameters: NumpyArray1D,
) -> NumpyArray1D:
    """
    Returns:
        Array of shape (n_observations, ), where the i-th entry is the variational entropy
        of the latent variable associated with the i-th observation.
    """
    n_obs = np.shape(z_natural_parameters)[0]
    entropy_zs = np.zeros(n_obs)
    for i, (label, z_natural_parameter) in enumerate(zip(labels, z_natural_parameters)):
        if label == 1:
            entropy_zs[i] = compute_entropy_normal_plus(mu=z_natural_parameter)
        elif label == 0:
            entropy_zs[i] = compute_entropy_normal_minus(mu=z_natural_parameter)
        else:
            raise ValueError(
                f"Found label of {label}, but the only legal values are 0 and 1."
            )
    return entropy_zs


def compute_variational_entropy_of_z(
    labels: NumpyArray1D,
    covariates: NumpyArray2D,
    beta_mean: NumpyArray1D,
) -> float:
    """
    Returns:
        The entropy of z, the vector of latent variables associated with the observations
    """
    z_natural_parameters = covariates @ beta_mean
    return np.sum(
        compute_variational_entropy_of_individual_entries_of_z(
            labels, z_natural_parameters
        )
    )


def compute_variational_entropy_of_beta(
    beta_mean: NumpyArray1D,
    beta_cov: NumpyArray2D,
) -> float:
    return scipy.stats.multivariate_normal(beta_mean, beta_cov).entropy()


def compute_variational_entropy(
    labels: NumpyArray1D,
    covariates: NumpyArray2D,
    beta_mean: NumpyArray1D,
    beta_cov: NumpyArray2D,
) -> float:
    entropy_z = compute_variational_entropy_of_z(labels, covariates, beta_mean)
    entropy_beta = compute_variational_entropy_of_beta(beta_mean, beta_cov)
    return entropy_z + entropy_beta


def compute_variational_expectation_of_complete_data_likelihood(
    z_mean: NumpyArray1D,
    beta_mean: NumpyArray1D,
    beta_cov: NumpyArray2D,
    covariates: NumpyArray2D,
):
    N = np.shape(covariates)[0]
    x = covariates
    z_mean_shifts = z_mean - x @ beta_mean
    # `z_mean_shifts` are the amount by which the variational expected value of the
    # latent z_i's, as truncated normal random variables, differ from the expected value
    # of the pre-truncated parent distributions.
    return (
        -0.5 * N * (np.log(2 * np.pi) + 1)
        + 0.5 * np.sum(z_mean_shifts * (x @ beta_mean))
        - 0.5 * np.sum([x[i].T @ beta_cov @ x[i] for i in range(len(x))])
    )


def compute_variational_expectation_of_prior(beta_mean, beta_cov):
    """
    This is the cross entropy of two gaussians.
    Note that the form is simpler than in general since the prior is zero mean, unit variance.
    """
    M = np.shape(beta_mean)[0]
    trace_beta_cov = np.sum(np.diag(beta_cov))
    return -0.5 * (beta_mean @ beta_mean + trace_beta_cov + M * np.log(2 * np.pi))


def compute_variational_energy(
    beta_mean: NumpyArray1D,
    beta_cov: NumpyArray2D,
    z_mean: NumpyArray1D,
    covariates: NumpyArray2D,
    verbose: bool = False,
) -> float:
    print(
        f"prior energy: {compute_variational_expectation_of_prior(beta_mean, beta_cov)} "
        f"likelihood energy:  { compute_variational_expectation_of_complete_data_likelihood(z_mean, beta_mean, beta_cov, covariates)}"
    ) if verbose else None
    return compute_variational_expectation_of_prior(
        beta_mean, beta_cov
    ) + compute_variational_expectation_of_complete_data_likelihood(
        z_mean, beta_mean, beta_cov, covariates
    )


def compute_elbo(
    beta_mean: NumpyArray1D,
    beta_cov: NumpyArray2D,
    z_mean: NumpyArray1D,
    covariates: NumpyArray2D,
    labels: NumpyArray1D,
    verbose: bool = False,
):
    entropy = compute_variational_entropy(labels, covariates, beta_mean, beta_cov)
    energy = compute_variational_energy(
        beta_mean, beta_cov, z_mean, covariates, verbose
    )
    return energy + entropy


###
# Monte Carlo Approximations to ELBO -- Used in unit testing
###


def sample_z_from_natural_parameters(
    labels: NumpyArray1D, natural_parameters: NumpyArray1D, n_samples: int
):
    """
    Returns:
        np.array of shape ((n_observations, n_samples))
    """

    n_obs = np.shape(natural_parameters)[0]
    z_samples = np.zeros((n_obs, n_samples))
    for i in range(n_obs):
        if labels[i] == 1:
            z_samples[i, :] = sample_normal_plus(natural_parameters[i], size=n_samples)
        elif labels[i] == 0:
            z_samples[i, :] = sample_normal_minus(natural_parameters[i], size=n_samples)
        else:
            raise ValueError("Label must be either 0 or 1")
    return z_samples


def compute_density_of_z_entry(z: float, natural_parameter: float, label: int):
    if label == 1:
        return compute_density_normal_plus(natural_parameter, z)
    elif label == 0:
        return compute_density_normal_minus(natural_parameter, z)
    else:
        raise ValueError("Label must be either 0 or 1")


def compute_monte_carlo_approximate_entropy_for_z(
    z_samples: NumpyArray2D,
    z_natural_params: NumpyArray1D,
    labels: NumpyArray1D,
) -> float:
    """
    Arguments
        z_samples: has shape (n_observations, n_samples)
            The i-th row gives a bunch of Monte carlo samples for i-th latent variable z_i,
            for i=1,.., N observations.
        z_natural_params: has shape (n_observations, )
            The i-th entry is the variational natural parameter for the i-th distribution on z.
        labels: has shape (n_observations, )
            These are the binary observations for the probit regression

    Returns:
        An estimate of the entropy of the z-vector
    """
    approximate_entropy_for_z = 0
    for z_samples_i, label_i, z_natural_parameter_i in zip(
        z_samples, labels, z_natural_params
    ):
        # take the mean of many monte carlo samples to get an estimate
        approximate_entropy_for_z_entry = np.mean(
            -np.log(
                compute_density_of_z_entry(z_samples_i, z_natural_parameter_i, label_i)
            )
        )
        approximate_entropy_for_z += approximate_entropy_for_z_entry
    return approximate_entropy_for_z


def compute_monte_carlo_approximate_likelihood_energy(
    z_samples,
    beta_samples,
    covariates,
) -> float:
    """
    Forms a Monte Carlo approximation of the variational expectation of the complete
    data log likelihood,  i.e. sum_{i=1}^N log p(y_i, z_i \cond \beta)

    Arguments:
        z_samples: np.array of shape (n_observations, n_samples)
        beta_samples: np.array of shape (n_samples, beta_dim)
        covariates:  np.array of shape (n_observations, beta_dim)
    """
    n_observations, n_mc_samples = np.shape(z_samples)
    expected_ll_for_monte_carlo_parameter_samples = (
        np.ones(n_mc_samples) * -0.5 * n_observations * np.log(2 * np.pi)
    )
    # TODO: Re: the transpose --
    # Should I restructure z_samples using n_samples as the first array dimension?
    for s, (z_sample, beta_sample) in enumerate(zip(z_samples.T, beta_samples)):
        expected_ll_for_monte_carlo_parameter_samples[s] += -0.5 * np.sum(
            (z_sample - covariates @ beta_sample) ** 2
        )
    return np.mean(expected_ll_for_monte_carlo_parameter_samples)
