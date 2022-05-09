import numpy as np
import scipy.stats

from categorical_from_binary.kl import (
    compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument,
    compute_kl_inverse_wishart,
    compute_kl_mvn,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def test_compute_kl_mvn():
    # TODO: What else can I test about KL besides its non-negativity and 0 property?
    dim, df = 2, 3
    m0 = scipy.stats.multivariate_normal(mean=np.zeros(dim)).rvs(random_state=0)
    S0 = scipy.stats.wishart.rvs(df=df, scale=np.eye(dim), random_state=0)
    m1 = scipy.stats.multivariate_normal(mean=np.zeros(dim)).rvs(random_state=1)
    S1 = scipy.stats.wishart.rvs(df=df, scale=np.eye(dim), random_state=1)

    assert compute_kl_mvn(m0, S0, m1, S1) > 0
    assert compute_kl_mvn(m1, S1, m0, S0) > 0
    assert np.isclose(compute_kl_mvn(m0, S0, m0, S0), 0.0)
    assert np.isclose(compute_kl_mvn(m1, S1, m1, S1), 0.0)


def test_compute_kl_inverse_wishart():
    # TODO: What else can I test about KL besides its non-negativity and 0 property?
    dim = 2
    df_gen = 2
    S0 = scipy.stats.wishart.rvs(df=df_gen, scale=np.eye(dim), random_state=0)
    S1 = scipy.stats.wishart.rvs(df=df_gen, scale=np.eye(dim), random_state=1)

    df0, df1 = 4, 6
    assert compute_kl_inverse_wishart(df0, S0, df1, S1) > 0
    assert compute_kl_inverse_wishart(df1, S1, df0, S0) > 0
    assert np.isclose(compute_kl_inverse_wishart(df0, S0, df0, S0), 0.0)
    assert np.isclose(compute_kl_inverse_wishart(df1, S1, df1, S1), 0.0)


def compute_approximate_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument(
    mu_0: NumpyArray1D,
    Sigma_0: NumpyArray2D,
    m: NumpyArray1D,
    V: NumpyArray2D,
    nu: float,
    S: NumpyArray2D,
    num_mc_samples: int,
) -> float:
    """
    Use Monte Carlo samples to compute an approximation to the term computed by the function
    `compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument`

    Arguments
        ADD

    Returns:
        An estimate of ADD
    """
    mu_1_samples = scipy.stats.multivariate_normal(mean=m, cov=V).rvs(
        random_state=0, size=num_mc_samples
    )
    Sigma_1_samples = scipy.stats.invwishart.rvs(
        df=nu, scale=S, random_state=0, size=num_mc_samples
    )

    N = S.shape[0]
    log_det_Sigma_0 = np.log(np.linalg.det(Sigma_0))

    mc_sum = 0
    for (mu_1_sample, Sigma_1_sample) in zip(mu_1_samples, Sigma_1_samples):
        iSigma_1_sample = np.linalg.inv(Sigma_1_sample)
        mc_sum += 0.5 * (
            np.log(np.linalg.det(Sigma_1_sample))
            - log_det_Sigma_0
            - N
            + (mu_0 - mu_1_sample).T @ iSigma_1_sample @ (mu_0 - mu_1_sample)
            + np.trace(iSigma_1_sample @ Sigma_0)
        )

    mc_approx = mc_sum / num_mc_samples
    return mc_approx


def test_compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument():
    """
    This test checks (most of) the derivation of the propostion on expected KL divergence in the categorical models
    document, as well as its implementation.
    """
    dim, df = 2, 3
    mu_0 = scipy.stats.multivariate_normal(mean=np.zeros(dim)).rvs(random_state=0)
    Sigma_0 = scipy.stats.wishart.rvs(df=df, scale=np.eye(dim), random_state=0)
    m = scipy.stats.multivariate_normal(mean=np.zeros(dim)).rvs(random_state=0)
    V = scipy.stats.wishart.rvs(df=df, scale=np.eye(dim), random_state=0)
    S = scipy.stats.wishart.rvs(df=df, scale=np.eye(dim), random_state=0)
    nu = 10

    num_monte_carlo_samples = 10000

    expected_kl_approx = compute_approximate_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument(
        mu_0,
        Sigma_0,
        m,
        V,
        nu,
        S,
        num_monte_carlo_samples,
    )
    expected_kl_computed = compute_expected_kl_divergence_wrt_independent_Normal_and_IW_distributions_on_params_of_second_argument(
        mu_0, Sigma_0, m, V, nu, S
    )
    assert np.isclose(expected_kl_approx, expected_kl_computed, atol=0.33)
