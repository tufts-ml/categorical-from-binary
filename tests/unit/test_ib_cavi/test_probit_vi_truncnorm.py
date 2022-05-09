from typing import Callable

import numpy as np
import scipy
from scipy.integrate import simps

from categorical_from_binary.ib_cavi.trunc_norm import (
    compute_density_normal_minus,
    compute_density_normal_plus,
    compute_entropy_normal_minus,
    compute_entropy_normal_plus,
    compute_expected_value_normal_minus,
    compute_expected_value_normal_plus,
    compute_inv_cdf_normal_minus,
    compute_inv_cdf_normal_plus,
    compute_variance_normal_minus,
    compute_variance_normal_plus,
    sample_normal_minus,
    sample_normal_plus,
)


###
# Test moments of truncated normal distributions
###

# We could have just used scipy.stats.truncnorm for everything, ideally, but their parameterization
# seems to be neither in terms of the moments of the truncated distribution, nor in terms of the moments
# of the parent (non-truncated) distribution.... so seems useful in only a very limited way, namely
# to check what happens when we set the (parent) mu =0
#
# So instead we just sample from a normal and discard samples that don't meet the condition, but
# note that this limits how far away we can set mu from 0 without dealing with sampling computation
# times that could be intolerable for a unit test.


def _samples_of_normal_plus(mu, target_n_samples):
    sample_rejection_rate = 1 - scipy.stats.norm(mu, 1).cdf(0)
    num_samples_to_use = int(target_n_samples / sample_rejection_rate)
    samples_of_normal = scipy.stats.norm(mu, 1).rvs(size=num_samples_to_use)
    return samples_of_normal[samples_of_normal > 0]


def _samples_of_normal_minus(mu, target_n_samples):
    sample_rejection_rate = 1 - scipy.stats.norm(mu, 1).cdf(0)
    num_samples_to_use = int(target_n_samples / sample_rejection_rate)
    samples_of_normal = scipy.stats.norm(mu, 1).rvs(size=num_samples_to_use)
    return samples_of_normal[samples_of_normal < 0]


def test_compute_expected_value_normal_plus():
    mu_values = [-1, -0.5, 0, 0.5, 1]
    target_n_samples = 10000
    for mu in mu_values:
        sample_mean = np.mean(_samples_of_normal_plus(mu, target_n_samples))
        computed_mean = compute_expected_value_normal_plus(mu)
        assert np.isclose(sample_mean, computed_mean, atol=0.05)


def test_compute_expected_value_normal_minus():
    mu_values = [-1, -0.5, 0, 0.5, 1]
    target_n_samples = 10000
    for mu in mu_values:
        sample_mean = np.mean(_samples_of_normal_minus(mu, target_n_samples))
        computed_mean = compute_expected_value_normal_minus(mu)
        assert np.isclose(sample_mean, computed_mean, atol=0.05)


def test_compute_variance_normal_plus():
    mu_values = [-1, -0.5, 0, 0.5, 1]
    target_n_samples = 10000
    for mu in mu_values:
        sample_var = np.var(_samples_of_normal_plus(mu, target_n_samples))
        computed_var = compute_variance_normal_plus(mu)
        assert np.isclose(sample_var, computed_var, atol=0.05)


def test_compute_variance_normal_minus():
    mu_values = [-1, -0.5, 0, 0.5, 1]
    target_n_samples = 10000
    for mu in mu_values:
        sample_var = np.var(_samples_of_normal_minus(mu, target_n_samples))
        computed_var = compute_variance_normal_minus(mu)
        assert np.isclose(sample_var, computed_var, atol=0.05)


def _compute_entropy_contribution(density_at_point):
    if density_at_point == 0:
        return 0
    else:
        return -density_at_point * np.log(density_at_point)


def _numerically_approximate_entropy(mu: float, density_function: Callable):
    xs = np.linspace(-10, 10, 500)
    densities = [density_function(mu, x) for x in xs]
    entropy_contributions = [_compute_entropy_contribution(d) for d in densities]
    return simps(entropy_contributions, xs)


def test_compute_entropy_normal_plus():
    mu_values = [-1, 1]
    for mu in mu_values:
        entropy_numeric = _numerically_approximate_entropy(
            mu, compute_density_normal_plus
        )
        entropy_computed = compute_entropy_normal_plus(mu)
        assert np.isclose(entropy_computed, entropy_numeric, atol=0.005)


def test_compute_entropy_normal_minus():
    mu_values = [-1, 1]
    for mu in mu_values:
        entropy_numeric = _numerically_approximate_entropy(
            mu, compute_density_normal_minus
        )
        entropy_computed = compute_entropy_normal_minus(mu)
        assert np.isclose(entropy_computed, entropy_numeric, atol=0.005)


def test_compute_inv_cdf_normal_plus():
    for parent_mean in [-5, -2, -1, 0, 1, 2, 5]:
        # a normal distribution truncated from the left at 0 should have invcdf(0)=0
        # regardless of the mean of the parent distribution
        inv_cdf_computed = compute_inv_cdf_normal_plus(parent_mean, 0)
        inv_cdf_expected = 0.0
        assert np.isclose(inv_cdf_computed, inv_cdf_expected, atol=0.001)

        # a normal distribution truncated from the left at 0 should have invcdf(.5)>parent_mean;
        # i.e. the mean should be shifted upwards due to the truncation
        inv_cdf_computed = compute_inv_cdf_normal_plus(parent_mean, 0.5)
        assert inv_cdf_computed > parent_mean


def test_compute_inv_cdf_normal_minus():
    for parent_mean in [-5, -2, -1, 0, 1, 2, 5]:
        # a normal distribution truncated from the right at 0 should have invcdf(1)=0
        # regardless of the mean of the parent distribution
        inv_cdf_computed = compute_inv_cdf_normal_minus(parent_mean, 1)
        inv_cdf_expected = 0.0
        assert np.isclose(inv_cdf_computed, inv_cdf_expected, atol=0.001)

        # a normal distribution truncated from the right at 0 should have invcdf(.5)<parent_mean;
        # i.e. the mean should be shifted downwards due to the truncation
        inv_cdf_computed = compute_inv_cdf_normal_minus(parent_mean, 0.5)
        assert inv_cdf_computed < parent_mean


def test_sample_normal_plus():
    for parent_mean in [-5, -2, -1, 0, 1, 2, 5]:
        samples = sample_normal_plus(parent_mean, size=10000, random_state=1)
        mean_empirical = np.mean(samples)
        mean_expected = compute_expected_value_normal_plus(parent_mean)
        assert np.isclose(mean_empirical, mean_expected, atol=0.05)


def test_sample_normal_minus():
    for parent_mean in [-5, -2, -1, 0, 1, 2, 5]:
        samples = sample_normal_minus(parent_mean, size=10000, random_state=1)
        mean_empirical = np.mean(samples)
        mean_expected = compute_expected_value_normal_minus(parent_mean)
        assert np.isclose(mean_empirical, mean_expected, atol=0.05)
