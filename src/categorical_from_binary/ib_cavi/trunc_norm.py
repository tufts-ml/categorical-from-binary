"""
Basic expressions (mean, var, entropy) for the truncated normals with general variance 
    N_+(mu, var), i.e. a univariate normal with mean mu and variance var that is truncated from below at zero.
        We call this "normal plus"
    N_-(mu, var), i.e. a univariate normal with mean mu and variance var that is truncated from above at zero.
        We call this "normal minus"

Note:
    If var is not specified, we assume it is 1. 

References:
    - Categorical models report 
    - Burkardt, John. "The truncated normal distribution." 
        Department of Scientific Computing Website, Florida State University (2014): 1-35.
"""

from typing import Iterable, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm as norm, truncnorm, uniform as uniform

from categorical_from_binary.math import logdiffexp
from categorical_from_binary.types import NumpyArray1D


def compute_density_normal_plus(mu: float, x: Union[float, Iterable]):
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        x: The point or points at which we wish to take the density

    Returns:
        The density of the truncated distribution, evaluated at x
    """
    if not isinstance(x, Iterable):
        if x < 0:
            return 0.0
        return norm(mu, 1).pdf(x) / (1 - norm(mu, 1).cdf(0))
    else:
        densities = norm(mu, 1).pdf(x) / (1 - norm(mu, 1).cdf(0))
        densities[x < 0] = 0.0
        return densities


def compute_density_normal_minus(mu: float, x: Union[float, Iterable]):
    """
    "Normal minus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        x: The point or points at which we wish to take the density

    Returns:
        The density of the truncated distribution, evaluated at x
    """
    if not isinstance(x, Iterable):
        if x > 0:
            return 0.0
        return norm(mu, 1).pdf(x) / norm(mu, 1).cdf(0)
    else:
        densities = norm(mu, 1).pdf(x) / norm(mu, 1).cdf(0)
        densities[x > 0] = 0.0
        return densities


def compute_mean_shift_normal_plus(mu: float, var: Optional[float] = None) -> float:
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and variance var
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        var: The variance of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The "mean shift", i.e. the difference between the mean of the truncated random variable
        and the mean of the parent random variable.

        In other words, we return E[X]-mu, where X ~ N_+(mu, 1)
    """
    if var is None:
        var = 1.0

    std = np.sqrt(var)
    snr = mu / std  # signal noise ratio
    # more intutively, the return value of the computation is
    #   (norm.pdf(-snr) / (1 - norm.cdf(-snr))) * std
    # but this can lead to numerical precision issues
    return np.exp(norm.logpdf(-snr) - logdiffexp(0, norm.logcdf(-snr))) * std


def compute_mean_shift_normal_minus(mu: float, var: Optional[float] = None) -> float:
    """
    "Normal minus" is the distribution obtained by taking a univariate  normal distribution with mean mu and variance var
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        var: The variance of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The "mean shift", i.e. the difference between the mean of the truncated random variable
        and the mean of the parent random variable.

        In other words, we return E[X]-mu, where X ~ N_-(mu, 1)
    """
    if var is None:
        var = 1.0

    std = np.sqrt(var)
    snr = mu / std  # signal noise ratio
    # more intutively, the return value of the computation is
    #   (-norm.pdf(-snr) / norm.cdf(-snr)) * std
    # but this can lead to numerical precision issues
    return -np.exp(norm.logpdf(-snr) - norm.logcdf(-snr)) * std


def compute_expected_value_normal_plus(mu: float, var: Optional[float] = None) -> float:
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and variance var
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        var: The variance of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The expected value of a random variable following a normal plus distribution
    """
    if var is None:
        var = 1.0
    mean_shift = compute_mean_shift_normal_plus(mu, var)
    return mu + mean_shift


def compute_expected_value_normal_minus(
    mu: float, var: Optional[float] = None
) -> float:
    """
    "Normal minus" is the distribution obtained by taking a univariate normal distribution with mean mu and variance var
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        var: The variance of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The expected value of a random variable following a normal minus distribution
    """
    if var is None:
        var = 1.0
    mean_shift = compute_mean_shift_normal_minus(mu, var)
    return mu + mean_shift


def compute_variance_normal_plus(mu: float, var: Optional[float] = None) -> float:
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and variance var
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        var: The variance of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The variance of the truncated distribution
    """
    if var is None:
        var = 1.0

    expectation_shift = compute_mean_shift_normal_plus(mu, var)
    return var - mu * expectation_shift - expectation_shift**2


def compute_variance_normal_minus(mu: float, var: Optional[float] = None) -> float:
    """
    "Normal minus" is the distribution obtained by taking a univariate normal distribution with mean mu and variance var
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        var: The variance of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The variance of the truncated distribution
    """
    if var is None:
        var = 1.0
    expectation_shift = compute_mean_shift_normal_minus(mu, var)
    return var - mu * expectation_shift - expectation_shift**2


def compute_entropy_normal_plus(mu: float):
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The entropy of the truncated distribution
    """
    expectation_shift = compute_expected_value_normal_plus(mu) - mu
    return (
        np.log(np.sqrt(2 * np.pi * np.e) * (1 - norm.cdf(-mu)))
        - mu * expectation_shift / 2
    )


def compute_entropy_normal_minus(mu: float):
    """
    "Normal minus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        The variance of the truncated distribution
    """
    expectation_shift = compute_expected_value_normal_minus(mu) - mu
    return (
        np.log(np.sqrt(2 * np.pi * np.e) * norm.cdf(-mu)) - mu * expectation_shift / 2
    )


def compute_inv_cdf_normal_plus(mu: float, p: Union[float, np.array]):
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        p: The percentile or percentiles whose inverse we want to compute

    Returns:
        The inverse cdf of the truncated distribution at percentile p
    """
    parent = norm(mu, 1)
    # ppf is what scipy.stats calls an inverse cdf
    return parent.ppf(parent.cdf(0) + p * (1 - parent.cdf(0)))


def compute_inv_cdf_normal_minus(mu: float, p: Union[float, np.array]):
    """
    "Normal minus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        p: The percentile or percentiles whose inverse we want to compute

    Returns:
        The inverse cdf of the truncated distribution at percentile p
    """
    parent = norm(mu, 1)
    # ppf is what scipy.stats calls an inverse cdf
    return parent.ppf(p * parent.cdf(0))


def sample_normal_plus(
    mu: float,
    size: Optional[Union[int, Tuple[int]]] = None,
    random_state: Optional = None,
):
    """
    "Normal plus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN left truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.

    Returns:
        A random sample of size `size` from the Normal plus distribution
    """
    uniform_samples = uniform.rvs(size=size, random_state=random_state)
    return compute_inv_cdf_normal_plus(mu, uniform_samples)


def sample_normal_minus(
    mu: float,
    size: Optional[Union[int, Tuple[int]]] = None,
    random_state: Optional = None,
):
    """
    "Normal minus" is the distribution obtained by taking a univariate normal distribution with mean mu and sd 1
    and THEN right truncating it at 0.

    Arguments:
        mu: The expected value of the PARENT gaussian distribution (i.e., pre-truncation)
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.

    Returns:
        A random sample of size `size` from the Normal minus distribution
    """
    uniform_samples = uniform.rvs(size=size, random_state=random_state)
    return compute_inv_cdf_normal_minus(mu, uniform_samples)


def sample_from_a_normal_plus_distribution_with_non_unity_parent_standard_deviation(
    mu: Union[float, Iterable[float]],
    sigma: Union[float, Iterable[float]],
) -> Union[float, NumpyArray1D]:
    """
    An alternate method for sampling from the normal plus distribution.  Advantages over `sample_normal_plus`:
        1. Allows for drawing  multiple samples simultaneously.  Draws a single sample for each value of mu
        2. Can sample from truncated normal whose parent standard deviation is something other than 1.

    Documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

    Arguments:
        mu: The expected value(s) of the PARENT gaussian distribution (i.e., pre-truncation)
        sigma: The standard deviation(s) of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        Float if mu was float, np.array if mu was iterable
    """
    lower = 0
    upper = np.inf

    mus = list(mu)
    sigmas = list(sigma)

    compute_a = lambda mu, sigma: (lower - mu) / sigma
    compute_b = lambda mu, sigma: (upper - mu) / sigma
    a_vec = [compute_a(mu, sigma) for (mu, sigma) in zip(mus, sigmas)]
    b_vec = [compute_b(mu, sigma) for (mu, sigma) in zip(mus, sigmas)]

    samples = truncnorm.rvs(a_vec, b_vec, loc=mus, scale=sigmas, size=len(mus))

    if len(samples) == 1:
        return samples[0]
    return samples


def sample_from_a_normal_minus_distribution_with_non_unity_parent_standard_deviation(
    mu: Union[float, Iterable[float]],
    sigma: Union[float, Iterable[float]],
) -> Union[float, NumpyArray1D]:
    """
    An alternate method for sampling from the normal minus distribution.  Advantages over `sample_normal_minus`:
        1. Allows for drawing  multiple samples simultaneously.  Draws a single sample for each value of mu
        2. Can sample from truncated normal whose parent standard deviation is something other than 1.

    Documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

    Arguments:
        mu: The expected value(s) of the PARENT gaussian distribution (i.e., pre-truncation)
        sigma: The standard deviation(s) of the PARENT gaussian distribution (i.e., pre-truncation)

    Returns:
        Float if mu was float, np.array if mu was iterable
    """

    lower = -np.inf
    upper = 0

    mus = list(mu)
    sigmas = list(sigma)

    compute_a = lambda mu, sigma: (lower - mu) / sigma
    compute_b = lambda mu, sigma: (upper - mu) / sigma
    a_vec = [compute_a(mu, sigma) for (mu, sigma) in zip(mus, sigmas)]
    b_vec = [compute_b(mu, sigma) for (mu, sigma) in zip(mus, sigmas)]

    samples = truncnorm.rvs(a_vec, b_vec, loc=mus, scale=sigmas, size=len(mus))

    if len(samples) == 1:
        return samples[0]
    return samples
